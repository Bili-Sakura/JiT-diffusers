from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from diffusers import DiffusionPipeline
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor

from .modeling_jit_transformer_2d import JiTTransformer2DModel
from .scheduling_jit import JiTScheduler


@dataclass
class JiTPipelineOutput(BaseOutput):
    images: List["PIL.Image.Image"] | np.ndarray | torch.Tensor


class JiTPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "transformer"

    def __init__(self, transformer: JiTTransformer2DModel, scheduler: JiTScheduler | None = None):
        super().__init__()
        self.register_modules(transformer=transformer, scheduler=scheduler or JiTScheduler())

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        model_kwargs = dict(kwargs)
        transformer_subfolder = model_kwargs.pop("transformer_subfolder", None)
        scheduler_subfolder = model_kwargs.pop("scheduler_subfolder", None)
        scheduler_kwargs = model_kwargs.pop("scheduler_kwargs", {})
        if transformer_subfolder is not None:
            transformer_path = str(Path(pretrained_model_name_or_path) / transformer_subfolder)
        else:
            transformer_path = pretrained_model_name_or_path
        transformer = JiTTransformer2DModel.from_pretrained(transformer_path, **model_kwargs)
        try:
            scheduler = JiTScheduler.from_pretrained(
                pretrained_model_name_or_path,
                subfolder=scheduler_subfolder,
                **scheduler_kwargs,
            )
        except Exception:
            scheduler = JiTScheduler(**scheduler_kwargs)
        return cls(transformer=transformer, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        class_labels: int | List[int] | torch.Tensor,
        num_inference_steps: int = 50,
        guidance_scale: float = 2.9,
        guidance_interval_min: float = 0.1,
        guidance_interval_max: float = 1.0,
        noise_scale: float = 2.0,
        t_eps: float = 5e-2,
        sampling_method: str | None = None,
        generator: torch.Generator | List[torch.Generator] | None = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> JiTPipelineOutput | ImagePipelineOutput | Tuple:
        if output_type not in {"pil", "np", "pt"}:
            raise ValueError("output_type must be one of: 'pil', 'np', 'pt'.")
        if sampling_method is not None and sampling_method not in {"heun", "euler"}:
            raise ValueError("sampling_method must be one of: 'heun', 'euler'.")
        if num_inference_steps < 2:
            raise ValueError("num_inference_steps must be >= 2.")
        if sampling_method is not None and sampling_method != self.scheduler.config.solver:
            self.scheduler = JiTScheduler.from_config(self.scheduler.config, solver=sampling_method)

        if isinstance(class_labels, int):
            class_labels = [class_labels]
        if isinstance(class_labels, list):
            class_labels = torch.tensor(class_labels, device=self._execution_device, dtype=torch.long)
        else:
            class_labels = class_labels.to(self._execution_device, dtype=torch.long).reshape(-1)

        batch_size = class_labels.shape[0]
        latent_size = int(self.transformer.config.sample_size)
        latent_channels = int(getattr(self.transformer.config, "in_channels", 3))
        num_classes = int(self.transformer.config.num_class_embeds)

        class_labels = class_labels.clamp(0, num_classes - 1)
        class_null = torch.full_like(class_labels, num_classes)

        latents = randn_tensor(
            shape=(batch_size, latent_channels, latent_size, latent_size),
            generator=generator,
            device=self._execution_device,
            dtype=self.transformer.dtype,
        ) * noise_scale
        self.scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=self._execution_device)
        timesteps = self.scheduler.timesteps.to(device=self._execution_device, dtype=latents.dtype)

        def forward_cfg(z_value: torch.Tensor, t: torch.Tensor | float) -> torch.Tensor:
            t = torch.as_tensor(t, device=self._execution_device, dtype=latents.dtype)
            x_cond = self.transformer(sample=z_value, timestep=t.flatten(), class_labels=class_labels).sample
            v_cond = (x_cond - z_value) / (1.0 - t).clamp_min(t_eps)

            x_uncond = self.transformer(sample=z_value, timestep=t.flatten(), class_labels=class_null).sample
            v_uncond = (x_uncond - z_value) / (1.0 - t).clamp_min(t_eps)

            interval_mask = (t < guidance_interval_max) & (t > guidance_interval_min)
            scale = torch.where(
                interval_mask,
                torch.tensor(guidance_scale, device=self._execution_device, dtype=latents.dtype),
                torch.tensor(1.0, device=self._execution_device, dtype=latents.dtype),
            )
            return v_uncond + scale * (v_cond - v_uncond)

        for i in self.progress_bar(range(num_inference_steps - 1)):
            t, t_next = timesteps[i], timesteps[i + 1]
            model_output = forward_cfg(latents, t)
            if self.scheduler.config.solver == "heun":
                latents = self.scheduler.step(
                    model_output=model_output,
                    timestep=t,
                    next_timestep=t_next,
                    sample=latents,
                    model_fn=forward_cfg,
                ).prev_sample
            else:
                latents = self.scheduler.step(
                    model_output=model_output,
                    timestep=t,
                    next_timestep=t_next,
                    sample=latents,
                ).prev_sample

        # Match the original JiT implementation: always use Euler for the final step.
        t, t_next = timesteps[-2], timesteps[-1]
        model_output = forward_cfg(latents, t)
        latents = self.scheduler.euler_step(
            model_output=model_output,
            timestep=t,
            next_timestep=t_next,
            sample=latents,
        ).prev_sample

        images_pt = ((latents.float().clamp(-1, 1) + 1.0) / 2.0).cpu()
        if output_type == "pt":
            images = images_pt
        else:
            images_np = images_pt.permute(0, 2, 3, 1).numpy()
            if output_type == "np":
                images = images_np
            else:
                images = self.numpy_to_pil(images_np)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (images,)
        return JiTPipelineOutput(images=images)
