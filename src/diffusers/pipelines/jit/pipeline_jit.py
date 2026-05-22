from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor

from ...models.transformers.transformer_jit import JiTTransformer2DModel
from ...schedulers.scheduling_jit import JiTScheduler

RECOMMENDED_NOISE_BY_SIZE = {
    256: 1.0,
    512: 2.0,
}


class JiTPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "transformer"

    def __init__(
        self,
        transformer: JiTTransformer2DModel,
        scheduler: JiTScheduler | None = None,
        id2label: Optional[Dict[int, str]] = None,
        id2label_cn: Optional[Dict[int, str]] = None,
    ):
        super().__init__()
        self.register_modules(transformer=transformer, scheduler=scheduler or JiTScheduler())
        self._id2label = id2label or {}
        self._id2label_cn = id2label_cn or {}
        self.labels = self._build_label2id(self._id2label)
        self.labels_cn = self._build_label2id(self._id2label_cn)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        model_kwargs = dict(kwargs)
        transformer_subfolder = model_kwargs.pop("transformer_subfolder", None)
        scheduler_subfolder = model_kwargs.pop("scheduler_subfolder", None)
        scheduler_kwargs = model_kwargs.pop("scheduler_kwargs", {})
        base_path = Path(pretrained_model_name_or_path)

        if transformer_subfolder is None and (base_path / "transformer").exists():
            transformer_subfolder = "transformer"
        if scheduler_subfolder is None and (base_path / "scheduler").exists():
            scheduler_subfolder = "scheduler"

        try:
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        except Exception:
            if transformer_subfolder is not None:
                transformer_path = str(base_path / transformer_subfolder)
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

            id2label, id2label_cn = cls._load_labels_for_variant(str(base_path))
            return cls(
                transformer=transformer,
                scheduler=scheduler,
                id2label=id2label,
                id2label_cn=id2label_cn,
            )

    def _ensure_labels_loaded(self) -> None:
        if self._id2label or self._id2label_cn:
            return
        loaded_en, loaded_cn = self._load_labels_for_variant(getattr(self.config, "_name_or_path", None))
        if loaded_en:
            self._id2label = loaded_en
            self.labels = self._build_label2id(self._id2label)
        if loaded_cn:
            self._id2label_cn = loaded_cn
            self.labels_cn = self._build_label2id(self._id2label_cn)

    @staticmethod
    def _labels_dir_for_variant(variant_path: Optional[str]) -> Optional[Path]:
        if not variant_path:
            return None
        variant_dir = Path(variant_path).resolve()
        local_labels_dir = variant_dir / "labels"
        if local_labels_dir.is_dir():
            return local_labels_dir
        parent_labels_dir = variant_dir.parent / "labels"
        if parent_labels_dir.is_dir():
            return parent_labels_dir
        return None

    @staticmethod
    def _read_id2label(labels_dir: Path, lang: str = "en") -> Dict[int, str]:
        filename = "id2label_en.json" if lang == "en" else "id2label_cn.json"
        path = labels_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {int(key): value for key, value in raw.items()}

    @classmethod
    def _load_labels_for_variant(
        cls,
        variant_path: Optional[str],
    ) -> Tuple[Optional[Dict[int, str]], Optional[Dict[int, str]]]:
        labels_dir = cls._labels_dir_for_variant(variant_path)
        if labels_dir is None:
            return None, None
        try:
            return cls._read_id2label(labels_dir, "en"), cls._read_id2label(labels_dir, "cn")
        except FileNotFoundError:
            return None, None

    @staticmethod
    def _build_label2id(id2label: Dict[int, str]) -> Dict[str, int]:
        label2id: Dict[str, int] = {}
        for class_id, value in id2label.items():
            for synonym in value.split(","):
                synonym = synonym.strip()
                if synonym:
                    label2id[synonym] = int(class_id)
        return dict(sorted(label2id.items()))

    @property
    def id2label(self) -> Dict[int, str]:
        self._ensure_labels_loaded()
        return self._id2label

    @property
    def id2label_cn(self) -> Dict[int, str]:
        self._ensure_labels_loaded()
        return self._id2label_cn

    def get_label_ids(self, label: Union[str, List[str]], lang: str = "en") -> List[int]:
        if lang not in ("en", "cn"):
            raise ValueError(f"`lang` must be 'en' or 'cn', got {lang!r}.")

        self._ensure_labels_loaded()
        label2id = self.labels if lang == "en" else self.labels_cn
        if not label2id:
            raise ValueError(
                f"No {lang} labels loaded. Ensure `labels/id2label_{lang}.json` exists next to the model directory."
            )

        if isinstance(label, str):
            label = [label]

        missing = [item for item in label if item not in label2id]
        if missing:
            preview = ", ".join(list(label2id.keys())[:8])
            raise ValueError(f"Unknown label(s) for lang={lang!r}: {missing}. Example valid labels: {preview}, ...")
        return [label2id[item] for item in label]

    def _normalize_class_labels(
        self,
        class_labels: Union[int, str, List[Union[int, str]]],
    ) -> List[int]:
        if isinstance(class_labels, int):
            return [class_labels]

        if isinstance(class_labels, str):
            return self.get_label_ids(class_labels)

        if class_labels and isinstance(class_labels[0], str):
            self._ensure_labels_loaded()
            if all(label in self.labels for label in class_labels):
                return self.get_label_ids(class_labels, lang="en")
            if all(label in self.labels_cn for label in class_labels):
                return self.get_label_ids(class_labels, lang="cn")
            raise ValueError(
                "Could not resolve string `class_labels`. Use English synonyms from `pipe.labels` "
                "or Chinese synonyms from `pipe.labels_cn`."
            )

        return list(class_labels)

    def _predict_velocity(
        self,
        z_value: torch.Tensor,
        t: torch.Tensor,
        class_labels: torch.Tensor,
        class_null: torch.Tensor,
        do_classifier_free_guidance: bool,
        guidance_scale: float,
        guidance_interval_min: float,
        guidance_interval_max: float,
    ) -> torch.Tensor:
        t = torch.as_tensor(t, device=z_value.device, dtype=z_value.dtype)
        if do_classifier_free_guidance:
            z_in = torch.cat([z_value, z_value], dim=0)
            labels = torch.cat([class_labels, class_null], dim=0)
        else:
            z_in = z_value
            labels = class_labels

        t_batch = t.flatten().expand(z_in.shape[0])
        x_pred = self.transformer(z_in, timestep=t_batch, class_labels=labels).sample
        v = self.scheduler.velocity_from_prediction(z_in, x_pred, t)

        if not do_classifier_free_guidance:
            return v

        v_cond, v_uncond = v.chunk(2, dim=0)
        interval_mask = t < guidance_interval_max
        if guidance_interval_min != 0.0:
            interval_mask = interval_mask & (t > guidance_interval_min)
        scale = torch.where(
            interval_mask,
            torch.tensor(guidance_scale, device=z_value.device, dtype=z_value.dtype),
            torch.tensor(1.0, device=z_value.device, dtype=z_value.dtype),
        )
        return v_uncond + scale * (v_cond - v_uncond)

    def _run_sampler(
        self,
        latents: torch.Tensor,
        class_labels: torch.Tensor,
        class_null: torch.Tensor,
        num_inference_steps: int,
        do_classifier_free_guidance: bool,
        guidance_scale: float,
        guidance_interval_min: float,
        guidance_interval_max: float,
        sampling_method: str,
    ) -> torch.Tensor:
        device = latents.device
        self.scheduler.set_timesteps(num_inference_steps, device=device, solver=sampling_method)
        timesteps = self.scheduler.timesteps

        for i in self.progress_bar(range(num_inference_steps - 1)):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            v = self._predict_velocity(
                latents,
                t,
                class_labels,
                class_null,
                do_classifier_free_guidance,
                guidance_scale,
                guidance_interval_min,
                guidance_interval_max,
            )

            if sampling_method == "heun":
                latents_euler = latents + (t_next - t) * v
                v_next = self._predict_velocity(
                    latents_euler,
                    t_next,
                    class_labels,
                    class_null,
                    do_classifier_free_guidance,
                    guidance_scale,
                    guidance_interval_min,
                    guidance_interval_max,
                )
                latents = self.scheduler.step(v, t, latents, model_output_next=v_next).prev_sample
            else:
                latents = self.scheduler.step(v, t, latents).prev_sample

        t = timesteps[-2]
        t_next = timesteps[-1]
        v = self._predict_velocity(
            latents,
            t,
            class_labels,
            class_null,
            do_classifier_free_guidance,
            guidance_scale,
            guidance_interval_min,
            guidance_interval_max,
        )
        return latents + (t_next - t) * v

    @torch.inference_mode()
    def __call__(
        self,
        class_labels: Union[int, str, List[Union[int, str]]],
        guidance_scale: Optional[float] = None,
        guidance_interval_min: float = 0.1,
        guidance_interval_max: float = 1.0,
        noise_scale: Optional[float] = None,
        t_eps: Optional[float] = None,
        sampling_method: Optional[str] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 50,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        solver = sampling_method or self.scheduler.config.solver
        if solver not in {"heun", "euler"}:
            raise ValueError("sampling_method must be one of: 'heun', 'euler'.")
        if num_inference_steps < 2:
            raise ValueError("num_inference_steps must be >= 2.")
        if output_type not in {"pil", "np", "pt"}:
            raise ValueError("output_type must be one of: 'pil', 'np', 'pt'.")

        if t_eps is not None:
            self.scheduler.register_to_config(t_eps=t_eps)

        class_label_ids = self._normalize_class_labels(class_labels)
        do_classifier_free_guidance = guidance_scale is not None and guidance_scale > 1.0

        batch_size = len(class_label_ids)
        image_size = int(self.transformer.config.sample_size)
        channels = int(self.transformer.config.in_channels)
        null_class_val = int(
            getattr(self.transformer.config, "num_classes", getattr(self.transformer.config, "num_class_embeds", 1000))
        )

        if guidance_scale is None:
            guidance_scale = 1.0
        if noise_scale is None:
            noise_scale = RECOMMENDED_NOISE_BY_SIZE.get(image_size, 1.0)

        latents = randn_tensor(
            shape=(batch_size, channels, image_size, image_size),
            generator=generator,
            device=self._execution_device,
            dtype=self.transformer.dtype,
        ) * noise_scale

        class_labels_t = torch.tensor(class_label_ids, device=self._execution_device, dtype=torch.long).reshape(-1)
        class_labels_t = class_labels_t.clamp(0, null_class_val - 1)
        class_null = torch.full_like(class_labels_t, null_class_val)

        latents = self._run_sampler(
            latents,
            class_labels_t,
            class_null,
            num_inference_steps,
            do_classifier_free_guidance,
            guidance_scale,
            guidance_interval_min,
            guidance_interval_max,
            solver,
        )

        images_pt = ((latents.float().clamp(-1, 1) + 1.0) / 2.0).cpu()
        if output_type == "pt":
            images = images_pt
        elif output_type == "np":
            images = images_pt.permute(0, 2, 3, 1).numpy()
        else:
            images = self.numpy_to_pil(images_pt.permute(0, 2, 3, 1).numpy())

        self.maybe_free_model_hooks()

        if not return_dict:
            return (images,)
        return ImagePipelineOutput(images=images)


JiTPipelineOutput = ImagePipelineOutput
