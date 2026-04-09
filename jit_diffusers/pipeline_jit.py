from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput

from .modeling_jit_transformer_2d import JiTTransformer2DModel


@dataclass
class JiTPipelineOutput(BaseOutput):
    images: torch.Tensor


class JiTPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "transformer"

    def __init__(self, transformer: JiTTransformer2DModel):
        super().__init__()
        self.register_modules(transformer=transformer)

    @torch.no_grad()
    def __call__(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        class_labels: torch.Tensor,
        return_dict: bool = True,
    ) -> JiTPipelineOutput | Tuple[torch.Tensor]:
        output = self.transformer(
            sample=sample,
            timestep=timestep,
            class_labels=class_labels,
            return_dict=return_dict,
        )
        if not return_dict:
            if isinstance(output, tuple):
                return output
            return (output.sample,)
        return JiTPipelineOutput(images=output.sample)
