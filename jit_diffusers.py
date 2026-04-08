from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Literal, Tuple

import torch
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from model_jit import JiT_models


def _extract_net_state_dict(state_dict: Dict[str, torch.Tensor], prefix: str = "net.") -> Dict[str, torch.Tensor]:
    if all(key.startswith(prefix) for key in state_dict.keys()):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def _build_jit_kwargs(
    model_name: str,
    image_size: int,
    num_classes: int,
    attn_dropout: float,
    proj_dropout: float,
) -> Dict[str, object]:
    return {
        "input_size": image_size,
        "in_channels": 3,
        "num_classes": num_classes,
        "attn_drop": attn_dropout,
        "proj_drop": proj_dropout,
    }


@dataclass
class JiTCheckpointConfig:
    model_name: str
    image_size: int
    num_classes: int
    attn_dropout: float
    proj_dropout: float


def _config_from_checkpoint(ckpt_args: argparse.Namespace) -> JiTCheckpointConfig:
    return JiTCheckpointConfig(
        model_name=getattr(ckpt_args, "model"),
        image_size=int(getattr(ckpt_args, "img_size")),
        num_classes=int(getattr(ckpt_args, "class_num")),
        attn_dropout=float(getattr(ckpt_args, "attn_dropout", 0.0)),
        proj_dropout=float(getattr(ckpt_args, "proj_dropout", 0.0)),
    )


class JiTDiffusersModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        model_name: str = "JiT-B/16",
        image_size: int = 256,
        num_classes: int = 1000,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        if model_name not in JiT_models:
            raise ValueError(f"Unknown model '{model_name}'. Available: {list(JiT_models.keys())}")

        self.net = JiT_models[model_name](
            **_build_jit_kwargs(
                model_name=model_name,
                image_size=image_size,
                num_classes=num_classes,
                attn_dropout=attn_dropout,
                proj_dropout=proj_dropout,
            )
        )

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor, class_labels: torch.Tensor, return_dict: bool = True):
        if timestep.ndim == 0:
            timestep = timestep[None].expand(sample.shape[0])
        elif timestep.ndim > 1:
            timestep = timestep.reshape(-1)

        denoised = self.net(sample, timestep, class_labels)
        if not return_dict:
            return (denoised,)
        return Transformer2DModelOutput(sample=denoised)

    @classmethod
    def from_jit_checkpoint(
        cls,
        checkpoint_path: str,
        weights: Literal["model", "ema1", "ema2"] = "ema1",
        map_location: str = "cpu",
        strict: bool = True,
    ) -> Tuple["JiTDiffusersModel", Dict[str, object]]:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        if "args" not in checkpoint:
            raise ValueError("Checkpoint is missing 'args', cannot infer JiT architecture config.")

        config = _config_from_checkpoint(checkpoint["args"])
        model = cls(
            model_name=config.model_name,
            image_size=config.image_size,
            num_classes=config.num_classes,
            attn_dropout=config.attn_dropout,
            proj_dropout=config.proj_dropout,
        )

        key = "model" if weights == "model" else f"model_{weights}"
        if key not in checkpoint:
            raise ValueError(f"Checkpoint key '{key}' not found. Available keys: {list(checkpoint.keys())}")

        model_state = _extract_net_state_dict(checkpoint[key])
        model.net.load_state_dict(model_state, strict=strict)

        metadata = {
            "checkpoint_path": checkpoint_path,
            "weights": weights,
            "epoch": checkpoint.get("epoch"),
            "source_args": checkpoint.get("args"),
        }
        return model, metadata

    def to_jit_checkpoint(
        self,
        include_ema: bool = True,
        prefix: str = "net.",
    ) -> Dict[str, object]:
        base_state = {f"{prefix}{k}": v.detach().cpu() for k, v in self.net.state_dict().items()}
        checkpoint = {"model": base_state}
        if include_ema:
            checkpoint["model_ema1"] = {k: v.clone() for k, v in base_state.items()}
            checkpoint["model_ema2"] = {k: v.clone() for k, v in base_state.items()}
        return checkpoint
