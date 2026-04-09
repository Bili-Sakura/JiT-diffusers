from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Literal, Tuple

import torch
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from model_jit import JiT_models


def _extract_module_state_dict(
    state_dict: Dict[str, torch.Tensor], prefixes: Tuple[str, ...] = ("transformer.", "net.")
) -> Dict[str, torch.Tensor]:
    """Extract a module state dict by stripping the first fully-matching prefix.

    Prefix precedence is left-to-right; `"transformer."` is preferred over legacy `"net."`.
    """
    for prefix in prefixes:
        if all(key.startswith(prefix) for key in state_dict.keys()):
            return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def _build_jit_kwargs(
    image_size: int,
    num_classes: int,
    attn_dropout: float,
    proj_dropout: float,
    model_name: str | None = None,
) -> Dict[str, object]:
    _ = model_name
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
    if isinstance(ckpt_args, argparse.Namespace):
        args_dict = vars(ckpt_args)
    elif isinstance(ckpt_args, dict):
        args_dict = ckpt_args
    else:
        raise TypeError(f"Unsupported checkpoint args type: {type(ckpt_args)}")

    def _get_first_available(*keys: str, default=None):
        for key in keys:
            if key in args_dict and args_dict[key] is not None:
                return args_dict[key]
        return default

    model_name = _get_first_available("model", "model_name", "model_type")
    image_size = _get_first_available("img_size", "image_size", "sample_size")
    num_classes = _get_first_available("class_num", "num_classes", "num_class_embeds")
    if model_name is None or image_size is None or num_classes is None:
        raise ValueError("Checkpoint args are missing model/image_size/num_classes information.")

    return JiTCheckpointConfig(
        model_name=str(model_name),
        image_size=int(image_size),
        num_classes=int(num_classes),
        attn_dropout=float(_get_first_available("attn_dropout", "attention_dropout", default=0.0)),
        proj_dropout=float(_get_first_available("proj_dropout", "dropout", default=0.0)),
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
        model_type: str | None = None,
        sample_size: int | None = None,
        num_class_embeds: int | None = None,
        attention_dropout: float | None = None,
        dropout: float | None = None,
    ):
        super().__init__()
        resolved_model_type = model_name if model_type is None else model_type
        resolved_sample_size = image_size if sample_size is None else sample_size
        resolved_num_class_embeds = num_classes if num_class_embeds is None else num_class_embeds
        resolved_attention_dropout = attn_dropout if attention_dropout is None else attention_dropout
        resolved_dropout = proj_dropout if dropout is None else dropout

        if resolved_model_type not in JiT_models:
            raise ValueError(f"Unknown model '{resolved_model_type}'. Available: {list(JiT_models.keys())}")

        self.transformer = JiT_models[resolved_model_type](
            **_build_jit_kwargs(
                model_name=resolved_model_type,
                image_size=resolved_sample_size,
                num_classes=resolved_num_class_embeds,
                attn_dropout=resolved_attention_dropout,
                proj_dropout=resolved_dropout,
            )
        )

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor, class_labels: torch.Tensor, return_dict: bool = True):
        timestep = torch.as_tensor(timestep, device=sample.device)
        if timestep.ndim == 0:
            timestep = timestep.repeat(sample.shape[0])
        else:
            timestep = timestep.reshape(-1)
            if timestep.shape[0] == 1 and sample.shape[0] > 1:
                timestep = timestep.repeat(sample.shape[0])

        denoised = self.transformer(sample, timestep, class_labels)
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
            model_type=config.model_name,
            sample_size=config.image_size,
            num_class_embeds=config.num_classes,
            attention_dropout=config.attn_dropout,
            dropout=config.proj_dropout,
        )

        key = "model" if weights == "model" else f"model_{weights}"
        if key not in checkpoint:
            raise ValueError(f"Checkpoint key '{key}' not found. Available keys: {list(checkpoint.keys())}")

        model_state = _extract_module_state_dict(checkpoint[key])
        model.transformer.load_state_dict(model_state, strict=strict)

        metadata = {
            "checkpoint_path": checkpoint_path,
            "weights": weights,
            "epoch": checkpoint.get("epoch"),
            "source_args": checkpoint.get("args"),
        }
        return model, metadata

    def to_jit_checkpoint(
        self,
        ema_mode: Literal["none", "copy_to_both"] = "copy_to_both",
        prefix: str = "net.",
    ) -> Dict[str, object]:
        base_state = {f"{prefix}{k}": v.detach().cpu() for k, v in self.transformer.state_dict().items()}
        checkpoint = {"model": base_state}
        if ema_mode == "copy_to_both":
            checkpoint["model_ema1"] = {k: v.clone() for k, v in base_state.items()}
            checkpoint["model_ema2"] = {k: v.clone() for k, v in base_state.items()}
        elif ema_mode != "none":
            raise ValueError(f"Unsupported ema_mode='{ema_mode}'.")
        return checkpoint

    @property
    def net(self):
        return self.transformer

    @net.setter
    def net(self, module):
        self.transformer = module
