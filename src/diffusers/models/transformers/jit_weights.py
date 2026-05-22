"""State-dict remapping from legacy JiT checkpoints to native JiTTransformer2DModel keys."""

from __future__ import annotations

from typing import Dict

import torch

JIT_PRESET_CONFIGS: Dict[str, Dict[str, object]] = {
    "JiT-B/16": {
        "sample_size": 256,
        "patch_size": 16,
        "hidden_size": 768,
        "num_layers": 12,
        "num_attention_heads": 12,
        "bottleneck_dim": 128,
        "in_context_len": 32,
        "in_context_start": 4,
        "attention_dropout": 0.0,
        "dropout": 0.0,
    },
    "JiT-B/32": {
        "sample_size": 512,
        "patch_size": 32,
        "hidden_size": 768,
        "num_layers": 12,
        "num_attention_heads": 12,
        "bottleneck_dim": 128,
        "in_context_len": 32,
        "in_context_start": 4,
        "attention_dropout": 0.0,
        "dropout": 0.0,
    },
    "JiT-L/16": {
        "sample_size": 256,
        "patch_size": 16,
        "hidden_size": 1024,
        "num_layers": 24,
        "num_attention_heads": 16,
        "bottleneck_dim": 128,
        "in_context_len": 32,
        "in_context_start": 8,
        "attention_dropout": 0.0,
        "dropout": 0.0,
    },
    "JiT-L/32": {
        "sample_size": 512,
        "patch_size": 32,
        "hidden_size": 1024,
        "num_layers": 24,
        "num_attention_heads": 16,
        "bottleneck_dim": 128,
        "in_context_len": 32,
        "in_context_start": 8,
        "attention_dropout": 0.0,
        "dropout": 0.0,
    },
    "JiT-H/16": {
        "sample_size": 256,
        "patch_size": 16,
        "hidden_size": 1280,
        "num_layers": 32,
        "num_attention_heads": 16,
        "bottleneck_dim": 256,
        "in_context_len": 32,
        "in_context_start": 10,
        "attention_dropout": 0.0,
        "dropout": 0.2,
    },
    "JiT-H/32": {
        "sample_size": 512,
        "patch_size": 32,
        "hidden_size": 1280,
        "num_layers": 32,
        "num_attention_heads": 16,
        "bottleneck_dim": 256,
        "in_context_len": 32,
        "in_context_start": 10,
        "attention_dropout": 0.0,
        "dropout": 0.2,
    },
}


def remap_legacy_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    remapped: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "transformer.", "net."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
                break

        new_key = new_key.replace(".adaLN_modulation.1.", ".adaLN_modulation.")
        if new_key.startswith("final_layer."):
            new_key = new_key.replace("final_layer.norm_final", "norm_final")
            new_key = new_key.replace("final_layer.linear", "linear_final")
            new_key = new_key.replace("final_layer.adaLN_modulation", "adaLN_modulation_final")

        remapped[new_key] = value
    return remapped
