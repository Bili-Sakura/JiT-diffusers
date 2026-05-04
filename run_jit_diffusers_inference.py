import argparse
from pathlib import Path

import torch

from jit_diffusers import JiTPipeline


RECOMMENDED_CFG_BY_MODEL = {
    "JiT-B/16": 3.0,
    "JiT-L/16": 2.4,
    "JiT-H/16": 2.2,
    "JiT-B/32": 3.0,
    "JiT-L/32": 2.5,
    "JiT-H/32": 2.3,
}

RECOMMENDED_NOISE_BY_RESOLUTION = {
    256: 1.0,
    512: 2.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-image JiT diffusers inference.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to converted diffusers model directory.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save output PNG image.")
    parser.add_argument("--class_label", type=int, default=207, help="ImageNet class id for conditional generation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--steps", type=int, default=50, help="Number of ODE sampling steps.")
    parser.add_argument(
        "--cfg",
        type=float,
        default=None,
        help="Classifier-free guidance scale. Defaults to paper recommendation for the loaded model.",
    )
    parser.add_argument("--interval_min", type=float, default=0.1, help="CFG interval min.")
    parser.add_argument("--interval_max", type=float, default=1.0, help="CFG interval max.")
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=None,
        help="Initial Gaussian noise scale. Defaults to paper recommendation for the loaded resolution.",
    )
    parser.add_argument("--t_eps", type=float, default=5e-2, help="Small epsilon for timestep denominator.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Inference device.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp32"],
        help="Inference dtype. Defaults to bf16 on CUDA.",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="scheduler",
        choices=["scheduler", "heun", "euler"],
        help="Sampling solver. Use scheduler to keep pipeline default.",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    return torch.float32


def resolve_generation_defaults(pipe: JiTPipeline, cfg: float | None, noise_scale: float | None) -> tuple[float, float]:
    model_type = str(getattr(pipe.transformer.config, "model_type", ""))
    sample_size = int(getattr(pipe.transformer.config, "sample_size", 256))
    resolved_cfg = cfg if cfg is not None else RECOMMENDED_CFG_BY_MODEL.get(model_type, 2.9)
    resolved_noise_scale = noise_scale if noise_scale is not None else RECOMMENDED_NOISE_BY_RESOLUTION.get(sample_size, 1.0)
    return resolved_cfg, resolved_noise_scale


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    pipe = JiTPipeline.from_pretrained(args.model_path).to(device)
    pipe.transformer = pipe.transformer.to(device=device, dtype=dtype)
    pipe.transformer.eval()
    sampling_method = None if args.solver == "scheduler" else args.solver
    cfg, noise_scale = resolve_generation_defaults(pipe, args.cfg, args.noise_scale)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    output = pipe(
        class_labels=[args.class_label],
        num_inference_steps=args.steps,
        guidance_scale=cfg,
        guidance_interval_min=args.interval_min,
        guidance_interval_max=args.interval_max,
        noise_scale=noise_scale,
        t_eps=args.t_eps,
        sampling_method=sampling_method,
        generator=generator,
        output_type="pil",
    )
    image = output.images[0]

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Used sampling hyperparameters: cfg={cfg}, noise_scale={noise_scale}")
    print(f"Saved image to: {output_path}")


if __name__ == "__main__":
    main()
