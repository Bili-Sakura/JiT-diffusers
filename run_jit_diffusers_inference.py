import argparse
from pathlib import Path

import torch

from jit_diffusers import JiTPipeline


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-image JiT diffusers inference.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to converted diffusers model directory.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save output PNG image.")
    parser.add_argument("--class_label", type=int, default=207, help="ImageNet class id for conditional generation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--steps", type=int, default=8, help="Number of ODE sampling steps.")
    parser.add_argument("--cfg", type=float, default=2.9, help="Classifier-free guidance scale.")
    parser.add_argument("--interval_min", type=float, default=0.1, help="CFG interval min.")
    parser.add_argument("--interval_max", type=float, default=1.0, help="CFG interval max.")
    parser.add_argument("--noise_scale", type=float, default=2.0, help="Initial Gaussian noise scale.")
    parser.add_argument("--t_eps", type=float, default=5e-2, help="Small epsilon for timestep denominator.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Inference device.",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    args = get_args()
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    pipe = JiTPipeline.from_pretrained(args.model_path).to(device)
    pipe.transformer = pipe.transformer.to(device=device, dtype=torch.float32)
    pipe.transformer.eval()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    output = pipe(
        class_labels=[args.class_label],
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        guidance_interval_min=args.interval_min,
        guidance_interval_max=args.interval_max,
        noise_scale=args.noise_scale,
        t_eps=args.t_eps,
        sampling_method="heun",
        generator=generator,
        output_type="pil",
    )
    image = output.images[0]

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Saved image to: {output_path}")


if __name__ == "__main__":
    main()
