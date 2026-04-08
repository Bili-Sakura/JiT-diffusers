import argparse
import os

import torch

from jit_diffusers import JiTDiffusersModel


def get_args():
    parser = argparse.ArgumentParser(description="Convert diffusers JiT model back to JiT checkpoint format.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to diffusers model directory.")
    parser.add_argument("--output_path", type=str, required=True, help="Output JiT checkpoint path (.pth).")
    parser.add_argument(
        "--no_ema",
        action="store_true",
        help="Do not include model_ema1/model_ema2 in converted checkpoint.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=0,
        help="Epoch value to store in the output checkpoint.",
    )
    return parser.parse_args()


def main():
    args = get_args()

    model = JiTDiffusersModel.from_pretrained(args.model_path)
    checkpoint = model.to_jit_checkpoint(include_ema=not args.no_ema)
    checkpoint["epoch"] = args.epoch
    checkpoint["optimizer"] = {}
    checkpoint["args"] = argparse.Namespace(
        model=model.config.model_name,
        img_size=model.config.image_size,
        class_num=model.config.num_classes,
        attn_dropout=model.config.attn_dropout,
        proj_dropout=model.config.proj_dropout,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    torch.save(checkpoint, args.output_path)
    print(f"Saved JiT checkpoint to: {args.output_path}")


if __name__ == "__main__":
    main()
