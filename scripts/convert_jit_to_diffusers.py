import argparse
import json
import os

from jit_diffusers import JiTDiffusersModel


def get_args():
    parser = argparse.ArgumentParser(description="Convert JiT training checkpoint to diffusers format.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to JiT checkpoint-*.pth file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for diffusers model.")
    parser.add_argument(
        "--weights",
        type=str,
        default="ema1",
        choices=["model", "ema1", "ema2"],
        help="Which checkpoint weights to export.",
    )
    parser.add_argument(
        "--safe_serialization",
        action="store_true",
        help="Save weights in safetensors format.",
    )
    parser.add_argument("--variant", type=str, default=None, help="Optional variant name passed to save_pretrained.")
    return parser.parse_args()


def main():
    args = get_args()
    model, metadata = JiTDiffusersModel.from_jit_checkpoint(
        checkpoint_path=args.checkpoint_path,
        weights=args.weights,
        map_location="cpu",
    )

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(
        args.output_dir,
        safe_serialization=args.safe_serialization,
        variant=args.variant,
    )

    metadata_path = os.path.join(args.output_dir, "conversion_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        model_type = getattr(model.config, "model_type", getattr(model.config, "model_name"))
        sample_size = getattr(model.config, "sample_size", getattr(model.config, "image_size"))
        num_class_embeds = getattr(model.config, "num_class_embeds", getattr(model.config, "num_classes"))
        attention_dropout = getattr(model.config, "attention_dropout", getattr(model.config, "attn_dropout", 0.0))
        dropout = getattr(model.config, "dropout", getattr(model.config, "proj_dropout", 0.0))
        json.dump(
            {
                "source_checkpoint": metadata["checkpoint_path"],
                "weights": metadata["weights"],
                "epoch": metadata["epoch"],
                "jit_args": {
                    "model_type": model_type,
                    "sample_size": sample_size,
                    "num_class_embeds": num_class_embeds,
                    "attention_dropout": attention_dropout,
                    "dropout": dropout,
                    "model": model_type,
                    "img_size": sample_size,
                    "class_num": num_class_embeds,
                    "attn_dropout": attention_dropout,
                    "proj_dropout": dropout,
                },
            },
            f,
            indent=2,
        )

    print(f"Saved diffusers model to: {args.output_dir}")


if __name__ == "__main__":
    main()
