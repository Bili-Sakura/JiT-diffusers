# Training JiT in diffusers style

This example ports JiT training into a Hugging Face Diffusers + Accelerate workflow, mirroring the structure of `examples/unconditional_image_generation/train_unconditional.py` while using JiT's class-conditional transformer and flow-style sampling pipeline.

## Install dependencies

```bash
pip install -r examples/unconditional_image_generation/requirements.txt
```

If you run from the repository root, keep `src` importable so the local JiT modules are used:

```bash
set PYTHONPATH=.
```

## Dataset format

The trainer expects:

- an image column (default: `image`)
- a class-id label column (auto-detected from `label`, `labels`, `class_label`, `class`, `target`, `y`)

For local data, use `imagefolder` format with class subfolders.

## Launch training

```bash
accelerate launch examples/unconditional_image_generation/train_jit_unconditional.py ^
  --train_data_dir path\to\imagenet_like_folder ^
  --model_type "JiT-B/16" ^
  --train_batch_size 16 ^
  --num_epochs 100 ^
  --gradient_accumulation_steps 1 ^
  --learning_rate 1e-4 ^
  --lr_warmup_steps 500 ^
  --class_dropout_prob 0.1 ^
  --prediction_target sample ^
  --num_inference_steps 50 ^
  --guidance_scale 4.0 ^
  --output_dir jit-b16-checkpoints
```

## Key JiT-specific flags

- `--prediction_target`: `sample` (x0) or `velocity`
- `--class_dropout_prob`: classifier-free guidance dropout during training
- `--t_eps`: clamp for stable velocity conversion near `t=1`
- `--solver`: `heun` or `euler` for evaluation sampling
- `--model_type`: JiT preset (`JiT-B/16`, `JiT-B/32`, `JiT-L/16`, `JiT-L/32`, `JiT-H/16`, `JiT-H/32`)

## Save outputs

The script periodically saves:

- Accelerator resume states in `checkpoint-*`
- final diffusers pipeline in `output_dir` (`transformer`, `scheduler`, `model_index.json`, etc.)

You can sample from the saved model with `scripts/sample_jit.py`.
