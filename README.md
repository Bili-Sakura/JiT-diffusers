# JiT Diffusers Refactor

This repository is now fully organized around a Diffusers-style package layout, following the same migration pattern used in `Bili-Sakura/NiT-diffusers`.

It now includes a Diffusers + Accelerate training entrypoint for JiT in addition to reusable components and checkpoint conversion scripts.

## Package layout

- `src/diffusers/models/transformers/transformer_jit.py`: `JiTTransformer2DModel` (`ModelMixin`/`ConfigMixin`) class-conditional transformer.
- `src/diffusers/schedulers/scheduling_jit.py`: `JiTScheduler` with Euler/Heun flow-matching updates.
- `src/diffusers/pipelines/jit/pipeline_jit.py`: `JiTPipeline` with classifier-free guidance and native-resolution latent sampling.
- `scripts/convert_jit_to_diffusers.py`: converts legacy JiT training checkpoints to Diffusers model directories.
- `scripts/convert_diffusers_to_jit.py`: converts Diffusers JiT models back to legacy JiT checkpoint format.
- `scripts/sample_jit.py`: single-image sampling script for converted models.
- `examples/unconditional_image_generation/train_jit_unconditional.py`: Diffusers-style Accelerate training script for class-conditional JiT.
- `examples/unconditional_image_generation/README.md`: training commands and dataset requirements.

## Convert a checkpoint

```bash
python scripts/convert_jit_to_diffusers.py \
  --checkpoint_path checkpoints/checkpoint-last.pth \
  --output_dir jit-diffusers \
  --weights ema1 \
  --safe_serialization
```

The generated `conversion_metadata.json` includes both Diffusers-style fields and JiT legacy aliases for compatibility.

## Convert back to legacy checkpoint

```bash
python scripts/convert_diffusers_to_jit.py \
  --model_path jit-diffusers \
  --output_path checkpoint-converted.pth \
  --ema_mode copy_to_both
```

## Sample

```bash
python scripts/sample_jit.py \
  --model jit-diffusers \
  --output demo.png \
  --class-label 207 \
  --num-inference-steps 50 \
  --solver heun
```

## Train

```bash
accelerate launch examples/unconditional_image_generation/train_jit_unconditional.py \
  --train_data_dir path/to/data \
  --model_type "JiT-B/16" \
  --output_dir jit-train-out
```

## Notes

- This repository is intended for Diffusers integration, training, and checkpoint conversion workflows.
- For direct upstreaming, copy files under `src/diffusers` into matching paths in `huggingface/diffusers` and register lazy imports there.

## Citation

```bibtex
@article{li2025jit,
  title={Back to Basics: Let Denoising Generative Models Denoise},
  author={Li, Tianhong and He, Kaiming},
  journal={arXiv preprint arXiv:2511.13720},
  year={2025}
}
```
