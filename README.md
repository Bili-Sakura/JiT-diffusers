# JiT Diffusers Refactor

This repository is now fully organized around a Diffusers-style package layout, following the same migration pattern used in `Bili-Sakura/NiT-diffusers`.

Legacy standalone training/evaluation codepaths have been removed so the tree is focused on reusable Diffusers components and checkpoint conversion.

## Package layout

- `src/diffusers/models/transformers/transformer_jit.py`: `JiTTransformer2DModel` (`ModelMixin`/`ConfigMixin`) class-conditional transformer.
- `src/diffusers/schedulers/scheduling_jit.py`: `JiTScheduler` with Euler/Heun flow-matching updates.
- `src/diffusers/pipelines/jit/pipeline_jit.py`: `JiTPipeline` with classifier-free guidance and native-resolution latent sampling.
- `scripts/convert_jit_to_diffusers.py`: converts legacy JiT training checkpoints to Diffusers model directories.
- `scripts/convert_diffusers_to_jit.py`: converts Diffusers JiT models back to legacy JiT checkpoint format.
- `scripts/sample_jit.py`: single-image sampling script for converted models.

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

## Notes

- This repository is intended for Diffusers integration and checkpoint conversion workflows.
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
