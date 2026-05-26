import argparse
import logging
import math
import os
import shutil
from datetime import timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from datasets import ClassLabel, load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_accelerate_version, is_tensorboard_available, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from src.diffusers import JiTPipeline, JiTScheduler, JiTTransformer2DModel
from src.diffusers.models.transformers.jit_weights import JIT_PRESET_CONFIGS


check_min_version("0.35.0")
logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Train JiT with a diffusers-style Accelerate loop.")
    parser.add_argument("--dataset_name", type=str, default=None, help="Hugging Face dataset name.")
    parser.add_argument("--dataset_config_name", type=str, default=None, help="Dataset config name.")
    parser.add_argument("--train_data_dir", type=str, default=None, help="Path to training folder (imagefolder format).")
    parser.add_argument("--cache_dir", type=str, default=None, help="Dataset/model cache directory.")
    parser.add_argument("--output_dir", type=str, default="jit-model", help="Output directory for checkpoints and model.")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--image_column", type=str, default="image", help="Name of the dataset image column.")
    parser.add_argument(
        "--label_column",
        type=str,
        default=None,
        help="Name of class label column. If omitted, auto-detected from common names.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="Optional pretrained JiT model directory; if provided, loads transformer from this path.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="JiT-B/16",
        choices=sorted(JIT_PRESET_CONFIGS.keys()),
        help="JiT architecture preset when training from scratch.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Train image resolution. Defaults to preset sample_size when omitted.",
    )
    parser.add_argument("--num_classes", type=int, default=None, help="Override class count. Auto-inferred if omitted.")
    parser.add_argument("--center_crop", action="store_true", help="Use center crop instead of random crop.")
    parser.add_argument("--random_flip", action="store_true", help="Random horizontal flip.")
    parser.add_argument("--preserve_input_precision", action="store_true", help="Keep 16/32-bit input precision.")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Per-device train batch size.")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="Batch size for evaluation sampling.")
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--num_epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--save_images_epochs", type=int, default=10, help="Image sampling frequency by epoch.")
    parser.add_argument("--save_model_epochs", type=int, default=10, help="Model saving frequency by epoch.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Optimizer learning rate.")
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help='Scheduler type in ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"].',
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Learning-rate warmup steps.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="AdamW beta1.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="AdamW beta2.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="AdamW epsilon.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument("--use_ema", action="store_true", help="Enable EMA model weights.")
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0, help="EMA inverse gamma.")
    parser.add_argument("--ema_power", type=float, default=0.75, help="EMA power.")
    parser.add_argument("--ema_max_decay", type=float, default=0.9999, help="EMA max decay.")
    parser.add_argument("--class_dropout_prob", type=float, default=0.1, help="CFG label dropout probability during training.")
    parser.add_argument("--t_eps", type=float, default=5e-2, help="Clamp minimum for (1 - t) in velocity conversion.")
    parser.add_argument("--noise_scale", type=float, default=1.0, help="Initial latent noise scale.")
    parser.add_argument("--min_timestep", type=float, default=0.0, help="Minimum sampled t in [0, 1).")
    parser.add_argument("--max_timestep", type=float, default=1.0, help="Maximum sampled t in (0, 1].")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Sampling steps for eval images.")
    parser.add_argument("--guidance_scale", type=float, default=4.0, help="CFG scale for eval sampling.")
    parser.add_argument("--guidance_interval_min", type=float, default=0.1, help="Lower bound of CFG interval.")
    parser.add_argument("--guidance_interval_max", type=float, default=1.0, help="Upper bound of CFG interval.")
    parser.add_argument("--solver", type=str, default="heun", choices=["heun", "euler"], help="Sampler solver.")
    parser.add_argument(
        "--prediction_target",
        type=str,
        default="sample",
        choices=["sample", "velocity"],
        help="Train JiT to predict x0 (`sample`) or flow velocity (`velocity`).",
    )
    parser.add_argument("--checkpointing_steps", type=int, default=500, help="Save Accelerator state every N steps.")
    parser.add_argument("--checkpoints_total_limit", type=int, default=None, help="Max retained checkpoints.")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help='Resume from checkpoint path or "latest".',
    )
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Enable xFormers attention.")
    parser.add_argument("--push_to_hub", action="store_true", help="Upload output directory to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="Hugging Face Hub token.")
    parser.add_argument("--hub_model_id", type=str, default=None, help="Hub repo id.")
    parser.add_argument("--hub_private_repo", action="store_true", help="Create private Hub repo.")
    parser.add_argument("--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb"], help="Experiment logger.")
    parser.add_argument("--logging_dir", type=str, default="logs", help="Tracker logging directory.")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training.")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], help="Mixed precision mode.")
    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Provide either `--dataset_name` or `--train_data_dir`.")
    if not 0.0 <= args.class_dropout_prob < 1.0:
        raise ValueError("`--class_dropout_prob` must be in [0, 1).")
    if not 0.0 <= args.min_timestep < args.max_timestep <= 1.0:
        raise ValueError("Need 0 <= min_timestep < max_timestep <= 1.")
    if args.t_eps <= 0:
        raise ValueError("`--t_eps` must be positive.")

    return args


def _ensure_three_channels(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    channels = tensor.shape[0]
    if channels == 3:
        return tensor
    if channels == 1:
        return tensor.repeat(3, 1, 1)
    if channels == 2:
        return torch.cat([tensor, tensor[:1]], dim=0)
    if channels > 3:
        return tensor[:3]
    raise ValueError(f"Unsupported number of channels: {channels}")


def _auto_label_column(dataset) -> Optional[str]:
    candidates = ["label", "labels", "class_label", "class", "target", "y"]
    for candidate in candidates:
        if candidate in dataset.column_names:
            return candidate
    return None


def _build_label_maps(dataset, label_column: str, num_classes_override: Optional[int]) -> Tuple[int, Dict[int, str]]:
    features = dataset.features
    id2label = {}
    if label_column in features and isinstance(features[label_column], ClassLabel):
        names = features[label_column].names
        id2label = {idx: name for idx, name in enumerate(names)}
        inferred = len(names)
    else:
        unique_labels = sorted(set(dataset[label_column]))
        inferred = int(max(unique_labels)) + 1 if unique_labels else 0
        id2label = {int(i): str(i) for i in range(inferred)}

    num_classes = int(num_classes_override) if num_classes_override is not None else inferred
    if num_classes <= 0:
        raise ValueError("`num_classes` must be positive.")
    return num_classes, id2label


def _build_model(args, num_classes: int) -> JiTTransformer2DModel:
    if args.model_name_or_path is not None:
        return JiTTransformer2DModel.from_pretrained(args.model_name_or_path)

    config = dict(JIT_PRESET_CONFIGS[args.model_type])
    if args.resolution is not None:
        config["sample_size"] = int(args.resolution)
    config["num_classes"] = num_classes
    config["model_type"] = args.model_type
    return JiTTransformer2DModel(**config)


def _compute_training_target(
    noisy: torch.Tensor,
    clean: torch.Tensor,
    noise: torch.Tensor,
    t: torch.Tensor,
    t_eps: float,
    target_type: str,
) -> torch.Tensor:
    if target_type == "sample":
        return clean
    if target_type == "velocity":
        denom = (1.0 - t).clamp_min(t_eps)
        while denom.ndim < noisy.ndim:
            denom = denom.unsqueeze(-1)
        return (clean - noisy) / denom
    raise ValueError(f"Unsupported target type: {target_type}")


def _make_noisy_input(clean_images: torch.Tensor, noise_scale: float, min_t: float, max_t: float):
    bsz = clean_images.shape[0]
    noise = torch.randn_like(clean_images) * noise_scale
    t = torch.rand((bsz,), device=clean_images.device, dtype=clean_images.dtype)
    t = t * (max_t - min_t) + min_t
    while t.ndim < clean_images.ndim:
        t = t.unsqueeze(-1)
    noisy = (1.0 - t) * noise + t * clean_images
    return noisy, noise, t


def main(args):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if args.logger == "tensorboard":
        if not is_tensorboard_available():
            raise ImportError("Install tensorboard to use `--logger tensorboard`.")
    elif args.logger == "wandb":
        if not is_wandb_available():
            raise ImportError("Install wandb to use `--logger wandb`.")
        import wandb

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_model.save_pretrained(os.path.join(output_dir, "transformer_ema"))
                for _ in range(len(models)):
                    model = models.pop()
                    model.save_pretrained(os.path.join(output_dir, "transformer"))
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "transformer_ema"), JiTTransformer2DModel
                )
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model

            while len(models) > 0:
                model = models.pop()
                load_model = JiTTransformer2DModel.from_pretrained(input_dir, subfolder="transformer")
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
                private=args.hub_private_repo,
            ).repo_id

    if args.dataset_name is not None:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            split="train",
        )
    else:
        dataset = load_dataset("imagefolder", data_dir=args.train_data_dir, cache_dir=args.cache_dir, split="train")

    label_column = args.label_column or _auto_label_column(dataset)
    if label_column is None:
        raise ValueError(
            "Unable to infer label column. Pass `--label_column` and ensure labels are integer class ids."
        )
    if label_column not in dataset.column_names:
        raise ValueError(f"Label column '{label_column}' not found. Available: {dataset.column_names}")
    if args.image_column not in dataset.column_names:
        raise ValueError(f"Image column '{args.image_column}' not found. Available: {dataset.column_names}")

    num_classes, id2label = _build_label_maps(dataset, label_column=label_column, num_classes_override=args.num_classes)
    model = _build_model(args, num_classes=num_classes)

    if args.resolution is not None and int(model.config.sample_size) != int(args.resolution):
        raise ValueError(
            f"Configured model sample_size={model.config.sample_size} does not match requested resolution={args.resolution}."
        )
    args.resolution = int(model.config.sample_size)
    null_class_id = int(num_classes)

    if args.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=args.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            model_cls=JiTTransformer2DModel,
            model_config=model.config,
        )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning("xFormers 0.0.16 may be unstable during training. Consider upgrading to >=0.0.17.")
            if hasattr(model, "enable_xformers_memory_efficient_attention"):
                model.enable_xformers_memory_efficient_attention()
            else:
                logger.warning("JiTTransformer2DModel has no xFormers toggle; skipping.")
        else:
            raise ValueError("xformers is not available; install it before enabling this flag.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    spatial_augmentations = [
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
        transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
    ]

    augmentations = transforms.Compose(
        spatial_augmentations
        + [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    precision_augmentations = transforms.Compose(
        [
            transforms.PILToTensor(),
            transforms.Lambda(_ensure_three_channels),
            transforms.ConvertImageDtype(torch.float32),
        ]
        + spatial_augmentations
        + [transforms.Normalize([0.5], [0.5])]
    )

    def transform_images(examples):
        processed = []
        labels = []
        for image, label in zip(examples[args.image_column], examples[label_column]):
            if not args.preserve_input_precision:
                processed.append(augmentations(image.convert("RGB")))
            else:
                precise_image = image
                if precise_image.mode == "P":
                    precise_image = precise_image.convert("RGB")
                processed.append(precision_augmentations(precise_image))
            labels.append(int(label))
        return {"input": processed, "class_labels": labels}

    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"Image column: {args.image_column}; label column: {label_column}; classes: {num_classes}")
    dataset.set_transform(transform_images)
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, lr_scheduler)

    if args.use_ema:
        ema_model.to(accelerator.device)

    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    logger.info(f"  Prediction target = {args.prediction_target}")

    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint.rstrip("/\\"))
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint-")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting from scratch.")
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    for epoch in range(first_epoch, args.num_epochs):
        model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            clean_images = batch["input"].to(weight_dtype)
            labels = batch["class_labels"].to(device=clean_images.device, dtype=torch.long)
            labels = labels.clamp(min=0, max=num_classes - 1)

            noisy_images, pure_noise, t = _make_noisy_input(
                clean_images,
                noise_scale=args.noise_scale,
                min_t=args.min_timestep,
                max_t=args.max_timestep,
            )
            target = _compute_training_target(
                noisy=noisy_images,
                clean=clean_images,
                noise=pure_noise,
                t=t,
                t_eps=args.t_eps,
                target_type=args.prediction_target,
            )

            t_flat = t.reshape(-1)
            if args.class_dropout_prob > 0.0:
                drop_mask = torch.rand_like(t_flat) < args.class_dropout_prob
                labels_train = labels.clone()
                labels_train[drop_mask] = null_class_id
            else:
                labels_train = labels

            with accelerator.accumulate(model):
                model_output = model(noisy_images, timestep=t_flat, class_labels=labels_train).sample
                loss = F.mse_loss(model_output.float(), target.float())

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(model.parameters())
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                    if args.checkpoints_total_limit is not None:
                        checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[:num_to_remove]
                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} old checkpoints"
                            )
                            for removing_checkpoint in removing_checkpoints:
                                shutil.rmtree(os.path.join(args.output_dir, removing_checkpoint))

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            if args.use_ema:
                logs["ema_decay"] = float(ema_model.cur_decay_value)
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

        progress_bar.close()
        accelerator.wait_for_everyone()

        if accelerator.is_main_process and (epoch % args.save_images_epochs == 0 or epoch == args.num_epochs - 1):
            transformer = accelerator.unwrap_model(model)
            if args.use_ema:
                ema_model.store(transformer.parameters())
                ema_model.copy_to(transformer.parameters())

            pipeline = JiTPipeline(
                transformer=transformer,
                scheduler=JiTScheduler(t_eps=args.t_eps, solver=args.solver),
                id2label=id2label,
            )
            pipeline = pipeline.to(accelerator.device)
            generator = torch.Generator(device=pipeline.device).manual_seed(0)

            if len(id2label) > 0:
                eval_classes = list(np.linspace(0, len(id2label) - 1, num=args.eval_batch_size, dtype=int))
            else:
                eval_classes = [0] * args.eval_batch_size

            images = pipeline(
                class_labels=eval_classes,
                guidance_scale=args.guidance_scale,
                guidance_interval_min=args.guidance_interval_min,
                guidance_interval_max=args.guidance_interval_max,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                output_type="np",
            ).images

            if args.use_ema:
                ema_model.restore(transformer.parameters())

            images_processed = (images * 255).round().astype("uint8")
            if args.logger == "tensorboard":
                if is_accelerate_version(">=", "0.17.0.dev0"):
                    tracker = accelerator.get_tracker("tensorboard", unwrap=True)
                else:
                    tracker = accelerator.get_tracker("tensorboard")
                tracker.add_images("test_samples", images_processed.transpose(0, 3, 1, 2), epoch)
            elif args.logger == "wandb":
                accelerator.get_tracker("wandb").log(
                    {"test_samples": [wandb.Image(img) for img in images_processed], "epoch": epoch},
                    step=global_step,
                )

        if accelerator.is_main_process and (epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1):
            transformer = accelerator.unwrap_model(model)
            if args.use_ema:
                ema_model.store(transformer.parameters())
                ema_model.copy_to(transformer.parameters())

            pipeline = JiTPipeline(
                transformer=transformer,
                scheduler=JiTScheduler(t_eps=args.t_eps, solver=args.solver),
                id2label=id2label,
            )
            pipeline.save_pretrained(args.output_dir)

            if args.use_ema:
                ema_model.restore(transformer.parameters())

            if args.push_to_hub:
                upload_folder(
                    repo_id=repo_id,
                    folder_path=args.output_dir,
                    commit_message=f"Epoch {epoch}",
                    ignore_patterns=["checkpoint-*", "logs/*", "runs/*"],
                )

    accelerator.end_training()


if __name__ == "__main__":
    parsed_args = parse_args()
    main(parsed_args)
