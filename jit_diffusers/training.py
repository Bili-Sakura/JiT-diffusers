from __future__ import annotations

import copy
import math
import os
import shutil
import sys
from collections.abc import Mapping

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch_fidelity

import util.lr_sched as lr_sched
import util.misc as misc

from .modeling_jit_transformer_2d import JiTTransformer2DModel
from .scheduling_jit import JiTScheduler


def remap_training_state_dict_keys(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map legacy JiT checkpoint keys to Diffusers-native training keys."""

    remapped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        normalized = key
        if normalized.startswith("module."):
            normalized = normalized[len("module.") :]

        if normalized.startswith("net."):
            normalized = f"transformer.transformer.{normalized[len('net.'):]}"
        elif normalized.startswith("transformer.transformer."):
            pass
        elif normalized.startswith("transformer."):
            normalized = f"transformer.{normalized}"
        else:
            normalized = f"transformer.transformer.{normalized}"

        remapped[normalized] = value
    return remapped


class JiTDiffusersDenoiser(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.transformer = JiTTransformer2DModel(
            model_type=args.model,
            sample_size=args.img_size,
            num_class_embeds=args.class_num,
            attention_dropout=args.attn_dropout,
            dropout=args.proj_dropout,
        )
        self.img_size = args.img_size
        self.num_classes = args.class_num

        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        # EMA tracking
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1: list[torch.Tensor] | None = None
        self.ema_params2: list[torch.Tensor] | None = None

        # Generation hyperparameters
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

    def drop_labels(self, labels: torch.Tensor) -> torch.Tensor:
        drop_mask = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        return torch.where(drop_mask, torch.full_like(labels, self.num_classes), labels)

    def sample_t(self, n: int, device: torch.device | None = None) -> torch.Tensor:
        gaussian = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(gaussian)

    @staticmethod
    def _expand_scalar_timestep(timestep: torch.Tensor | float, batch_size: int, device: torch.device, dtype: torch.dtype):
        timestep = torch.as_tensor(timestep, device=device, dtype=dtype).reshape(-1)
        if timestep.shape[0] == 1 and batch_size > 1:
            timestep = timestep.repeat(batch_size)
        return timestep

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels_dropped = self.drop_labels(labels) if self.training else labels

        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)

        x_pred = self.transformer(
            sample=z,
            timestep=t.flatten(),
            class_labels=labels_dropped,
            return_dict=True,
        ).sample
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

        loss = (v - v_pred).pow(2).mean(dim=(1, 2, 3)).mean()
        return loss

    @torch.no_grad()
    def _forward_cfg(self, latents: torch.Tensor, timestep: torch.Tensor | float, labels: torch.Tensor) -> torch.Tensor:
        timestep_batch = self._expand_scalar_timestep(
            timestep=timestep,
            batch_size=latents.shape[0],
            device=latents.device,
            dtype=latents.dtype,
        )
        x_cond = self.transformer(
            sample=latents,
            timestep=timestep_batch,
            class_labels=labels,
            return_dict=True,
        ).sample
        v_cond = (x_cond - latents) / (1.0 - timestep_batch.view(-1, 1, 1, 1)).clamp_min(self.t_eps)

        null_labels = torch.full_like(labels, self.num_classes)
        x_uncond = self.transformer(
            sample=latents,
            timestep=timestep_batch,
            class_labels=null_labels,
            return_dict=True,
        ).sample
        v_uncond = (x_uncond - latents) / (1.0 - timestep_batch.view(-1, 1, 1, 1)).clamp_min(self.t_eps)

        low, high = self.cfg_interval
        guidance_mask = timestep_batch < high
        if low != 0:
            guidance_mask = guidance_mask & (timestep_batch > low)
        guidance_scale = torch.where(
            guidance_mask,
            torch.full_like(timestep_batch, self.cfg_scale),
            torch.ones_like(timestep_batch),
        ).view(-1, 1, 1, 1)
        return v_uncond + guidance_scale * (v_cond - v_uncond)

    @torch.no_grad()
    def generate(self, labels: torch.Tensor) -> torch.Tensor:
        device = labels.device
        batch_size = labels.size(0)
        latents = self.noise_scale * torch.randn(batch_size, 3, self.img_size, self.img_size, device=device)

        scheduler = JiTScheduler(solver=self.method)
        scheduler.set_timesteps(num_inference_steps=self.steps, device=device)
        timesteps = scheduler.timesteps.to(device=device, dtype=latents.dtype)

        for index in range(self.steps - 1):
            t, t_next = timesteps[index], timesteps[index + 1]
            model_output = self._forward_cfg(latents, t, labels)
            if scheduler.config.solver == "heun":
                latents = scheduler.step(
                    model_output=model_output,
                    timestep=t,
                    next_timestep=t_next,
                    sample=latents,
                    model_fn=lambda sample, step_t: self._forward_cfg(sample, step_t, labels),
                ).prev_sample
            else:
                latents = scheduler.step(
                    model_output=model_output,
                    timestep=t,
                    next_timestep=t_next,
                    sample=latents,
                ).prev_sample

        # Keep parity with the reference implementation: final step is Euler.
        t, t_next = timesteps[-2], timesteps[-1]
        model_output = self._forward_cfg(latents, t, labels)
        latents = scheduler.euler_step(
            model_output=model_output,
            timestep=t,
            next_timestep=t_next,
            sample=latents,
        ).prev_sample
        return latents

    @torch.no_grad()
    def update_ema(self):
        if self.ema_params1 is None or self.ema_params2 is None:
            raise RuntimeError("EMA params are not initialized.")

        source_params = list(self.parameters())
        for target, source in zip(self.ema_params1, source_params):
            target.detach().mul_(self.ema_decay1).add_(source, alpha=1 - self.ema_decay1)
        for target, source in zip(self.ema_params2, source_params):
            target.detach().mul_(self.ema_decay2).add_(source, alpha=1 - self.ema_decay2)

    @torch.no_grad()
    def initialize_ema_from_model(self):
        self.ema_params1 = copy.deepcopy(list(self.parameters()))
        self.ema_params2 = copy.deepcopy(list(self.parameters()))


def train_one_epoch(model, model_without_ddp, data_loader, optimizer, device, epoch, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print(f"log_dir: {log_writer.log_dir}")

    for data_iter_step, (x, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # Normalize image to [-1, 1]
        x = x.to(device, non_blocking=True).to(torch.float32).div_(255)
        x = x * 2.0 - 1.0
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x, labels)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        model_without_ddp.update_ema()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if data_iter_step % args.log_freq == 0:
                log_writer.add_scalar("train_loss", loss_value_reduce, epoch_1000x)
                log_writer.add_scalar("lr", lr, epoch_1000x)


def evaluate(model_without_ddp, args, epoch, batch_size=64, log_writer=None):
    model_without_ddp.eval()
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    num_steps = args.num_images // (batch_size * world_size) + 1

    save_folder = os.path.join(
        args.output_dir,
        "{}-steps{}-cfg{}-interval{}-{}-image{}-res{}".format(
            model_without_ddp.method,
            model_without_ddp.steps,
            model_without_ddp.cfg_scale,
            model_without_ddp.cfg_interval[0],
            model_without_ddp.cfg_interval[1],
            args.num_images,
            args.img_size,
        ),
    )
    print("Save to:", save_folder)
    if misc.get_rank() == 0 and not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Switch to the first EMA.
    model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    for index, (name, _value) in enumerate(model_without_ddp.named_parameters()):
        ema_state_dict[name] = model_without_ddp.ema_params1[index]
    print("Switch to ema")
    model_without_ddp.load_state_dict(ema_state_dict)

    class_num = args.class_num
    assert args.num_images % class_num == 0, "Number of images per class must be the same"
    class_label_gen_world = np.arange(0, class_num).repeat(args.num_images // class_num)
    class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])

    for step_id in range(num_steps):
        print(f"Generation step {step_id}/{num_steps}")

        start_idx = world_size * batch_size * step_id + local_rank * batch_size
        end_idx = start_idx + batch_size
        labels_gen = class_label_gen_world[start_idx:end_idx]
        labels_gen = torch.Tensor(labels_gen).long().cuda()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            sampled_images = model_without_ddp.generate(labels_gen)

        torch.distributed.barrier()

        sampled_images = (sampled_images + 1) / 2
        sampled_images = sampled_images.detach().cpu()

        for batch_index in range(sampled_images.size(0)):
            img_id = (
                step_id * sampled_images.size(0) * world_size
                + local_rank * sampled_images.size(0)
                + batch_index
            )
            if img_id >= args.num_images:
                break
            generated_image = np.round(np.clip(sampled_images[batch_index].numpy().transpose([1, 2, 0]) * 255, 0, 255))
            generated_image = generated_image.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(save_folder, f"{str(img_id).zfill(5)}.png"), generated_image)

    torch.distributed.barrier()

    print("Switch back from ema")
    model_without_ddp.load_state_dict(model_state_dict)

    if log_writer is not None:
        if args.img_size == 256:
            fid_statistics_file = "fid_stats/jit_in256_stats.npz"
        elif args.img_size == 512:
            fid_statistics_file = "fid_stats/jit_in512_stats.npz"
        else:
            raise NotImplementedError
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=save_folder,
            input2=None,
            fid_statistics_file=fid_statistics_file,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=False,
        )
        fid = metrics_dict["frechet_inception_distance"]
        inception_score = metrics_dict["inception_score_mean"]
        postfix = f"_cfg{model_without_ddp.cfg_scale}_res{args.img_size}"
        log_writer.add_scalar(f"fid{postfix}", fid, epoch)
        log_writer.add_scalar(f"is{postfix}", inception_score, epoch)
        print(f"FID: {fid:.4f}, Inception Score: {inception_score:.4f}")
        shutil.rmtree(save_folder)

    torch.distributed.barrier()
