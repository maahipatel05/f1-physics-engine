"""
train.py
========
Main training script for the Language-Conditioned SVD model.

Features
--------
- HuggingFace Accelerate for single-GPU and multi-GPU/mixed-precision training
- Min-SNR loss weighting (Hang et al., 2023) for stable convergence
- Sliding-window cosine LR schedule with warm restarts
- Checkpoint saving (LoRA weights + optimiser state)
- Optional WandB logging
- Gradient checkpointing + xformers for VRAM efficiency
- Compatible with SLURM (see slurm/train.sh)

Usage
-----
Single GPU:
    python train.py --config configs/default.yaml

With Accelerate (multi-GPU):
    accelerate launch --num_processes=4 train.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import DDPMScheduler, EulerDiscreteScheduler
from diffusers.training_utils import compute_snr
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm.auto import tqdm

from data import F1VideoDataset, SyntheticF1Dataset, create_dataloader
from models import build_model, LanguageConditionedSVD

logger = get_logger(__name__, log_level="INFO")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Language-Conditioned SVD for F1 Physics Simulation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to OmegaConf YAML config file.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use SyntheticF1Dataset instead of real data (for debugging).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume from.",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Optional WandB run name.",
    )
    return parser.parse_args()


def compute_loss_with_snr(
    noise_pred: torch.Tensor,
    noise_target: torch.Tensor,
    timesteps: torch.Tensor,
    noise_scheduler: DDPMScheduler,
    snr_gamma: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute the denoising loss, optionally with min-SNR weighting.

    Parameters
    ----------
    noise_pred    : [B, F, 4, H, W] predicted noise
    noise_target  : [B, F, 4, H, W] actual noise added
    timesteps     : [B] int64 timestep indices
    noise_scheduler
    snr_gamma     : if not None, applies min-SNR weighting
    """
    if snr_gamma is None:
        # Standard unweighted MSE
        return F.mse_loss(noise_pred.float(), noise_target.float(), reduction="mean")

    # Compute SNR weights per timestep
    snr = compute_snr(noise_scheduler, timesteps)                     # [B]
    mse_loss_weights = torch.minimum(snr, torch.full_like(snr, snr_gamma)) / snr

    # Per-element MSE then weighted mean over the batch
    loss = F.mse_loss(noise_pred.float(), noise_target.float(), reduction="none")
    # loss shape: [B, F, 4, H, W] → mean over everything except batch
    loss = loss.mean(dim=list(range(1, loss.ndim)))  # [B]
    loss = (loss * mse_loss_weights).mean()
    return loss


def build_added_time_ids(
    batch_size: int,
    fps_id: int = 10,
    motion_bucket_id: int = 127,
    noise_aug_strength: float = 0.02,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Build the SVD 'added_time_ids' tensor used for temporal conditioning.

    Returns [B, 3] float tensor.
    """
    ids = torch.tensor(
        [fps_id, motion_bucket_id, noise_aug_strength],
        dtype=dtype,
        device=device,
    )
    return ids.unsqueeze(0).repeat(batch_size, 1)


def save_checkpoint(
    model: LanguageConditionedSVD,
    accelerator: Accelerator,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: object,
    cfg: DictConfig,
    global_step: int,
) -> None:
    """Save LoRA weights, optimizer, and scheduler state."""
    ckpt_dir = Path(cfg.training.output_dir) / f"checkpoint-{global_step:07d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Unwrap from Accelerate to get the base module
    unwrapped: LanguageConditionedSVD = accelerator.unwrap_model(model)
    unwrapped.save_lora_weights(str(ckpt_dir))

    # Save optimiser + scheduler state via Accelerate
    accelerator.save_state(str(ckpt_dir / "accelerate_state"))

    # Also save a "latest" symlink / text file for easy resumption
    latest_path = Path(cfg.training.output_dir) / "latest"
    latest_path.write_text(str(ckpt_dir))

    logger.info("Checkpoint saved → %s", ckpt_dir)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    cfg: DictConfig = OmegaConf.load(args.config)
    if args.resume:
        cfg.training.resume_from_checkpoint = args.resume

    # ---- Accelerator setup -------------------------------------------------
    project_cfg = ProjectConfiguration(
        project_dir=cfg.training.output_dir,
        logging_dir=os.path.join(cfg.training.output_dir, "logs"),
    )
    is_mps = torch.backends.mps.is_available() and not torch.cuda.is_available()
    if is_mps:
        # Accelerate does not support MPS as a distributed backend
        # Force it to CPU mode for the accelerator wrapper but use MPS manually
        os.environ["ACCELERATE_USE_MPS_DEVICE"] = "1"

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision=cfg.training.mixed_precision,  # should be "no" in config
        log_with="wandb" if os.environ.get("WANDB_API_KEY") else None,
        project_config=project_cfg,
    )

    # Initialise loggers
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    if accelerator.is_local_main_process:
        os.makedirs(cfg.training.output_dir, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(cfg.training.output_dir, "config.yaml"))

    set_seed(cfg.training.seed)

    # ---- WandB ---------------------------------------------------------------
    if accelerator.is_main_process and os.environ.get("WANDB_API_KEY"):
        run_name = args.wandb_run_name or "f1-lora-svd"
        accelerator.init_trackers(
            project_name="f1_physics_engine",
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": {"name": run_name}},
        )

    # ---- Dataset / DataLoader -----------------------------------------------
    logger.info("Building dataset …")
    if args.synthetic:
        logger.info("Using SyntheticF1Dataset for debugging.")
        train_dataset = SyntheticF1Dataset(
            num_samples=1024,
            num_frames=cfg.data.num_frames,
            image_size=cfg.data.image_size,
            seed=cfg.training.seed,
        )
    else:
        train_dataset = F1VideoDataset(
            data_root=cfg.data.data_root,
            num_frames=cfg.data.num_frames,
            image_size=cfg.data.image_size,
            stride=cfg.data.stride,
            fps=cfg.data.fps,
            augment=True,
        )

    # MPS does not support pin_memory — detect and disable
    use_pin_memory = cfg.data.pin_memory and torch.cuda.is_available()

    train_loader = create_dataloader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=use_pin_memory,
        drop_last=True,
    )

    logger.info(
        "Dataset: %d clips | Batch size: %d | Steps/epoch: %d",
        len(train_dataset),
        cfg.training.batch_size,
        len(train_loader),
    )

    # ---- Model ---------------------------------------------------------------
    logger.info("Building LanguageConditionedSVD …")
    model = build_model(cfg.model)

    if cfg.training.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    # ---- Noise Scheduler (DDPM for training) --------------------------------
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
    )

    # ---- Optimizer (only trainable parameters) ------------------------------
    trainable_params = model.trainable_parameters()
    logger.info(
        "Trainable parameters: %d (%.2f M)",
        sum(p.numel() for p in trainable_params),
        sum(p.numel() for p in trainable_params) / 1e6,
    )

    optimizer = AdamW(
        trainable_params,
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # ---- LR Scheduler -------------------------------------------------------
    total_steps = (
        cfg.training.num_epochs
        * len(train_loader)
        // cfg.training.gradient_accumulation_steps
    )
    lr_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=max(1, total_steps // 3),
        T_mult=1,
        eta_min=1e-7,
    )

    # ---- Accelerate prepare -------------------------------------------------
    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, lr_scheduler
    )

    # ---- Resume from checkpoint --------------------------------------------
    global_step = 0
    first_epoch = 0

    if cfg.training.resume_from_checkpoint:
        resume_dir = Path(cfg.training.resume_from_checkpoint)
        if resume_dir.exists():
            logger.info("Resuming from %s", resume_dir)
            accelerator.load_state(str(resume_dir / "accelerate_state"))
            # Parse step from directory name
            try:
                global_step = int(resume_dir.name.split("-")[-1])
                first_epoch = global_step // len(train_loader)
            except (ValueError, IndexError):
                logger.warning("Cannot parse step from checkpoint name.")
        else:
            logger.warning(
                "Checkpoint dir %s does not exist; training from scratch.",
                resume_dir,
            )

    # ---- Training Loop ------------------------------------------------------
    logger.info("Starting training for %d epochs …", cfg.training.num_epochs)
    progress_bar = tqdm(
        range(global_step, total_steps),
        disable=not accelerator.is_local_main_process,
        desc="Training",
    )

    for epoch in range(first_epoch, cfg.training.num_epochs):
        model.train()
        epoch_loss = 0.0
        num_steps_this_epoch = 0

        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):

                # ---- Unpack batch ------------------------------------------
                frames: torch.Tensor = batch["frames"]   # [B, F, 3, H, W]
                prompts: List[str]   = batch["prompt"]
                B = frames.shape[0]

                # ---- Encode frames to latent space -------------------------
                # VAE is always in fp32 to avoid overflow; cast back after
                # ---- Encode frames to latent space -------------------------
                with torch.no_grad():
                    unwrapped = accelerator.unwrap_model(model)

                    latents = unwrapped.encode_frames(
                        frames.float()
                    )  # [B, F, 4, H//8, W//8]

                    # SVD requires 8-channel input: noisy latents + conditioning latents
                    # Use the first frame repeated across all F frames as the conditioning signal
                    first_frame = frames[:, 0:1, :, :, :]                        # [B, 1, 3, H, W]
                    first_frame_repeated = first_frame.expand(
                        -1, frames.shape[1], -1, -1, -1
                    ).contiguous()                                                # [B, F, 3, H, W]

                    image_latents = unwrapped.encode_frames(
                        first_frame_repeated.float()
                    )  # [B, F, 4, H//8, W//8]

                # ---- Sample noise and timestep -----------------------------
                noise = torch.randn_like(latents)

                # Slightly bias towards noisier timesteps for video diffusion
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (B,),
                    device=latents.device,
                ).long()

                # ---- Forward diffusion (add noise to latents) --------------
                noisy_latents = noise_scheduler.add_noise(
                    latents, noise, timesteps
                )

                # ---- Text conditioning -------------------------------------
                text_embeddings = accelerator.unwrap_model(model).encode_text(
                    prompts, device=latents.device
                )
                text_embeddings = text_embeddings.to(latents.dtype)

                # ---- added_time_ids ----------------------------------------
                added_time_ids = build_added_time_ids(
                    batch_size=B,
                    fps_id=cfg.data.fps,
                    motion_bucket_id=cfg.inference.motion_bucket_id,
                    noise_aug_strength=cfg.inference.noise_aug_strength,
                    device=latents.device,
                    dtype=latents.dtype,
                )

                # ---- UNet forward pass ------------------------------------
                # ---- UNet forward pass ------------------------------------
                # Pass image_latents so UNet gets the required 8-channel input
                noise_pred = model(
                    noisy_latents,
                    timesteps,
                    text_embeddings,
                    added_time_ids,
                    image_latents=image_latents,
                )

                # ---- Loss --------------------------------------------------
                loss = compute_loss_with_snr(
                    noise_pred,
                    noise,
                    timesteps,
                    noise_scheduler,
                    snr_gamma=cfg.training.snr_gamma,
                )

                # ---- Backward + gradient clip -------------------------------
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), cfg.training.max_grad_norm
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # ---- Logging (every log_steps) --------------------------------
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                epoch_loss += loss.item()
                num_steps_this_epoch += 1

                if global_step % cfg.training.log_steps == 0:
                    avg_loss = epoch_loss / max(num_steps_this_epoch, 1)
                    cur_lr   = optimizer.param_groups[0]["lr"]

                    log_dict = {
                        "train/loss":        loss.item(),
                        "train/avg_loss":    avg_loss,
                        "train/lr":          cur_lr,
                        "train/epoch":       epoch,
                        "train/global_step": global_step,
                    }
                    accelerator.log(log_dict, step=global_step)
                    progress_bar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        lr=f"{cur_lr:.2e}",
                        epoch=epoch,
                    )

                # ---- Checkpoint -------------------------------------------
                if (
                    global_step % cfg.training.save_steps == 0
                    and accelerator.is_main_process
                ):
                    save_checkpoint(
                        model, accelerator, optimizer, lr_scheduler,
                        cfg, global_step,
                    )

        logger.info(
            "Epoch %d complete | avg_loss=%.4f",
            epoch,
            epoch_loss / max(num_steps_this_epoch, 1),
        )

    # ---- Final save ---------------------------------------------------------
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_checkpoint(
            model, accelerator, optimizer, lr_scheduler, cfg, global_step
        )
        logger.info("Training complete.")

    accelerator.end_training()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    train(args)