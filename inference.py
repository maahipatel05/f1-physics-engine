"""
inference.py
============
Load a trained LanguageConditionedSVD checkpoint and generate a predicted
video clip given a starting frame and a text prompt.

Usage
-----
python inference.py \\
    --checkpoint ./checkpoints/checkpoint-0005000 \\
    --start_frame ./data/test_frame.png \\
    --prompt "The driver brakes aggressively and turns sharply right." \\
    --output ./outputs/predicted_clip.mp4 \\
    --num_frames 8 \\
    --num_inference_steps 25

The script:
1. Loads the SVD pipeline (VAE + scheduler + UNet with LoRA weights).
2. Encodes the starting frame as the first latent.
3. Runs the DDIM/Euler reverse diffusion loop conditioned on the text prompt.
4. Decodes latents → frames and writes an MP4.
5. Optionally extracts and saves cross-attention heatmaps.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import EulerDiscreteScheduler
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

from models import LanguageConditionedSVD, build_model
from utils import AttentionHeatmapExtractor, frames_to_uint8

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate F1 video clip from a starting frame and text prompt."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory (output of train.py).",
    )
    parser.add_argument(
        "--start_frame",
        type=str,
        required=True,
        help="Path to the conditioning PNG frame.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Natural language action description.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs/generated.mp4",
        help="Output MP4 path.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="Classifier-free guidance scale. Set to 1.0 to disable.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--motion_bucket_id",
        type=int,
        default=127,
    )
    parser.add_argument(
        "--noise_aug_strength",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--save_heatmaps",
        action="store_true",
        help="Extract and save cross-attention heatmaps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    def _default_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    parser.add_argument(
        "--device",
        type=str,
        default=_default_device(),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Frame utilities
# ---------------------------------------------------------------------------

def load_and_preprocess_frame(
    path: str,
    image_size: int = 128,
) -> torch.Tensor:
    """
    Load a PNG and return a [1, 3, H, W] float tensor in [-1, 1].
    """
    transform = transforms.Compose([
        transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)   # [1, 3, H, W]


def save_video(
    frames: np.ndarray,
    output_path: str,
    fps: int = 10,
) -> None:
    """
    Save a numpy array of frames [F, H, W, 3] uint8 as an MP4.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264")
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    logger.info("Saved video → %s", output_path)


# ---------------------------------------------------------------------------
# Denoising loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_video(
    model: LanguageConditionedSVD,
    start_frame: torch.Tensor,          # [1, 3, H, W]
    prompt: str,
    num_frames: int = 8,
    num_inference_steps: int = 25,
    guidance_scale: float = 3.5,
    fps_id: int = 10,
    motion_bucket_id: int = 127,
    noise_aug_strength: float = 0.02,
    device: torch.device = torch.device("cuda"),
    seed: int = 42,
) -> torch.Tensor:
    """
    Run the reverse diffusion loop to generate a video clip.

    Parameters
    ----------
    model           : LanguageConditionedSVD (with LoRA loaded)
    start_frame     : [1, 3, H, W] conditioning frame in [-1, 1]
    prompt          : text action description
    num_frames      : number of frames to generate
    num_inference_steps : DDPM/Euler steps
    guidance_scale  : CFG scale (>=1.0)
    ...

    Returns
    -------
    Tensor [1, F, 3, H, W] float in [-1, 1]
    """
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)

    # ---- Build noise scheduler (Euler ancestral for inference) ------------
    scheduler = EulerDiscreteScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
    )
    scheduler.set_timesteps(num_inference_steps, device=device)

    # ---- Encode start frame -----------------------------------------------
    start_frame = start_frame.to(device)
    # Repeat the start frame to create a base video (all frames identical)
    # We tile to [1, F, 3, H, W] then encode
    start_video = start_frame.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
    base_latents = model.encode_frames(start_video)   # [1, F, 4, H//8, W//8]

    # ---- Initial noise latents -------------------------------------------
    _, F, lC, lH, lW = base_latents.shape
    latents = torch.randn(
        (1, F, lC, lH, lW),
        generator=generator,   # CPU generator
        dtype=base_latents.dtype,
    ).to(device)              # then move to MPS
    latents = latents * scheduler.init_noise_sigma

    # ---- Text embedding ---------------------------------------------------
    text_emb = model.encode_text([prompt], device=device)          # [1, 77, 1024]
    text_emb = text_emb.to(latents.dtype)

    # Null embedding for CFG (empty prompt)
    null_emb = model.encode_text([""], device=device)
    null_emb = null_emb.to(latents.dtype)

    # ---- added_time_ids ---------------------------------------------------
    added_time_ids = torch.tensor(
        [[fps_id, motion_bucket_id, noise_aug_strength]],
        dtype=latents.dtype,
        device=device,
    )

    # ---- Denoising loop ---------------------------------------------------
    logger.info(
        "Running %d denoising steps (CFG scale=%.1f) …",
        num_inference_steps, guidance_scale,
    )
    for t in tqdm(scheduler.timesteps, desc="Denoising", leave=False):
        # Scale model input
        latent_model_input = scheduler.scale_model_input(latents, t)

        # Conditional noise prediction
        t_batch = t.unsqueeze(0).to(device)
        image_latents = base_latents   # [1, F, 4, H//8, W//8] — first frame repeated

        # then inside the loop:
        noise_pred_cond = model(
            latent_model_input, t_batch, text_emb, added_time_ids,
            image_latents=image_latents,
        )
        # ...
        noise_pred_uncond = model(
            latent_model_input, t_batch, null_emb, added_time_ids,
            image_latents=image_latents,
        )

        # Unconditional noise prediction (for CFG)
        if guidance_scale > 1.0:
            noise_pred_uncond = model(
                latent_model_input, t_batch, null_emb, added_time_ids
            )
            # Classifier-Free Guidance
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )
        else:
            noise_pred = noise_pred_cond

        # Scheduler step
        latents = scheduler.step(noise_pred, t, latents, generator=generator).prev_sample

    # ---- Decode latents → frames -----------------------------------------
    generated_frames = model.decode_latents(latents, num_frames=num_frames)
    return generated_frames   # [1, F, 3, H, W]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg: DictConfig = OmegaConf.load(args.config)

    device = torch.device(args.device)
    logger.info("Running inference on device: %s", device)

    # ---- Build and load model -------------------------------------------
    logger.info("Building model …")
    model = build_model(cfg.model)

    logger.info("Loading LoRA weights from %s …", args.checkpoint)
    model.load_lora_weights(args.checkpoint)
    model = model.to(device)
    model.eval()

    # ---- Load starting frame --------------------------------------------
    logger.info("Loading start frame: %s", args.start_frame)
    start_frame = load_and_preprocess_frame(args.start_frame, args.image_size)

    # ---- Optional: install attention hooks ------------------------------
    extractor: Optional[AttentionHeatmapExtractor] = None
    if args.save_heatmaps:
        extractor = AttentionHeatmapExtractor(model)
        extractor.install_hooks(max_layers=cfg.evaluation.num_heatmap_layers)
        logger.info("Attention hooks installed for heatmap extraction.")

    # ---- Generate video --------------------------------------------------
    logger.info("Generating video with prompt:\n  '%s'", args.prompt)
    generated = generate_video(
        model=model,
        start_frame=start_frame,
        prompt=args.prompt,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        fps_id=args.fps,
        motion_bucket_id=args.motion_bucket_id,
        noise_aug_strength=args.noise_aug_strength,
        device=device,
        seed=args.seed,
    )   # [1, F, 3, H, W]

    # ---- Save video ------------------------------------------------------
    arr = frames_to_uint8(generated[0])   # [F, 3, H, W] uint8
    frames_hwc = [np.transpose(arr[t], (1, 2, 0)) for t in range(arr.shape[0])]
    save_video(np.stack(frames_hwc), args.output, fps=args.fps)

    # ---- Save individual frames as PNGs ---------------------------------
    frame_dir = Path(args.output).with_suffix("")
    frame_dir.mkdir(parents=True, exist_ok=True)
    for t, frame in enumerate(frames_hwc):
        Image.fromarray(frame).save(frame_dir / f"frame_{t:04d}.png")
    logger.info("Saved individual frames → %s", frame_dir)

    # ---- Attention heatmaps (qualitative evaluation) --------------------
    if extractor is not None:
        attn_maps = extractor.collect_maps()
        heatmap_dir = str(Path(args.output).parent / "heatmaps")
        extractor.plot(
            attn_maps,
            generated[0],   # [F, 3, H, W]
            args.prompt,
            save_dir=heatmap_dir,
            global_step=0,
        )
        extractor.remove_hooks()
        logger.info("Attention heatmaps saved → %s", heatmap_dir)

    # ---- Quantitative self-evaluation (flow MSE vs starting frame) ------
    _self_eval(generated[0], start_frame[0], args.prompt)


def _self_eval(
    generated: torch.Tensor,   # [F, 3, H, W]
    start_frame: torch.Tensor, # [3, H, W]
    prompt: str,
) -> None:
    """
    A quick diagnostic: compute flow MSE between generated frames
    and a trivial static baseline (all frames = start frame).
    A good model should have LOWER MSE than the trivial baseline
    when given an action prompt that implies motion.
    """
    from utils import OpticalFlowEvaluator, VideoMetrics

    F_count = generated.shape[0]
    static_baseline = start_frame.unsqueeze(0).expand(F_count, -1, -1, -1)

    evaluator = OpticalFlowEvaluator()
    metrics_gen  = evaluator.compute_flow_mse(generated,       static_baseline)
    metrics_stat = evaluator.compute_flow_mse(static_baseline, static_baseline)

    logger.info("=== Inference Self-Evaluation ===")
    logger.info("Prompt: '%s'", prompt)
    logger.info(
        "Generated  flow MSE: %.4f | EPE: %.4f",
        metrics_gen["mse_total"], metrics_gen["epe"],
    )
    logger.info(
        "Static-BL  flow MSE: %.4f | EPE: %.4f  (should be ~0)",
        metrics_stat["mse_total"], metrics_stat["epe"],
    )
    if metrics_gen["epe"] > metrics_stat["epe"] + 0.01:
        logger.info(
            "✓ Model generates motion (EPE delta=%.4f)",
            metrics_gen["epe"] - metrics_stat["epe"],
        )
    else:
        logger.warning(
            "⚠  Generated video shows minimal motion. "
            "Consider more training steps or adjusting guidance_scale."
        )


if __name__ == "__main__":
    main()