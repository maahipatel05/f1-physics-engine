"""
utils/evaluation.py
===================
Quantitative and qualitative evaluation utilities.

Classes
-------
OpticalFlowEvaluator
    Computes dense optical flow (Farneback) between consecutive frames
    and calculates MSE between predicted and ground-truth flow fields.

AttentionHeatmapExtractor
    Installs CaptureAttnProcessor hooks onto a LanguageConditionedSVD
    model, runs a single forward pass, gathers the stored attention
    weight tensors, upsamples them to pixel resolution, and produces
    matplotlib figures showing which spatial regions each text token
    attends to.

VideoMetrics
    Aggregates per-clip metrics (flow MSE, SSIM, PSNR) into summary
    statistics and writes them to a JSON file.

Standalone helpers
------------------
frames_to_uint8(frames)   convert [-1,1] tensor → uint8 numpy
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import zoom as scipy_zoom

logger = logging.getLogger(__name__)

matplotlib.use("Agg")   # non-interactive backend (safe on SLURM)


# ---------------------------------------------------------------------------
# Utility: tensor → uint8 numpy
# ---------------------------------------------------------------------------

def frames_to_uint8(frames: torch.Tensor) -> np.ndarray:
    """
    Convert a float tensor in [-1, 1] to uint8 numpy array [0, 255].

    Parameters
    ----------
    frames : torch.Tensor of shape [..., 3, H, W] or [..., H, W, 3]

    Returns
    -------
    np.ndarray uint8 with the same leading dimensions
    """
    arr = frames.detach().cpu().float().numpy()
    arr = (arr + 1.0) / 2.0          # [-1,1] → [0,1]
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr


# ---------------------------------------------------------------------------
# 1. OpticalFlowEvaluator
# ---------------------------------------------------------------------------

class OpticalFlowEvaluator:
    """
    Compute and compare dense optical flow between consecutive video frames.

    Flow is estimated using the Gunnar Farneback algorithm from OpenCV.
    The resulting flow fields (one per consecutive pair) are compared between
    a predicted clip and a ground-truth clip using Mean Squared Error.

    Parameters
    ----------
    pyr_scale  : scale between pyramid levels (0.5 = classical half-scale)
    levels     : number of pyramid levels
    winsize    : averaging window size
    iterations : number of iterations per level
    poly_n     : neighbourhood size for polynomial expansion
    poly_sigma : standard deviation of Gaussian used for smoothing
    """

    def __init__(
        self,
        pyr_scale: float = 0.5,
        levels: int = 3,
        winsize: int = 15,
        iterations: int = 3,
        poly_n: int = 5,
        poly_sigma: float = 1.2,
    ) -> None:
        self.pyr_scale  = pyr_scale
        self.levels     = levels
        self.winsize    = winsize
        self.iterations = iterations
        self.poly_n     = poly_n
        self.poly_sigma = poly_sigma

    def _tensor_to_gray_frames(self, frames: torch.Tensor) -> List[np.ndarray]:
        """
        Convert [F, 3, H, W] float tensor → list of F grayscale uint8 images.
        """
        arr = frames_to_uint8(frames)   # [F, 3, H, W]
        gray_list: List[np.ndarray] = []
        for t in range(arr.shape[0]):
            frame_chw = arr[t]          # [3, H, W]
            frame_hwc = np.transpose(frame_chw, (1, 2, 0))   # [H, W, 3]
            gray = cv2.cvtColor(frame_hwc, cv2.COLOR_RGB2GRAY)
            gray_list.append(gray)
        return gray_list

    def compute_flow_sequence(
        self,
        frames: torch.Tensor,
    ) -> List[np.ndarray]:
        """
        Compute dense optical flow for every consecutive frame pair.

        Parameters
        ----------
        frames : [F, 3, H, W] float tensor in [-1, 1]

        Returns
        -------
        List of (F-1) flow arrays each of shape [H, W, 2]
        """
        gray_frames = self._tensor_to_gray_frames(frames)
        flows: List[np.ndarray] = []

        for i in range(len(gray_frames) - 1):
            flow = cv2.calcOpticalFlowFarneback(
                gray_frames[i],
                gray_frames[i + 1],
                None,
                self.pyr_scale,
                self.levels,
                self.winsize,
                self.iterations,
                self.poly_n,
                self.poly_sigma,
                0,
            )
            flows.append(flow)   # [H, W, 2]

        return flows

    def compute_flow_mse(
        self,
        pred_frames: torch.Tensor,
        gt_frames: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compare optical flow between predicted and ground-truth video clips.

        Parameters
        ----------
        pred_frames : [F, 3, H, W]
        gt_frames   : [F, 3, H, W]

        Returns
        -------
        dict with keys:
          mse_u       – MSE of horizontal (u) component
          mse_v       – MSE of vertical (v) component
          mse_total   – combined (u + v) MSE
          epe         – endpoint error (average magnitude of flow difference)
        """
        if pred_frames.shape != gt_frames.shape:
            raise ValueError(
                f"Shape mismatch: pred {pred_frames.shape} vs gt {gt_frames.shape}"
            )

        pred_flows = self.compute_flow_sequence(pred_frames)
        gt_flows   = self.compute_flow_sequence(gt_frames)

        mse_u_acc = 0.0
        mse_v_acc = 0.0
        epe_acc   = 0.0
        n = len(pred_flows)

        for pf, gf in zip(pred_flows, gt_flows):
            diff = pf.astype(np.float32) - gf.astype(np.float32)  # [H, W, 2]
            mse_u_acc += float(np.mean(diff[..., 0] ** 2))
            mse_v_acc += float(np.mean(diff[..., 1] ** 2))
            epe_acc   += float(np.mean(np.sqrt(np.sum(diff ** 2, axis=-1))))

        return {
            "mse_u":     mse_u_acc / n,
            "mse_v":     mse_v_acc / n,
            "mse_total": (mse_u_acc + mse_v_acc) / n,
            "epe":        epe_acc / n,
        }

    def visualize_flow(
        self,
        flow: np.ndarray,
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Render a single optical flow field as an HSV colour wheel image.

        Parameters
        ----------
        flow      : [H, W, 2] float32 array
        save_path : if provided, saves the image

        Returns
        -------
        rgb image as uint8 numpy [H, W, 3]
        """
        h, w = flow.shape[:2]
        magnitude, angle = cv2.cartToPolar(
            flow[..., 0], flow[..., 1], angleInDegrees=True
        )

        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 0] = (angle / 2).astype(np.uint8)   # hue ← direction
        hsv[..., 1] = 255                             # full saturation
        hsv[..., 2] = cv2.normalize(
            magnitude, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            cv2.imwrite(save_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        return rgb

    def batch_flow_mse(
        self,
        pred_batch: torch.Tensor,
        gt_batch: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Average flow MSE over a batch of clips.

        Parameters
        ----------
        pred_batch : [B, F, 3, H, W]
        gt_batch   : [B, F, 3, H, W]
        """
        B = pred_batch.shape[0]
        agg: Dict[str, float] = {
            "mse_u": 0.0, "mse_v": 0.0, "mse_total": 0.0, "epe": 0.0
        }
        for b in range(B):
            metrics = self.compute_flow_mse(pred_batch[b], gt_batch[b])
            for k in agg:
                agg[k] += metrics[k]
        for k in agg:
            agg[k] /= B
        return agg


# ---------------------------------------------------------------------------
# 2. AttentionHeatmapExtractor
# ---------------------------------------------------------------------------

class AttentionHeatmapExtractor:
    """
    Extract cross-attention maps from the LanguageConditionedSVD model and
    overlay them on video frames to visualise which spatial regions each
    text token governs.

    Usage
    -----
    extractor = AttentionHeatmapExtractor(model)
    extractor.install_hooks()

    # … run one forward pass of the model …

    maps = extractor.collect_maps()   # dict: layer_name → [B*F, heads, Q, K]
    fig  = extractor.plot(maps, frames, prompts, save_dir="./eval/heatmaps")
    extractor.remove_hooks()
    """

    def __init__(self, model: "LanguageConditionedSVD") -> None:
        from models.lora_svd import CaptureAttnProcessor
        self.model = model
        self.CaptureAttnProcessor = CaptureAttnProcessor
        self._hooks: Dict[str, "CaptureAttnProcessor"] = {}

    # ------------------------------------------------------------------

    def install_hooks(self, max_layers: int = 4) -> None:
        """
        Replace the attention processor on the first `max_layers` cross-
        attention blocks with a CaptureAttnProcessor.

        Only blocks whose cross_attention_dim != None are cross-attention.
        """
        from diffusers.models.attention_processor import Attention

        installed = 0
        for full_name, module in self.model.unet.named_modules():
            if isinstance(module, Attention):
                # Skip self-attention blocks (they have no encoder_hidden_states)
                if module.cross_attention_dim is None:
                    continue
                if installed >= max_layers:
                    break
                processor = self.CaptureAttnProcessor()
                module.set_processor(processor)
                self._hooks[full_name] = processor
                installed += 1
                logger.debug("Installed capture hook on %s", full_name)

        logger.info("Installed %d attention capture hooks.", installed)

    def remove_hooks(self) -> None:
        """Restore the default attention processor on all hooked modules."""
        from diffusers.models.attention_processor import Attention, AttnProcessor2_0

        for full_name, module in self.model.unet.named_modules():
            if isinstance(module, Attention) and full_name in self._hooks:
                module.set_processor(AttnProcessor2_0())

        self._hooks.clear()
        logger.info("Removed all attention capture hooks.")

    def collect_maps(self) -> Dict[str, Optional[torch.Tensor]]:
        """
        After a forward pass, collect stored attention weight tensors.

        Returns
        -------
        dict: layer_name → Tensor [B*F, heads, Q_len, K_len] | None
        """
        return {name: proc.maps for name, proc in self._hooks.items()}

    # ------------------------------------------------------------------

    def _avg_heads_and_upsample(
        self,
        attn_map: torch.Tensor,
        target_h: int,
        target_w: int,
    ) -> np.ndarray:
        """
        Average over attention heads and token sequence, then
        upsample to (target_h, target_w).

        attn_map : [N, heads, Q_len, K_len]  (K_len = 77 text tokens)

        Returns
        -------
        np.ndarray float32 [target_h, target_w]
        """
        # Mean over heads and key-dimension → [N, Q_len]
        avg = attn_map.float().mean(dim=(1, 3))   # [N, Q_len]

        # For video UNet the Q_len corresponds to spatial positions.
        # Find the closest square resolution.
        q_len = avg.shape[1]
        side = int(math.sqrt(q_len))
        if side * side != q_len:
            side = int(math.sqrt(q_len)) + 1

        # Mean over batch+frames → [Q_len]
        avg = avg.mean(dim=0).numpy()             # [Q_len]

        # Pad if needed and reshape
        pad = side * side - q_len
        if pad > 0:
            avg = np.concatenate([avg, np.zeros(pad)], axis=0)

        heatmap = avg.reshape(side, side)

        # Normalise to [0, 1]
        mn, mx = heatmap.min(), heatmap.max()
        if mx > mn:
            heatmap = (heatmap - mn) / (mx - mn)

        # Bilinear upsample to target resolution
        zoom_h = target_h / side
        zoom_w = target_w / side
        heatmap = scipy_zoom(heatmap, (zoom_h, zoom_w), order=1)
        return heatmap.astype(np.float32)

    # ------------------------------------------------------------------

    def plot(
        self,
        attn_maps: Dict[str, Optional[torch.Tensor]],
        frames: torch.Tensor,
        prompt: str,
        save_dir: str = "./eval/heatmaps",
        global_step: int = 0,
    ) -> List[plt.Figure]:
        """
        Produce one matplotlib figure per captured attention layer.

        Each figure has two rows:
          Row 0 – original video frames
          Row 1 – heatmap overlaid on frames

        Parameters
        ----------
        attn_maps   : output of collect_maps()
        frames      : [F, 3, H, W] float tensor in [-1, 1]
        prompt      : the text prompt used for generation
        save_dir    : directory to save PNG files
        global_step : training step (used for filenames)
        """
        import math as _math

        os.makedirs(save_dir, exist_ok=True)

        F_count = frames.shape[0]
        H = frames.shape[2]
        W = frames.shape[3]

        frame_arr = frames_to_uint8(frames)   # [F, 3, H, W]
        frame_rgb = [
            np.transpose(frame_arr[t], (1, 2, 0)) for t in range(F_count)
        ]   # list of [H, W, 3] uint8

        figs: List[plt.Figure] = []

        for layer_idx, (layer_name, amap) in enumerate(attn_maps.items()):
            if amap is None:
                logger.warning("Layer %s has no captured maps.", layer_name)
                continue

            # Compute spatial heatmap averaged over tokens + heads
            heatmap = self._avg_heads_and_upsample(amap, H, W)

            fig, axes = plt.subplots(
                2, F_count,
                figsize=(2 * F_count, 5),
                gridspec_kw={"hspace": 0.05, "wspace": 0.02},
            )

            fig.suptitle(
                f"Layer {layer_idx}: {layer_name.split('.')[-3]}\n"
                f'"{prompt[:80]}"',
                fontsize=7,
                y=1.01,
            )

            for t in range(F_count):
                # Row 0: original frame
                ax_orig = axes[0, t] if F_count > 1 else axes[0]
                ax_orig.imshow(frame_rgb[t])
                ax_orig.axis("off")
                if t == 0:
                    ax_orig.set_ylabel("Frame", fontsize=7)

                # Row 1: heatmap overlay
                ax_heat = axes[1, t] if F_count > 1 else axes[1]
                ax_heat.imshow(frame_rgb[t])
                ax_heat.imshow(
                    heatmap, cmap="jet", alpha=0.5,
                    vmin=0.0, vmax=1.0,
                )
                ax_heat.axis("off")
                if t == 0:
                    ax_heat.set_ylabel("Attention", fontsize=7)

            save_path = os.path.join(
                save_dir,
                f"step{global_step:06d}_layer{layer_idx:02d}.png",
            )
            fig.savefig(save_path, bbox_inches="tight", dpi=120)
            logger.info("Saved heatmap → %s", save_path)
            figs.append(fig)
            plt.close(fig)

        return figs


# ---------------------------------------------------------------------------
# 3. VideoMetrics
# ---------------------------------------------------------------------------

class VideoMetrics:
    """
    Accumulates per-clip evaluation metrics and writes summary statistics.

    Supported metrics
    -----------------
    - flow_mse    : optical flow MSE (via OpticalFlowEvaluator)
    - psnr        : peak signal-to-noise ratio
    - ssim        : structural similarity index (grayscale)
    """

    def __init__(self, flow_eval: Optional[OpticalFlowEvaluator] = None) -> None:
        self.flow_eval = flow_eval or OpticalFlowEvaluator()
        self._records: List[Dict] = []

    # ------------------------------------------------------------------

    @staticmethod
    def _psnr(pred: np.ndarray, gt: np.ndarray) -> float:
        """PSNR between two uint8 images."""
        mse_val = np.mean((pred.astype(np.float32) - gt.astype(np.float32)) ** 2)
        if mse_val < 1e-10:
            return 100.0
        return float(20.0 * np.log10(255.0 / np.sqrt(mse_val)))

    @staticmethod
    def _ssim_gray(
        pred_gray: np.ndarray,
        gt_gray: np.ndarray,
        k1: float = 0.01,
        k2: float = 0.03,
        L: int = 255,
    ) -> float:
        """Simple single-scale SSIM on grayscale uint8 images."""
        c1 = (k1 * L) ** 2
        c2 = (k2 * L) ** 2
        p  = pred_gray.astype(np.float64)
        g  = gt_gray.astype(np.float64)
        mu1, mu2 = p.mean(), g.mean()
        sigma1   = p.std()
        sigma2   = g.std()
        sigma12  = np.mean((p - mu1) * (g - mu2))
        num  = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
        den  = (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 ** 2 + sigma2 ** 2 + c2)
        return float(num / den)

    # ------------------------------------------------------------------

    def update(
        self,
        pred_frames: torch.Tensor,
        gt_frames: torch.Tensor,
        prompt: str = "",
        clip_id: str = "",
    ) -> Dict:
        """
        Compute all metrics for a single clip pair and record them.

        Parameters
        ----------
        pred_frames : [F, 3, H, W] float in [-1, 1]
        gt_frames   : [F, 3, H, W] float in [-1, 1]
        prompt      : text prompt (for logging)
        clip_id     : identifier string (for logging)

        Returns
        -------
        dict of metric values for this clip
        """
        flow_metrics = self.flow_eval.compute_flow_mse(pred_frames, gt_frames)

        pred_arr = frames_to_uint8(pred_frames)   # [F, 3, H, W]
        gt_arr   = frames_to_uint8(gt_frames)

        psnr_vals: List[float] = []
        ssim_vals: List[float] = []

        for t in range(pred_arr.shape[0]):
            p_hwc = np.transpose(pred_arr[t], (1, 2, 0))
            g_hwc = np.transpose(gt_arr[t], (1, 2, 0))
            psnr_vals.append(self._psnr(p_hwc, g_hwc))

            p_gray = cv2.cvtColor(p_hwc, cv2.COLOR_RGB2GRAY)
            g_gray = cv2.cvtColor(g_hwc, cv2.COLOR_RGB2GRAY)
            ssim_vals.append(self._ssim_gray(p_gray, g_gray))

        record = {
            "clip_id":    clip_id,
            "prompt":     prompt,
            "flow_mse_u": flow_metrics["mse_u"],
            "flow_mse_v": flow_metrics["mse_v"],
            "flow_mse":   flow_metrics["mse_total"],
            "epe":        flow_metrics["epe"],
            "psnr":       float(np.mean(psnr_vals)),
            "ssim":       float(np.mean(ssim_vals)),
        }
        self._records.append(record)
        return record

    def summary(self) -> Dict[str, float]:
        """Return mean ± std for every metric across all recorded clips."""
        if not self._records:
            return {}

        keys = [k for k in self._records[0] if k not in ("clip_id", "prompt")]
        summary: Dict[str, float] = {}
        for k in keys:
            vals = np.array([r[k] for r in self._records], dtype=np.float32)
            summary[f"{k}_mean"] = float(vals.mean())
            summary[f"{k}_std"]  = float(vals.std())
        return summary

    def save(self, path: str) -> None:
        """Write all records + summary to a JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        out = {
            "summary": self.summary(),
            "records": self._records,
        }
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        logger.info("Saved evaluation metrics → %s", path)