"""
models/lora_svd.py
==================
Architecture for Language-Conditioned Stable Video Diffusion.

Components
----------
1. TextConditioner
   - Wraps a frozen CLIP text encoder.
   - Projects CLIP hidden states (768-d) → SVD cross-attention dim (1024-d).

2. LanguageConditionedSVD
   - Loads the frozen SVD VAE + UNet.
   - Injects PEFT LoRA layers into every cross-attention block of the UNet.
   - Exposes:
       encode_frames(frames)           → latents
       decode_latents(latents)         → frames
       encode_text(prompts)            → conditioned embeddings
       forward(latents, t, text_emb, added_time_ids) → noise prediction

3. build_model(cfg)
   - Factory that reads an OmegaConf config and returns a ready-to-use
     LanguageConditionedSVD.

NOTE: You must have accepted the Stability AI model terms before calling
      huggingface_hub.snapshot_download for the SVD model.
      Run: huggingface-cli login
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from omegaconf import DictConfig
from peft import LoraConfig, get_peft_model
from transformers import CLIPTextModel, CLIPTokenizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default model identifiers
# ---------------------------------------------------------------------------
_DEFAULT_SVD_MODEL_ID  = "stabilityai/stable-video-diffusion-img2vid-xt"
_DEFAULT_TEXT_ENC_ID   = "openai/clip-vit-large-patch14"
_SVD_CROSS_ATTN_DIM    = 1024   # hard-wired in SVD-XT UNet
_CLIP_LARGE_HIDDEN_DIM = 768


# ---------------------------------------------------------------------------
# 1. TextConditioner
# ---------------------------------------------------------------------------

class TextConditioner(nn.Module):
    """
    Encodes a list of text prompts into cross-attention context tensors
    compatible with the SVD UNet (1024-d hidden dim).

    The CLIP text encoder is kept frozen; only the linear projection is
    trainable.

    Parameters
    ----------
    text_encoder_id : HuggingFace model ID for the CLIP text model
    input_dim       : CLIP output hidden dimension (768 for ViT-L/14)
    output_dim      : Target dimension for SVD cross-attention (1024)
    max_length      : Maximum tokenisation length
    """

    def __init__(
        self,
        text_encoder_id: str = _DEFAULT_TEXT_ENC_ID,
        input_dim: int = _CLIP_LARGE_HIDDEN_DIM,
        output_dim: int = _SVD_CROSS_ATTN_DIM,
        max_length: int = 77,
    ) -> None:
        super().__init__()
        self.max_length = max_length
        self.output_dim = output_dim

        logger.info("Loading CLIP text encoder: %s", text_encoder_id)
        self.tokenizer = CLIPTokenizer.from_pretrained(text_encoder_id)
        self.encoder   = CLIPTextModel.from_pretrained(text_encoder_id)

        # Freeze the CLIP encoder – only the projection is trainable
        for param in self.encoder.parameters():
            param.requires_grad_(False)

        # Learnable linear projection: CLIP hidden dim → SVD cross-attn dim
        self.projection = nn.Linear(input_dim, output_dim, bias=True)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

        # Layer norm for stable training
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        prompts: List[str],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        prompts : list of B strings
        device  : target device; if None uses self.projection's device

        Returns
        -------
        torch.Tensor of shape [B, max_length, output_dim]
        """
        if device is None:
            device = next(self.projection.parameters()).device

        tokens = self.tokenizer(
            prompts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = tokens.input_ids.to(device)
        attention_mask = tokens.attention_mask.to(device)

        with torch.no_grad():
            clip_out = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # last_hidden_state: [B, max_length, 768]
        hidden = clip_out.last_hidden_state.to(device)

        # Project to SVD dim: [B, max_length, 1024]
        projected = self.projection(hidden)
        projected = self.layer_norm(projected)

        return projected


# ---------------------------------------------------------------------------
# Attention processor for heatmap capture (evaluation use)
# ---------------------------------------------------------------------------

class CaptureAttnProcessor(AttnProcessor2_0):
    """
    A drop-in replacement for the default attention processor that
    additionally stores the cross-attention weight tensor in `self.maps`.

    Attach via:
        for name, module in unet.named_modules():
            if isinstance(module, Attention):
                module.set_processor(CaptureAttnProcessor())

    Retrieve:
        for name, module in unet.named_modules():
            if isinstance(module.processor, CaptureAttnProcessor):
                maps = module.processor.maps  # None until first forward pass
    """

    def __init__(self) -> None:
        super().__init__()
        self.maps: Optional[torch.Tensor] = None  # [B*F, heads, Q, K]

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch, channel, height * width).transpose(1, 2)

        batch_size, seq_len, _ = hidden_states.shape

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key   = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim  = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key   = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Compute raw attention weights for capture
        scale  = 1.0 / math.sqrt(head_dim)
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale

        if attention_mask is not None:
            scores = scores + attention_mask

        weights = torch.softmax(scores, dim=-1)

        # Store only cross-attention maps
        if encoder_hidden_states is not hidden_states:
            self.maps = weights.detach().cpu()

        hidden_states = torch.matmul(weights, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


# ---------------------------------------------------------------------------
# 2. LanguageConditionedSVD
# ---------------------------------------------------------------------------

class LanguageConditionedSVD(nn.Module):
    """
    Language-Conditioned Stable Video Diffusion model.

    Architecture
    ------------
    - VAE (AutoencoderKLTemporalDecoder) – frozen, encodes/decodes frames
    - UNet (UNetSpatioTemporalConditionModel) – backbone, frozen base weights
    - LoRA layers – injected into UNet cross-attention Q/K/V/Out projections
    - TextConditioner – CLIP text encoder + linear projection (trainable)

    Only the LoRA matrices and the text projection are updated during training.

    Parameters
    ----------
    svd_model_id        : HuggingFace ID of the SVD model
    text_encoder_id     : HuggingFace ID of the CLIP text encoder
    lora_r              : LoRA rank
    lora_alpha          : LoRA scaling factor (alpha / r = effective scale)
    lora_dropout        : Dropout on LoRA adapter
    lora_target_modules : which weight matrices to inject LoRA into
    conditioning_mode   : "replace" uses only text embeddings;
                          "concat"  concatenates text onto image embeddings
    """

    def __init__(
        self,
        svd_model_id: str = _DEFAULT_SVD_MODEL_ID,
        text_encoder_id: str = _DEFAULT_TEXT_ENC_ID,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        conditioning_mode: str = "replace",
    ) -> None:
        super().__init__()

        if lora_target_modules is None:
            lora_target_modules = ["to_q", "to_k", "to_v", "to_out.0"]

        self.conditioning_mode = conditioning_mode

        # ---- Load frozen VAE ------------------------------------------------
        logger.info("Loading SVD VAE from %s …", svd_model_id)
        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(
            svd_model_id,
            subfolder="vae",
        )
        self.vae.requires_grad_(False)
        self.vae.eval()

        # ---- Load frozen UNet -----------------------------------------------
        logger.info("Loading SVD UNet from %s …", svd_model_id)
        unet_base = UNetSpatioTemporalConditionModel.from_pretrained(
            svd_model_id,
            subfolder="unet",
        )
        unet_base.requires_grad_(False)

        # ---- Inject LoRA into the UNet -------------------------------------
        logger.info(
            "Injecting LoRA (r=%d, alpha=%d) into: %s",
            lora_r, lora_alpha, lora_target_modules,
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
        )
        self.unet: nn.Module = get_peft_model(unet_base, lora_config)
        self.unet.print_trainable_parameters()

        # ---- Text conditioner (projection is trainable) --------------------
        self.text_conditioner = TextConditioner(
            text_encoder_id=text_encoder_id,
            output_dim=_SVD_CROSS_ATTN_DIM,
        )

        # ---- VAE scaling factor -------------------------------------------
        self.vae_scale_factor: float = self.vae.config.scaling_factor

    # ------------------------------------------------------------------
    # Encoding / Decoding helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_frames(
        self,
        frames: torch.Tensor,
        sample: bool = True,
    ) -> torch.Tensor:
        """
        Encode a batch of video clips to latent space.

        Parameters
        ----------
        frames : [B, F, 3, H, W] in [-1, 1]
        sample : if True samples from the posterior, else uses mode

        Returns
        -------
        latents : [B, F, 4, H//8, W//8]
        """
        B, F, C, H, W = frames.shape
        flat = frames.view(B * F, C, H, W)

        posterior = self.vae.encode(flat).latent_dist
        if sample:
            latents_flat = posterior.sample()
        else:
            latents_flat = posterior.mode()

        latents_flat = latents_flat * self.vae_scale_factor
        _, lC, lH, lW = latents_flat.shape
        return latents_flat.view(B, F, lC, lH, lW)

    @torch.no_grad()
    def decode_latents(
        self,
        latents: torch.Tensor,
        num_frames: int = 8,
    ) -> torch.Tensor:
        """
        Decode latent video to pixel space.

        Parameters
        ----------
        latents : [B, F, 4, H//8, W//8]

        Returns
        -------
        frames : [B, F, 3, H, W] in [-1, 1]
        """
        B, F, lC, lH, lW = latents.shape
        flat = latents.view(B * F, lC, lH, lW)
        flat = flat / self.vae_scale_factor

        # AutoencoderKLTemporalDecoder.decode expects (flat, num_frames)
        decoded = self.vae.decode(flat, num_frames=num_frames).sample
        _, C, H, W = decoded.shape
        return decoded.view(B, F, C, H, W)

    def encode_text(
        self,
        prompts: List[str],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Encode text prompts into cross-attention context vectors.

        Returns
        -------
        [B, 77, 1024]
        """
        return self.text_conditioner(prompts, device=device)

    # ------------------------------------------------------------------
    # Forward pass (denoising UNet)
    # ------------------------------------------------------------------

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: torch.Tensor,
        added_time_ids: torch.Tensor,
        image_latents: Optional[torch.Tensor] = None,
        image_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run the denoising UNet forward pass.

        Parameters
        ----------
        noisy_latents  : [B, F, 4, H//8, W//8]  — the noisy video latents
        timesteps      : [B] int64
        text_embeddings: [B, 77, 1024]
        added_time_ids : [B, 3]
        image_latents  : [B, F, 4, H//8, W//8]  — conditioning frame latents
                        (first frame repeated across F). SVD REQUIRES this.
                        If None, zeros are used (degraded quality).
        image_embeddings: unused, kept for API compatibility
        """
        # ----------------------------------------------------------------
        # SVD UNet conv_in expects 8 channels:
        #   channels 0:4 = noisy latents
        #   channels 4:8 = conditioning image latents (first frame repeated)
        # ----------------------------------------------------------------
        if image_latents is not None:
            unet_input = torch.cat([noisy_latents, image_latents], dim=2)
        else:
            # Fallback: pad with zeros (signals no visual conditioning)
            unet_input = torch.cat(
                [noisy_latents, torch.zeros_like(noisy_latents)], dim=2
            )
        # unet_input shape: [B, F, 8, H//8, W//8]  ← UNet expects this

        # Build encoder_hidden_states
        if self.conditioning_mode == "concat" and image_embeddings is not None:
            encoder_hidden = torch.cat([image_embeddings, text_embeddings], dim=1)
        else:
            encoder_hidden = text_embeddings

        noise_pred = self.unet(
            unet_input,
            timesteps,
            encoder_hidden_states=encoder_hidden,
            added_time_ids=added_time_ids,
        ).sample

        return noise_pred

    # ------------------------------------------------------------------
    # Parameter access utilities
    # ------------------------------------------------------------------

    def trainable_parameters(self) -> List[nn.Parameter]:
        """Returns only the parameters that require gradients."""
        return [p for p in self.parameters() if p.requires_grad]

    def save_lora_weights(self, path: str) -> None:
        """
        Persist LoRA weights + text projection to disk.
        Stores a dict with two keys:
          - "lora_state_dict"      : LoRA adapter weights
          - "text_projection_state": TextConditioner projection + LN
        """
        import os
        os.makedirs(path, exist_ok=True)

        # Save LoRA weights via PEFT
        self.unet.save_pretrained(os.path.join(path, "lora_unet"))

        # Save text projection weights
        torch.save(
            {
                "projection": self.text_conditioner.projection.state_dict(),
                "layer_norm": self.text_conditioner.layer_norm.state_dict(),
            },
            os.path.join(path, "text_projection.pt"),
        )
        logger.info("Saved LoRA weights → %s", path)

    def load_lora_weights(self, path: str) -> None:
        """Load previously saved LoRA + projection weights."""
        import os
        from peft import PeftModel

        # Reload LoRA adapter
        unet_path = os.path.join(path, "lora_unet")
        if os.path.isdir(unet_path):
            self.unet.load_adapter(unet_path, adapter_name="default")
            logger.info("Loaded LoRA weights from %s", unet_path)

        # Reload text projection
        proj_path = os.path.join(path, "text_projection.pt")
        if os.path.isfile(proj_path):
            ckpt = torch.load(proj_path, map_location="cpu")
            self.text_conditioner.projection.load_state_dict(ckpt["projection"])
            self.text_conditioner.layer_norm.load_state_dict(ckpt["layer_norm"])
            logger.info("Loaded text projection from %s", proj_path)

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing on the UNet to reduce VRAM."""
        self.unet.enable_gradient_checkpointing()
        logger.info("Gradient checkpointing enabled on UNet.")

    def enable_xformers(self) -> None:
        """Enable memory-efficient attention via xformers."""
        logger.warning("xformers not available on Apple M1 MPS — skipping.")


# ---------------------------------------------------------------------------
# 3. build_model factory
# ---------------------------------------------------------------------------

def build_model(cfg: DictConfig) -> LanguageConditionedSVD:
    """
    Construct a LanguageConditionedSVD from an OmegaConf config node.

    Expected config keys (see configs/default.yaml → model section):
      svd_model_id, text_encoder_id, lora_r, lora_alpha, lora_dropout,
      lora_target_modules, conditioning_mode
    """
    model = LanguageConditionedSVD(
        svd_model_id=cfg.svd_model_id,
        text_encoder_id=cfg.text_encoder_id,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        lora_target_modules=list(cfg.lora_target_modules),
        conditioning_mode=cfg.conditioning_mode,
    )

    if cfg.get("gradient_checkpointing", False):
        model.enable_gradient_checkpointing()

    if cfg.get("use_xformers", False):
        model.enable_xformers()

    return model