"""
Transformer-based Flow Matching Policy.

Architecture:
  - DINOv2 image encoder (frozen) → image tokens per camera
  - State MLP projection
  - Fused obs embedding
  - Transformer denoiser: takes (obs_emb, x_t, t) → velocity field v_t
  - Euler ODE integrator for inference (flow matching: x_0 = x_1 + ∫ v dt)

Flow matching formulation (Lipman et al. 2022 / Liu et al. 2022):
  - Data: x_0 ~ p_data, Noise: x_1 ~ N(0, I)
  - Straight path: x_t = (1-t)*x_0 + t*x_1,  t in [0,1]
  - Target velocity: v* = x_0 - x_1  (constant along path)
  - At inference we integrate from t=1 → t=0 (noise → data)
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from openpi.noise_policy.adaln_attention import AdaLNAttentionBlock, AdaLNFinalLayer
from openpi.noise_policy.utils import SinusoidalPosEmb, init_weights

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embedding for scalar time values in [0, 1]."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,)
        device = t.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(
            torch.arange(half_dim, device=device, dtype=torch.float32) * -emb_scale
        )
        emb = t[:, None].float() * freqs[None, :]  # (B, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (B, dim)
        return emb


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for sequences."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# DINOv2 loader
# ---------------------------------------------------------------------------

DINOV2_REGISTERS_MODELS = {
    "small":  "facebook/dinov2-with-registers-small",
    "base":   "facebook/dinov2-with-registers-base",
    "large":  "facebook/dinov2-with-registers-large",
    "giant":  "facebook/dinov2-with-registers-giant",
}

# ImageNet normalisation constants (same as used by the HF processor)
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])

_UNUSED_TOKENS = 5  # 1 CLS + 4 register tokens


class Dinov2WithNorm(nn.Module):
    """
    DINOv2-with-registers image encoder.

    • Loads ``Dinov2WithRegistersModel`` from a local path or HF hub.
    • Backbone is frozen (``requires_grad_(False)``).
    • Optionally disables the final LayerNorm affine parameters so the output
      is purely normalised without learned scale/shift.
    • Returns **patch tokens** (strips the 5 special tokens at the front):
          shape  (B, N_patches, hidden_size)

    Input contract:
        x : (B, C, H, W)  float32 in [0, 1]  –or–  (B, H, W, C) / uint8.
        Internal ImageNet normalisation is applied automatically.
    """

    def __init__(self, dinov2_path: str = 'facebook/dinov2-with-registers-base', normalize: bool = True):
        super().__init__()
        try:
            from transformers import Dinov2WithRegistersModel
        except ImportError:
            raise ImportError(
                "transformers >= 4.39 is required for Dinov2WithRegistersModel. "
                "Install with: pip install -U transformers"
            )

        try:
            self.encoder = Dinov2WithRegistersModel.from_pretrained(
                dinov2_path, local_files_only=True
            )
        except (OSError, ValueError, AttributeError):
            self.encoder = Dinov2WithRegistersModel.from_pretrained(
                dinov2_path, local_files_only=False
            )

        self.encoder.requires_grad_(False)

        if normalize:
            self.encoder.layernorm.elementwise_affine = False
            self.encoder.layernorm.weight = None
            self.encoder.layernorm.bias = None

        self.patch_size  = self.encoder.config.patch_size
        self.hidden_size = self.encoder.config.hidden_size

        # Register ImageNet stats as buffers so they move with .to(device)
        self.register_buffer("_mean", _IMAGENET_MEAN.view(1, 3, 1, 1))
        self.register_buffer("_std",  _IMAGENET_STD.view(1, 3, 1, 1))

    def _normalise(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / self._std

    def _to_bchw_float(self, x: torch.Tensor) -> torch.Tensor:
        """Accept (B,H,W,C) or (H,W,C) or (B,C,H,W) → (B,C,H,W) float [0,1]."""
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.size(-1) in (1, 3):      # BHWC → BCHW
            x = x.permute(0, 3, 1, 2)
        x = x.float()
        if x.max() > 1.0:
            x = x / 255.0
        return x

    def dinov2_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) float, already ImageNet-normalised.
        Returns:
            patch_tokens: (B, N_patches, hidden_size)
        """
        out = self.encoder(x, output_hidden_states=True)
        return out.last_hidden_state[:, _UNUSED_TOKENS:]  # strip CLS + registers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) or (B, H, W, C) — float [0,1] or uint8.
        Returns:
            patch_tokens: (B, N_patches, hidden_size)
        """
        x = self._to_bchw_float(x).to(self._mean.device)
        x = self._normalise(x)
        return self.dinov2_forward(x)

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()   # backbone always frozen
        return self


def load_dinov2(
    size: str = "base",
    dinov2_path: Optional[str] = None,
    device: str | torch.device = "cpu",
    normalize: bool = True,
) -> Dinov2WithNorm:
    """
    Load a Dinov2WithNorm encoder.

    Args:
        size:         one of "small", "base", "large", "giant" — used when
                      ``dinov2_path`` is None to pick the HuggingFace model id.
        dinov2_path:  local path or full HuggingFace model id (overrides ``size``).
        device:       device to move the encoder to.
        normalize:    if True, disables the final LayerNorm affine params.

    Returns:
        Dinov2WithNorm instance (frozen, eval, on ``device``)

    Example::

        enc    = load_dinov2("base", device="cuda")
        policy = FlowMatchingPolicy(config, dino_encoder=enc)
    """
    path = dinov2_path or DINOV2_REGISTERS_MODELS.get(size)
    if path is None:
        raise ValueError(
            f"Unknown size '{size}'. Choose from {list(DINOV2_REGISTERS_MODELS)} "
            "or pass dinov2_path explicitly."
        )
    print(f"Loading Dinov2WithRegisters from '{path}' …")
    encoder = Dinov2WithNorm(dinov2_path=path, normalize=normalize)
    encoder.eval()
    encoder.to(device)
    print(f"  hidden_size={encoder.hidden_size}  |  device={device}")
    return encoder


# ---------------------------------------------------------------------------
# Multi-camera fuser
# ---------------------------------------------------------------------------

class MultiCameraEncoder(nn.Module):
    """
    Encode each camera independently with a shared Dinov2WithNorm encoder,
    mean-pool the patch tokens, then fuse all cameras by concatenation + linear.

    Data flow per camera:
        image (B,C,H,W) → Dinov2WithNorm → (B, N_patches, hidden_size)
                        → mean pool      → (B, hidden_size)
                        → per-cam proj   → (B, image_feature_dim)
    All cameras concatenated → (B, image_feature_dim * n_cams)
                             → fuse linear → (B, fused_dim)
    """

    def __init__(
        self,
        camera_names: Sequence[str],
        dino_encoder: Dinov2WithNorm,
        image_feature_dim: int = 256,
        fused_dim: int = 256,
    ):
        super().__init__()
        self.camera_names = list(camera_names)
        n_cams = len(camera_names)

        # Shared frozen backbone
        self.dino = dino_encoder
        hidden_size = dino_encoder.hidden_size

        # Learned projection from pooled patch features → image_feature_dim
        self.cam_proj = nn.Linear(hidden_size, image_feature_dim)

        # Fusion across cameras
        self.fuse = nn.Linear(image_feature_dim * n_cams, fused_dim)
        self.output_dim = fused_dim

    def _cams_from_image_dict(self, images: Mapping[str, torch.Tensor]) -> list[torch.Tensor]:
        cams = []
        for name in self.camera_names:
            if name not in images:
                raise KeyError(
                    f"Camera '{name}' not found in images dict. "
                    f"Available: {list(images.keys())}"
                )
            cams.append(images[name])
        return cams

    def forward(self, images: Mapping[str, torch.Tensor]) -> torch.Tensor:
        cams = self._cams_from_image_dict(images)
        cams = [c.unsqueeze(0) if c.ndim == 3 else c for c in cams]

        feats = []
        for c in cams:
            patch_tokens = self.dino(c)           # (B, N_patches, hidden_size)
            pooled       = patch_tokens.mean(1)   # (B, hidden_size)  — mean pool
            feats.append(self.cam_proj(pooled))   # (B, image_feature_dim)

        fused = torch.cat(feats, dim=-1)          # (B, image_feature_dim * n_cams)
        return self.fuse(fused)                   # (B, fused_dim)

class KeypointTrajectoryEncoder(nn.Module):
    """
    Encodes a human keypoint trajectory + extra proprioceptive dims using a 1D CNN over the time axis.
    """
    def __init__(
        self,
        num_kp_timesteps: int,      # 110
        kp_dim: int,                # 9  (3 keypoints × 3 coords)
        extra_dim: int,             # 8  (robot proprio / gripper)
        channels: Sequence[int] = (64, 128, 256),
        kernel_size: int = 3,
        extra_out_dim: int = 32,
    ):
        super().__init__()
        self.num_kp_timesteps = num_kp_timesteps
        self.kp_dim = kp_dim
        self.extra_dim = extra_dim
        self.state_dim = num_kp_timesteps * kp_dim + extra_dim

        # Build 1D conv stack
        layers = []
        in_ch = kp_dim
        for i, out_ch in enumerate(channels):
            stride = 1 if i == 0 else 2
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size,
                          stride=stride, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
            ]
            in_ch = out_ch

        self.cnn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        cnn_out_dim = channels[-1]

        # Extra dims (robot proprio / gripper)
        self.extra_proj = nn.Sequential(
            nn.Linear(extra_dim, extra_out_dim),
            nn.SiLU(),
        ) if extra_dim > 0 else None

        self.output_dim = cnn_out_dim + (extra_out_dim if extra_dim > 0 else 0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        B = state.shape[0]
        kp_flat = state[:, self.extra_dim:]   # (B, T*kp_dim)
        extra   = state[:,  :self.extra_dim]   # (B, extra_dim)

        kp = kp_flat.view(B, self.num_kp_timesteps, self.kp_dim)    # (B, T, kp_dim)
        kp = kp.transpose(1, 2)                                      # (B, kp_dim, T)

        feat   = self.cnn(kp)                                        # (B, 256, T')
        kp_emb = self.pool(feat).squeeze(-1)                         # (B, 256)

        if self.extra_proj is not None and extra.shape[-1] > 0:
            extra_emb = self.extra_proj(extra)                       # (B, extra_out_dim)
            return torch.cat([kp_emb, extra_emb], dim=-1)            # (B, output_dim)
        return kp_emb

# ---------------------------------------------------------------------------
# Observation encoder
# ---------------------------------------------------------------------------


class ObservationEncoder(nn.Module):
    """Encodes (images, state) → obs_emb. Supports conditioning on keypoints or one-hot task name."""

    def __init__(
        self,
        camera_names: Sequence[str],
        num_kp_timesteps,
        kp_dim,
        extra_dim,
        dino_encoder: Dinov2WithNorm,
        image_feature_dim: int = 256,
        fused_image_dim: int = 256,
        kp_channels=(64,128,256), kp_kernel_size=3, extra_out_dim=32,
        conditioning: str = "keypoints",  # "keypoints" or "onehot"
        task_names: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        if conditioning == "keypoints":
            self.state_dim = num_kp_timesteps * kp_dim + extra_dim
        self.conditioning = conditioning
        self.task_names = list(task_names) if task_names is not None else None

        self.camera_encoder = MultiCameraEncoder(
            camera_names=camera_names,
            dino_encoder=dino_encoder,
            image_feature_dim=image_feature_dim,
            fused_dim=fused_image_dim,
        )
        if conditioning == "keypoints":
            self.state_encoder = KeypointTrajectoryEncoder(
                num_kp_timesteps, kp_dim, extra_dim,
                channels=kp_channels, kernel_size=kp_kernel_size, extra_out_dim=extra_out_dim
            )
            self.output_dim = fused_image_dim + self.state_encoder.output_dim
        elif conditioning == "onehot":
            if self.task_names is None:
                raise ValueError("task_names must be provided for onehot conditioning")
            self.task_name_to_idx = {name: i for i, name in enumerate(self.task_names)}
            self.onehot_dim = len(self.task_names)
            self.output_dim = fused_image_dim + self.onehot_dim
        else:
            raise ValueError(f"Unknown conditioning type: {conditioning}")

    def _get_obs(self, observation: Any) -> Tuple[Mapping[str, torch.Tensor], torch.Tensor, Optional[str]]:
        # If onehot, expect observation to have 'task_name' key
        if hasattr(observation, "images") and hasattr(observation, "state"):
            images = observation.images
            state = observation.state
            task_name = getattr(observation, "task_name", None)
        elif isinstance(observation, Mapping):
            if "images" not in observation or "state" not in observation:
                raise ValueError("Expected dict with keys {'images','state'}")
            images = observation["images"]
            state = observation["state"]
            task_name = observation.get("task_name", None)
        else:
            raise TypeError(f"Expected Observation-like or dict, got {type(observation)}")

        if not isinstance(images, Mapping):
            raise TypeError(f"images must be Mapping, got {type(images)}")
        if not isinstance(state, torch.Tensor):
            raise TypeError(f"state must be torch.Tensor, got {type(state)}")
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if self.conditioning == "keypoints":
            if state.shape[-1] != self.state_dim:
                raise ValueError(
                    f"Expected state last dim={self.state_dim}, got {tuple(state.shape)}"
                )
        return images, state, task_name

    def forward(self, observation: Any) -> torch.Tensor:
        images, state, task_name = self._get_obs(observation)
        fused = self.camera_encoder(images)       # (B, fused_image_dim)
        if self.conditioning == "keypoints":
            s_emb = self.state_encoder(state.float())    # (B, state_proj_dim)
            return torch.cat([fused, s_emb], dim=-1)  # (B, obs_dim)
        elif self.conditioning == "onehot":
            # task_name must be provided for each sample in batch
            if task_name is None:
                raise ValueError("task_name must be provided in observation for onehot conditioning")
            # Support batch or single
            if isinstance(task_name, str):
                task_name = [task_name] * state.shape[0]
            onehot = torch.zeros((state.shape[0], self.onehot_dim), device=state.device)
            for i, name in enumerate(task_name):
                if name not in self.task_name_to_idx:
                    raise ValueError(f"Unknown task_name '{name}' for onehot encoding")
                onehot[i, self.task_name_to_idx[name]] = 1.0
            return torch.cat([fused, onehot], dim=-1)
        else:
            raise ValueError(f"Unknown conditioning type: {self.conditioning}")


# ---------------------------------------------------------------------------
# Transformer denoiser
# ---------------------------------------------------------------------------

def _act_fn(name: str) -> nn.Module:
    return {"relu": nn.ReLU(), "gelu": nn.GELU(), "silu": nn.SiLU(), "tanh": nn.Tanh()}[name.lower()]


class TransformerDenoiser(nn.Module):
    def __init__(self, 
        obs_dim: int = 320,
        action_dim: int = 7,
        action_horizon: int = 16,
        action_embedding_dim: int = 64,
        time_embedding_dim: int = 128,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        mlp_ratio: float = 4.0,
        positional_dropout: float = 0.1,
        output_hidden_dims: List[int] = [256],
        activation: str = "gelu",
        use_layer_norm: bool = True,
        dropout_rate: float = 0.0):

        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.d_model = d_model

        # ---- Time embedding (single timestep) ----
        self.time_embedding = SinusoidalPositionEmbedding(time_embedding_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_embedding_dim, time_embedding_dim * 2),
            nn.Mish(),
            nn.Linear(time_embedding_dim * 2, time_embedding_dim),
        )

        # ---- Action token embedding (no time concatenation) ----
        hidden_dim = d_model // 2
        self.action_embedding = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, d_model),
        )

        # ---- Learned positional embedding (like reference) ----
        self.pos_embed = nn.Parameter(
            torch.empty(1, action_horizon, d_model).normal_(std=0.02)
        )

        # ---- AdaLN conditioning dim: t_emb + obs_emb ----
        cond_dim = time_embedding_dim + obs_dim

        # ---- AdaLN DiT blocks ----
        self.blocks = nn.ModuleList([
            AdaLNAttentionBlock(
                dim=d_model,
                cond_dim=cond_dim,
                num_heads=nhead,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
            )
            for _ in range(num_encoder_layers)
        ])

        self.head = AdaLNFinalLayer(dim=d_model, cond_dim=cond_dim)

        # ---- Output decoder ----
        self.action_decoder = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        # Action embedding init
        w = self.action_embedding[0].weight.data
        nn.init.normal_(w.view([w.shape[0], -1]), mean=0.0, std=0.02)
        nn.init.constant_(self.action_embedding[0].bias, 0)

        # Zero-init AdaLN modulation (stable training start)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-init output head
        nn.init.constant_(self.head.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.head.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.head.linear.weight, 0)
        nn.init.constant_(self.head.linear.bias, 0)

    def forward(
        self,
        obs_emb: torch.Tensor,   # (B, obs_dim)
        x_t: torch.Tensor,       # (B, H, action_dim)
        t: torch.Tensor,         # (B,)
    ) -> torch.Tensor:

        B, H, _ = x_t.shape

        # ---- Time embedding ----
        t_emb = self.time_proj(self.time_embedding(t))   # (B, time_embedding_dim)

        # ---- Global conditioning: cat(t_emb, obs_emb) ----
        cond = torch.cat([t_emb, obs_emb], dim=-1)       # (B, cond_dim)

        # ---- Action tokens ----
        x = self.action_embedding(x_t)                   # (B, H, d_model)
        x = x + self.pos_embed                           # (B, H, d_model)

        # ---- DiT blocks ----
        for block in self.blocks:
            x = block(x, cond)

        x = self.head(x, cond)                           # (B, H, d_model)

        # ---- Decode ----
        velocity = self.action_decoder(x)                # (B, H, action_dim)

        return velocity

class MultiCamTransformerFlowMatchingPolicy(nn.Module):
    def __init__(
        self,
        num_kp_timesteps: int,
        kp_dim: int,
        extra_dim: int,
        action_dim: int,
        action_horizon: int,
        image_feature_dim: int = 256,
        fused_image_dim: int = 256,
        state_proj_dim: int = 64,
        num_flow_steps: int = 10,
        camera_names: List[str] = None,
        action_embedding_dim: int = 64,
        time_embedding_dim: int = 128,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        positional_dropout: float = 0.1,
        output_hidden_dims: List[int] = None,
        use_layer_norm: bool = True,
        dropout_rate: float = 0.0,
        img_encoder: str = "Dinov2WithNorm",
        conditioning: str = "keypoints",
        task_names: Optional[List[str]] = ["paired_purple", "paired_pan", "paired_yellow"],
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.dino_encoder = Dinov2WithNorm() if img_encoder == "Dinov2WithNorm" else None
        self.dino_encoder.eval()
        self.obs_encoder = ObservationEncoder(
            camera_names=camera_names,
            num_kp_timesteps=num_kp_timesteps,
            kp_dim=kp_dim,
            extra_dim=extra_dim,
            dino_encoder=self.dino_encoder,
            image_feature_dim=image_feature_dim,
            fused_image_dim=fused_image_dim,
            conditioning=conditioning,
            task_names=task_names if conditioning == "onehot" else None,
        )
        if output_hidden_dims is None:
            output_hidden_dims = [256]
            
        self.denoiser = TransformerDenoiser(
            obs_dim=self.obs_encoder.output_dim,
            action_dim=action_dim,
            action_horizon=action_horizon,
            action_embedding_dim=action_embedding_dim,
            time_embedding_dim=time_embedding_dim,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            mlp_ratio=4.0,
            positional_dropout=positional_dropout,
            output_hidden_dims=output_hidden_dims,
            activation="gelu",
            use_layer_norm=use_layer_norm,
            dropout_rate=dropout_rate,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self, observation: Any):
        return self.obs_encoder._get_obs(observation)

    def _encode_observation(self, observation: Any) -> torch.Tensor:
        return self.obs_encoder(observation)  # (B, obs_dim)

    def _denoise_step(
        self,
        obs_emb: torch.Tensor,  # (B, obs_dim)
        x_t: torch.Tensor,      # (B, H, action_dim)
        t: torch.Tensor,        # (B,)
    ) -> torch.Tensor:
        return self.denoiser(obs_emb, x_t, t)  # (B, H, action_dim)

    @staticmethod
    def sample_noise(shape: tuple, device: torch.device) -> torch.Tensor:
        return torch.randn(*shape, device=device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        observation: Any,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict velocity field.

        Args:
            observation: observation dict/object with 'images' and 'state'
            x_t:         noisy actions (B, H, action_dim)
            t:           time values (B,) in [0, 1]

        Returns:
            velocity (B, H, action_dim)
        """
        obs_emb = self._encode_observation(observation)
        if t.ndim == 0:
            t = t.expand(x_t.shape[0])
        return self._denoise_step(obs_emb, x_t, t)

    @torch.no_grad()
    def sample_actions(
        self,
        observation: Any,
        device: torch.device,
        noise: Optional[torch.Tensor] = None,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """
        Generate actions by integrating the learned velocity field from t=1 → t=0.

        Args:
            observation: observation dict/object
            device:      target device
            noise:       optional initial noise (B, H, action_dim); sampled if None
            num_steps:   number of Euler integration steps

        Returns:
            actions (B, H, action_dim)
        """
        _, state, _ = self._get_obs(observation)
        b = state.shape[0]

        if noise is None:
            noise = self.sample_noise(
                (b, self.action_horizon, self.action_dim), device
            )

        obs_emb = self._encode_observation(observation)

        dt = -1.0 / num_steps
        dt_t = torch.tensor(dt, dtype=torch.float32, device=device)
        x_t = noise.to(device=device, dtype=torch.float32)
        time = torch.tensor(1.0, dtype=torch.float32, device=device)

        while time > dt_t / 2:
            t_expanded = time.expand(b)
            v_t = self._denoise_step(obs_emb, x_t, t_expanded)
            x_t = x_t + dt_t * v_t
            time = time + dt_t

        return x_t  # (B, H, action_dim)


if __name__ == "__main__":
    import torch

    device = "cuda" 

    # --- Build policy for keypoints conditioning ---
    dino = Dinov2WithNorm()
    dino.eval().to(device)

    print("\nTesting keypoints conditioning:")
    policy_keypoints = MultiCamTransformerFlowMatchingPolicy(
        num_kp_timesteps=110,
        kp_dim=9,
        extra_dim=8,
        action_dim=7,
        action_horizon=16,
        camera_names=["front", "wrist"],
        image_feature_dim=256,
        fused_image_dim=256,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        img_encoder="Dinov2WithNorm",
        conditioning='keypoints'
    ).to(device)

    print(f"obs_encoder output_dim: {policy_keypoints.obs_encoder.output_dim}")

    # --- Fake batch ---
    B = 2
    obs = {
        "images": {
            "front": torch.randint(0, 256, (B, 3, 224, 224), dtype=torch.uint8).to(device),
            "wrist": torch.randint(0, 256, (B, 3, 224, 224), dtype=torch.uint8).to(device),
        },
        "state": torch.randn(B, 110 * 9 + 8).to(device),
        "task": "paired_purple",
    }
    actions = torch.randn(B, 16, 7).to(device)

    # --- Forward (training) ---
    t = torch.rand(B).to(device)
    x_1 = torch.randn_like(actions)
    x_t = (1 - t[:, None, None]) * actions + t[:, None, None] * x_1
    v_pred = policy_keypoints(obs, x_t, t)
    loss = torch.nn.functional.mse_loss(v_pred, actions - x_1)
    print(f"velocity shape: {v_pred.shape}")  # (2, 16, 7)
    print(f"loss: {loss.item():.4f}")

    # --- Inference ---
    policy_keypoints.eval()
    with torch.no_grad():
        sampled = policy_keypoints.sample_actions(obs, device=device, num_steps=10)
    print(f"sampled actions shape: {sampled.shape}")  # (2, 16, 7)

    # --- Build policy for onehot conditioning ---
    print("\nTesting onehot conditioning:")
    policy_onehot = MultiCamTransformerFlowMatchingPolicy(
        num_kp_timesteps=110,
        kp_dim=9,
        extra_dim=8,
        action_dim=7,
        action_horizon=16,
        camera_names=["front", "wrist"],
        image_feature_dim=256,
        fused_image_dim=256,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        img_encoder="Dinov2WithNorm",
        conditioning='onehot',
    ).to(device)

    obs_onehot = {
        "images": {
            "front": torch.randint(0, 256, (B, 3, 224, 224), dtype=torch.uint8).to(device),
            "wrist": torch.randint(0, 256, (B, 3, 224, 224), dtype=torch.uint8).to(device),
        },
        "state": torch.randn(B, 110 * 9 + 8).to(device),
        "task_name": "paired_purple",
    }
    actions = torch.randn(B, 16, 7).to(device)
    t = torch.rand(B).to(device)
    x_1 = torch.randn_like(actions)
    x_t = (1 - t[:, None, None]) * actions + t[:, None, None] * x_1
    v_pred = policy_onehot(obs_onehot, x_t, t)
    loss = torch.nn.functional.mse_loss(v_pred, actions - x_1)
    print(f"velocity shape: {v_pred.shape}")  # (2, 16, 7)
    print(f"loss: {loss.item():.4f}")

    # --- Inference ---
    policy_onehot.eval()
    with torch.no_grad():
        sampled = policy_onehot.sample_actions(obs_onehot, device=device, num_steps=10)
    print(f"sampled actions shape: {sampled.shape}")  # (2, 16, 7)