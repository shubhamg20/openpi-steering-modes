from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal, Tuple, List, Optional, Sequence
import torch
import torch.nn as nn
from torchvision import models


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embedding for time (standard in diffusion/flow-matching models)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) scalar time values in [0, 1]
        device = t.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb_scale)
        emb = t[:, None].float() * emb[None, :]  # (B, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (B, dim)
        return emb


class ResNetBackbone(nn.Module):
    """ResNet -> feature vector (B, D), with optional frozen backbone."""
    def __init__(self, size=34, pretrained=True, train_backbone=True, imagenet_norm=True):
        super().__init__()
        self.train_backbone = train_backbone
        self.imagenet_norm = imagenet_norm

        if size == 50:
            net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            self.out_dim = 2048
        elif size == 34:
            net = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
            self.out_dim = 512
        elif size == 18:
            net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            self.out_dim = 512
        else:
            raise ValueError(f"Unsupported size: {size}")

        # Remove final fc, keep avgpool => output (B, D, 1, 1)
        self.model = nn.Sequential(*list(net.children())[:-1])

        if self.imagenet_norm:
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.set_train_backbone(train_backbone)

    def set_train_backbone(self, train_backbone: bool):
        self.train_backbone = train_backbone
        for p in self.model.parameters():
            p.requires_grad = train_backbone
        # Freeze BN running stats if frozen
        if train_backbone:
            self.model.train()
        else:
            self.model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input: (B,3,H,W) float in [0,1] or [-1,1] (or uint8 [0,255]).
        # Also accepts a single image (3,H,W).
        if x.ndim == 3:
            x = x.unsqueeze(0)

        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected 3 channels, got shape={tuple(x.shape)}")

        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        else:
            x = x.float()

        if self.imagenet_norm:
            # If caller provides [-1,1], convert to [0,1] before ImageNet norm.
            if x.min() < 0:
                x = (x + 1.0) / 2.0
            x = (x - self.mean) / self.std

        with torch.set_grad_enabled(self.train_backbone):
            feat = self.model(x).flatten(1)  # (B, D)
        return feat


class MultiCamResNetPolicy(nn.Module):
    """
    Deterministic multi-camera ResNet policy producing an action HORIZON:
      per-cam ResNet -> fuse -> MLP -> linear head -> (B, T, A) (unbounded)

    Input (simplified, state is required):
      - Observation-like object with `.images` (dict) and `.state` tensor
      - dict in OpenPI repacked format: {"images": {cam_key: (B,3,H,W), ...}, "state": (B,S)}

    Output:
      - actions_hat: (B, T, A) (typically normalized actions during training)
    """
    def __init__(
        self,
        action_dim: int,                    # A
        action_horizon: int,                # T
        num_cameras: int = 3,
        resnet_size: int = 34,
        pretrained_backbone: bool = True,
        train_backbone: bool = True,
        imagenet_norm: bool = True,
        hidden_dims: Tuple[int, int] = (256, 256),
        fusion: Literal["concat", "mean", "max"] = "concat",
        shared_encoder: bool = True,
        camera_keys: Optional[Sequence[str]] = None,
        state_dim: int = 1,
        state_key: str = "state",
        state_proj_dim: Optional[int] = None,
    ):
        super().__init__()
        assert num_cameras >= 1
        assert action_dim >= 1
        assert action_horizon >= 1

        self.num_cameras = num_cameras
        self.fusion = fusion
        self.shared_encoder = shared_encoder
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.camera_keys = camera_keys
        self.state_dim = int(state_dim)
        self.state_key = state_key

        if self.state_dim <= 0:
            raise ValueError(f"state_dim must be > 0 (state is required), got {state_dim}")

        # Encoders
        if shared_encoder:
            self.encoder = ResNetBackbone(
                size=resnet_size,
                pretrained=pretrained_backbone,
                train_backbone=train_backbone,
                imagenet_norm=imagenet_norm,
            )
            enc_out = self.encoder.out_dim
        else:
            self.encoders = nn.ModuleList([
                ResNetBackbone(
                    size=resnet_size,
                    pretrained=pretrained_backbone,
                    train_backbone=train_backbone,
                    imagenet_norm=imagenet_norm,
                )
                for _ in range(num_cameras)
            ])
            enc_out = self.encoders[0].out_dim

        # Fusion output dim
        if fusion == "concat":
            fused_dim = enc_out * num_cameras
        elif fusion in ("mean", "max"):
            fused_dim = enc_out
        else:
            raise ValueError(f"Unknown fusion: {fusion}")

        # Optional state encoder (proprio / robot state)
        if state_proj_dim is None:
            self.state_proj = nn.Identity()
            state_out_dim = self.state_dim
        else:
            if state_proj_dim <= 0:
                raise ValueError(f"state_proj_dim must be > 0 when provided, got {state_proj_dim}")
            self.state_proj = nn.Sequential(
                nn.Linear(self.state_dim, state_proj_dim),
                nn.ReLU(inplace=True),
            )
            state_out_dim = int(state_proj_dim)

        # MLP trunk
        layers: List[nn.Module] = []
        last = fused_dim + state_out_dim
        for h in hidden_dims:
            layers += [nn.Linear(last, h), nn.ReLU(inplace=True)]
            last = h
        self.mlp = nn.Sequential(*layers)

        # Horizon head (unbounded)
        self.head = nn.Linear(last, action_horizon * action_dim)

    def set_train_backbone(self, train_backbone: bool):
        """Toggle frozen <-> end-to-end for the ResNet(s)."""
        if self.shared_encoder:
            self.encoder.set_train_backbone(train_backbone)
        else:
            for enc in self.encoders:
                enc.set_train_backbone(train_backbone)

    def _encode(self, cams: List[torch.Tensor]) -> torch.Tensor:
        feats = []
        if self.shared_encoder:
            for cam in cams:
                feats.append(self.encoder(cam))  # (B, D)
        else:
            for i, cam in enumerate(cams):
                feats.append(self.encoders[i](cam))  # (B, D)

        Fstack = torch.stack(feats, dim=1)  # (B, N, D)

        if self.fusion == "concat":
            return Fstack.flatten(1)        # (B, N*D)
        if self.fusion == "mean":
            return Fstack.mean(dim=1)       # (B, D)
        if self.fusion == "max":
            return Fstack.max(dim=1).values # (B, D)
        raise RuntimeError("unreachable")

    def _cams_from_image_dict(
        self,
        images: Mapping[str, torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Convert an OpenPI-style image dict to a list of camera tensors.

        images: dict[name -> tensor], typically name in {"base_0_rgb","left_wrist_0_rgb","right_wrist_0_rgb"}.
        """
        if self.camera_keys is not None:
            keys = list(self.camera_keys)
        else:
            # Default OpenPI camera ordering for the common 3-cam case.
            if self.num_cameras != 3:
                raise ValueError(
                    "num_cameras != 3 requires explicit camera_keys to map image dict -> camera list. "
                    f"Got num_cameras={self.num_cameras}."
                )
            keys = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]

        missing = [k for k in keys if k not in images]
        if missing:
            raise ValueError(f"images dict missing keys: {missing}. Present keys: {list(images.keys())}")

        return [images[k] for k in keys]

    def forward(
        self,
        pixels: Any,  # supports OpenPI Observation-like objects with .images/.state or a repacked dict
    ) -> torch.Tensor:
        """
        Returns:
          actions: (B, T, A) unbounded
        """
        # Observation-like path
        if hasattr(pixels, "images") and hasattr(pixels, "state"):
            images = pixels.images
            state = pixels.state
        # OpenPI repacked dict path: {"images": {...}, "state": ...}
        elif isinstance(pixels, Mapping):
            if "images" not in pixels or self.state_key not in pixels:
                raise ValueError(f"Expected dict with keys 'images' and '{self.state_key}', got keys={list(pixels.keys())}")
            images = pixels["images"]
            state = pixels[self.state_key]
        else:
            raise TypeError(
                "Expected an Observation-like object with .images/.state or a dict "
                f"{{'images': ..., '{self.state_key}': ...}}, got {type(pixels)}"
            )

        if not isinstance(images, Mapping):
            raise TypeError(f"Expected images to be a dict-like mapping, got {type(images)}")
        if not isinstance(state, torch.Tensor):
            raise TypeError(f"Expected state to be a torch.Tensor, got {type(state)}")

        cams = self._cams_from_image_dict(images)

        # Ensure batch size is consistent for dict/list inputs.
        B = cams[0].shape[0] if cams[0].ndim >= 4 else 1
        for i, cam in enumerate(cams):
            if cam.ndim == 3:
                cams[i] = cam.unsqueeze(0)
            if cams[i].shape[0] != B:
                raise ValueError(f"Camera batch sizes differ: cam0 has B={B}, cam{i} has B={cams[i].shape[0]}")

        fused = self._encode(cams)          # (B, fused_dim)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        state = state.float()
        if state.shape[0] != fused.shape[0]:
            raise ValueError(f"State batch size {state.shape[0]} != image batch size {fused.shape[0]}")
        if state.shape[-1] != self.state_dim:
            raise ValueError(f"Expected state last dim={self.state_dim}, got {tuple(state.shape)}")
        state_emb = self.state_proj(state)
        fused = torch.cat([fused, state_emb], dim=-1)

        x = self.mlp(fused)                 # (B, hidden)
        y = self.head(x)                    # (B, T*A)
        return y.view(B, self.action_horizon, self.action_dim)


class MultiCamResNetFlowMatchingPolicy(nn.Module):
    """Flow-matching policy that models a multi-modal distribution over noise/action trajectories.

    This is intentionally simple:
      - Reuses the same multi-cam ResNet encoder + fusion logic as `MultiCamResNetPolicy`
      - Learns a velocity field v_theta(x_t, t | obs)
      - Samples by explicit Euler integrating from t=1 -> t=0

    Expected inputs:
      - observation-like object with `.images` (dict) and `.state` tensor, OR
      - dict: {"images": {cam_key: (B,3,H,W), ...}, "state": (B,S)}

    Training call:
      v_t = model(observation, x_t, t)  where x_t is (B,T,A), t is (B,) in [0,1]
    """

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        action_horizon: int,
        num_cameras: int = 3,
        resnet_size: int = 34,
        pretrained_backbone: bool = True,
        train_backbone: bool = True,
        imagenet_norm: bool = True,
        fusion: Literal["concat", "mean", "max"] = "concat",
        shared_encoder: bool = True,
        camera_keys: Optional[Sequence[str]] = None,
        state_proj_dim: int = 64,
        hidden_dims: Tuple[int, ...] = (512, 512),
        use_film: bool = False,
        time_embed_dim: int = 128,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.num_cameras = int(num_cameras)
        self.camera_keys = camera_keys
        self.fusion = fusion
        self.shared_encoder = shared_encoder
        self.use_film = bool(use_film)
        self.time_embed_dim = int(time_embed_dim)

        # Encoders (same as MultiCamResNetPolicy)
        if shared_encoder:
            self.encoder = ResNetBackbone(
                size=resnet_size,
                pretrained=pretrained_backbone,
                train_backbone=train_backbone,
                imagenet_norm=imagenet_norm,
            )
            enc_out = self.encoder.out_dim
        else:
            self.encoders = nn.ModuleList(
                [
                    ResNetBackbone(
                        size=resnet_size,
                        pretrained=pretrained_backbone,
                        train_backbone=train_backbone,
                        imagenet_norm=imagenet_norm,
                    )
                    for _ in range(num_cameras)
                ]
            )
            enc_out = self.encoders[0].out_dim

        if fusion == "concat":
            fused_dim = enc_out * num_cameras
        elif fusion in ("mean", "max"):
            fused_dim = enc_out
        else:
            raise ValueError(f"Unknown fusion: {fusion}")

        self.state_proj = nn.Sequential(
            nn.Linear(self.state_dim, state_proj_dim),
            nn.ReLU(inplace=True),
        )

        # Time embedding: sinusoidal -> MLP (standard diffusion/flow-matching approach)
        time_hidden_dim = time_embed_dim
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_hidden_dim),
            nn.SiLU(),
            nn.Linear(time_hidden_dim, time_embed_dim),
        )

        # Velocity network:
        # - Default (use_film=False): concat conditioning [obs_emb, x_t_flat, time_emb] -> v
        # - FiLM (use_film=True): mlp_x([x_t_flat, time_emb]) then FiLM(obs_emb) then out -> v
        x_dim = self.action_horizon * self.action_dim
        obs_dim = fused_dim + state_proj_dim
        full_in_dim = obs_dim + x_dim + time_embed_dim
        x_in_dim = x_dim + time_embed_dim

        if not self.use_film:
            layers: List[nn.Module] = []
            last = full_in_dim
            for h in hidden_dims:
                layers += [nn.Linear(last, h), nn.LeakyReLU(negative_slope=0.01, inplace=True)]
                last = h
            layers.append(nn.Linear(last, x_dim))
            self.mlp = nn.Sequential(*layers)
        else:
            layers = []
            last = x_in_dim
            for h in hidden_dims:
                layers += [nn.Linear(last, h), nn.LeakyReLU(negative_slope=0.01, inplace=True)]
                last = h
            self.mlp_x = nn.Sequential(*layers)
            self.film = nn.Linear(obs_dim, 2 * last)
            self.out = nn.Linear(last, x_dim)

    def set_train_backbone(self, train_backbone: bool):
        if self.shared_encoder:
            self.encoder.set_train_backbone(train_backbone)
        else:
            for enc in self.encoders:
                enc.set_train_backbone(train_backbone)

    def sample_noise(self, shape, device):
        return torch.randn(shape, device=device, dtype=torch.float32)

    def _cams_from_image_dict(self, images: Mapping[str, torch.Tensor]) -> List[torch.Tensor]:
        if self.camera_keys is not None:
            keys = list(self.camera_keys)
        else:
            if self.num_cameras != 3:
                raise ValueError("num_cameras != 3 requires explicit camera_keys")
            keys = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
        missing = [k for k in keys if k not in images]
        if missing:
            raise ValueError(f"images dict missing keys: {missing}. Present keys: {list(images.keys())}")
        return [images[k] for k in keys]

    def _encode(self, cams: List[torch.Tensor]) -> torch.Tensor:
        feats = []
        if self.shared_encoder:
            for cam in cams:
                feats.append(self.encoder(cam))
        else:
            for i, cam in enumerate(cams):
                feats.append(self.encoders[i](cam))
        fstack = torch.stack(feats, dim=1)  # (B,N,D)
        if self.fusion == "concat":
            return fstack.flatten(1)
        if self.fusion == "mean":
            return fstack.mean(dim=1)
        if self.fusion == "max":
            return fstack.max(dim=1).values
        raise RuntimeError("unreachable")

    def _get_obs(self, observation: Any) -> tuple[Mapping[str, torch.Tensor], torch.Tensor]:
        # Observation-like
        if hasattr(observation, "images") and hasattr(observation, "state"):
            images = observation.images
            state = observation.state
        # Repacked dict
        elif isinstance(observation, Mapping):
            if "images" not in observation or "state" not in observation:
                raise ValueError("Expected dict with keys {'images','state'}")
            images = observation["images"]
            state = observation["state"]
        else:
            raise TypeError(f"Expected Observation-like or dict, got {type(observation)}")

        if not isinstance(images, Mapping):
            raise TypeError(f"images must be mapping, got {type(images)}")
        if not isinstance(state, torch.Tensor):
            raise TypeError(f"state must be torch.Tensor, got {type(state)}")
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.shape[-1] != self.state_dim:
            raise ValueError(f"Expected state last dim={self.state_dim}, got {tuple(state.shape)}")
        return images, state

    def _encode_observation(self, observation: Any) -> torch.Tensor:
        images, state = self._get_obs(observation)
        cams = self._cams_from_image_dict(images)
        cams = [c.unsqueeze(0) if c.ndim == 3 else c for c in cams]
        b = cams[0].shape[0]
        if any(c.shape[0] != b for c in cams):
            raise ValueError("Camera batch sizes differ")
        if state.shape[0] != b:
            raise ValueError(f"State batch size {state.shape[0]} != image batch size {b}")

        fused = self._encode(cams)  # (B,F)
        s_emb = self.state_proj(state.float())  # (B,S')
        return torch.cat([fused, s_emb], dim=-1)  # (B, obs_dim)

    def _denoise_step(self, obs_emb: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # obs_emb: (B,obs_dim), x_t: (B,T,A), t: (B,)
        b = x_t.shape[0] if x_t.ndim >= 3 else 1
        x_flat = x_t.reshape(b, -1)
        # Embed time with sinusoidal + MLP
        t_emb = self.time_embed(t.reshape(b).to(device=x_flat.device))  # (B, time_embed_dim)
        if not self.use_film:
            inp = torch.cat([obs_emb, x_flat, t_emb], dim=-1)
            v_flat = self.mlp(inp)
        else:
            h = self.mlp_x(torch.cat([x_flat, t_emb], dim=-1))
            gamma, beta = self.film(obs_emb).chunk(2, dim=-1)
            # Stabilize FiLM scale a bit.
            gamma = torch.tanh(gamma)
            h = (1.0 + gamma) * h + beta
            v_flat = self.out(h)
        return v_flat.view_as(x_t)

    def forward(self, observation: Any, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        obs_emb = self._encode_observation(observation)
        if t.ndim == 0:
            t = t.expand(x_t.shape[0])
        return self._denoise_step(obs_emb, x_t, t)

    @torch.no_grad()
    def sample_actions(self, observation: Any, device,      noise: torch.Tensor | None = None, num_steps: int = 10) -> torch.Tensor:
        b = self._get_obs(observation)[1].shape[0]
        if noise is None:
            noise = self.sample_noise((b, self.action_horizon, self.action_dim), device)
        obs_emb = self._encode_observation(observation)

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)
        x_t = noise.to(device=device, dtype=torch.float32)
        time = torch.tensor(1.0, dtype=torch.float32, device=device)

        while time >= -dt / 2:
            expanded_time = time.expand(b)
            v_t = self._denoise_step(obs_emb, x_t, expanded_time)
            x_t = x_t + dt * v_t
            time = time + dt
        return x_t


if __name__ == "__main__":
    B, H, W = 4, 224, 224
    num_cams = 3
    T, A = 50, 32
    S = 14

    policy = MultiCamResNetPolicy(
        action_dim=A,
        action_horizon=T,
        num_cameras=num_cams,
        resnet_size=34,
        pretrained_backbone=True,
        train_backbone=False,  # start frozen
        fusion="concat",
        shared_encoder=True,
        camera_keys=["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"],
        state_dim=S,
    )

    images = {
        "base_0_rgb": torch.rand(B, 3, H, W),
        "left_wrist_0_rgb": torch.rand(B, 3, H, W),
        "right_wrist_0_rgb": torch.rand(B, 3, H, W),
    }
    batch = {"images": images, "state": torch.randn(B, S)}
    out_batch = policy(batch)
    print("repacked-dict:", out_batch.shape)  # (B, T, A)
