"""Train MultiCamResNetFlowMatchingPolicy on ACT noise_action.

Flow-matching setup (very simple):
  - x0 = noise_action (normalized)
  - x1 ~ N(0, I)
  - t ~ Uniform(0, 1)
  - x_t = (1-t) * x0 + t * x1
  - target velocity v* = x1 - x0
  - train v_theta(x_t, t | obs) with MSE

This models a multi-modal distribution: sampling different x1 produces different outputs.
"""

from __future__ import annotations

import argparse
import datetime
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple
import random

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset
import tqdm
import time
import wandb  # type: ignore
import openpi.training.config as _config
from openpi.shared import normalize as _norm
from openpi.training.act_dataset import ActDataset

from openpi.noise_policy.multicam_resnet_policy import MultiCamResNetFlowMatchingPolicy
from openpi.noise_policy.multicam_transformer_policy import MultiCamTransformerFlowMatchingPolicy
from openpi.models_pytorch.pi0_pytorch import make_dct_basis

def _to_torch_1d(x) -> torch.Tensor:
    t = torch.as_tensor(x)
    if t.ndim != 1:
        t = t.reshape(-1)
    return t


def _apply_norm(x: torch.Tensor, *, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / (std + 1e-6)


class NoiseDataset(Dataset):
    """Returns dict: {"images":{...}, "state":..., "target": noise_action}."""

    def __init__(
        self,
        base: ActDataset,
        *,
        state_idxs: List[int],
        cam_keys: List[str],
        dct_basis: torch.Tensor | None = None,
        siglip_cache: Dict[int, torch.Tensor] | None = None,
    ):
        self.base = base
        self.state_idxs = state_idxs
        self.cam_keys = cam_keys
        self.dct_basis = dct_basis
        self.siglip_cache = siglip_cache  # {episode_id: (T, n_cams, 2048)} on CPU

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.base[idx]

        canonical = ["exterior_image_1_left", "wrist_image_left"]
        images: Dict[str, torch.Tensor] = {}

        if self.siglip_cache is not None:
            episode_id = item["episode_index"].item() if isinstance(item["episode_index"], torch.Tensor) else item["episode_index"]
            t = item["frame_index"].item() if isinstance(item["frame_index"], torch.Tensor) else item["frame_index"]
            pooled = self.siglip_cache[episode_id]  # (T, n_cams, 2048)
            for i, k in enumerate(canonical):
                images[k] = pooled[t, i]  # (2048,) pre-pooled feature
        else:
            for i in range(2):
                if i < len(self.cam_keys):
                    k = self.cam_keys[i]
                    img = item[f"observation.images.{k}"]
                else:
                    img = torch.zeros(3, 224, 224, dtype=torch.float32)
                images[canonical[i]] = img

        state = torch.cat([item["observation.joint_position"], item["observation.gripper_position"], item["observation.human_traj"]], dim=-1)
        state = state[self.state_idxs]
        assert state.shape[0] == len(self.state_idxs)
        if "noise_action" not in item:
            raise KeyError("Expected 'noise_action' in episode pickle.")
        target = item["noise_action"]  # (T, A)
        if self.dct_basis is not None:
            target = torch.matmul(self.dct_basis.transpose(0, 1), target)
        return {"images": images, "state": state, "target": target, "task_name": item["task_name"]}

def _precompute_siglip_features(
    episode_paths: List[str],
    cam_data_keys: List[str],
    encoder,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Dict[int, torch.Tensor]:
    """Precompute mean-pooled SigLIP features for all episodes into CPU memory.

    Uses a single flat DataLoader over all (episode, timestep) pairs for efficiency.

    Returns:
        cache: {episode_id: tensor(T, n_cams, 2048)} on CPU
    """
    import pickle
    import numpy as np
    from torch.utils.data import DataLoader as _DL, Dataset as _DS

    n_cams = len(cam_data_keys)

    class _FlatImgDS(_DS):
        """Flat dataset: one item per (episode, timestep), returns (episode_id, t, n_cams, 3, H, W)."""
        def __init__(self, ep_paths, keys):
            self.keys = keys
            self.index = []  # list of (episode_id, t, ep_path)
            for ep_id, ep_path in enumerate(ep_paths):
                with open(ep_path, "rb") as f:
                    data = pickle.load(f)
                if "robot" in data and "timesteps" in data["robot"]:
                    ts = data["robot"]["timesteps"]
                elif "timesteps" in data:
                    ts = data["timesteps"]
                else:
                    raise ValueError(f"No timesteps in {ep_path}")
                for t in range(len(ts)):
                    self.index.append((ep_id, t, ts[t]))

        def __len__(self):
            return len(self.index)

        def __getitem__(self, i):
            ep_id, t, timestep = self.index[i]
            imgs = []
            for k in self.keys:
                img = np.array(timestep["observations"]["image"][k])
                if img.shape[-1] == 4:
                    img = img[:, :, :3]
                imgs.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)
            return ep_id, t, torch.stack(imgs, dim=0)  # (n_cams, 3, H, W)

    encoder.eval()
    n_total = sum(1 for _ in open(episode_paths[0], "rb") or [])  # just for print
    print(f"[precompute] Building flat dataset over {len(episode_paths)} episodes ...")
    flat_ds = _FlatImgDS(episode_paths, cam_data_keys)
    print(f"[precompute] {len(flat_ds)} total timesteps — running SigLIP in one pass ...")
    loader = _DL(flat_ds, batch_size=batch_size, shuffle=False,
                 num_workers=num_workers, pin_memory=True)

    # Accumulate per-episode lists
    ep_feats: Dict[int, List[torch.Tensor]] = {}
    ep_ts: Dict[int, List[int]] = {}

    with torch.no_grad():
        for ep_ids, ts, imgs in tqdm.tqdm(loader, desc="siglip", dynamic_ncols=True):
            imgs = imgs.to(device)  # (B, n_cams, 3, H, W)
            cam_feats = []
            for c in range(n_cams):
                tokens = encoder(imgs[:, c])   # (B, N_tokens, 2048)
                cam_feats.append(tokens.mean(1).cpu())  # (B, 2048)
            pooled = torch.stack(cam_feats, dim=1)  # (B, n_cams, 2048)
            for b in range(pooled.shape[0]):
                eid = int(ep_ids[b])
                t = int(ts[b])
                ep_feats.setdefault(eid, []).append((t, pooled[b]))

    # Sort by timestep and pack into tensors
    cache: Dict[int, torch.Tensor] = {}
    for eid, items in ep_feats.items():
        items.sort(key=lambda x: x[0])
        cache[eid] = torch.stack([v for _, v in items], dim=0)  # (T, n_cams, 2048)

    print(f"[precompute] Done. {len(cache)} episodes cached.")
    return cache


def _infinite(loader: DataLoader) -> Iterator[Dict[str, Any]]:
    while True:
        for batch in loader:
            yield batch


def _parse_hidden_dims(s: str) -> tuple[int, ...]:
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    dims = tuple(int(p) for p in parts)
    if not dims or any(d <= 0 for d in dims):
        raise ValueError(f"Invalid --hidden-dims {s!r}. Example: '512,512'")
    return dims


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config-name", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dist-backend", default="nccl", help="DDP backend (nccl/gloo).")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--persistent-workers", action="store_true")
    p.add_argument("--num-steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--out-dir", default="/gpfs/scrubbed/shubham/chkpts-sft/noise_transformer_flow_matching")
    p.add_argument("--run-prefix", default=None)
    p.add_argument("--train-backbone", action="store_true")
    p.add_argument("--resnet-size", type=int, default=34)
    p.add_argument("--hidden-dims", default="512,512")
    p.add_argument("--state-proj-dim", type=int, default=64)
    p.add_argument("--shared-encoder", action="store_true", help="Use a shared ResNet encoder across cameras.")
    p.add_argument(
        "--deterministic-test",
        action="store_true",
        help="Debug: set x1=0 and t=0.5 so loss should go ~0 on tiny overfit.",
    )
    p.add_argument(
        "--l1-sample-flow",
        action="store_true",
        help="Use L1 sample prediction (predict clean x1 from noisy x_t). Default: velocity flow matching.",
    )
    p.add_argument("--val-fraction", type=float, default=0.1, help="Hold-out fraction for validation loss logging.")
    p.add_argument("--val-every", type=int, default=1000, help="Validate every N steps (if val-fraction>0).")
    p.add_argument("--wandb-project", type=str, default=None, help="If set, log to this W&B project.")
    p.add_argument("--use-film", action="store_true", help="Enable FiLM conditioning (default: off).")
    p.add_argument("--overfit-num", type=int, default=0)
    p.add_argument("--num-demos", type=int, default=0, help="Limit number of episodes used for training.")
    p.add_argument("--num-steps-euler", type=int, default=50, help="Euler steps used for sampling (saved in ckpt).")
    p.add_argument("--dct-k", type=int, default=0, help="If >0, train on DCT coeffs (KxA) instead of HxA.")
    p.add_argument("--img-encoder", type=str, default="Pi0SigLIP", help="Image encoder: 'Dinov2WithNorm' or 'Pi0SigLIP'.")
    p.add_argument("--pi0-checkpoint", type=str, default='/gpfs/scrubbed/shubham/chkpts/pi0_droid_no_lang/pytorch_160000/model.safetensors', help="Path to pi0 checkpoint (.pt) when --img-encoder=Pi0SigLIP.")
    return p.parse_args()


def _init_distributed(args: argparse.Namespace) -> Tuple[bool, int, int, int]:
    env_keys = {"RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "TORCHELASTIC_RUN_ID"}
    has_env = any(k in os.environ for k in env_keys)
    if not has_env:
        return False, 0, 1, 0

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
    elif "LOCAL_WORLD_SIZE" in os.environ:
        world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    else:
        raise RuntimeError("DDP env detected but WORLD_SIZE/LOCAL_WORLD_SIZE missing.")

    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
    else:
        node_rank = int(os.environ.get("GROUP_RANK", os.environ.get("NODE_RANK", "0")))
        rank = node_rank * world_size + local_rank

    if world_size <= 1:
        return False, 0, 1, 0

    dist.init_process_group(backend=str(args.dist_backend), init_method="env://")
    return True, rank, world_size, local_rank


def main() -> None:
    args = parse_args()
    distributed, rank, world_size, local_rank = _init_distributed(args)

    cfg = _config.get_config(args.config_name)

    norm_dir = Path(cfg.data.assets.assets_dir) / cfg.data.assets.asset_id
    norm_stats = _norm.load(norm_dir)
    
    # state slice (Vega)
    # state_joint_names = meta["state_joint_names"]
    # action_joint_names = meta["action_joint_names"]
    # state_idxs = list(vega_policy.make_action_idxs_in_state(state_joint_names, action_joint_names))
    state_idxs = list(range(8))
    state_dim = len(state_idxs)

    data_action_horizon = int(cfg.model.action_horizon)
    action_dim = int(cfg.model.action_dim)
    use_dct = int(args.dct_k) > 0
    dct_k = int(args.dct_k) if use_dct else data_action_horizon
    if use_dct and dct_k > data_action_horizon:
        raise ValueError(f"--dct-k must be <= action_horizon ({data_action_horizon}), got {dct_k}")


    dct_basis_cpu = None
    if use_dct:
        dct_basis_cpu = make_dct_basis(data_action_horizon, dct_k, device="cpu", dtype=torch.float32)

    base = ActDataset(
        root_dir=args.data_root,
        action_horizon=data_action_horizon,
        with_noise=True,
        extra_episode_keys=["noise_action"],
    )
    episode_paths = base.episode_paths
    episode_lengths = base.episode_lengths
    cam_keys: List[str] = base.image_keys

    # siglip_cache built later (after device + model are ready); placeholder here
    full_dataset: Dataset = NoiseDataset(
        base, state_idxs=state_idxs, cam_keys=cam_keys, dct_basis=dct_basis_cpu
    )
    _siglip_cache_ref: List[Dict[int, torch.Tensor]] = [None]  # filled after model is on device
    if args.overfit_num and int(args.overfit_num) > 0:
        n = min(int(args.overfit_num), len(full_dataset))
        full_dataset = Subset(full_dataset, list(range(n)))
        print(f"[debug] overfit-num={args.overfit_num} -> dataset_len={len(full_dataset)}")

    val_dataset = None
    train_dataset = full_dataset
    if float(args.val_fraction) > 0 and len(full_dataset) > 1:
        val_size = max(1, int(len(full_dataset) * float(args.val_fraction)))
        train_size = len(full_dataset) - val_size
        if train_size <= 0:
            train_size = len(full_dataset) - 1
            val_size = 1
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        print(f"[split] train={len(train_dataset)} val={len(val_dataset)}")

    batch_size = int(args.batch_size or cfg.batch_size)
    num_workers = int(args.num_workers or cfg.num_workers)
    if distributed and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device)

    loader_kwargs: Dict[str, Any] = {}
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(args.prefetch_factor)
        loader_kwargs["persistent_workers"] = bool(args.persistent_workers)

    train_sampler = None
    if distributed:
        train_sampler = torch.utils.data.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False
        )
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        **loader_kwargs,
    )
    it = _infinite(loader)
    val_loader = None
    if val_dataset is not None:
        if not distributed or rank == 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=device.type == "cuda",
                **loader_kwargs,
            )

    # norms: slice state stats; noise uses full dim
    state_mean = _to_torch_1d(norm_stats["state"].mean).float()[state_idxs].to(device)
    state_std = _to_torch_1d(norm_stats["state"].std).float()[state_idxs].to(device)
    noise_mean_raw = _to_torch_1d(norm_stats["noise_action"].mean).float()
    noise_std_raw = _to_torch_1d(norm_stats["noise_action"].std).float()

    if use_dct:
        # Expand stats to (H, A) for DCT projection if needed.
        if noise_mean_raw.numel() == action_dim:
            noise_mean_raw = noise_mean_raw.view(1, action_dim).repeat(data_action_horizon, 1)
            noise_std_raw = noise_std_raw.view(1, action_dim).repeat(data_action_horizon, 1)
        elif noise_mean_raw.numel() == data_action_horizon * action_dim:
            noise_mean_raw = noise_mean_raw.view(data_action_horizon, action_dim)
            noise_std_raw = noise_std_raw.view(data_action_horizon, action_dim)
        else:
            raise ValueError(
                "noise_action stats shape mismatch: "
                f"expected {action_dim} or {data_action_horizon * action_dim} values, "
                f"got {noise_mean_raw.numel()}"
            )

        dct_basis = make_dct_basis(data_action_horizon, dct_k, device=device, dtype=torch.float32)
        noise_mean = torch.matmul(dct_basis.transpose(0, 1), noise_mean_raw.to(device))
        noise_var = torch.matmul((dct_basis**2).transpose(0, 1), (noise_std_raw.to(device) ** 2))
        noise_std = torch.sqrt(noise_var.clamp_min(1e-12))
    else:
        if noise_mean_raw.numel() == data_action_horizon * action_dim:
            noise_mean = noise_mean_raw.view(data_action_horizon, action_dim).to(device)
            noise_std = noise_std_raw.view(data_action_horizon, action_dim).to(device)
        else:
            noise_mean = noise_mean_raw.to(device)
            noise_std = noise_std_raw.to(device)

    # model = MultiCamResNetFlowMatchingPolicy(
    #     state_dim=state_dim,
    #     action_dim=action_dim,
    #     action_horizon=dct_k,
    #     resnet_size=int(args.resnet_size),
    #     pretrained_backbone=True,
    #     train_backbone=bool(args.train_backbone),
    #     # camera_keys=("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
    #     num_cameras=2,
    #     camera_keys=("exterior_image_1_left", "wrist_image_left"),
    #     fusion="concat",
    #     shared_encoder=bool(args.shared_encoder),
    #     state_proj_dim=int(args.state_proj_dim),
    #     hidden_dims=_parse_hidden_dims(args.hidden_dims),
    #     use_film=bool(args.use_film),
    # ).to(device)
    
    model = MultiCamTransformerFlowMatchingPolicy(
        num_kp_timesteps=110,
        kp_dim=9,
        extra_dim=0,
        action_dim=action_dim,
        action_horizon=dct_k,
        camera_names=["exterior_image_1_left", "wrist_image_left"],
        image_feature_dim=256,   #does not matter for Pi0SigLIP
        fused_image_dim=256,     #does not matter for Pi0SigLIP
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        img_encoder=args.img_encoder,
        pi0_checkpoint_path=args.pi0_checkpoint,
        conditioning="onehot",
        state_dim=state_dim,
    ).to(device)

    # Precompute SigLIP features — all ranks work in parallel, then share results.
    my_episodes = base.episode_paths[rank::world_size] if distributed else base.episode_paths
    my_episode_ids = list(range(rank, len(base.episode_paths), world_size)) if distributed else list(range(len(base.episode_paths)))
    partial_cache = _precompute_siglip_features(
        episode_paths=my_episodes,
        cam_data_keys=base.data_keys,
        encoder=model.dino_encoder,
        device=device,
        batch_size=256,
        num_workers=num_workers,
    )
    # Re-key partial cache to global episode ids
    partial_cache_global = {my_episode_ids[k]: v for k, v in partial_cache.items()}
    if distributed:
        all_partial: List[Dict[int, torch.Tensor]] = [None] * world_size
        dist.all_gather_object(all_partial, partial_cache_global)
        siglip_cache: Dict[int, torch.Tensor] = {}
        for d in all_partial:
            siglip_cache.update(d)
    else:
        siglip_cache = partial_cache_global
    # Inject cache into the dataset (works through Subset wrappers too)
    _ds = full_dataset
    while hasattr(_ds, "dataset"):
        _ds = _ds.dataset
    _ds.siglip_cache = siglip_cache

    if distributed:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    num_steps = int(args.num_steps or cfg.num_train_steps)
    if isinstance(model, MultiCamTransformerFlowMatchingPolicy):
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        warmup = LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=1000)
        cosine = CosineAnnealingLR(opt, T_max=num_steps - 1000, eta_min=1e-6)
        scheduler = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[1000])

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = str(args.run_prefix).strip() if args.run_prefix is not None else ""
    run_name = f"{prefix}_{ts}" if prefix else ts
    out_dir = Path(args.out_dir) / args.config_name / run_name
    if not distributed or rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[train] run_dir={out_dir}")
    wb = None
    if args.wandb_project and (not distributed or rank == 0):
        wb = wandb.init(
            project=args.wandb_project,
            entity="shubham2-university-of-washington",
            name=run_name,
        )

    num_steps = int(args.num_steps or cfg.num_train_steps)
    model.train()
    if not distributed or rank == 0:
        pbar = tqdm.tqdm(range(1, num_steps + 1), dynamic_ncols=True, desc="train_fm")
    else:
        pbar = range(1, num_steps + 1)
    loss_ema = None
    last_t = time.perf_counter()
    data_time_ema = None
    step_time_ema = None
    prev_epoch_int = -1
    last_sample_mse = None  # Track latest validation sample MSE for tqdm display
    for step in pbar:
        if train_sampler is not None:
            steps_per_epoch = max(1, math.ceil(len(train_dataset) / (batch_size * world_size)))
            epoch_int = (step - 1) // steps_per_epoch
            if epoch_int != prev_epoch_int:
                train_sampler.set_epoch(epoch_int)
                prev_epoch_int = epoch_int

        batch = next(it)
        now = time.perf_counter()
        data_dt = now - last_t
        last_t = now

        images = {k: v.to(device, non_blocking=True) for k, v in batch["images"].items()}
        state = batch["state"].to(device, non_blocking=True).float()
        x0 = batch["target"].to(device, non_blocking=True).float()  # (B,T,A)

        if step == 1 and (not distributed or rank == 0):
            img0 = images["exterior_image_1_left"]
            print(
                f"[sanity] exterior_image_1_left shape={tuple(img0.shape)} "
                f"min={float(img0.min()):.4f} max={float(img0.max()):.4f} mean={float(img0.mean()):.4f}"
            )

        # Normalize to match your existing pipeline.
        state = _apply_norm(state, mean=state_mean, std=state_std)
        x0 = _apply_norm(x0, mean=noise_mean, std=noise_std)

        b = x0.shape[0]
        obs = {"images": images, "state": state, "task_name": batch["task_name"]}

        if args.l1_sample_flow:
            # L1 sample prediction: predict clean x1 from noisy interpolation x_t.
            x1 = x0  # clean target (normalized)
            if args.deterministic_test:
                x0_noise = torch.zeros_like(x1)
                t = torch.full((b,), 0.5, device=device, dtype=torch.float32)
            else:
                x0_noise = torch.randn_like(x1)
                t = torch.rand(b, device=device, dtype=torch.float32) 
            t_view = t.view(b, 1, 1)
            x_t = (1.0 - t_view) * x0_noise + t_view * x1
            x1_hat = model(obs, x_t, t)  # (B,T,A)
            loss = F.l1_loss(x1_hat, x1)
            baseline = F.l1_loss(torch.zeros_like(x1), x1).item()
        else:
            # Original velocity flow matching.
            x1 = torch.randn_like(x0)
            t = torch.distributions.Beta(1.5, 1).sample((b,)).to(device=device, dtype=torch.float32) * 0.999 + 0.001  # (B,) Beta(1.5,1) like pi0
            t_view = t.view(b, 1, 1)
            x_t = (1.0 - t_view) * x0 + t_view * x1
            v_target = x1 - x0

            v_pred = model(obs, x_t, t)  # (B,T,A)
            loss = F.smooth_l1_loss(v_pred, v_target, beta=0.1)
            baseline = F.smooth_l1_loss(torch.zeros_like(v_target), v_target, beta=0.1).item()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if isinstance(model, MultiCamTransformerFlowMatchingPolicy):
            scheduler.step()
        step_dt = time.perf_counter() - now

        loss_item = torch.as_tensor(loss.detach())
        if distributed:
            dist.all_reduce(loss_item, op=dist.ReduceOp.SUM)
            loss_item = loss_item / float(world_size)
        loss_item = float(loss_item.item())

        if loss_ema is None:
            loss_ema = loss_item
            data_time_ema = data_dt
            step_time_ema = step_dt
        else:
            loss_ema = 0.9 * loss_ema + 0.1 * loss_item
            assert data_time_ema is not None and step_time_ema is not None
            alpha = 0.1
            data_time_ema = (1 - alpha) * data_time_ema + alpha * data_dt
            step_time_ema = (1 - alpha) * step_time_ema + alpha * step_dt

        # Epoch estimate based on actual batch size.
        actual_bs = int(x0.shape[0])
        epoch = (step * actual_bs) / max(1, len(train_dataset))

        # Always show live metrics in tqdm (like train_noise_resnet.py).
        if not distributed or rank == 0:
            postfix_dict = {
                "loss": f"{loss_ema:.4f}",
                "base": f"{baseline:.4f}",
                "epoch": f"{epoch:.2f}",
            }
            if last_sample_mse is not None:
                postfix_dict["smse"] = f"{last_sample_mse:.4f}"
            pbar.set_postfix(**postfix_dict)

        if wb is not None and step % int(args.log_every) == 0:
            wb.log({"train/l1_loss": loss_item, "train/baseline": baseline, "train/epoch_est": epoch}, step=step)

        if val_loader is not None and step % int(args.val_every) == 0:
            model.eval()
            # Get the underlying model for sampling (handle DDP)
            sample_model = model.module if distributed else model
            with torch.no_grad():
                val_losses = []
                sample_mse_sum = 0.0
                sample_count = 0
                for vbatch in val_loader:
                    v_images = {k: v.to(device, non_blocking=True) for k, v in vbatch["images"].items()}
                    v_state = vbatch["state"].to(device, non_blocking=True).float()
                    vx0 = vbatch["target"].to(device, non_blocking=True).float()
                    v_state = _apply_norm(v_state, mean=state_mean, std=state_std)
                    vx0 = _apply_norm(vx0, mean=noise_mean, std=noise_std)
                    bval = vx0.shape[0]
                    v_obs = {"images": v_images, "state": v_state, "task_name": vbatch["task_name"]}
                    if args.l1_sample_flow:
                        vx1 = vx0  # clean target
                        if args.deterministic_test:
                            vx0_noise = torch.zeros_like(vx1)
                            vt = torch.full((bval,), 0.5, device=device, dtype=torch.float32)
                        else:
                            vx0_noise = torch.randn_like(vx1)
                            vt = torch.rand(bval, device=device, dtype=torch.float32)
                        vt_view = vt.view(bval, 1, 1)
                        vx_t = (1.0 - vt_view) * vx0_noise + vt_view * vx1
                        vx1_hat = model(v_obs, vx_t, vt)
                        vloss = F.l1_loss(vx1_hat, vx1)
                    else:
                        vx1 = torch.randn_like(vx0)
                        vt = torch.rand(bval, device=device, dtype=torch.float32)
                        vt_view = vt.view(bval, 1, 1)
                        vx_t = (1.0 - vt_view) * vx0 + vt_view * vx1
                        vv_target = vx1 - vx0
                        vv_pred = model(v_obs, vx_t, vt)
                        vloss = F.smooth_l1_loss(vv_pred, vv_target, beta=0.1)
                    val_losses.append(float(vloss.item()))

                    # Sample reconstruction MSE: run full Euler integration and compare to target
                    pred_sample = sample_model.sample_actions(
                        v_obs, device, num_steps=int(args.num_steps_euler)
                    )  # (B, T, A)
                    sample_mse = F.mse_loss(pred_sample, vx0, reduction="sum").item()
                    sample_mse_sum += sample_mse
                    sample_count += vx0.numel()

                val_loss_mean = float(sum(val_losses) / max(1, len(val_losses)))
                sample_mse_mean = sample_mse_sum / max(1, sample_count)
                last_sample_mse = sample_mse_mean
            print(f"Validation: loss={val_loss_mean:.4f}, sample_mse={sample_mse_mean:.4f}")
            model.train()
            if wb is not None:
                wb.log(
                    {"val/fm_loss": val_loss_mean, "val/sample_mse": sample_mse_mean},
                    step=step,
                )

        if step % int(args.save_every) == 0 and (not distributed or rank == 0):
            torch.save(
                {
                    "step": step,
                    "model": model.module.state_dict() if distributed else model.state_dict(),
                    "opt": opt.state_dict(),
                    "config_name": args.config_name,
                    "state_dim": state_dim,
                    "action_dim": action_dim,
                    "action_horizon": dct_k,
                    "dct_k": int(dct_k) if use_dct else 0,
                    "data_action_horizon": int(data_action_horizon),
                    "resnet_size": int(args.resnet_size),
                    "hidden_dims": list(_parse_hidden_dims(args.hidden_dims)),
                    "state_proj_dim": int(args.state_proj_dim),
                    "use_film": bool(args.use_film),
                    "num_steps_euler": int(args.num_steps_euler),
                    "norm": {
                        "state_idxs": list(state_idxs),
                        "state_mean": state_mean.detach().cpu(),
                        "state_std": state_std.detach().cpu(),
                        "noise_mean": noise_mean.detach().cpu(),
                        "noise_std": noise_std.detach().cpu(),
                    },
                },
                out_dir / f"ckpt_{step:07d}.pt",
            )
            print(f"[checkpoint] step={step} saved to {out_dir / f'ckpt_{step:07d}.pt'}")

    if not distributed or rank == 0:
        torch.save(
            {"step": num_steps, "model": model.module.state_dict() if distributed else model.state_dict()},
            out_dir / "final.pt",
        )

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()


