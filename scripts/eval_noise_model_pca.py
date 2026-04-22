"""
Evaluate the trained noise model on a random sample of dataset episodes.

- Loads N random episodes from the paired DROID dataset
- Runs the noise model (MultiCamTransformerFlowMatchingPolicy) in batches
- PCA overlay of GT noise vs predicted noise, labelled with per-sample L1 and MSE
"""

from __future__ import annotations

import argparse
import math
import pickle
import random
from pathlib import Path
from typing import Dict, Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from openpi.noise_policy.multicam_transformer_policy import MultiCamTransformerFlowMatchingPolicy


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT = Path("/gpfs/scrubbed/shubham/data/data_paired_droid")

TASKS = {
    "paired_pan":    DATA_ROOT / "paired_pan",
    "paired_purple": DATA_ROOT / "paired_purple",
    "paired_yellow": DATA_ROOT / "paired_yellow",
}

TASK_COLORS = {
    "paired_pan":    "#e07b39",
    "paired_purple": "#9b59b6",
    "paired_yellow": "#f1c40f",
}

TASK_SHORT = {
    "paired_pan":    "pan",
    "paired_purple": "purple",
    "paired_yellow": "yellow",
}

# Human trajectory is padded to this length; kp_dim = 9
HUMAN_T  = 110
HUMAN_KP = 9


# ---------------------------------------------------------------------------
# DCT helpers (same as pi0_inference.py)
# ---------------------------------------------------------------------------

def make_dct_basis(horizon: int, k: int, device, dtype=torch.float32) -> torch.Tensor:
    H, K = int(horizon), int(k)
    t  = torch.arange(H, device=device, dtype=dtype).view(H, 1)
    kk = torch.arange(K, device=device, dtype=dtype).view(1, K)
    B  = torch.cos((math.pi / H) * (t + 0.5) * kk)
    B[:, 0] *= math.sqrt(1.0 / H)
    if K > 1:
        B[:, 1:] *= math.sqrt(2.0 / H)
    return B  # (H, K)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_noise_model(checkpoint_path: str, device):
    print(f"Loading noise model from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    print(f"  action_dim={ckpt['action_dim']}  action_horizon={ckpt['action_horizon']}")

    model = MultiCamTransformerFlowMatchingPolicy(
        num_kp_timesteps=HUMAN_T,
        kp_dim=HUMAN_KP,
        extra_dim=8,
        action_dim=int(ckpt["action_dim"]),
        action_horizon=int(ckpt["action_horizon"]),
        camera_names=["exterior_image_1_left", "wrist_image_left"],
        image_feature_dim=256,
        fused_image_dim=256,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        img_encoder="Dinov2WithNorm",
        conditioning="onehot",
    ).to(device)

    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    norm = ckpt["norm"]
    state_mean  = norm["state_mean"].to(device)
    state_std   = norm["state_std"].to(device)
    noise_mean  = norm["noise_mean"].to(device)
    noise_std   = norm["noise_std"].to(device)
    print("  Noise model loaded OK")
    return model, state_mean, state_std, noise_mean, noise_std, int(ckpt["action_dim"]), int(ckpt["action_horizon"])


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def pad_human_traj(timesteps) -> np.ndarray:
    poses = np.stack([np.array(ts["hand_pose"]) for ts in timesteps], axis=0)  # (T, 9)
    if poses.shape[0] < HUMAN_T:
        pad = np.repeat(poses[-1:], HUMAN_T - poses.shape[0], axis=0)
        poses = np.concatenate([poses, pad], axis=0)
    return poses[:HUMAN_T].flatten()  # (990,)


def load_sample(fpath: Path, task_name: str):
    """Load one pkl file and return dict with all arrays needed for inference."""
    with open(fpath, "rb") as f:
        data = pickle.load(f)

    robot_ts = data["robot"]["timesteps"]

    # Use index 0 (first timestep)
    idx = 0
    joint_pos = np.array(robot_ts[idx]["observations"]["robot_state"]["joint_positions"])  # (7,)
    gripper   = np.array([robot_ts[idx]["observations"]["robot_state"]["gripper_position"]])  # (1,)
    cam1_img  = np.ascontiguousarray(robot_ts[idx]["observations"]["image"]["23804457_left"]).astype(np.uint8)
    cam2_img  = np.ascontiguousarray(robot_ts[idx]["observations"]["image"]["13263313_left"]).astype(np.uint8)

    human_traj = pad_human_traj(data["human"]["timesteps"])  # (990,)
    state = np.concatenate([joint_pos, gripper, human_traj])  # (998,)

    gt_noise = np.array(data["noise_action"][0])  # (action_horizon, action_dim)

    return {
        "state":    state,
        "cam1":     cam1_img[..., :3],
        "cam2":     cam2_img[..., :3],
        "gt_noise": gt_noise,
        "task":     task_name,
    }


def collect_samples(n: int, seed: int = 42):
    rng = random.Random(seed)
    all_files = []
    for task_name, task_dir in TASKS.items():
        files = sorted(task_dir.glob("*_noise.pkl"))
        print(f"  [{task_name}] {len(files)} files found")
        all_files.extend((f, task_name) for f in files)

    if len(all_files) < n:
        print(f"Warning: only {len(all_files)} files available, using all.")
        n = len(all_files)

    chosen = rng.sample(all_files, n)
    print(f"Loading {n} samples …")
    samples = []
    for fpath, task_name in chosen:
        try:
            samples.append(load_sample(fpath, task_name))
        except Exception as e:
            print(f"  Skip {fpath}: {e}")
    return samples


# ---------------------------------------------------------------------------
# Batched inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference_batched(
    samples,
    model,
    state_mean,
    state_std,
    noise_mean,
    noise_std,
    action_dim: int,
    action_horizon: int,
    device,
    batch_size: int = 16,
    dct_k: int = 0,
    num_flow_steps: int = 50,
):
    all_pred = []

    dct_basis = None
    flow_horizon = action_horizon if dct_k == 0 else dct_k
    if dct_k > 0:
        dct_basis = make_dct_basis(action_horizon, dct_k, device)  # (H, K)

    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        B = len(batch)

        # --- state ---
        states = torch.tensor(
            np.stack([s["state"] for s in batch]), dtype=torch.float32, device=device
        )  # (B, state_dim)
        states_norm = (states - state_mean) / (state_std + 1e-6)

        # --- images ---
        def to_tensor(imgs):
            return (
                torch.from_numpy(np.stack(imgs))  # (B, H, W, 3)
                .permute(0, 3, 1, 2)              # (B, 3, H, W)
                .to(device)
            )

        images: Dict[str, torch.Tensor] = {
            "exterior_image_1_left": to_tensor([s["cam1"] for s in batch]),
            "wrist_image_left":      to_tensor([s["cam2"] for s in batch]),
        }

        task_names = [TASK_SHORT[s["task"]] for s in batch]

        flow_obs: Mapping[str, Any] = {
            "images":    images,
            "state":     states_norm,
            "task_name": task_names[0],   # model may use this for conditioning
        }

        # Full ODE integration via sample_actions (num_flow_steps Euler steps)
        noise_seed = torch.randn(B, flow_horizon, action_dim, device=device)
        noise_norm = model.sample_actions(flow_obs, device, noise_seed, num_steps=num_flow_steps)

        noise_out = noise_norm * (noise_std + 1e-6) + noise_mean  # denormalize
        noise_l2 = noise_out.norm(dim=-1).mean().item()
        print(f"  Batch {start}–{start+B-1} | Pred noise L2 norm mean={noise_l2:.4f}")
        if dct_basis is not None:
            # noise_out: (B, K, action_dim) → (B, H, action_dim)
            noise_out = torch.einsum("hk,bkd->bhd", dct_basis, noise_out)

        all_pred.append(noise_out.cpu().numpy())
        # print(f"  Batch {start}–{start+B-1} done | pred mean={noise_out.mean():.4f}")

    return np.concatenate(all_pred, axis=0)  # (N, action_horizon, action_dim)


# ---------------------------------------------------------------------------
# PCA plot
# ---------------------------------------------------------------------------

def plot_pca_overlay(
    gt: np.ndarray,
    pred: np.ndarray,
    task_labels: list[str],
    out_path: Path,
    l1_per_sample: np.ndarray,
    mse_per_sample: np.ndarray,
):
    """PCA overlay of GT vs predicted noise, one plot per task + one combined."""
    N = len(gt)
    gt_flat   = gt.reshape(N, -1)    # (N, H*D)
    pred_flat = pred.reshape(N, -1)  # (N, H*D)

    combined = np.concatenate([gt_flat, pred_flat], axis=0)  # (2N, H*D)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(combined)
    var = pca.explained_variance_ratio_ * 100

    gt_2d   = coords[:N]
    pred_2d = coords[N:]

    tasks = list(TASKS.keys())
    ncols = len(tasks) + 1  # one per task + combined
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))

    def _panel(ax, mask, title_suffix):
        if mask is None:
            mask = np.ones(N, dtype=bool)
        ax.scatter(
            gt_2d[mask, 0], gt_2d[mask, 1],
            c="#2ecc71", alpha=0.7, s=50, edgecolors="white", linewidths=0.4,
            label="GT noise",
        )
        ax.scatter(
            pred_2d[mask, 0], pred_2d[mask, 1],
            c="#e74c3c", alpha=0.7, s=50, marker="^", edgecolors="white", linewidths=0.4,
            label="Predicted noise",
        )
        # Draw lines from GT to prediction for each sample
        for i in np.where(mask)[0]:
            ax.plot(
                [gt_2d[i, 0], pred_2d[i, 0]],
                [gt_2d[i, 1], pred_2d[i, 1]],
                color="gray", alpha=0.25, linewidth=0.6,
            )
        l1_m  = l1_per_sample[mask].mean()
        mse_m = mse_per_sample[mask].mean()
        ax.set_title(
            f"{title_suffix}\nL1={l1_m:.4f}  MSE={mse_m:.4f}",
            fontsize=11, fontweight="bold",
        )
        ax.set_xlabel(f"PC1 ({var[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({var[1]:.1f}%)")
        ax.legend(fontsize=9, framealpha=0.85)
        ax.grid(True, alpha=0.2)

    # Per-task panels
    for col, task_name in enumerate(tasks):
        mask = np.array([l == task_name for l in task_labels])
        _panel(axes[col], mask, TASK_SHORT[task_name].capitalize())

    # Combined panel
    _panel(axes[-1], None, "All tasks combined")

    overall_l1  = l1_per_sample.mean()
    overall_mse = mse_per_sample.mean()
    fig.suptitle(
        f"Noise model PCA — GT (●) vs Predicted (▲)\n"
        f"Overall  L1={overall_l1:.4f}   MSE={overall_mse:.6f}   N={N}",
        fontsize=13,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out_path.resolve()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate noise model with PCA overlay.")
    p.add_argument("--checkpoint", default="/gpfs/projects/weirdlab/shubham/openpi-steering-modes/runs/noise_transformer_flow_matching/pi0_droid_lora_finetune_data/no_robot_one_hot_real_world_20260415_035510/ckpt_0012000.pt")
    p.add_argument("--n_samples", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--dct_k", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="plots/noise_model_pca.png")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Device: {device}")

    # 1. Load model
    model, state_mean, state_std, noise_mean, noise_std, action_dim, action_horizon = \
        load_noise_model(args.checkpoint, device)

    # 2. Collect random samples
    print(f"\nCollecting {args.n_samples} random samples (seed={args.seed}) …")
    samples = collect_samples(args.n_samples, seed=args.seed)
    print(f"  Loaded {len(samples)} samples")

    # 3. Batched inference
    print(f"\nRunning inference (batch_size={args.batch_size}, dct_k={args.dct_k}) …")
    pred_noise = run_inference_batched(
        samples, model, state_mean, state_std, noise_mean, noise_std,
        action_dim, action_horizon, device,
        batch_size=args.batch_size,
        dct_k=args.dct_k,
    )  # (N, H, D)

    gt_noise = np.stack([s["gt_noise"] for s in samples])  # (N, H, D)

    # Align shapes if GT has more timesteps (e.g. raw vs DCT-expanded pred)
    H = min(gt_noise.shape[1], pred_noise.shape[1])
    gt_noise   = gt_noise[:, :H, :]
    pred_noise = pred_noise[:, :H, :]

    task_labels = [s["task"] for s in samples]

    # 4. Per-sample errors
    diff = gt_noise - pred_noise  # (N, H, D)
    l1_per_sample  = np.abs(diff).mean(axis=(1, 2))         # (N,)
    mse_per_sample = (diff ** 2).mean(axis=(1, 2))           # (N,)

    print(f"\nError summary over {len(samples)} samples:")
    print(f"  L1  mean={l1_per_sample.mean():.4f}  std={l1_per_sample.std():.4f}")
    print(f"  MSE mean={mse_per_sample.mean():.6f}  std={mse_per_sample.std():.6f}")

    # 5. PCA overlay plot
    print("\nGenerating PCA overlay …")
    plot_pca_overlay(
        gt_noise, pred_noise, task_labels,
        out_path=Path(args.out),
        l1_per_sample=l1_per_sample,
        mse_per_sample=mse_per_sample,
    )


if __name__ == "__main__":
    main()
