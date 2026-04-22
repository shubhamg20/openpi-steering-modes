"""
For a random sample of DROID dataset episodes, run pi0 implicit inversion to get the
noise vector at t=1, compute per-sample L1 norm, and plot a histogram.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run scripts/droid_noise_histogram.py \
        --policy_config_name pi0_droid_lora_finetune \
        --pytorch_weight_path /path/to/pytorch_checkpoint_dir \
        --droid_data_dir /gpfs/scrubbed/hongmm/droid/ \
        --n_episodes 200 \
        --num_steps 150 \
        --out plots/droid_noise_histogram.png
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import openpi.models.model as _model
import openpi.transforms as transforms
from openpi.policies import policy_config
from openpi.training import config as _config


# ──────────────────────────────────────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--policy_config_name", default="pi0_droid_lora_finetune")
    p.add_argument("--pytorch_weight_path",
                   default="/gpfs/scrubbed/shubham/chkpts/pi0_droid_no_lang/pytorch_160000/",
                   help="Dir containing model.safetensors for pi0.")
    p.add_argument("--droid_data_dir", default="/gpfs/scrubbed/hongmm/droid/1.0.1/",
                   help="Directory containing dataset_info.json and tfrecords (passed to tfds.builder_from_directory).")
    p.add_argument("--n_episodes", type=int, default=100,
                   help="Number of DROID episodes to sample.")
    p.add_argument("--steps_per_episode", type=int, default=1,
                   help="How many timesteps to take from each episode (1 = first only).")
    p.add_argument("--num_steps", type=int, default=150,
                   help="Implicit inversion ODE steps (used if --num_steps_list not set).")
    p.add_argument("--num_steps_list", type=str, default="10,50,150",
                   help="Comma-separated list of inversion steps to compare.")
    p.add_argument("--action_horizon", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="plots/droid_noise_histogram.png")
    p.add_argument("--action_stats_out", default="plots/droid_action_stats.png")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# DROID data loading (via tensorflow / dlimp)
# ──────────────────────────────────────────────────────────────────────────────

def load_droid_samples(
    data_dir: str,
    n_episodes: int,
    steps_per_episode: int,
    action_horizon: int,
    seed: int,
) -> list[dict]:
    """Load raw samples from DROID tfrecords.

    Returns list of dicts with keys:
        joint_position  (7,)
        gripper_position (1,)
        exterior_image  (H, W, 3) uint8
        wrist_image     (H, W, 3) uint8
        actions         (action_horizon, 8)  joint_position (7) + gripper (1)
        prompt          str

    data_dir should be the directory that contains dataset_info.json (and the
    tfrecord files), e.g. /gpfs/scrubbed/hongmm/droid/1.0.1/
    """
    import tensorflow as tf
    import tensorflow_datasets as tfds
    import dlimp as dl

    tf.config.set_visible_devices([], "GPU")

    # Use builder_from_directory so we don't need the dataset class registered.
    builder = tfds.builder_from_directory(data_dir)
    dataset = dl.DLataset.from_rlds(builder, split="train", shuffle=True,
                                     num_parallel_reads=tf.data.AUTOTUNE)

    # Keep only successful episodes
    dataset = dataset.filter(
        lambda traj: tf.strings.regex_full_match(
            traj["traj_metadata"]["episode_metadata"]["file_path"][0], ".*success.*"
        )
    )

    def restructure(traj):
        traj_len = tf.shape(traj["action"])[0]
        # action chunk indices per timestep, capped at end
        idx_base = tf.broadcast_to(tf.range(action_horizon)[None], [traj_len, action_horizon])
        idx_offset = tf.broadcast_to(tf.range(traj_len)[:, None], [traj_len, action_horizon])
        chunk_idx = tf.minimum(idx_base + idx_offset, traj_len - 1)
        # actions: joint_position (7) + gripper_position (1)
        all_actions = tf.concat([
            traj["action_dict"]["joint_position"],
            traj["action_dict"]["gripper_position"],
        ], axis=-1)  # (T, 8)
        actions_chunked = tf.gather(all_actions, chunk_idx)  # (T, H, 8)
        # randomly pick exterior image
        exterior = tf.cond(
            tf.random.uniform([]) > 0.5,
            lambda: traj["observation"]["exterior_image_1_left"],
            lambda: traj["observation"]["exterior_image_2_left"],
        )
        instruction = tf.random.shuffle([
            traj["language_instruction"],
            traj["language_instruction_2"],
            traj["language_instruction_3"],
        ])[0]
        return {
            "joint_position": traj["observation"]["joint_position"],       # (T, 7)
            "gripper_position": traj["observation"]["gripper_position"],    # (T, 1)
            "exterior_image": exterior,                                     # (T, H, W, 1) encoded
            "wrist_image": traj["observation"]["wrist_image_left"],         # (T, H, W, 1) encoded
            "actions": actions_chunked,                                     # (T, H, 8)
            "prompt": instruction,
        }

    def decode_images(frame):
        frame["exterior_image"] = tf.io.decode_image(
            frame["exterior_image"], expand_animations=False, dtype=tf.uint8)
        frame["wrist_image"] = tf.io.decode_image(
            frame["wrist_image"], expand_animations=False, dtype=tf.uint8)
        return frame

    dataset = dataset.traj_map(restructure, tf.data.AUTOTUNE)
    dataset = dataset.flatten(num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.frame_map(decode_images, tf.data.AUTOTUNE)
    dataset = dataset.shuffle(10_000, seed=seed)

    samples = []
    target = n_episodes * steps_per_episode
    print(f"Collecting {target} samples from DROID ...")
    for item in dataset.take(target).as_numpy_iterator():
        def to_chw_float(img):
            return (img.astype(np.float32) / 255.0).transpose(2, 0, 1)  # (H,W,3) -> (3,H,W) float32
        samples.append({
            "joint_position":   item["joint_position"].astype(np.float32),      # (7,)
            "gripper_position": item["gripper_position"].astype(np.float32),     # (1,)
            "exterior_image":   to_chw_float(item["exterior_image"]),            # (3,H,W) float32
            "wrist_image":      to_chw_float(item["wrist_image"]),               # (3,H,W) float32
            "actions":          item["actions"].astype(np.float32),              # (H, 8)
            "prompt":           "",  # no-lang model: always use empty prompt
        })
        if len(samples) % 50 == 0:
            print(f"  {len(samples)}/{target}")
    print(f"Loaded {len(samples)} samples.")
    return samples


# ──────────────────────────────────────────────────────────────────────────────
# Format samples into the pi0 observation format
# ──────────────────────────────────────────────────────────────────────────────

def samples_to_batch(samples: list[dict], transform, device: torch.device) -> tuple:
    """Apply the full openpi transform pipeline and batch samples."""
    processed = []
    for s in samples:
        # Must match ActDataset output format: RepackTransform looks up dot-separated keys.
        item = {
            "observation.joint_position":                      s["joint_position"],
            "observation.gripper_position":                    s["gripper_position"],
            "observation.images.exterior_image_1_left":        s["exterior_image"],
            "observation.images.wrist_image_left":             s["wrist_image"],
            "observation.human_traj":                          np.full(110 * 4, -1.0, dtype=np.float32),
            "action":                                          s["actions"],
            "action_is_pad":                                   np.zeros(s["actions"].shape[0], dtype=bool),
            "prompt":                                          s["prompt"],
            "task_name":                                       "",
        }
        processed.append(transform(item))

    # Collate into tensors
    def to_tensor(x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        if isinstance(x, bool):
            return torch.tensor(x, dtype=torch.bool)
        return x

    def collate(items):
        keys = items[0].keys()
        out = {}
        for k in keys:
            vals = [it[k] for it in items]
            if isinstance(vals[0], dict):
                out[k] = collate(vals)
            elif isinstance(vals[0], np.ndarray):
                out[k] = torch.from_numpy(np.stack(vals))
            elif isinstance(vals[0], (bool, np.bool_)):
                out[k] = torch.tensor(vals, dtype=torch.bool)
            else:
                out[k] = vals
        return out

    batch = collate(processed)

    def move(obj):
        if isinstance(obj, torch.Tensor):
            return obj.float().to(device) if obj.is_floating_point() else obj.to(device)
        if isinstance(obj, dict):
            return {k: move(v) for k, v in obj.items()}
        return obj

    return move(batch)


# ──────────────────────────────────────────────────────────────────────────────
# Inversion loop
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inversion(
    policy,
    samples: list[dict],
    transform,
    device: torch.device,
    batch_size: int,
    num_steps: int,
) -> np.ndarray:
    """Returns (l2_norms, l1_norms) arrays of shape (N,)."""
    all_l2 = []
    all_l1 = []
    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start : start + batch_size]
        batch = samples_to_batch(batch_samples, transform, device)
        obs = _model.Observation.from_dict(batch)
        actions = batch["actions"]
        if start == 0:
            print(f"  [sanity] actions mean={actions.mean():.3f} std={actions.std():.3f} "
                  f"min={actions.min():.3f} max={actions.max():.3f} shape={tuple(actions.shape)}")

        noise = policy._model.sample_noise_action_implicit(
            device=device,
            observation=obs,
            action=actions,
            num_steps=num_steps,
        )  # (B, H, D)

        noise_flat = noise.flatten(start_dim=1).cpu().numpy()  # (B, H*D)
        l2 = np.linalg.norm(noise_flat, axis=1)
        l1 = np.abs(noise_flat).mean(axis=1)
        all_l2.extend(l2.tolist())
        all_l1.extend(l1.tolist())
        print(f"  batch {start}–{start+len(batch_samples)-1}  L2 mean={l2.mean():.4f}  L1 mean={l1.mean():.4f}")

    return np.array(all_l2), np.array(all_l1)


# ──────────────────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────────────────

COLORS = ["#e74c3c", "#2980b9", "#27ae60", "#8e44ad", "#f39c12", "#16a085"]

ACTION_DIM_NAMES = [f"joint_{i}" for i in range(7)] + ["gripper"]


def plot_action_stats(samples: list[dict], out_path: Path) -> None:
    """Plot per-dimension mean and std of raw expert actions across all samples."""
    # actions: (H, 8) per sample → stack to (N, H, 8), then flatten over N*H
    actions = np.stack([s["actions"] for s in samples], axis=0)  # (N, H, 8)
    N, H, D = actions.shape
    flat = actions.reshape(-1, D)  # (N*H, 8)

    means = flat.mean(axis=0)  # (8,)
    stds = flat.std(axis=0)    # (8,)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(D)
    dim_labels = ACTION_DIM_NAMES[:D]

    ax = axes[0]
    ax.bar(x, means, color=COLORS[1], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(dim_labels, rotation=30, ha="right")
    ax.set_ylabel("Mean")
    ax.set_title(f"Expert action mean per dim\n(N={N} samples, H={H})", fontweight="bold")
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(x, stds, color=COLORS[0], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(dim_labels, rotation=30, ha="right")
    ax.set_ylabel("Std")
    ax.set_title(f"Expert action std per dim\n(N={N} samples, H={H})", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved action stats → {out_path.resolve()}")


def plot_comparison(
    results: dict[int, tuple[np.ndarray, np.ndarray]],
    out_path: Path,
    action_horizon: int = 10,
    action_dim: int = 32,
) -> None:
    """Plot L2 and L1 norm distributions for multiple inversion step counts."""
    d = action_horizon * action_dim
    n = next(iter(results.values()))[0].shape[0]
    gauss_flat = np.random.randn(n, d)
    gauss_l2 = np.linalg.norm(gauss_flat, axis=1)
    gauss_l1 = np.abs(gauss_flat).mean(axis=1)

    step_counts = sorted(results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, norm_idx, xlabel, title in [
        (axes[0], 0, "L2 norm", "Noise L2 norm vs inversion steps"),
        (axes[1], 1, "mean |noise| (L1)", "Noise L1 norm vs inversion steps"),
    ]:
        ax.hist(gauss_l2 if norm_idx == 0 else gauss_l1, bins=40,
                color="gray", alpha=0.3, edgecolor="none", label="Gaussian N(0,I)")
        ax.axvline((gauss_l2 if norm_idx == 0 else gauss_l1).mean(),
                   color="gray", linestyle="--", linewidth=1.2)

        for i, steps in enumerate(step_counts):
            vals = results[steps][norm_idx]
            color = COLORS[i % len(COLORS)]
            ax.hist(vals, bins=40, color=color, alpha=0.5, edgecolor="none",
                    label=f"{steps} steps (μ={vals.mean():.3f})")
            ax.axvline(vals.mean(), color=color, linestyle="--", linewidth=1.5)

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(f"{title}\n(N={n} DROID samples)", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out_path.resolve()}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # Build transform pipeline from config (normalization etc.)
    policy_cfg = _config.get_config(args.policy_config_name)
    data_cfg = policy_cfg.data.create(policy_cfg.data.assets.assets_dir, policy_cfg.model)

    if data_cfg.norm_stats is None:
        raise RuntimeError(
            "Norm stats failed to load (likely GCS download failure). "
            "Actions will be unnormalized, making inversion meaningless. "
            "Check that gs://openpi-assets/checkpoints/pi0_droid/assets/droid is accessible, "
            "or point --policy_config_name to a config with local assets."
        )
    print(f"Norm stats keys: {list(data_cfg.norm_stats.keys())}")
    for k, v in data_cfg.norm_stats.items():
        print(f"  {k}: mean shape={np.array(v.mean).shape}  mean[:4]={np.array(v.mean).flat[:4]}")

    norm = transforms.Normalize(data_cfg.norm_stats, use_quantiles=data_cfg.use_quantile_norm)
    transform = transforms.compose([
        *data_cfg.repack_transforms.inputs,
        *data_cfg.data_transforms.inputs,
        norm,
        *data_cfg.model_transforms.inputs,
    ])

    # Load DROID samples
    samples = load_droid_samples(
        data_dir=args.droid_data_dir,
        n_episodes=args.n_episodes,
        steps_per_episode=args.steps_per_episode,
        action_horizon=args.action_horizon,
        seed=args.seed,
    )

    plot_action_stats(samples, Path(args.action_stats_out))

    # Load policy
    print(f"Loading policy from: {args.pytorch_weight_path}")
    checkpoint_dir = Path(args.pytorch_weight_path).expanduser().resolve()
    policy = policy_config.create_trained_policy(
        policy_cfg,
        checkpoint_dir,
        pytorch_device=args.device,
    )

    step_counts = [int(s) for s in args.num_steps_list.split(",")]
    results: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for num_steps in step_counts:
        print(f"\nRunning implicit inversion ({num_steps} steps, batch_size={args.batch_size}) ...")
        l2_norms, l1_norms = run_inversion(
            policy, samples, transform, device,
            batch_size=args.batch_size,
            num_steps=num_steps,
        )
        results[num_steps] = (l2_norms, l1_norms)
        print(f"  L2: mean={l2_norms.mean():.4f}  std={l2_norms.std():.4f}  median={np.median(l2_norms):.4f}")
        print(f"  L1: mean={l1_norms.mean():.4f}  std={l1_norms.std():.4f}  median={np.median(l1_norms):.4f}")

    plot_comparison(results, Path(args.out), action_horizon=args.action_horizon)


if __name__ == "__main__":
    main()
