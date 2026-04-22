"""Compare inversion quality across different num_steps for noise inversion.

For a small number of dataset samples, runs implicit inversion with each of the
specified num_steps values, then cycle-verifies by reconstructing actions and
computing MSE/L1 vs ground truth. Produces a summary plot and table.

Usage:
    uv run scripts/compare_inversion_steps.py \
        --policy_config_name pi0_droid_lora_finetune \
        --data_config_name pi0_droid_lora_finetune \
        --pytorch_weight_path /gpfs/scrubbed/shubham/chkpts/pi0_droid_discrete_time/pytorch_80000/ \
        --num_samples 50 \
        --steps 10,50,100,150,200 \
        --method implicit
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import openpi.models.model as _model
import openpi.transforms as transforms
from openpi.policies import policy_config
from openpi.training import config as _config
from openpi.training.act_dataset import ActDataset


DEFAULT_STEPS = [10, 50, 100, 150, 200, 250, 300, 500, 1000]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare inversion quality vs num_steps.")
    parser.add_argument("--policy_config_name", required=True)
    parser.add_argument("--data_config_name", required=True)
    parser.add_argument("--pytorch_weight_path", required=True)
    parser.add_argument(
        "--data_root",
        default=os.environ.get("ACT_DATA_ROOT", "/gpfs/scrubbed/shubham/data/data_paired_droid"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--steps",
        default=",".join(str(s) for s in DEFAULT_STEPS),
        help="Comma-separated list of num_steps values to compare.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=64,
        help="Total number of dataset samples to evaluate over.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--method",
        choices=["implicit", "explicit"],
        default="implicit",
        help="Inversion method to benchmark.",
    )
    parser.add_argument(
        "--true_action_dim",
        type=int,
        default=8,
        help="Number of real (non-padded) action dims to use for metrics.",
    )
    parser.add_argument(
        "--out_dir",
        default="plots/inversion_steps_compare",
        help="Output directory for plots and results.",
    )
    return parser.parse_args()


def move_to_device(obj: Any, device: torch.device):
    if isinstance(obj, (bool, np.bool_)):
        return torch.tensor(bool(obj), dtype=torch.bool, device=device)
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj).to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    return obj


class TransformingDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        return self.transform(self.base_dataset[idx])


def evaluate_steps(
    policy,
    loader: DataLoader,
    num_steps: int,
    method: str,
    true_action_dim: int,
    device: torch.device,
) -> dict[str, float]:
    """Run inversion + reconstruction + re-inversion for all batches; return aggregate metrics."""
    mse_vals: list[float] = []
    l1_vals: list[float] = []
    noise_mse_vals: list[float] = []
    noise_l1_vals: list[float] = []
    noise_l2_norm_vals: list[float] = []
    noise_hat_l2_norm_vals: list[float] = []
    elapsed_inversion = 0.0

    for batch in loader:
        batch = move_to_device(batch, device)
        batch = {
            k: (v.float() if isinstance(v, torch.Tensor) and torch.is_floating_point(v) else v)
            for k, v in batch.items()
        }
        obs = _model.Observation.from_dict(batch)
        actions_norm = batch["actions"]
        actions_eval = actions_norm[..., :true_action_dim]
        reduce_dims = tuple(range(1, actions_eval.ndim))

        with torch.no_grad():
            t0 = time.perf_counter()
            if method == "implicit":
                noise = policy._model.sample_noise_action_implicit(
                    device=device,
                    observation=obs,
                    action=actions_norm,
                    num_steps=num_steps,
                )
            else:
                noise = policy._model.sample_noise_action(
                    device=device,
                    observation=obs,
                    action=actions_norm,
                    num_steps=num_steps,
                )
            elapsed_inversion += time.perf_counter() - t0

            # Reconstruct actions from inverted noise (use default forward steps).
            actions_hat = policy._model.sample_actions(
                device=device,
                observation=obs,
                noise=noise,
            )

            # Re-invert the reconstructed actions -> noise_hat, then compare to original noise.
            if method == "implicit":
                noise_hat = policy._model.sample_noise_action_implicit(
                    device=device,
                    observation=obs,
                    action=actions_hat,
                    num_steps=num_steps,
                )
            else:
                noise_hat = policy._model.sample_noise_action(
                    device=device,
                    observation=obs,
                    action=actions_hat,
                    num_steps=num_steps,
                )

        actions_hat_eval = actions_hat[..., :true_action_dim]
        mse_per = ((actions_hat_eval - actions_eval) ** 2).mean(dim=reduce_dims)
        l1_per = (actions_hat_eval - actions_eval).abs().mean(dim=reduce_dims)
        mse_vals.extend(mse_per.detach().cpu().numpy().tolist())
        l1_vals.extend(l1_per.detach().cpu().numpy().tolist())

        noise_flat = noise.flatten(start_dim=1)
        noise_hat_flat = noise_hat.flatten(start_dim=1)
        noise_reduce_dims = tuple(range(1, noise.ndim))
        noise_mse_per = ((noise_hat - noise) ** 2).mean(dim=noise_reduce_dims)
        noise_l1_per = (noise_hat - noise).abs().mean(dim=noise_reduce_dims)
        noise_mse_vals.extend(noise_mse_per.detach().cpu().numpy().tolist())
        noise_l1_vals.extend(noise_l1_per.detach().cpu().numpy().tolist())
        noise_l2_norm_vals.extend(noise_flat.norm(dim=1).detach().cpu().numpy().tolist())
        noise_hat_l2_norm_vals.extend(noise_hat_flat.norm(dim=1).detach().cpu().numpy().tolist())

    return {
        "mse_mean": float(np.mean(mse_vals)),
        "mse_median": float(np.median(mse_vals)),
        "mse_max": float(np.max(mse_vals)),
        "l1_mean": float(np.mean(l1_vals)),
        "l1_median": float(np.median(l1_vals)),
        "l1_max": float(np.max(l1_vals)),
        "noise_mse_mean": float(np.mean(noise_mse_vals)),
        "noise_mse_median": float(np.median(noise_mse_vals)),
        "noise_mse_max": float(np.max(noise_mse_vals)),
        "noise_l1_mean": float(np.mean(noise_l1_vals)),
        "noise_l1_median": float(np.median(noise_l1_vals)),
        "noise_l1_max": float(np.max(noise_l1_vals)),
        "noise_l2_norm_mean": float(np.mean(noise_l2_norm_vals)),
        "noise_l2_norm_median": float(np.median(noise_l2_norm_vals)),
        "noise_hat_l2_norm_mean": float(np.mean(noise_hat_l2_norm_vals)),
        "noise_hat_l2_norm_median": float(np.median(noise_hat_l2_norm_vals)),
        "inversion_time_s": elapsed_inversion,
        "n_samples": len(mse_vals),
    }


def save_results_plot(results: dict[int, dict], out_dir: Path, method: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] matplotlib not available, skipping plots: {e}")
        return

    steps_sorted = sorted(results.keys())

    inversion_times = [results[s]["inversion_time_s"] for s in steps_sorted]

    fig, axes = plt.subplots(6, 1, figsize=(10, 24))

    # --- Action MSE ---
    ax = axes[0]
    for label, key in [("MSE (mean)", "mse_mean"), ("MSE (median)", "mse_median"), ("MSE (max)", "mse_max")]:
        ax.plot(steps_sorted, [results[s][key] for s in steps_sorted], marker="o", label=label)
    ax.set_xlabel("num_steps (inversion)")
    ax.set_ylabel("MSE")
    ax.set_title(f"[{method}] Action cycle-verify MSE vs inversion steps")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # --- Action L1 ---
    ax = axes[1]
    for label, key in [("L1 (mean)", "l1_mean"), ("L1 (median)", "l1_median")]:
        ax.plot(steps_sorted, [results[s][key] for s in steps_sorted], marker="o", label=label)
    ax.set_xlabel("num_steps (inversion)")
    ax.set_ylabel("L1 (MAE)")
    ax.set_title(f"[{method}] Action cycle-verify L1 vs inversion steps")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # --- Noise MSE ---
    ax = axes[2]
    for label, key in [("Noise MSE (mean)", "noise_mse_mean"), ("Noise MSE (median)", "noise_mse_median"), ("Noise MSE (max)", "noise_mse_max")]:
        ax.plot(steps_sorted, [results[s][key] for s in steps_sorted], marker="o", label=label)
    ax.set_xlabel("num_steps (inversion)")
    ax.set_ylabel("MSE")
    ax.set_title(f"[{method}] Noise cycle-verify MSE vs inversion steps")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # --- Noise L1 ---
    ax = axes[3]
    for label, key in [("Noise L1 (mean)", "noise_l1_mean"), ("Noise L1 (median)", "noise_l1_median")]:
        ax.plot(steps_sorted, [results[s][key] for s in steps_sorted], marker="o", label=label)
    ax.set_xlabel("num_steps (inversion)")
    ax.set_ylabel("L1 (MAE)")
    ax.set_title(f"[{method}] Noise cycle-verify L1 vs inversion steps")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # --- Noise L2 norms ---
    ax = axes[4]
    ax.plot(steps_sorted, [results[s]["noise_l2_norm_mean"] for s in steps_sorted], marker="o", label="noise L2 norm (mean)")
    ax.plot(steps_sorted, [results[s]["noise_l2_norm_median"] for s in steps_sorted], marker="o", linestyle="--", label="noise L2 norm (median)")
    ax.plot(steps_sorted, [results[s]["noise_hat_l2_norm_mean"] for s in steps_sorted], marker="s", label="noise_hat L2 norm (mean)")
    ax.plot(steps_sorted, [results[s]["noise_hat_l2_norm_median"] for s in steps_sorted], marker="s", linestyle="--", label="noise_hat L2 norm (median)")
    ax.set_xlabel("num_steps (inversion)")
    ax.set_ylabel("L2 norm")
    ax.set_title(f"[{method}] Noise L2 norms vs inversion steps")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # --- Inversion time ---
    ax = axes[5]
    ax.plot(steps_sorted, inversion_times, marker="s", color="purple", label="inversion time (s)")
    ax.set_xlabel("num_steps (inversion)")
    ax.set_ylabel("Wall time (s)")
    ax.set_title(f"[{method}] Total inversion time vs num_steps")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    out_path = out_dir / f"inversion_steps_compare_{method}.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[compare] saved plot -> {out_path}")


def main() -> None:
    args = parse_args()
    steps_list = [int(s.strip()) for s in args.steps.split(",")]
    out_dir = Path(args.out_dir).expanduser().resolve()

    if args.data_root:
        os.environ["ACT_DATA_ROOT"] = args.data_root

    device = torch.device(args.device)

    # ---- dataset & transforms ----
    data_cfg_root = _config.get_config(args.data_config_name)
    data_cfg = data_cfg_root.data.create(data_cfg_root.data.assets.assets_dir, data_cfg_root.model)

    norm = transforms.Normalize(data_cfg.norm_stats, use_quantiles=data_cfg.use_quantile_norm)
    transform = transforms.compose(
        [
            *data_cfg.repack_transforms.inputs,
            *data_cfg.data_transforms.inputs,
            norm,
            *data_cfg.model_transforms.inputs,
        ]
    )

    dataset = ActDataset(root_dir=args.data_root, action_horizon=10)
    transformed_dataset = TransformingDataset(dataset, transform)

    # Take first num_samples samples deterministically.
    n = min(args.num_samples, len(transformed_dataset))
    subset = Subset(transformed_dataset, list(range(n)))
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    print(f"[compare] evaluating {n} samples, method={args.method}, steps={steps_list}")

    # ---- policy ----
    checkpoint_dir = Path(args.pytorch_weight_path).expanduser().resolve()
    policy_cfg = _config.get_config(args.policy_config_name)
    policy = policy_config.create_trained_policy(
        policy_cfg,
        checkpoint_dir,
        pytorch_device=args.device,
    )

    # ---- run comparison ----
    results: dict[int, dict] = {}
    for num_steps in steps_list:
        print(f"[compare] num_steps={num_steps} ...", flush=True)
        t_start = time.perf_counter()
        metrics = evaluate_steps(
            policy=policy,
            loader=loader,
            num_steps=num_steps,
            method=args.method,
            true_action_dim=args.true_action_dim,
            device=device,
        )
        elapsed = time.perf_counter() - t_start
        results[num_steps] = metrics
        print(
            f"  mse_mean={metrics['mse_mean']:.6g}  l1_mean={metrics['l1_mean']:.6g}"
            f"  inversion_time={metrics['inversion_time_s']:.2f}s  total={elapsed:.2f}s"
        )

    # ---- print table ----
    print("\n" + "=" * 130)
    print(f"{'steps':>8}  {'mse_mean':>12}  {'l1_mean':>10}  {'n_mse_mean':>12}  {'n_l1_mean':>10}  {'noise_norm':>12}  {'nhat_norm':>12}  {'inv_time':>10}")
    print("-" * 130)
    for s in sorted(results.keys()):
        r = results[s]
        print(
            f"{s:>8}  {r['mse_mean']:>12.6g}  {r['l1_mean']:>10.6g}"
            f"  {r['noise_mse_mean']:>12.6g}  {r['noise_l1_mean']:>10.6g}"
            f"  {r['noise_l2_norm_mean']:>12.4f}  {r['noise_hat_l2_norm_mean']:>12.4f}"
            f"  {r['inversion_time_s']:>10.2f}s"
        )
    print("=" * 130)

    # ---- save ----
    save_results_plot(results, out_dir, args.method)

    # Save raw numbers as JSON for later analysis.
    import json

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"inversion_steps_results_{args.method}.json"
    json_results = {str(s): results[s] for s in sorted(results.keys())}
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"[compare] saved raw results -> {json_path}")


if __name__ == "__main__":
    main()
