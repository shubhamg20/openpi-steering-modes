"""Generate noise_action for each episode and save alongside data.pkl.

Uses the config assets (dataset_metadata + norm_stats) and data_dirs from the
TrainConfig to locate episodes.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import openpi.models.model as _model
import openpi.transforms as transforms
from openpi.policies import policy_config
from openpi.training import config as _config
from openpi.training.act_dataset import ActDataset

# python get_libero_dataset.py
# ACT_DATA_ROOT=converted_dataset uv run scripts/prepare_act_dataset.py --config-name pi0_libero_90_low_mem_finetune_franka_data
# python examples/convert_jax_model_to_pytorch.py --checkpoint_dir /gpfs/scrubbed/shubham/chkpts/pi0_libero_90_low_mem_finetune/pi0_libero_90_low_mem_finetune_20251218_165604/29999/ --config_name pi0_libero_90_low_mem_finetune --output_path /gpfs/scrubbed/shubham/chkpts/pi0_libero_90_low_mem_finetune/pi0_libero_90_low_mem_finetune_20251218_165604/pytorch_chkpts/29999_2/model.safetensors
#chmod +x *.model.safetensors
# for i in 0 1 2 3; do   CUDA_VISIBLE_DEVICES=$i ACT_DATA_ROOT=converted_dataset  uv run scripts/gen_noise_data.py     --policy_config_name=pi0_libero_90_low_mem_finetune_policy     --data_config_name=pi0_droid_lora_finetune_data     --pytorch_weight_path=/gpfs/scrubbed/shubham/chkpts/pi0_libero_90_low_mem_finetune/pi0_libero_90_low_mem_finetune_20251218_165604/pytorch_chkpts/29999/model.safetensors    --output_name data_with_noise_test.pkl     --batch_size 128 --num_workers 16 --cycle_verify     --implicit --num_steps 150 --num_demos 200    --num_shards 8 --shard_id=$i & done
# ACT_DATA_ROOT=converted_dataset uv run scripts/compute_norm_stats_pickles.py  --config-name pi0_libero_90_low_mem_finetune_franka --extra-keys noise_action --data-pkl-name data_with_noise.pkl
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ACT_DATA_ROOT=converted_dataset uv run  torchrun --nproc_per_node=8  scripts/train_noise_flow_matching.py --config-name pi0_libero_90_low_mem_finetune_franka --data-pkl-name data_with_noise_full.pkl --batch-size 256 --num-steps 1000000 --num-workers 8 --run-prefix test_noshare_dct12_nonorm --hidden-dims 512,512,512 --train-backbone --lr 3e-4 --use-film --wandb-project sft-noise --val-fraction 0.1 --dct-k 6 --l1-sample-flow


#for i in 0; do   CUDA_VISIBLE_DEVICES=$i  uv run scripts/gen_noise_data.py     --policy_config_name=pi0_droid_lora_finetune     --data_config_name=pi0_droid_lora_finetune     --pytorch_weight_path=/gscratch/weirdlab/shubham2/IsaacLab/source/openpi/checkpoints/pi0_droid_lora_finetune/droid-finetune-lora/9999_pytorch/      --batch_size 128 --num_workers 16 --cycle_verify     --implicit --num_steps 150     --num_shards 1 --shard_id=$i & done
#uv run scripts/compute_norm_stats_pickle.py --root-dir ../recorded_runs/data_paired/ --config-name pi0_droid_lora_finetune_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate noise_action for each episode.")
    parser.add_argument(
        "--policy_config_name",
        required=True,
        help="TrainConfig name for loading the policy.",
    )
    parser.add_argument(
        "--data_config_name",
        required=True,
        help="TrainConfig name for loading the dataset/transforms.",
    )
    parser.add_argument(
        "--pytorch_weight_path",
        required=True,
        help="Directory containing the PyTorch checkpoint (model.safetensors).",
    )
    parser.add_argument(
        "--data_root",
        default=os.environ.get("ACT_DATA_ROOT", "/gscratch/weirdlab/shubham2/IsaacLab/source/recorded_runs/data_paired"),
        help="Override ACT_DATA_ROOT for loading the dataset.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (default: auto-detect).",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=10,
        help="Number of integration steps for sample_noise.",
    )
    parser.add_argument(
        "--explicit",
        action="store_true",
        help="Generate explicit noise_action (and optionally cycle-verify).",
    )
    parser.add_argument(
        "--implicit",
        action="store_true",
        help="Generate implicit noise_action (and optionally cycle-verify).",
    )
    parser.add_argument(
        "--cem",
        action="store_true",
        help="Generate CEM noise_action (and optionally cycle-verify).",
    )
    parser.add_argument(
        "--cem_blocks",
        type=int,
        default=1,
        help="Number of block parameters K used for CEM noise search.",
    )
    parser.add_argument(
        "--cem_pop",
        type=int,
        default=256,
        help="Population size sampled each CEM iteration.",
    )
    parser.add_argument(
        "--cem_elites",
        type=int,
        default=32,
        help="Number of elite samples to update the CEM distribution.",
    )
    parser.add_argument(
        "--cem_init_std",
        type=float,
        default=0.5,
        help="Initial stddev for CEM parameter sampling.",
    )
    parser.add_argument(
        "--cem_min_std",
        type=float,
        default=0.05,
        help="Lower bound on stddev during CEM updates.",
    )
    parser.add_argument(
        "--cem_alpha",
        type=float,
        default=0.25,
        help="Exponential moving average factor for the CEM mean/std.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for sample_noise_action (default: 1).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of DataLoader workers (default: 0).",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Number of shards to split episodes across processes (for multi-GPU).",
    )
    parser.add_argument(
        "--shard_id",
        type=int,
        default=0,
        help="Shard id in [0, num_shards) for this process.",
    )
    parser.add_argument(
        "--cycle_verify",
        action="store_true",
        help=(
            "If set: for each batch, run cycle verification for the triggered noise methods by sampling actions from the "
            "corresponding noise (policy._model.sample_actions(noise=...)) and compute per-step MSE vs batch['actions']. "
            "After the episode, print min/max MSE for each triggered method and save max-MSE visualizations."
        ),
    )
    parser.add_argument(
        "--dump_actions_episode_idx",
        type=int,
        default=-1,
        help=(
            "If >= 0, dump the FULL action trajectories for this episode index during cycle_verify: "
            "1) ground truth actions_norm, 2) explicit reconstructed actions_hat, 3) implicit reconstructed actions_hat. "
            "Saved under ./plots/trajectories/ as a .npz (requires --explicit and --implicit)."
        ),
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
    def __init__(self, base_dataset: torch.utils.data.Dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        return self.transform(self.base_dataset[idx])


def _repeat_interleave_batch(x: torch.Tensor | None, repeats: int):
    if x is None:
        return None
    return x.repeat_interleave(repeats, dim=0)


def _slice_batch(x: torch.Tensor | None, idx: int):
    if x is None:
        return None
    return x[idx : idx + 1]


def repeat_observation(obs: _model.Observation, repeats: int) -> _model.Observation:
    """Repeat an Observation along the batch dimension."""
    return _model.Observation(
        images={k: _repeat_interleave_batch(v, repeats) for k, v in obs.images.items()},
        image_masks={k: _repeat_interleave_batch(v, repeats) for k, v in obs.image_masks.items()},
        state=_repeat_interleave_batch(obs.state, repeats),
        action_is_pad=_repeat_interleave_batch(obs.action_is_pad, repeats),
        tokenized_prompt=_repeat_interleave_batch(obs.tokenized_prompt, repeats),
        tokenized_prompt_mask=_repeat_interleave_batch(obs.tokenized_prompt_mask, repeats),
        token_ar_mask=_repeat_interleave_batch(obs.token_ar_mask, repeats),
        token_loss_mask=_repeat_interleave_batch(obs.token_loss_mask, repeats),
    )


def slice_observation(obs: _model.Observation, idx: int) -> _model.Observation:
    """Take a single element from the batch dimension (keeps batch dim=1)."""
    return _model.Observation(
        images={k: _slice_batch(v, idx) for k, v in obs.images.items()},
        image_masks={k: _slice_batch(v, idx) for k, v in obs.image_masks.items()},
        state=_slice_batch(obs.state, idx),
        action_is_pad=_slice_batch(obs.action_is_pad, idx),
        tokenized_prompt=_slice_batch(obs.tokenized_prompt, idx),
        tokenized_prompt_mask=_slice_batch(obs.tokenized_prompt_mask, idx),
        token_ar_mask=_slice_batch(obs.token_ar_mask, idx),
        token_loss_mask=_slice_batch(obs.token_loss_mask, idx),
    )


def _unnormalize_actions_only(
    actions_norm: np.ndarray,
    *,
    action_stats,
    use_quantiles: bool,
) -> np.ndarray:
    """
    Unnormalize just the action array using the per-key action stats.
    This avoids `transforms.Unnormalize` strict tree-matching requirements.
    """
    x = np.asarray(actions_norm)
    if action_stats is None:
        return x
    mean = np.asarray(action_stats.mean)
    std = np.asarray(action_stats.std)
    # Pad mean/std to match last dim if needed (same semantics as openpi.transforms.Unnormalize).
    if mean.shape[-1] < x.shape[-1]:
        mean = np.pad(mean, [(0, 0)] * (mean.ndim - 1) + [(0, x.shape[-1] - mean.shape[-1])], constant_values=0.0)
    if std.shape[-1] < x.shape[-1]:
        std = np.pad(std, [(0, 0)] * (std.ndim - 1) + [(0, x.shape[-1] - std.shape[-1])], constant_values=1.0)
    mean = mean[..., : x.shape[-1]]
    std = std[..., : x.shape[-1]]
    return x * (std + 1e-6) + mean


def _dump_action_trajectories_npz(
    out_path: str | Path,
    *,
    actions_true_norm: np.ndarray,  # (T, H, A)
    actions_hat_explicit_norm: np.ndarray,  # (T, H, A)
    actions_hat_implicit_norm: np.ndarray,  # (T, H, A)
    action_stats,
    use_quantiles: bool,
    meta: dict[str, Any],
) -> Path:
    """Dump action trajectories to a .npz (both normalized and denormalized)."""
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    true_norm = np.asarray(actions_true_norm)
    exp_norm = np.asarray(actions_hat_explicit_norm)
    imp_norm = np.asarray(actions_hat_implicit_norm)

    true = _unnormalize_actions_only(true_norm, action_stats=action_stats, use_quantiles=use_quantiles)
    exp = _unnormalize_actions_only(exp_norm, action_stats=action_stats, use_quantiles=use_quantiles)
    imp = _unnormalize_actions_only(imp_norm, action_stats=action_stats, use_quantiles=use_quantiles)

    np.savez_compressed(
        out_path,
        actions_true_norm=true_norm,
        actions_hat_explicit_norm=exp_norm,
        actions_hat_implicit_norm=imp_norm,
        actions_true=true,
        actions_hat_explicit=exp,
        actions_hat_implicit=imp,
        meta_json=json.dumps(meta),
    )
    print(f"[cycle_verify] wrote {out_path}")
    return out_path


def _try_save_cycle_verify_plot(
    out_path: str | Path,
    actions_hat: np.ndarray,
    actions_true: np.ndarray,
    mse_value: float,
    step_idx: int,
    label: str = "",
) -> None:
    """Best-effort visualization; if matplotlib isn't installed, we just skip."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        print(f"[warn] cycle_verify visualization skipped (matplotlib not available): {e}")
        return

    # Expected shapes: (H, A) or (A,) or anything array-like. Normalize to (H, A).
    a_hat = np.asarray(actions_hat)
    a_true = np.asarray(actions_true)
    if a_hat.ndim == 1:
        a_hat = a_hat[None, :]
    if a_true.ndim == 1:
        a_true = a_true[None, :]
    if a_hat.shape != a_true.shape:
        print(f"[warn] cycle_verify visualization skipped (shape mismatch): {a_hat.shape} vs {a_true.shape}")
        return

    h, a = a_hat.shape
    max_dims = min(8, a)
    x = np.arange(h)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=False)

    # Plot a few dims over horizon.
    for d in range(max_dims):
        ax1.plot(x, a_true[:, d], linewidth=1.6, label=f"true[d={d}]" if d == 0 else None, alpha=0.9)
        ax1.plot(x, a_hat[:, d], linestyle="--", linewidth=1.3, label=f"hat[d={d}]" if d == 0 else None, alpha=0.9)
    prefix = f"{label} " if label else ""
    ax1.set_title(f"{prefix}cycle_verify @ step={step_idx}  mse={mse_value:.6g}  (showing first {max_dims}/{a} dims)")
    ax1.set_xlabel("horizon t")
    ax1.set_ylabel("action (normalized)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best")

    # Scatter all values hat vs true.
    ax2.scatter(a_true.reshape(-1), a_hat.reshape(-1), s=8, alpha=0.4)
    lo = float(min(a_true.min(), a_hat.min()))
    hi = float(max(a_true.max(), a_hat.max()))
    ax2.plot([lo, hi], [lo, hi], color="black", linewidth=1.0, alpha=0.7)
    ax2.set_title("hat vs true scatter (all horizon*dims)")
    ax2.set_xlabel("true")
    ax2.set_ylabel("hat")
    ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[cycle_verify] wrote {out_path}")


def _try_save_mse_list_plot(out_path: str | Path, mse_list: list[float]) -> None:
    """Best-effort plot of per-step MSE over an episode."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        print(f"[warn] cycle_verify MSE plot skipped (matplotlib not available): {e}")
        return

    if not mse_list:
        return

    y = np.asarray(mse_list, dtype=np.float32)
    x = np.arange(len(y))

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(x, y, linewidth=1.4)
    ax.set_title(f"cycle_verify MSE per step (len={len(y)})")
    ax.set_xlabel("episode step")
    ax.set_ylabel("MSE")
    ax.grid(True, alpha=0.25)

    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[cycle_verify] wrote {out_path}")


def _try_save_mse_lists_plot(
    out_path: str | Path,
    series: dict[str, list[float]],
    title: str,
    ylabel: str,
) -> None:
    """Best-effort plot of multiple per-step series over an episode."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        print(f"[warn] cycle_verify MSE plot skipped (matplotlib not available): {e}")
        return

    # Keep non-empty series only.
    series = {k: v for k, v in series.items() if v}
    if not series:
        return

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    for label, vals in series.items():
        y = np.asarray(vals, dtype=np.float32)
        x = np.arange(len(y))
        ax.plot(x, y, linewidth=1.4, label=label)

    ax.set_title(title)
    ax.set_xlabel("episode step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[cycle_verify] wrote {out_path}")


def main() -> None:
    args = parse_args()
    triggered_methods = []
    for method_name, enabled in (("explicit", args.explicit), ("implicit", args.implicit), ("cem", args.cem)):
        if enabled:
            triggered_methods.append(method_name)
    if not triggered_methods:
        raise ValueError("No noise method selected. Use --explicit, --implicit, and/or --cem.")
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1.")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError("--shard_id must be in [0, num_shards).")
    checkpoint_dir = Path(args.pytorch_weight_path).expanduser().resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    if args.data_root:
        os.environ["ACT_DATA_ROOT"] = args.data_root

    device = torch.device(args.device)

    data_cfg_root = _config.get_config(args.data_config_name)
    data_cfg = data_cfg_root.data.create(data_cfg_root.data.assets.assets_dir, data_cfg_root.model)
    print(data_cfg.repack_transforms.inputs)


    # Real (non-padded) action dimension, used for metrics. If the model pads actions (e.g. 14->32),
    # we should not include the padded dims in L1/MSE.
    # Restrict metrics to first 7 action dims
    true_action_dim = 8
    print(f"[cycle_verify] true_action_dim={true_action_dim} (metrics use actions[..., :8])")
    norm = transforms.Normalize(data_cfg.norm_stats, use_quantiles=data_cfg.use_quantile_norm)
    unnorm = transforms.Unnormalize(data_cfg.norm_stats, use_quantiles=data_cfg.use_quantile_norm)
    print(f"Data normalization stats: {data_cfg.norm_stats}")
    transform = transforms.compose(
        [
            *data_cfg.repack_transforms.inputs,
            *data_cfg.data_transforms.inputs,
            norm,
            *data_cfg.model_transforms.inputs,
        ]
    )

    dataset = ActDataset(
        root_dir=args.data_root,
        action_horizon=16
    )
    episode_paths = dataset.episode_paths
    episode_lengths = dataset.episode_lengths

    # sharding stays the same
    selected_episode_indices = [
        i for i in range(len(episode_paths)) if i % args.num_shards == args.shard_id
    ]
    # episode_start_offsets stays the same - still needed for Subset
    episode_start_offsets = np.concatenate(([0], np.cumsum(episode_lengths[:-1])))
    if not selected_episode_indices:
        print(f"[warn] no episodes assigned to shard_id={args.shard_id} / num_shards={args.num_shards}")
    transformed_dataset = TransformingDataset(dataset, transform)

    policy_cfg = _config.get_config(args.policy_config_name)
    policy = policy_config.create_trained_policy(
        policy_cfg,
        checkpoint_dir,
        pytorch_device=args.device, 
    )
    action_horizon = policy_cfg.model.action_horizon

    # Track global worst-case (across all episodes) for cycle verification.
    global_max_mse_value = float("-inf")
    global_max_episode_idx = -1
    global_max_step = -1
    global_max_actions_hat: np.ndarray | None = None
    global_max_actions_true: np.ndarray | None = None
    global_max_mse_value_implicit = float("-inf")
    global_max_episode_idx_implicit = -1
    global_max_step_implicit = -1
    global_max_actions_hat_implicit: np.ndarray | None = None
    global_max_actions_true_implicit: np.ndarray | None = None
    global_max_mse_value_cem = float("-inf")
    global_max_episode_idx_cem = -1
    global_max_step_cem = -1
    global_max_actions_hat_cem: np.ndarray | None = None
    global_max_actions_true_cem: np.ndarray | None = None
    for episode_idx in selected_episode_indices:
        episode_rel_path = episode_paths[episode_idx]
        episode_len = episode_lengths[episode_idx]
        if os.environ.get("ACT_DATA_ROOT"):
            episode_path = os.path.join(os.environ["ACT_DATA_ROOT"], episode_rel_path)
        else:
            episode_path = episode_rel_path

        episode_path = Path(episode_paths[episode_idx])
        output_path = episode_path.parent / (episode_path.stem + "_noise.pkl")

        with open(episode_path, "rb") as f:
            data = pickle.load(f)

        episode_timer = time.perf_counter()
        data_prep_time = 0.0
        model_time = 0.0
        episode_start = int(episode_start_offsets[episode_idx])
        episode_indices = list(range(episode_start, episode_start + episode_len))
        episode_dataset = Subset(transformed_dataset, episode_indices)
        loader = DataLoader(
            episode_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

        noise_actions: dict[str, list[np.ndarray]] = {m: [] for m in triggered_methods}
        mse_list: list[float] = []
        mse_list_implicit: list[float] = []
        mse_list_cem: list[float] = []
        l1_list: list[float] = []
        l1_list_implicit: list[float] = []
        l1_list_cem: list[float] = []
        max_mse_value = float("-inf")
        max_mse_step = -1
        max_actions_hat: np.ndarray | None = None
        max_actions_true: np.ndarray | None = None
        max_mse_value_implicit = float("-inf")
        max_mse_step_implicit = -1
        max_actions_hat_implicit: np.ndarray | None = None
        max_actions_true_implicit: np.ndarray | None = None
        max_mse_value_cem = float("-inf")
        max_mse_step_cem = -1
        max_actions_hat_cem: np.ndarray | None = None
        max_actions_true_cem: np.ndarray | None = None
        # Optional: dump full action trajectories (length=episode_len) for one selected episode.
        dump_actions_true: list[np.ndarray] = []
        dump_actions_hat_explicit: list[np.ndarray] = []
        dump_actions_hat_implicit: list[np.ndarray] = []
        batch_timer = time.perf_counter()
        for batch in loader:
            data_prep_time += time.perf_counter() - batch_timer
            batch = move_to_device(batch, device)
            batch = {
                k: (v.float() if isinstance(v, torch.Tensor) and torch.is_floating_point(v) else v)
                for k, v in batch.items()
            }

            obs_on_device = _model.Observation.from_dict(batch)
            actions_norm = batch["actions"]
            model_timer = time.perf_counter()
            with torch.no_grad():
                noise_norm_explicit = None
                noise_norm_implicit = None
                noise_norm_cem = None
                if args.explicit:
                    noise_norm_explicit = policy._model.sample_noise_action(
                        device=device,
                        observation=obs_on_device,
                        action=actions_norm,
                        num_steps=args.num_steps,
                    )
                    
                if args.implicit:
                    noise_norm_implicit = policy._model.sample_noise_action_implicit(
                        device=device,
                        observation=obs_on_device,
                        action=actions_norm,
                        num_steps=args.num_steps,
                    )
                    # print(f"[debug] mean {noise_norm_implicit.mean(dim=(0, 1))}")
                    per_dim_mean = noise_norm_implicit.mean(dim=(0, 1))
                    worst_dim = per_dim_mean.abs().argmax()
                    print(f"[debug] worst dim={worst_dim.item()} val={per_dim_mean[worst_dim].item():.4f}")

                if args.cem:
                    per_sample_noises = []
                    batch_size = int(actions_norm.shape[0])

                    for bi in range(batch_size):
                        obs_single = slice_observation(obs_on_device, bi)

                        def sample_actions_fn(noise_candidates: torch.Tensor) -> torch.Tensor:
                            obs_rep = repeat_observation(obs_single, int(noise_candidates.shape[0]))
                            return policy._model.sample_actions(
                                device=device,
                                observation=obs_rep,
                                noise=noise_candidates,
                                num_steps=args.num_steps,
                            )


                        best_noise, _ = policy._model.cem_fit_noise_blocks(
                            sample_actions_fn,
                            actions_true=actions_norm[bi],
                            c_init=None,
                            true_action_dim=true_action_dim,
                            horizon=action_horizon,
                            action_dim=actions_norm.shape[-1],
                            k_blocks=args.cem_blocks,
                            pop=args.cem_pop,
                            elites=args.cem_elites,
                            init_std=args.cem_init_std,
                            min_std=args.cem_min_std,
                            alpha=args.cem_alpha,
                            verbose=args.cycle_verify,
                            device=device,
                        )
                        per_sample_noises.append(best_noise)

                    noise_norm_cem = torch.stack(per_sample_noises, dim=0)
                if args.cycle_verify:
                
                    actions_norm_eval = actions_norm[..., :true_action_dim]
                    reduce_dims = tuple(range(1, actions_norm_eval.ndim))

                    if args.explicit and noise_norm_explicit is not None:
                        actions_hat = policy._model.sample_actions(
                            device=device,
                            observation=obs_on_device,
                            noise=noise_norm_explicit,
                            # num_steps=args.num_steps,
                        )
                        # Metrics: only evaluate on real action dims (ignore zero-padded dims).
                        actions_hat_eval = actions_hat[..., :true_action_dim]
                        mse_per = ((actions_hat_eval - actions_norm_eval) ** 2).mean(dim=reduce_dims)  # (B,)
                        l1_per = (actions_hat_eval - actions_norm_eval).abs().mean(dim=reduce_dims)  # (B,)
                        mse_cpu = mse_per.detach().cpu().numpy().tolist()
                        l1_cpu = l1_per.detach().cpu().numpy().tolist()
                        mse_list.extend(float(x) for x in mse_cpu)
                        l1_list.extend(float(x) for x in l1_cpu)
                        for i, mse_val in enumerate(mse_cpu):
                            if float(mse_val) > max_mse_value:
                                max_mse_value = float(mse_val)
                                max_mse_step = len(mse_list) - len(mse_cpu) + i
                                max_actions_hat = actions_hat[i].detach().cpu().numpy()
                                max_actions_true = actions_norm[i].detach().cpu().numpy()

                    if args.implicit and noise_norm_implicit is not None:
                        actions_hat_implicit = policy._model.sample_actions(
                            device=device,
                            observation=obs_on_device,
                            noise=noise_norm_implicit,
                            # num_steps=args.num_steps,
                        )
                        actions_hat_impl_eval = actions_hat_implicit[..., :true_action_dim]
                        mse_per_impl = ((actions_hat_impl_eval - actions_norm_eval) ** 2).mean(dim=reduce_dims)  # (B,)
                        l1_per_impl = (actions_hat_impl_eval - actions_norm_eval).abs().mean(dim=reduce_dims)  # (B,)
                        mse_cpu_impl = mse_per_impl.detach().cpu().numpy().tolist()
                        l1_cpu_impl = l1_per_impl.detach().cpu().numpy().tolist()
                        mse_list_implicit.extend(float(x) for x in mse_cpu_impl)
                        l1_list_implicit.extend(float(x) for x in l1_cpu_impl)
                        for i, mse_val in enumerate(mse_cpu_impl):
                            if float(mse_val) > max_mse_value_implicit:
                                max_mse_value_implicit = float(mse_val)
                                max_mse_step_implicit = len(mse_list_implicit) - len(mse_cpu_impl) + i
                                max_actions_hat_implicit = actions_hat_implicit[i].detach().cpu().numpy()
                                max_actions_true_implicit = actions_norm[i].detach().cpu().numpy()

                    if args.cem and noise_norm_cem is not None:
                        actions_hat_cem = policy._model.sample_actions(
                            device=device,
                            observation=obs_on_device,
                            noise=noise_norm_cem,
                            num_steps=args.num_steps,
                        )
                        actions_hat_cem_eval = actions_hat_cem[..., :true_action_dim]
                        mse_per_cem = ((actions_hat_cem_eval - actions_norm_eval) ** 2).mean(dim=reduce_dims)
                        l1_per_cem = (actions_hat_cem_eval - actions_norm_eval).abs().mean(dim=reduce_dims)
                        mse_cpu_cem = mse_per_cem.detach().cpu().numpy().tolist()
                        l1_cpu_cem = l1_per_cem.detach().cpu().numpy().tolist()
                        mse_list_cem.extend(float(x) for x in mse_cpu_cem)
                        l1_list_cem.extend(float(x) for x in l1_cpu_cem)
                        for i, mse_val in enumerate(mse_cpu_cem):
                            if float(mse_val) > max_mse_value_cem:
                                max_mse_value_cem = float(mse_val)
                                max_mse_step_cem = len(mse_list_cem) - len(mse_cpu_cem) + i
                                max_actions_hat_cem = actions_hat_cem[i].detach().cpu().numpy()
                                max_actions_true_cem = actions_norm[i].detach().cpu().numpy()

                    # Dump full trajectories for a chosen episode index.
                    if args.dump_actions_episode_idx >= 0 and episode_idx == args.dump_actions_episode_idx:
                        if not (args.explicit and args.implicit):
                            print("[warn] dump_actions_episode_idx requires --explicit and --implicit enabled; skipping.")
                        else:
                            # Save per-sample chunks in order; each is (H, A)
                            bsz = int(actions_norm.shape[0])
                            for bi in range(bsz):
                                dump_actions_true.append(actions_norm[bi].detach().cpu().numpy())
                                dump_actions_hat_explicit.append(actions_hat[bi].detach().cpu().numpy())
                                dump_actions_hat_implicit.append(actions_hat_implicit[bi].detach().cpu().numpy())
            model_time += time.perf_counter() - model_timer
            if args.explicit and noise_norm_explicit is not None:
                noise_actions["explicit"].extend(noise_norm_explicit.detach().cpu().numpy())
            if args.implicit and noise_norm_implicit is not None:
                noise_actions["implicit"].extend(noise_norm_implicit.detach().cpu().numpy())
            if args.cem and noise_norm_cem is not None:
                noise_actions["cem"].extend(noise_norm_cem.detach().cpu().numpy())
            batch_timer = time.perf_counter()

        if len(triggered_methods) == 1:
            data["noise_action"] = np.stack(noise_actions[triggered_methods[0]], axis=0)
        else:
            for method in triggered_methods:
                data[f"noise_action_{method}"] = np.stack(noise_actions[method], axis=0)
        #TODO need to think about the realtionship betwen noise chunk and real action chunk
        # it always generate noise action for 50 steps
        
        with open(output_path, "wb") as f:
            pickle.dump(data, f)

        if args.cycle_verify:
            if args.explicit:
                if len(mse_list) != episode_len:
                    print(f"[warn] cycle_verify expected {episode_len} MSE values, got {len(mse_list)}")
                if mse_list:
                    print(
                        f"[cycle_verify] episode={episode_idx} mse: mean={np.mean(mse_list):.6g} min={min(mse_list):.6g} max={max(mse_list):.6g} "
                        f"(max_step={max_mse_step})"
                    )
                if len(l1_list) != episode_len:
                    print(f"[warn] cycle_verify expected {episode_len} L1 values, got {len(l1_list)}")
                if l1_list:
                    print(f"[cycle_verify] episode={episode_idx} l1: mean={np.mean(l1_list):.6g} min={min(l1_list):.6g} max={max(l1_list):.6g}")
            if (
                args.explicit
                and max_actions_hat is not None
                and max_actions_true is not None
                and max_mse_step >= 0
            ):
                plots_dir = Path.cwd() / "plots" / "explicit" / "max_mse"
                plots_dir.mkdir(parents=True, exist_ok=True)
                viz_path = plots_dir / f"cycle_verify_max_mse_ep{episode_idx:05d}.png"
                _try_save_cycle_verify_plot(
                    viz_path,
                    actions_hat=max_actions_hat,
                    actions_true=max_actions_true,
                    mse_value=max_mse_value,
                    step_idx=max_mse_step,
                    label="explicit",
                )
                # Also save the full error plot for the episode (all steps, explicit and implicit)
                # Use dump_actions_true and dump_actions_hat_explicit/implicit if available and lengths match
                if dump_actions_true and dump_actions_hat_explicit and len(dump_actions_true) == len(dump_actions_hat_explicit) == len(mse_list):
                    _save_full_error_plot(
                        action_errors=mse_list,
                        gt_actions=dump_actions_true,
                        reconstructed_actions=dump_actions_hat_explicit,
                        plot_out_path=plots_dir,
                        episode_idx=episode_idx,
                        method_name="explicit",
                    )
                if dump_actions_true and dump_actions_hat_implicit and len(dump_actions_true) == len(dump_actions_hat_implicit) == len(mse_list_implicit):
                    _save_full_error_plot(
                        action_errors=mse_list_implicit,
                        gt_actions=dump_actions_true,
                        reconstructed_actions=dump_actions_hat_implicit,
                        plot_out_path=plots_dir,
                        episode_idx=episode_idx,
                        method_name="implicit",
                    )
                # Update global max across all episodes.
                if max_mse_value > global_max_mse_value:
                    global_max_mse_value = max_mse_value
                    global_max_episode_idx = episode_idx
                    global_max_step = max_mse_step
                    global_max_actions_hat = max_actions_hat
                    global_max_actions_true = max_actions_true

            if args.implicit:
                if len(mse_list_implicit) != episode_len:
                    print(f"[warn] cycle_verify(implicit) expected {episode_len} MSE values, got {len(mse_list_implicit)}")
                if mse_list_implicit:
                    print(
                        f"[cycle_verify] episode={episode_idx} implicit mse: mean={np.mean(mse_list_implicit):.6g} min={min(mse_list_implicit):.6g} "
                        f"max={max(mse_list_implicit):.6g} (max_step={max_mse_step_implicit})"
                    )
                if len(l1_list_implicit) != episode_len:
                    print(f"[warn] cycle_verify(implicit) expected {episode_len} L1 values, got {len(l1_list_implicit)}")
                if l1_list_implicit:
                    print(
                        f"[cycle_verify] episode={episode_idx} implicit l1: mean={np.mean(l1_list_implicit):.6g} min={min(l1_list_implicit):.6g} "
                        f"max={max(l1_list_implicit):.6g}"
                    )

            if args.cem and noise_norm_cem is not None:
                if len(mse_list_cem) != episode_len:
                    print(f"[warn] cycle_verify(cem) expected {episode_len} MSE values, got {len(mse_list_cem)}")
                if mse_list_cem:
                    print(
                        f"[cycle_verify] episode={episode_idx} cem mse: min={min(mse_list_cem):.6g} "
                        f"max={max(mse_list_cem):.6g} (max_step={max_mse_step_cem})"
                    )
                if len(l1_list_cem) != episode_len:
                    print(f"[warn] cycle_verify(cem) expected {episode_len} L1 values, got {len(l1_list_cem)}")
                if l1_list_cem:
                    print(
                        f"[cycle_verify] episode={episode_idx} cem l1: min={min(l1_list_cem):.6g} "
                        f"max={max(l1_list_cem):.6g}"
                    )

                if (
                    max_actions_hat_cem is not None
                    and max_actions_true_cem is not None
                    and max_mse_step_cem >= 0
                ):
                    plots_dir = Path.cwd() / "plots" / "cem" / "max_mse"
                    plots_dir.mkdir(parents=True, exist_ok=True)
                    viz_path = plots_dir / f"cycle_verify_max_mse_ep{episode_idx:05d}.png"
                    _try_save_cycle_verify_plot(
                        viz_path,
                        actions_hat=max_actions_hat_cem,
                        actions_true=max_actions_true_cem,
                        mse_value=max_mse_value_cem,
                        step_idx=max_mse_step_cem,
                        label="cem",
                    )
                    if max_mse_value_cem > global_max_mse_value_cem:
                        global_max_mse_value_cem = max_mse_value_cem
                        global_max_episode_idx_cem = episode_idx
                        global_max_step_cem = max_mse_step_cem
                        global_max_actions_hat_cem = max_actions_hat_cem
                        global_max_actions_true_cem = max_actions_true_cem

            if (
                args.implicit
                and max_actions_hat_implicit is not None
                and max_actions_true_implicit is not None
                and max_mse_step_implicit >= 0
            ):
                plots_dir = Path.cwd() / "plots" / "implicit" / "max_mse"
                plots_dir.mkdir(parents=True, exist_ok=True)
                viz_path = plots_dir / f"cycle_verify_max_mse_ep{episode_idx:05d}.png"
                _try_save_cycle_verify_plot(
                    viz_path,
                    actions_hat=max_actions_hat_implicit,
                    actions_true=max_actions_true_implicit,
                    mse_value=max_mse_value_implicit,
                    step_idx=max_mse_step_implicit,
                    label="implicit",
                )
                if max_mse_value_implicit > global_max_mse_value_implicit:
                    global_max_mse_value_implicit = max_mse_value_implicit
                    global_max_episode_idx_implicit = episode_idx
                    global_max_step_implicit = max_mse_step_implicit
                    global_max_actions_hat_implicit = max_actions_hat_implicit
                    global_max_actions_true_implicit = max_actions_true_implicit

            # Always save a combined plot for easy comparison (only triggered methods).
            mse_series = {}
            l1_series = {}
            if args.explicit:
                mse_series["explicit"] = mse_list
                l1_series["explicit"] = l1_list
            if args.implicit:
                mse_series["implicit"] = mse_list_implicit
                l1_series["implicit"] = l1_list_implicit
            if args.cem:
                mse_series["cem"] = mse_list_cem
                l1_series["cem"] = l1_list_cem
            if mse_series:
                plots_dir = Path.cwd() / "plots" / "compare" / "mse_list"
                plots_dir.mkdir(parents=True, exist_ok=True)
                mse_plot_path = plots_dir / f"cycle_verify_mse_list_compare_ep{episode_idx:05d}.png"
                _try_save_mse_lists_plot(
                    mse_plot_path,
                    mse_series,
                    title="cycle_verify MSE per step",
                    ylabel="MSE",
                )
            if l1_series:
                plots_dir = Path.cwd() / "plots" / "compare" / "l1_list"
                plots_dir.mkdir(parents=True, exist_ok=True)
                l1_plot_path = plots_dir / f"cycle_verify_l1_list_compare_ep{episode_idx:05d}.png"
                _try_save_mse_lists_plot(
                    l1_plot_path,
                    l1_series,
                    title="cycle_verify L1 per step",
                    ylabel="L1 (MAE)",
                )

            # If requested, write the 3 full trajectories for this episode.
            if args.dump_actions_episode_idx >= 0 and episode_idx == args.dump_actions_episode_idx:
                if len(dump_actions_true) != episode_len:
                    print(
                        f"[warn] dump_actions_episode_idx={args.dump_actions_episode_idx}: "
                        f"expected {episode_len} steps, got {len(dump_actions_true)}"
                    )
                if dump_actions_true:
                    action_stats = None
                    if isinstance(data_cfg.norm_stats, dict):
                        action_stats = data_cfg.norm_stats.get("actions")
                    traj_dir = Path.cwd() / "plots" / "trajectories"
                    _dump_action_trajectories_npz(
                        traj_dir / f"episode_{episode_idx:05d}_actions.npz",
                        actions_true_norm=np.stack(dump_actions_true, axis=0),
                        actions_hat_explicit_norm=np.stack(dump_actions_hat_explicit, axis=0),
                        actions_hat_implicit_norm=np.stack(dump_actions_hat_implicit, axis=0),
                        action_stats=action_stats,
                        use_quantiles=data_cfg.use_quantile_norm,
                        meta={
                            "episode_idx": episode_idx,
                            "episode_path": episode_path,
                            "notes": "Arrays are (T, H, A) where T=episode_len, H=action_horizon, A=action_dim",
                        },
                    )

        episode_time = time.perf_counter() - episode_timer
        print(
            f"[{episode_idx + 1}/{len(episode_paths)}] wrote {output_path} "
            f"(episode_time={episode_time:.2f}s, data_prep={data_prep_time:.2f}s, model={model_time:.2f}s)"
        )

    # End-of-run summary for cycle verification.
    if (
        args.cycle_verify
        and args.explicit
        and global_max_episode_idx >= 0
        and global_max_actions_hat is not None
        and global_max_actions_true is not None
    ):
        plots_dir = Path.cwd() / "plots" / "explicit"
        plots_dir.mkdir(parents=True, exist_ok=True)
        global_viz_path = plots_dir / "cycle_verify_global_max_mse.png"
        _try_save_cycle_verify_plot(
            global_viz_path,
            actions_hat=global_max_actions_hat,
            actions_true=global_max_actions_true,
            mse_value=global_max_mse_value,
            step_idx=global_max_step,
            label="explicit GLOBAL",
        )
        summary_path = plots_dir / "cycle_verify_global_max_mse.json"
        summary = {
            "method": "explicit",
            "global_max_mse": global_max_mse_value,
            "episode_idx": global_max_episode_idx,
            "episode_step": global_max_step,
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"[cycle_verify] GLOBAL max mse={global_max_mse_value:.6g} episode={global_max_episode_idx} step={global_max_step}")
        print(f"[cycle_verify] wrote {summary_path}")


    if (
        args.cycle_verify
        and args.implicit
        and global_max_episode_idx_implicit >= 0
        and global_max_actions_hat_implicit is not None
        and global_max_actions_true_implicit is not None
    ):
        plots_dir = Path.cwd() / "plots" / "implicit"
        plots_dir.mkdir(parents=True, exist_ok=True)
        global_viz_path = plots_dir / "cycle_verify_global_max_mse.png"
        _try_save_cycle_verify_plot(
            global_viz_path,
            actions_hat=global_max_actions_hat_implicit,
            actions_true=global_max_actions_true_implicit,
            mse_value=global_max_mse_value_implicit,
            step_idx=global_max_step_implicit,
            label="implicit GLOBAL",
        )
        summary_path = plots_dir / "cycle_verify_global_max_mse.json"
        summary = {
            "method": "implicit",
            "global_max_mse": global_max_mse_value_implicit,
            "episode_idx": global_max_episode_idx_implicit,
            "episode_step": global_max_step_implicit,
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(
            f"[cycle_verify] GLOBAL implicit max mse={global_max_mse_value_implicit:.6g} "
            f"episode={global_max_episode_idx_implicit} step={global_max_step_implicit}"
        )
        print(f"[cycle_verify] wrote {summary_path}")

    if (
        args.cycle_verify
        and args.cem
        and global_max_episode_idx_cem >= 0
        and global_max_actions_hat_cem is not None
        and global_max_actions_true_cem is not None
    ):
        plots_dir = Path.cwd() / "plots" / "cem"
        plots_dir.mkdir(parents=True, exist_ok=True)
        global_viz_path = plots_dir / "cycle_verify_global_max_mse.png"
        _try_save_cycle_verify_plot(
            global_viz_path,
            actions_hat=global_max_actions_hat_cem,
            actions_true=global_max_actions_true_cem,
            mse_value=global_max_mse_value_cem,
            step_idx=global_max_step_cem,
            label="cem GLOBAL",
        )
        summary_path = plots_dir / "cycle_verify_global_max_mse.json"
        summary = {
            "method": "cem",
            "global_max_mse": global_max_mse_value_cem,
            "episode_idx": global_max_episode_idx_cem,
            "episode_step": global_max_step_cem,
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(
            f"[cycle_verify] GLOBAL cem max mse={global_max_mse_value_cem:.6g} "
            f"episode={global_max_episode_idx_cem} step={global_max_step_cem}"
        )
        print(f"[cycle_verify] wrote {summary_path}")

def _save_full_error_plot(
    action_errors,
    gt_actions,
    reconstructed_actions,
    plot_out_path,
    episode_idx,
    method_name="explicit",
):
    """
    Generate plots showing the difference between GT and reconstructed actions (like replay_reconstructed_actions.py).
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] error plot skipped (matplotlib not available): {e}")
        return
    import numpy as np
    gt_actions = np.array(gt_actions)
    reconstructed_actions = np.array(reconstructed_actions)
    action_errors = np.array(action_errors)
    episode_steps = np.arange(len(action_errors))

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    # Plot 1: Overall L2 error over time
    axes[0].plot(episode_steps, action_errors, linewidth=2, color='red')
    axes[0].set_xlabel('Episode Step')
    axes[0].set_ylabel('L2 Error')
    axes[0].set_title('L2 Error between GT and Reconstructed Actions')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Per-dimension error over time
    per_dim_errors = np.abs(gt_actions - reconstructed_actions)
    for dim in range(per_dim_errors.shape[1]):
        axes[1].plot(episode_steps, per_dim_errors[:, dim], label=f'Dim {dim}', alpha=0.7)
    axes[1].set_xlabel('Episode Step')
    axes[1].set_ylabel('Absolute Error')
    axes[1].set_title('Per-Dimension Absolute Error')
    axes[1].legend(loc='upper right', ncol=4)
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Action values comparison (first 3 dimensions)
    for dim in range(min(3, gt_actions.shape[1])):
        axes[2].plot(episode_steps, gt_actions[:, dim], '--', label=f'GT Dim {dim}', alpha=0.7)
        axes[2].plot(episode_steps, reconstructed_actions[:, dim], '-', label=f'Recon Dim {dim}', alpha=0.7)
    axes[2].set_xlabel('Episode Step')
    axes[2].set_ylabel('Action Value')
    axes[2].set_title('Action Values Comparison (First 3 Dimensions)')
    axes[2].legend(loc='upper right', ncol=3)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    time_str = time.strftime("%H%M%S")
    plot_filename = f"error_plot_episode{episode_idx}_{method_name}_{time_str}.png"
    plot_path = Path(plot_out_path) / plot_filename
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[cycle_verify] error plot saved to: {plot_path}")


if __name__ == "__main__":
    main()


# for i in 0 1 2 3; do   CUDA_VISIBLE_DEVICES=$i ACT_DATA_ROOT=converted_dataset  uv run scripts/gen_noise_data.py     --policy_config_name=pi0_libero_90_low_mem_finetune     --data_config_name=pi0_libero_90_low_mem_finetune_franka     --pytorch_weight_path=/gpfs/scrubbed/shubham/chkpts/pi0_libero_90_low_mem_finetune/pi0_libero_90_low_mem_finetune_20251218_165604/pytorch_chkpts/29999_3/model.safetensors    --output_name data_with_noise_full.pkl     --batch_size 128 --num_workers 16 --cycle_verify     --implicit --num_steps 150 --num_demos 200    --num_shards 4 --shard_id=$i & done