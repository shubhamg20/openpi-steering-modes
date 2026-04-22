"""Compute normalization statistics by loading episode pickle files directly.

This is a lightweight alternative to `compute_norm_stats.py` that bypasses the
dataset/dataloader/transforms pipeline. It is useful when you want stats for
keys that are present in the episode pickle (e.g. `noise_action`) but are not
exposed by `ActDataset`.
"""

from __future__ import annotations

import os
import pickle

import etils.epath as epath
import numpy as np
import smart_open
import tqdm
import tyro

import openpi.shared.normalize as normalize
import openpi.training.config as _config


def compute_stats_from_pickle_files(
    root_dir: str,
    keys: list[str],
    *,
    max_frames: int | None = None,
    compute_quantiles: bool = False,
) -> dict[str, normalize.NormStats]:
    """Compute stats directly from episode pickle files.

    Notes:
    - For ACT datasets, episode pickle keys are typically `state` and `action`.
      The model-facing names are `state` and `actions`, so we map:
        - requested "state"   -> dataset_metadata["state_key"] (default: "state")
        - requested "actions" -> dataset_metadata["action_key"] (default: "action")
    - Extra keys (e.g. "noise_action") are used as-is.
    - If a requested key is missing (or doesn't have enough samples), it is skipped.
    """
    if root_dir is None:
        raise ValueError("root_dir is None, cannot compute stats")
    from pathlib import Path
    root = Path(root_dir)
    episode_paths = []
    episode_lengths = []
    for task_dir in sorted(root.iterdir()):
        if not task_dir.is_dir():
            continue
        # files = sorted(task_dir.glob("*_pi0-droid_noise.pkl"))
        # files = sorted(task_dir.glob("*_pi0-droid_with_prompts_noise.pkl"))
        # files = sorted(task_dir.glob("*_pi0-droid-no-lang_noise.pkl"))
        # files = sorted(task_dir.glob("*_pi0-droid-no-lang-discrete-timestep_noise_40000.pkl"))
        files = sorted(task_dir.glob("*_pi0-droid-no-lang-discrete-timestep_noise_99999.pkl"))
        # files = sorted(task_dir.glob("*_pi0-droid-with-prompts-discrete-timestep_noise_40000.pkl"))

        print(f"Found {len(files)} episode files in {task_dir}")
        for episode_pkl in files:
            try:
                with smart_open.open(str(episode_pkl), "rb") as f:
                    data = pickle.load(f)
                if "robot" in data and "timesteps" in data["robot"]:
                    episode_len = len(data["robot"]["timesteps"])
                elif "timesteps" in data:
                    episode_len = len(data["timesteps"])
                else:
                    raise ValueError(f"Could not determine episode length for {episode_pkl}")
                
                episode_paths.append(str(episode_pkl))
                episode_lengths.append(episode_len)
            except Exception as e:
                print(f"Warning: failed to load {episode_pkl}: {e}")
                continue

    stats = {k: normalize.RunningStats() for k in keys}
    num_samples: dict[str, int] = {k: 0 for k in keys}

    import time
    total_frames = 0
    for episode_idx, (episode_path, episode_len) in enumerate(tqdm.tqdm(
        zip(episode_paths, episode_lengths, strict=True),
        total=len(episode_paths),
        desc="Loading pickle files",
    )):
        if max_frames is not None and total_frames >= max_frames:
            break

        t0 = time.time()
        try:
            with smart_open.open(episode_path, "rb") as f:
                episode_data = pickle.load(f)
        except Exception as e:
            print(f"Warning: failed to load {episode_path}: {e}")
            continue
        t1 = time.time()
        if "robot" in episode_data and "timesteps" in episode_data["robot"]:
            robot_timesteps = episode_data["robot"]["timesteps"]
        elif "timesteps" in episode_data:
            robot_timesteps = episode_data["timesteps"]
        else:
            raise ValueError(f"Could not find timesteps in episode data at {episode_path}")
        # human_timesteps = episode_data["human"]["timesteps"]
        episode_len = len(robot_timesteps)
        t2 = time.time()
        # human_traj = np.stack([np.array(ts["hand_pose"]) for ts in human_timesteps], axis=0)
        # if human_traj.shape[0] < 110:
        #     last_pose = np.array(human_timesteps[-1]["hand_pose"])
        #     pad_count = 110 - human_traj.shape[0]
        #     pad_vals = np.repeat(last_pose[None, :], pad_count, axis=0)
        #     human_traj = np.concatenate([human_traj, pad_vals], axis=0)
        # t3 = time.time()

        # Batch all states and actions for this episode
        states = []
        actions = []
        for t in range(episode_len):
            joint_pos = np.array(robot_timesteps[t]["observations"]["robot_state"]["joint_positions"])
            gripper_pos = np.array([robot_timesteps[t]["observations"]["robot_state"]["gripper_position"]])
            state = np.concatenate([
                joint_pos,
                gripper_pos,
                # human_traj.flatten()
            ])
            states.append(state)
            action = np.concatenate([
                np.array(robot_timesteps[t]["action"]["joint_velocity"]),
                np.array([robot_timesteps[t]["action"]["target_gripper_position"]])
            ])
            actions.append(action)

        states = np.stack(states, axis=0)
        actions = np.stack(actions, axis=0)
        t3 = time.time()
        stats["state"].update(states)
        num_samples["state"] += states.shape[0]
        stats["actions"].update(actions)
        num_samples["actions"] += actions.shape[0]

        # Handle extra keys like noise_action
        for k in keys:
            if k not in ["state", "actions"] and k in episode_data:
                arr = np.asarray(episode_data[k])
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                elif arr.ndim > 2:
                    arr = arr.reshape(-1, arr.shape[-1])
                stats[k].update(arr)
                num_samples[k] += int(arr.shape[0])

        t4 = time.time()
        print(f"Episode {episode_idx}: PKL load {t1-t0:.2f}s, numpy prep {t3-t2:.2f}s, stats update {t4-t3:.2f}s, total {t4-t0:.2f}s")
        total_frames += int(episode_len)

    norm_stats: dict[str, normalize.NormStats] = {}
    for k in keys:
        if num_samples[k] < 2:
            print(f"Warning: skipping {k!r} (only {num_samples[k]} samples)")
            continue
        try:
            norm_stats[k] = stats[k].get_statistics()
        except ValueError as e:
            print(f"Warning: skipping {k!r} ({e})")

    return norm_stats


def main(
    config_name: str,
    root_dir: str,
    *,
    extra_keys: list[str] | None = None,
    max_frames: int | None = None,
):
    keys = ["state", "actions", "noise_action"]
    if extra_keys:
        keys.extend(extra_keys)
    config = _config.get_config(config_name)
    output_path = epath.Path(config.data.assets.assets_dir) / config.data.assets.asset_id
    norm_stats = compute_stats_from_pickle_files(
        root_dir=root_dir,
        keys=keys,
        max_frames=max_frames,
    )
   
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)


#uv run scripts/compute_norm_stats_pickles.py  --config-name pi0_droid_lora_finetune_data --extra-keys noise_action --root-dir /gpfs/projects/weirdlab/shubham/openpi-steering-modes/data/data_paired_droid
