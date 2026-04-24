        
"""
Script for replaying reconstructed actions from a model in the Libero environment.

This script:
1. Loads a dataset and extracts ground truth actions
2. Resets the environment to match the dataset's initial state
3. Sends observations and actions to the model for reconstruction
4. Plays the reconstructed actions in the environment
5. Saves a video of the replay
6. Generates a plot showing the difference between GT and reconstructed actions

Usage:
uv run examples/libero/replay_reconstructed_actions.py \
    --dataset_path /path/to/dataset \
    --host 0.0.0.0 \
    --port 8000 \
    --task_suite_name libero_10 \
    --task_id 0 \
    --video_out_path data/libero/replay_videos
"""

import collections
import dataclasses
import logging
import math
import sys
import time
import pathlib
from pathlib import Path
from typing import Optional
import tqdm
import imageio

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VENDORED_LIBERO = _REPO_ROOT / "third_party" / "libero"
if _VENDORED_LIBERO.exists() and str(_VENDORED_LIBERO) not in sys.path:
    sys.path.insert(0, str(_VENDORED_LIBERO))

from libero.libero import benchmark
import wandb
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import matplotlib.pyplot as plt
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tyro
import torch

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:
    # Whether to use closed-loop (env obs) or open-loop (dataset obs) images
    close_loop: bool = False
    #################################################################################################################
    # Dataset parameters
    #################################################################################################################
    # dataset_path: str = "shubhamg20/libero_10_no_noops_task-turnonthestoveandputthemokapotonit-full"  # Path to the LeRobot dataset
    dataset_path: str = "shubhamg20/libero_goal_no_noops_tasks-0_1_7" #it wil first chk in local $HF_LEROBOT_HOME directory, if not found it will try to load from Hugging Face Hub
    action_horizon: int = 50  # Should not be changed, Action horizon for reconstruction
    replay_horizon:int = 30
    gt_replay: bool = False
    plot_errors: bool = False
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "172.16.0.60"
    port: int = 8000
    resize_size: int = 224

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_goal"  # Task suite for environment setup
    task_id: int = 1  # Task ID for environment setup
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize

    #################################################################################################################
    # Output parameters
    #################################################################################################################
    video_out_path: str = "data/libero/replay_videos"  # Path to save replay videos
    plot_out_path: str = "data/libero/replay_plots"  # Path to save error plots


    seed: int = 7  # Random seed
    num_rollouts: int = 40  # Number of rollouts to perform

    use_wandb: bool = False  # Whether to also log results in Weights & Biases
    wandb_project: str = "sft-noise"  # Name of W&B project to log to (use default!)
    wandb_group_prefix: str = None
    wandb_entity: str = "shubham2-university-of-washington"  # Name of entity to log under
    wandb_name_suffix: str = ""


def replay_reconstructed_actions(args: Args) -> None:
    """
    Replay reconstructed actions from a model in the Libero environment.
    """

    # Setup output directories and dataset once
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.plot_out_path).mkdir(parents=True, exist_ok=True)
    logging.info(f"Loading dataset from: {args.dataset_path}")
    dataset = LeRobotDataset(args.dataset_path)

    # wandb: init ONCE, log all videos/plots, finish at end
    if args.use_wandb:
        run_name = f"replay-recon-task{args.task_id}_date-{time.strftime('%Y-%m-%d')}_seed-{args.seed}"
        group = None
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=run_name, config=dataclasses.asdict(args), group=group)


    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    task = task_suite.get_task(args.task_id)
    task_description = task.language
    logging.info(f"Task: {task_description}")

    # Find the task_index in the dataset that matches this task's language instruction
    task_idx_in_dataset = next(
        idx for idx, desc in dataset.meta.tasks.items() if desc == task_description
    )
    # Load full task_index column once, then index by episode start frames
    all_task_indices = np.array(dataset.hf_dataset["task_index"])
    first_frame_indices = [int(dataset.episode_data_index["from"][ep]) for ep in range(dataset.num_episodes)]
    task_episode_indices = [
        ep for ep, fi in enumerate(first_frame_indices) if all_task_indices[fi] == task_idx_in_dataset
    ]
    logging.info(f"Found {len(task_episode_indices)} episodes for task '{task_description}'")

    all_rollout_status = []
    for rollout_idx in range(args.num_rollouts):
        ep_idx = task_episode_indices[rollout_idx % len(task_episode_indices)]
        # Set random seed for each rollout for reproducibility
        np.random.seed(args.seed + rollout_idx)

        # Initialize LIBERO environment
        env, _ = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed + rollout_idx)

        episode_start = int(dataset.episode_data_index["from"][ep_idx])
        episode_end = int(dataset.episode_data_index["to"][ep_idx]) + 50
        episode_length = episode_end - episode_start
        # Get initial states for this task
        initial_states = task_suite.get_task_init_states(args.task_id)

        # Reset environment to initial state
        env.reset()
        obs = env.set_init_state(initial_states[rollout_idx])  # Use first initial state

        # Initialize model client
        client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

        # Storage for replay data
        replay_images = []
        gt_actions = []
        reconstructed_actions = []
        action_errors = []

        logging.info(f"[Rollout {rollout_idx+1}/{args.num_rollouts}] Starting replay with reconstructed actions...")

        # Preload entire episode at once
        ep_slice = dataset.hf_dataset.select(range(episode_start, episode_end))
        ep_actions = np.array(ep_slice["actions"])   # (T, 7)
        ep_states = np.array(ep_slice["state"])      # (T, 8)
        ep_images = ep_slice["image"]                # list of PIL/tensors
        ep_wrist_images = ep_slice["wrist_image"]

        # Wait for objects to stabilize
        for t in range(args.num_steps_wait):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

        # Iterate through episode frames
        done = False
        for frame_idx in tqdm.tqdm(range(episode_start, episode_end, args.replay_horizon)):
            relative_idx = frame_idx - episode_start

            if args.plot_errors:
                gt_action = ep_actions[relative_idx]
                gt_state = ep_states[relative_idx]
            # Get current observation and state from environment or dataset (open/closed loop)
            if args.close_loop:
                # Closed-loop: use current obs and state from env 
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                state = np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )
                )
            else:
                # Open-loop: use preloaded images and state
                img = np.array(ep_images[relative_idx])
                wrist_img = np.array(ep_wrist_images[relative_idx])
                if img.ndim == 3 and img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                    wrist_img = np.transpose(wrist_img, (1, 2, 0))
                img = np.ascontiguousarray(img)
                wrist_img = np.ascontiguousarray(wrist_img)
                img = (img * 255).clip(0, 255).astype(np.uint8) if np.issubdtype(img.dtype, np.floating) else img.astype(np.uint8)
                wrist_img = (wrist_img * 255).clip(0, 255).astype(np.uint8) if np.issubdtype(wrist_img.dtype, np.floating) else wrist_img.astype(np.uint8)
                state = ep_states[relative_idx]

            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
            )
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
            )

            chunk_end = min(relative_idx + args.action_horizon, len(ep_actions))
            gt_action_chunk = ep_actions[relative_idx:chunk_end]

            if args.gt_replay:
                # Skip model inference entirely for gt replay
                for step_i in range(args.replay_horizon):
                    action = gt_action_chunk[min(step_i, len(gt_action_chunk) - 1), :7]
                    obs, reward, done, info = env.step(action.tolist())
                    replay_images.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
                    if done:
                        break
            else:
                if len(gt_action_chunk) < args.action_horizon:
                    last_action = gt_action_chunk[-1]
                    pad = np.tile(last_action, (args.action_horizon - len(gt_action_chunk), 1))
                    gt_action_chunk = np.concatenate([gt_action_chunk, pad], axis=0)

                element = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": state,
                    "prompt": task_description,
                    "actions": gt_action_chunk,
                    "invert": True,
                }
                element = tensor_and_dtype_to_numpy(element)
                return_dict = client.infer(element)
                reconstructed_action_chunk = return_dict["reconstructed_actions"]
                target_noise = return_dict.get("target_noise", None)

                for step_i in range(args.replay_horizon):
                    reconstructed_action = reconstructed_action_chunk[step_i, :7]
                    obs, reward, done, info = env.step(reconstructed_action.tolist())
                    replay_images.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
                    reconstructed_actions.append(reconstructed_action)
                    if args.plot_errors:
                        gt_actions.append(gt_action_chunk[step_i, :7])
                    if done:
                        break

                
                if done:
                    logging.info(f"Task completed at step {relative_idx} (replay step {step_i})!")
                    break
            
            if args.plot_errors and reconstructed_action_chunk is not None:
                error = np.linalg.norm(reconstructed_action_chunk - gt_action_chunk[:,:7])
                horizon = reconstructed_action_chunk.shape[0]
                error /= horizon
                action_errors.append(error)
                logging.info(f"Step {relative_idx}/{episode_length}: Action error = {error:.6f}")
            if done:
                break

        # Save replay video
        time_str = time.strftime("%H%M%S")
        video_filename = f"replay_episode{rollout_idx}_task{args.task_id}_rollout{rollout_idx}_{time_str}.mp4"
        video_path = pathlib.Path(args.video_out_path) / video_filename
        imageio.mimwrite(
            video_path,
            [np.asarray(x) for x in replay_images],
            fps=10,
            codec='libx264',
            format='mp4'
        )
        logging.info(f"Replay video saved to: {video_path}")
        # Log error plot and current overall success rate to wandb
        all_rollout_status.append(int(done))
        if args.use_wandb:
            current_success_rate = sum(all_rollout_status) / float(rollout_idx + 1)
            wandb.log({
                f"success_rate/episode{rollout_idx}_task{args.task_id}": current_success_rate
                })
        if args.plot_errors:
            # Generate error plot and get plot path
            plot_path = _generate_error_plot(
                action_errors,
                gt_actions,
                reconstructed_actions,
                args.plot_out_path,
                rollout_idx,
                args.task_id,
            )

            if args.use_wandb and plot_path is not None:
                wandb.log({
                    f"plots/error_plot_episode{rollout_idx}_task{args.task_id}_rollout{rollout_idx}": wandb.Image(str(plot_path)),
                })

        # Log video to wandb
        if args.use_wandb:
            wandb.log({
                f"videos/replay_episode{rollout_idx}_task{args.task_id}_rollout{rollout_idx}": wandb.Video(str(video_path), fps=10)
            })

        if args.plot_errors:
            # Print statistics
            logging.info("\n" + "=" * 60)
            logging.info(f"[Rollout {rollout_idx+1}/{args.num_rollouts}] RECONSTRUCTION STATISTICS")
            logging.info("=" * 60)
            logging.info(f"Mean action error: {np.mean(action_errors):.6f}")
            logging.info(f"Std action error: {np.std(action_errors):.6f}")
            logging.info(f"Max action error: {np.max(action_errors):.6f}")
            logging.info(f"Min action error: {np.min(action_errors):.6f}")
            logging.info("=" * 60)

        # Clean up environment after each rollout
        env.close()

    # wandb: finish ONCE after all rollouts
    if args.use_wandb:
        wandb.finish()

# Convert all PyTorch tensors in 'element' to numpy arrays before inference
def tensor_and_dtype_to_numpy(obj):
    import numpy as np
    import torch
    if isinstance(obj, torch.Tensor):
        arr = obj.cpu().numpy()
        # Convert dtype if not image
        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32)
        return arr
    elif isinstance(obj, np.ndarray):
        # Convert dtype if not image
        if obj.dtype != np.uint8:
            return obj.astype(np.float32)
        return obj
    elif isinstance(obj, dict):
        return {k: tensor_and_dtype_to_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_and_dtype_to_numpy(v) for v in obj]
    else:
        return obj

def _generate_error_plot(
    action_errors,
    gt_actions,
    reconstructed_actions,
    plot_out_path,
    episode_idx,
    task_id,
):
    """
    Generate plots showing the difference between GT and reconstructed actions.
    """
    gt_actions = np.array(gt_actions)
    reconstructed_actions = np.array(reconstructed_actions)
    episode_steps = np.arange(len(gt_actions))

    actual_inferred_steps = np.arange(len(action_errors)) 
    # Create figure with multiple subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot 1: Overall L2 error over time
    axes[0].plot(actual_inferred_steps, action_errors, linewidth=2, color='red')
    axes[0].set_xlabel('Episode Step')
    axes[0].set_ylabel('L2 Error')
    axes[0].set_title('L2 Error between GT and Reconstructed Actions')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Per-dimension error over time
    per_dim_errors = np.abs(gt_actions - reconstructed_actions)/gt_actions.shape[0]
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


    # Save plot
    time_str = time.strftime("%H%M%S")
    plot_filename = f"error_plot_episode{episode_idx}_task{task_id}_{time_str}.png"
    plot_path = pathlib.Path(plot_out_path) / plot_filename
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f"Error plot saved to: {plot_path}")
    return plot_path


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    print(task_description)
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":

    from libero.libero import benchmark
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite_name = "libero_goal"
    task_suite = benchmark_dict[task_suite_name]()
    print(f"Loaded task suite: {task_suite}")
    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        print(task_id, ":", task.language)
        logging.basicConfig(level=logging.INFO)

    tyro.cli(replay_reconstructed_actions)
