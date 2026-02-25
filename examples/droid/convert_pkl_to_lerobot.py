#!/usr/bin/env python3
"""
Convert pickle files with DROID-like structure to LeRobot dataset format.

This script converts your custom pickle dataset to the LeRobot format required by pi0.
Images are expected to be RGBA (224, 224, 4) and will be converted to RGB.

Camera mapping:
- 23804457_left -> exterior_image_1_left (base camera)
- 13263313_left -> wrist_image_left (wrist camera)
"""

import pickle
import json
import logging
from pathlib import Path
from typing import Optional
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import concurrent.futures

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_pkl_file(args):
    """Process a single pickle file and return frames."""
    pkl_file, action_dim = args
    try:
        with open(pkl_file, 'rb') as f:
            episode_data = pickle.load(f)
    except Exception as e:
        return None, f"Failed to load {pkl_file}: {e}"
    
    if not isinstance(episode_data, dict):
        return None, f"Invalid episode data in {pkl_file}: expected dict"
    
    timesteps = episode_data.get('timesteps')
    if not isinstance(timesteps, list) or len(timesteps) == 0:
        return None, f"No valid timesteps in {pkl_file}"
    
    episode_frames = []
    
    for t, timestep in enumerate(timesteps):
        try:
            obs = timestep.get("observations", {})
            action = timestep.get("action", {})
            
            # Extract images (RGBA -> RGB)
            image_dict = obs.get("image")
            if image_dict is None or "23804457_left" not in image_dict or "13263313_left" not in image_dict:
                continue
            
            base_img_rgba = np.asarray(image_dict["23804457_left"])
            wrist_img_rgba = np.asarray(image_dict["13263313_left"])
            
            if base_img_rgba.shape != (224, 224, 4) or wrist_img_rgba.shape != (224, 224, 4):
                continue
            
            base_img = base_img_rgba[..., :3].astype(np.uint8)
            wrist_img = wrist_img_rgba[..., :3].astype(np.uint8)
            
            robot_state = obs.get("robot_state")
            if robot_state is None or "joint_positions" not in robot_state or "gripper_position" not in robot_state:
                continue
            
            joint_positions = np.array(robot_state["joint_positions"], dtype=np.float32)
            gripper_position = np.array([robot_state["gripper_position"]], dtype=np.float32)
            
            if len(joint_positions) != 7:
                continue
            
            action_data = action.get("robot_state")
            if action_data is None or "joint_velocities" not in action_data or "gripper_velocity" not in action:
                continue
            
            action_joint_velocity = np.array(action_data["joint_velocities"], dtype=np.float32)
            action_gripper_velocity = np.array([action["gripper_velocity"]], dtype=np.float32)
            
            if len(action_joint_velocity) != 7:
                continue
            
            action_vector = np.concatenate([action_joint_velocity, action_gripper_velocity])
            if len(action_vector) < action_dim:
                action_vector = np.pad(action_vector, (0, action_dim - len(action_vector)))
            action_vector = action_vector[:action_dim].astype(np.float32)
            
            frame = {
                "exterior_image_1_left": base_img,
                "wrist_image_left": wrist_img,
                "joint_position": joint_positions,
                "gripper_position": gripper_position,
                "actions": action_vector,
                "task": str(pkl_file.parent.name),
            }
            
            episode_frames.append(frame)
            
        except Exception as e:
            continue
    
    if len(episode_frames) < 2:
        return None, f"Episode too short: {len(episode_frames)} frames"
    
    return episode_frames, None


def convert_pkl_to_lerobot(
    pkl_dir: str = "/path/to/pkl/files",
    output_dir: str = "./custom_droid_dataset",
    fps: int = 10,
    robot_type: str = "droid",
    action_dim: int = 32,
    repo_id: str = "shubhamg20/custom_three_tasks",
    push_to_hub: bool = False,
    private: bool = False,
):
    """
    Convert pickle files to LeRobot dataset format.
    
    Default settings are optimized for pi0 DROID finetuning.
    
    Args:
        pkl_dir: Directory containing .pkl files (default: ./pkl_files)
        output_dir: Output directory for LeRobot dataset (default: ./custom_droid_dataset)
        fps: Frames per second (default: 10)
        robot_type: Type of robot (default: "droid")
        action_dim: Action dimension, 32 for pi0/pi05, 8 for pi0-FAST (default: 32)
        repo_id: HuggingFace repo ID (default: shubhamg20/custom_three_tasks)
        push_to_hub: Whether to push to HuggingFace Hub (default: False)
        private: Make dataset private on Hub (default: False)
    """
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.datasets.utils import DEFAULT_FEATURES
    
    pkl_dir = Path(pkl_dir)
    output_dir = Path(output_dir)
    
    # Find all task folders and their pkl files
    task_folders = sorted([d for d in pkl_dir.iterdir() if d.is_dir()])
    if not task_folders:
        raise FileNotFoundError(f"No task folders found in {pkl_dir}")
    
    logger.info(f"Found {len(task_folders)} task folders: {[d.name for d in task_folders]}")
    
    # Collect all pkl files by task
    pkl_files_by_task = {}
    for task_folder in task_folders:
        pkl_files = sorted(task_folder.glob("*.pkl"))
        if pkl_files:
            pkl_files_by_task[task_folder.name] = pkl_files
            logger.info(f"  {task_folder.name}: {len(pkl_files)} episodes")
    
    if not pkl_files_by_task:
        raise FileNotFoundError(f"No .pkl files found in task folders under {pkl_dir}")
    
    total_pkl_files = sum(len(files) for files in pkl_files_by_task.values())
    logger.info(f"Total: {total_pkl_files} pickle files")
    
    # Define features for this dataset (following the reference convert_droid_data_to_lerobot.py)
    features = {
        "exterior_image_1_left": {
            "dtype": "image",
            "shape": (224, 224, 3),
            "names": ["height", "width", "channel"],
        },
        "wrist_image_left": {
            "dtype": "image",
            "shape": (224, 224, 3),
            "names": ["height", "width", "channel"],
        },
        "joint_position": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["joint_position"],
        },
        "gripper_position": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["gripper_position"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": ["actions"],
        },
    }
    
    # Create empty LeRobot dataset
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=output_dir,
        robot_type=robot_type,
        features=features,
        use_videos=False,  # Store as images, not videos
    )
    
    episode_index = 0
    total_frames = 0
    
    # Collect all pkl files for parallel processing
    all_pkl_files = []
    for task_name, pkl_files in pkl_files_by_task.items():
        all_pkl_files.extend(pkl_files)
    
    logger.info(f"\nProcessing {len(all_pkl_files)} episodes in parallel...")
    
    # Process all files in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(
            executor.map(process_pkl_file, [(f, action_dim) for f in all_pkl_files]),
            total=len(all_pkl_files),
            desc="Converting episodes"
        ))
    
    # Add results to dataset
    for episode_frames, error in results:
        if error:
            logger.warning(error)
            continue
        
        if episode_frames is None:
            continue
        
        try:
            for frame in episode_frames:
                dataset.add_frame(frame)
            
            dataset.save_episode()
            
            total_frames += len(episode_frames)
            logger.info(f"✓ Episode {episode_index}: {len(episode_frames)} frames")
            episode_index += 1
            
        except Exception as e:
            logger.error(f"Failed to save episode {episode_index}: {e}")
            continue
    
    logger.info(f"\n✓ Conversion complete!")
    logger.info(f"  Episodes: {episode_index}")
    logger.info(f"  Total frames: {total_frames}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Dataset repo: {repo_id}")
    
    if push_to_hub:
        logger.info(f"\nPushing to Hub: {repo_id}")
        dataset.repo_id = repo_id
        try:
            dataset.push_to_hub(
                private=private,
                tag_version=True,
                license="apache-2.0",
            )
            logger.info(f"✓ Successfully pushed to {repo_id}")
        except Exception as e:
            logger.error(f"Failed to push to Hub: {e}")
            logger.info("You can manually push later with:")
            logger.info(f"  huggingface-cli upload {repo_id} {dataset.root} --repo-type dataset")
    
    return dataset, episode_index, total_frames


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert pickle files to LeRobot dataset format (default: pi0 DROID finetuning)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pkl_dir", type=str, default="../recorded_runs/droid/data", help="Directory containing .pkl files")
    parser.add_argument("--output_dir", type=str, default="./custom_droid_dataset", help="Output directory")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second")
    parser.add_argument("--robot_type", type=str, default="droid", help="Robot type")
    parser.add_argument("--action_dim", type=int, default=32, help="Action dimension (32 for pi0/pi05, 8 for pi0-FAST)")
    parser.add_argument("--repo_id", type=str, default="shubhamg20/custom_three_tasks", help="HuggingFace repo ID")
    parser.add_argument("--push_to_hub", action="store_true", help="Push dataset to HuggingFace Hub")
    parser.add_argument("--private", action="store_true", help="Make dataset private on Hub")
    
    args = parser.parse_args()
    
    dataset, num_episodes, num_frames = convert_pkl_to_lerobot(
        pkl_dir=args.pkl_dir,
        output_dir=args.output_dir,
        fps=args.fps,
        robot_type=args.robot_type,
        action_dim=args.action_dim,
        repo_id=args.repo_id,
        push_to_hub=args.push_to_hub,
        private=args.private,
    )
