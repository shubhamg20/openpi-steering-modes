from functools import lru_cache
import os
import pickle
from pathlib import Path

import numpy as np
import torch


@lru_cache(maxsize=1000)
def _load_episode_data(episode_data_path: str) -> dict:
    with open(episode_data_path, "rb") as f:
        return pickle.load(f)


class ActDataset(torch.utils.data.Dataset):
    # Default prompts for each task subdirectory
    # DEFAULT_TASK_PROMPTS: dict[str, str] = {
    #     "paired_pan": "pick up the lid and place it on the pan",
    #     "paired_purple": "pick up the purple object and place it in the pan",
    #     "paired_yellow": "pick up the yellow object and place it in the pan",
    # }
    DEFAULT_TASK_PROMPTS: dict[str, str] = {
        "paired_pan": "",
        "paired_purple": "",
        "paired_yellow": "",
    }

    def __init__(
        self,
        root_dir: str,
        action_horizon: int,
        prompt: str = "",  # fallback prompt if task not in task_prompts
        task_prompts: dict[str, str] | None = None,  # per-task prompts keyed by subdir name
        with_noise: bool = False,
        extra_episode_keys: list[str] = [],  # extra keys to load from episode pickle (e.g. "noise_action" for noise dataset
    ):
        self.extra_episode_keys = extra_episode_keys
        self.root_dir = root_dir
        self.action_horizon = action_horizon
        self.prompt = prompt
        self._task_prompts = self.DEFAULT_TASK_PROMPTS

        # scan task folders for episode pkls
        self.episode_paths = []
        self.episode_lengths = []
        self.episode_prompts = []
        self.episode_task_names = []

        root = Path(root_dir)
        for task_dir in sorted(root.iterdir()):
            if not task_dir.is_dir():
                continue    
            task_name = task_dir.name
            if with_noise:
                # files = sorted(task_dir.glob("*_pi0-droid_noise.pkl"))[:]
                # files = sorted(task_dir.glob("*_pi0-droid_with_prompts_noise.pkl"))[:]
                # files = sorted(task_dir.glob("*_pi0-droid-no-lang_noise.pkl"))[:]
                # files = sorted(task_dir.glob("*_pi0-droid-no-lang-discrete-timestep_noise_40000.pkl"))[:]
                files = sorted(task_dir.glob("*_pi0-droid-no-lang-discrete-timestep_noise_99999.pkl"))[:]
                # files = sorted(task_dir.glob("*_pi0-droid-with-prompts-discrete-timestep_noise_40000.pkl"))[:]
            else:
               files = sorted(
                    f for f in task_dir.glob("*.pkl")
                    if "_noise" not in f.name
                )
            for episode_pkl in files:
                data = _load_episode_data(str(episode_pkl))
                if "robot" in data and "timesteps" in data["robot"]:
                    episode_len = len(data["robot"]["timesteps"])
                elif "timesteps" in data:
                    episode_len = len(data["timesteps"])
                else:
                    raise ValueError(f"Could not determine episode length for {episode_pkl}")
                self.episode_paths.append(str(episode_pkl))
                self.episode_lengths.append(episode_len)
                self.episode_prompts.append(self._task_prompts.get(task_name, prompt))
                self.episode_task_names.append(task_name)


        # flat index: (episode_id, t)
        self.idx_to_episode_and_step = []
        for episode_id, ep_len in enumerate(self.episode_lengths):
            for t in range(ep_len):
                self.idx_to_episode_and_step.append((episode_id, t))

        sample_data = _load_episode_data(self.episode_paths[0])
        if "robot" in sample_data and "timesteps" in sample_data["robot"] and len(sample_data["robot"]["timesteps"]) > 0:
            self.image_keys = list(sample_data["robot"]["timesteps"][0]["observations"]["image"].keys())
        elif "timesteps" in sample_data and len(sample_data["timesteps"]) > 0:
            self.image_keys = list(sample_data["timesteps"][0]["observations"]["image"].keys())
        else:
             raise ValueError(f"Could not find timesteps in sample episode data at {self.episode_paths[0]} to determine image keys")

        #TODO Find better way
        ###################################################################################
        self.image_keys = ["exterior_image_1_left", "wrist_image_left"]
        self.data_keys = ["23804457_left", "13263313_left"] 
        ##################################################################################
        print(f"[Act Dataset] found image keys: {self.data_keys}")

        print(f"[Act Dataset] found {len(self.episode_paths)} episodes, "
              f"{len(self.idx_to_episode_and_step)} total timesteps")

    def __len__(self):
        return len(self.idx_to_episode_and_step)

    def __getitem__(self, idx):
        episode_id, t = self.idx_to_episode_and_step[idx]
        episode_path = self.episode_paths[episode_id]
        episode_len = self.episode_lengths[episode_id]

        task_name = self.episode_task_names[episode_id]

        data = _load_episode_data(episode_path)
        if "robot" in data and "timesteps" in data["robot"]:
            timesteps = data["robot"]["timesteps"]
        elif "timesteps" in data:
            timesteps = data["timesteps"]
        else:
            raise ValueError(f"Could not find timesteps in episode data at {episode_path}")
        if "human" in data and "timesteps" in data["human"]:
            human_timesteps = data["human"]["timesteps"]
        else:
            human_timesteps = []

        # --- state ---
        joint_pos = torch.from_numpy(np.array(timesteps[t]["observations"]["robot_state"]["joint_positions"])).float()
        gripper_pos = torch.from_numpy(np.array([timesteps[t]["observations"]["robot_state"]["gripper_position"]])).float()
        if not human_timesteps:
            human_traj = torch.full((110, 4), -1.0)
        else:
            human_traj = torch.from_numpy(np.stack([np.array(ts["hand_pose"]) for ts in human_timesteps], axis=0)).float()
            # Pad to 110 timesteps with the last timestep's full pose
            if human_traj.shape[0] < 110:
                last_pose = human_traj[-1]
                pad_count = 110 - human_traj.shape[0]
                human_traj = torch.cat([human_traj, last_pose.unsqueeze(0).expand(pad_count, -1)], dim=0)

        # --- action chunk [t, t+action_horizon) ---
        action_idxs = list(range(t, t + self.action_horizon))
        action_is_pad = [i >= episode_len for i in action_idxs]
        action_idxs = [min(i, episode_len - 1) for i in action_idxs]  # clamp

        # Build action array for each timestep in action_idxs
        actions = torch.from_numpy(np.stack([
            np.concatenate([
                np.array(timesteps[i]["action"]["joint_velocity"]),
                np.array([timesteps[i]["action"]["target_gripper_position"]])
            ]) for i in action_idxs
        ], axis=0)).float()  # (action_horizon, action_dim)
        # print(self.episode_prompts[episode_id])
        result = {
            "observation.joint_position": joint_pos,
            "observation.gripper_position": gripper_pos, 
            "observation.human_traj": human_traj.flatten(),
            "action": actions, # (action_horizon, action_dim)
            "action_is_pad": torch.tensor(action_is_pad, dtype=torch.bool),
            "prompt": self.episode_prompts[episode_id],
            "index": idx,
            "episode_index": episode_id,
            "frame_index": t,
            "next.done": t == episode_len - 1,
            "task_name": task_name,
        }
        for i, image_key in enumerate(self.image_keys):
            image = np.array(timesteps[t]["observations"]["image"][self.data_keys[i]])
            if image.shape[-1] == 4:
                image = image[:, :, :3]  # drop alpha
            image = torch.from_numpy(image).permute(2, 0, 1).float()/255.0  
            if image.dtype != torch.float32:
                raise ValueError(f"Expected image dtype float32, got {image.dtype} in ActDataset for {episode_path} at timestep {t} image key {image_key}")
            result[f"observation.images.{image_key}"] = image

        for k in self.extra_episode_keys:
            if k not in data:
                continue
            result[k] = torch.from_numpy(data[k][t]).to(torch.float32)

        return result