"""
Script for generating a filtered Libero dataset for one or more task IDs.

Usage:
uv run examples/libero/get_libero_dataset.py --data_dir /gpfs/scrubbed/shubham/data/data/libero_tfds --task_suite_name libero_goal_no_noops --task_ids 0 1 7

libero_goal tasks:
  0  open the middle drawer of the cabinet
  1  put the bowl on the stove
  2  put the wine bottle on top of the cabinet
  3  open the top drawer and put the bowl inside
  4  put the bowl on top of the cabinet
  5  push the plate to the front of the stove
  6  put the cream cheese in the bowl
  7  turn on the stove
  8  put the bowl on the plate
  9  put the wine bottle on the rack

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $HF_LEROBOT_HOME directory.
"""

import shutil
from typing import List

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from libero.libero import benchmark
from libero.libero.benchmark import libero_suite_task_map
import tensorflow_datasets as tfds
import tyro


AVAILABLE_TASK_SUITES = [
    "libero_10_no_noops",
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_spatial_no_noops",
]

DATASET_TO_BENCHMARK_MAP = {
    "libero_10_no_noops": "libero_10",
    "libero_goal_no_noops": "libero_goal",
    "libero_object_no_noops": "libero_object",
    "libero_spatial_no_noops": "libero_spatial",
}


def main(
    data_dir: str,
    task_suite_name: str,
    task_ids: List[int],
    *,
    push_to_hub: bool = False,
    repo_name_prefix: str = "shubhamg20/"
):
    """
    Generate a filtered Libero dataset for one or more task IDs.

    Args:
        data_dir: Path to the directory containing raw Libero datasets
        task_suite_name: Name of the task suite (e.g., 'libero_goal_no_noops')
        task_ids: One or more task IDs to include (e.g., --task_ids 1 2 3)
        repo_name_prefix: Prefix for the output dataset repository name
    """
    assert task_suite_name in AVAILABLE_TASK_SUITES, (
        f"Invalid task_suite_name: {task_suite_name}. Available options: {AVAILABLE_TASK_SUITES}"
    )
    assert len(task_ids) > 0, "At least one task_id must be specified"

    benchmark_name = DATASET_TO_BENCHMARK_MAP[task_suite_name]
    task_map = libero_suite_task_map.libero_task_map[benchmark_name]
    num_tasks_in_suite = len(task_map)

    for tid in task_ids:
        assert 0 <= tid < num_tasks_in_suite, (
            f"task_id {tid} out of range. Suite '{task_suite_name}' has {num_tasks_in_suite} tasks (0-{num_tasks_in_suite - 1})"
        )

    # Build set of language instructions to include
    target_instructions = {
        task_map[tid].replace("_", " ") for tid in task_ids
    }
    print(f"Collecting episodes for task_ids {task_ids}:")
    for instr in sorted(target_instructions):
        print(f"  - {instr}")

    # Create repository name from task IDs
    ids_str = "_".join(str(t) for t in sorted(task_ids))
    repo_name = f"{repo_name_prefix}{task_suite_name}_tasks-{ids_str}"

    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        print(f"Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)

    print(f"Creating dataset: {repo_name}")
    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    print(f"Loading raw dataset: {task_suite_name} from {data_dir}")
    builder = tfds.builder_from_directory(f"{data_dir}/{task_suite_name}/1.0.0")
    raw_dataset = builder.as_dataset(split="train")
    print(f"Total episodes in raw dataset: {len(raw_dataset)}")

    episode_count = 0
    for episode in raw_dataset:
        first_step = next(iter(episode["steps"]))
        task_instruction = first_step["language_instruction"].numpy().decode().strip()

        if task_instruction in target_instructions:
            print(f"Adding episode {episode_count}: {task_instruction}")
            for step in episode["steps"].as_numpy_iterator():
                dataset.add_frame(
                    {
                        "image": step["observation"]["image"],
                        "wrist_image": step["observation"]["wrist_image"],
                        "state": step["observation"]["state"],
                        "actions": step["action"],
                        "task": step["language_instruction"].decode(),
                    }
                )
            dataset.save_episode()
            episode_count += 1

    print(f"Total episodes added: {episode_count}")

    if episode_count == 0:
        print(f"WARNING: No episodes found for task_ids {task_ids} in {task_suite_name}")

    if push_to_hub:
        print(f"Pushing dataset to Hugging Face Hub: {repo_name}")
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds", task_suite_name] + [f"task_{t}" for t in task_ids],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
        print(f"Dataset successfully pushed to: https://huggingface.co/datasets/{repo_name}")
    else:
        print(f"Dataset saved locally to: {output_path}")


if __name__ == "__main__":
    tyro.cli(main)
