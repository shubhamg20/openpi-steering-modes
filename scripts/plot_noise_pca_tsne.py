"""
Load the first noise chunk and first action chunk from every episode's *_noise.pkl
across three tasks, then plot PCA, t-SNE, and L2 norm histograms coloured by task.
"""

# CUDA_VISIBLE_DEVICES=0 uv run scripts/droid_noise_histogram.py \
#     --policy_config_name pi0_droid_lora_finetune \
#     --pytorch_weight_path /gpfs/scrubbed/shubham/chkpts/pi0_droid/pytorch/\
#     --droid_data_dir /gpfs/scrubbed/hongmm/droid/1.0.1 \
#     --n_episodes 200 \
#     --num_steps 150 \
#     --out plots/droid_noise_histogram.png

import pickle
from pathlib import Path
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

DATA_ROOT = Path("/gpfs/scrubbed/shubham/data/data_paired_droid")

TASKS = {
    "paired_pan": DATA_ROOT / "paired_pan",
    "paired_purple": DATA_ROOT / "paired_purple",
    "paired_yellow": DATA_ROOT / "paired_yellow",
}

COLORS = {
    "paired_pan": "#e07b39",
    "paired_purple": "#9b59b6",
    "paired_yellow": "#f1c40f",
}

LABELS = {
    "paired_pan": "Pan",
    "paired_purple": "Purple",
    "paired_yellow": "Yellow",
}

ACTION_HORIZON = 10
TRUE_ACTION_DIM = 32  # 7 joint_velocity + 1 target_gripper_position


def _load_one(args):
    fpath, task_name = args
    with open(fpath, "rb") as f:
        data = pickle.load(f)
    # noise — only grab the first chunk
    noise_vec = data["noise_action"][0].flatten()
    # action chunk
    if "robot" in data and "timesteps" in data["robot"]:
        timesteps = data["robot"]["timesteps"]
    elif "timesteps" in data:
        timesteps = data["timesteps"]
    else:
        timesteps = []
    chunk = []
    for i in range(ACTION_HORIZON):
        t = min(i, len(timesteps) - 1)
        jv = np.array(timesteps[t]["action"]["joint_velocity"])
        gp = np.array([timesteps[t]["action"]["target_gripper_position"]])
        chunk.append(np.concatenate([jv, gp]))
    action_vec = np.stack(chunk).flatten()
    return noise_vec, action_vec, task_name


def load_chunks():
    """Load noise and action chunks in parallel across all pkl files."""
    jobs = []
    for task_name, task_dir in TASKS.items():
        # noise_files = sorted(task_dir.glob("*_pi0-droid-no-lang_noise.pkl"))
        noise_files = sorted(task_dir.glob("*_pi0-droid_noise.pkl"))
        print(f"[{task_name}] found {len(noise_files)} noise files")
        jobs.extend((fpath, task_name) for fpath in noise_files)

    if not jobs:
        raise RuntimeError("No noise files found. Check DATA_ROOT and TASKS paths.")
    workers = min(cpu_count() or 4, len(jobs))
    print(f"Loading {len(jobs)} files with {workers} workers …")
    with Pool(workers) as pool:
        results = list(tqdm(pool.imap(_load_one, jobs), total=len(jobs), desc="loading pkls"))

    noise_vecs, action_vecs, labels = zip(*results)
    return np.stack(noise_vecs), np.stack(action_vecs), list(labels)


def scatter(ax, coords, labels, title):
    for task_name in TASKS:
        mask = [i for i, l in enumerate(labels) if l == task_name]
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=COLORS[task_name],
            label=LABELS[task_name],
            alpha=0.8,
            s=60,
            edgecolors="white",
            linewidths=0.4,
        )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(framealpha=0.9, fontsize=10)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True, alpha=0.2)


def run_pca_tsne(X, perplexity):
    pca2 = PCA(n_components=2, random_state=42)
    X_pca = pca2.fit_transform(X)
    var = pca2.explained_variance_ratio_ * 100
    # Pre-reduce to at most 50 dims before t-SNE — dramatically faster on high-dim data
    pre_dims = min(50, X.shape[1], X.shape[0] - 1)
    X_pre = PCA(n_components=pre_dims, random_state=42).fit_transform(X) if pre_dims < X.shape[1] else X
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                max_iter=500, init="pca", learning_rate="auto")
    X_tsne = tsne.fit_transform(X_pre)
    return X_pca, var, X_tsne


def _prep_flat(X, labels, true_action_dim, horizon):
    full_dim = X.shape[1] // horizon
    slice_dim = true_action_dim if true_action_dim is not None else full_dim
    X_true = X.reshape(len(X), horizon, full_dim)[:, :, :slice_dim]
    flat_dim = X_true.shape[1] * X_true.shape[2]
    X_flat = X_true.reshape(len(X), -1)
    return X_flat, flat_dim


def plot_l2_hist(ax, X, labels, title, true_action_dim=TRUE_ACTION_DIM, horizon=ACTION_HORIZON):
    X_flat, flat_dim = _prep_flat(X, labels, true_action_dim, horizon)
    print(f"[{title}] L2 norm flat_dim={flat_dim}")
    l2 = np.linalg.norm(X_flat, axis=1)
    gauss_l2 = np.linalg.norm(np.random.randn(len(X), flat_dim), axis=1)
    for task_name in TASKS:
        mask = [i for i, l in enumerate(labels) if l == task_name]
        ax.hist(l2[mask], bins=15, color=COLORS[task_name], label=LABELS[task_name],
                alpha=0.55, edgecolor="white", linewidth=0.5)
        ax.axvline(l2[mask].mean(), color=COLORS[task_name], linestyle="--", linewidth=1.5)
    ax.hist(gauss_l2, bins=15, color="gray", label=f"Gaussian N(0,1) d={flat_dim}",
            alpha=0.35, edgecolor="white", linewidth=0.5)
    ax.axvline(gauss_l2.mean(), color="gray", linestyle="--", linewidth=1.5)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("L2 norm")
    ax.set_ylabel("Count")
    ax.legend(framealpha=0.9, fontsize=10)
    ax.grid(True, alpha=0.2)


def plot_l1_hist(ax, X, labels, title, true_action_dim=TRUE_ACTION_DIM, horizon=ACTION_HORIZON):
    X_flat, flat_dim = _prep_flat(X, labels, true_action_dim, horizon)
    print(f"[{title}] L1 norm flat_dim={flat_dim}")
    l1 = np.abs(X_flat).mean(axis=1)
    gauss_l1 = np.abs(np.random.randn(len(X), flat_dim)).mean(axis=1)
    for task_name in TASKS:
        mask = [i for i, l in enumerate(labels) if l == task_name]
        ax.hist(l1[mask], bins=15, color=COLORS[task_name], label=LABELS[task_name],
                alpha=0.55, edgecolor="white", linewidth=0.5)
        ax.axvline(l1[mask].mean(), color=COLORS[task_name], linestyle="--", linewidth=1.5)
    ax.hist(gauss_l1, bins=15, color="gray", label=f"Gaussian N(0,1) d={flat_dim}",
            alpha=0.35, edgecolor="white", linewidth=0.5)
    ax.axvline(gauss_l1.mean(), color="gray", linestyle="--", linewidth=1.5)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("L1 norm (mean |x|)")
    ax.set_ylabel("Count")
    ax.legend(framealpha=0.9, fontsize=10)
    ax.grid(True, alpha=0.2)


ACTION_DIM_NAMES = [f"joint_{i}" for i in range(7)] + ["gripper"]


def plot_action_stats(X_act, labels, out_path: Path = Path("plots/teleoperated_action_stats_pertask.png")):
    """Per-dimension mean and std of expert actions, coloured by task."""
    # X_act: (N, H*8) → reshape to (N, H, 8), then avg over H → (N, 8)
    N = X_act.shape[0]
    D = 8  # joint(7) + gripper(1)
    H = X_act.shape[1] // D
    acts = X_act.reshape(N, H, D).mean(axis=1)  # (N, 8)

    x = np.arange(D)
    dim_labels = ACTION_DIM_NAMES[:D]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for task_name in TASKS:
        mask = [i for i, l in enumerate(labels) if l == task_name]
        vals = acts[mask]  # (n_task, 8)
        m = vals.mean(axis=0)
        s = vals.std(axis=0)
        color = COLORS[task_name]
        label = LABELS[task_name]
        axes[0].bar(x, m, alpha=0.55, label=label, color=color)
        axes[1].bar(x, s, alpha=0.55, label=label, color=color)

    for ax, ylabel, title in [
        (axes[0], "Mean", f"Expert action mean per dim (H-averaged)\nN={N}"),
        (axes[1], "Std",  f"Expert action std per dim (H-averaged)\nN={N}"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels(dim_labels, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.legend(framealpha=0.9, fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        if ylabel == "Mean":
            ax.axhline(0, color="k", linewidth=0.8, linestyle="--")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved action stats → {out_path.resolve()}")


def main():
    perplexity = 29  # min(30, 89//3 - 1)

    # ── Load (parallel) ──────────────────────────────────────────────────────
    X_noise, X_act, labels = load_chunks()
    print(f"  noise matrix:  {X_noise.shape}")
    print(f"  action matrix: {X_act.shape}")

    plot_action_stats(X_act, labels)

    # ── Noise ────────────────────────────────────────────────────────────────
    print("Running PCA + t-SNE (noise) …")
    Xn_pca, varn, Xn_tsne = run_pca_tsne(X_noise, perplexity)

    # ── Actions ──────────────────────────────────────────────────────────────
    print("Running PCA + t-SNE (actions) …")
    Xa_pca, vara, Xa_tsne = run_pca_tsne(X_act, perplexity)

    # ── Plot: 2 rows × 4 cols ────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))

    # Row 0 — noise
    scatter(axes[0, 0], Xn_pca, labels,
            f"[Noise] PCA\n(PC1={varn[0]:.1f}%, PC2={varn[1]:.1f}%)")
    scatter(axes[0, 1], Xn_tsne, labels,
            f"[Noise] t-SNE  (perplexity={perplexity})")
    plot_l2_hist(axes[0, 2], X_noise, labels, "[Noise] L2 norm histogram", true_action_dim=None)
    plot_l1_hist(axes[0, 3], X_noise, labels, "[Noise] L1 norm histogram", true_action_dim=None)

    # Row 1 — actions
    scatter(axes[1, 0], Xa_pca, labels,
            f"[Action] PCA\n(PC1={vara[0]:.1f}%, PC2={vara[1]:.1f}%)")
    scatter(axes[1, 1], Xa_tsne, labels,
            f"[Action] t-SNE  (perplexity={perplexity})")
    plot_l2_hist(axes[1, 2], X_act, labels, "[Action] L2 norm histogram")
    plot_l1_hist(axes[1, 3], X_act, labels, "[Action] L1 norm histogram")

    fig.suptitle(
        "First chunk embeddings — noise (top) vs raw actions (bottom) — coloured by task",
        fontsize=14,
    )
    fig.tight_layout()

    out = Path("plots/noise_pca_tsne.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out.resolve()}")


if __name__ == "__main__":
    main()
