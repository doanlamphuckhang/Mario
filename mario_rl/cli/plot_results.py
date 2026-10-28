"""Vẽ đường cong học và các chỉ số chẩn đoán PPO từ runs/<tên>/metrics.csv.

Ví dụ:
    python plot_results.py --runs ppo_1-1 --out results/figs
    python plot_results.py --runs ppo_1-1 ppo_noshape --labels "Reward đầy đủ" "Chỉ Δx" --out results/figs --tag so_sanh
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 130})


def smooth(y: np.ndarray, k: int) -> np.ndarray:
    if len(y) < k:
        return y
    return pd.Series(y).rolling(k, min_periods=1, center=False).mean().to_numpy()


def learning_curves(runs: list[str], labels: list[str], root: Path, out: Path, tag: str) -> Path:
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for i, (r, lab) in enumerate(zip(runs, labels)):
        m = pd.read_csv(root / r / "metrics.csv")
        x = m["global_step"] / 1e6
        c = PALETTE[i % len(PALETTE)]
        ax[0].plot(x, m["ep_return"], color=c, label=lab)
        ax[1].plot(x, m["ep_max_x"], color=c, label=lab)
        ax[2].plot(x, m["flag_rate"], color=c, label=lab)
    ax[0].set(title="Return trung bình (100 ván gần nhất)", xlabel="Bước huấn luyện (triệu)", ylabel="return (đã scale)")
    ax[1].set(title="Quãng đường x lớn nhất đạt được", xlabel="Bước huấn luyện (triệu)", ylabel="x (pixel)")
    ax[2].set(title="Tỉ lệ chạm cờ (100 ván gần nhất)", xlabel="Bước huấn luyện (triệu)", ylabel="tỉ lệ", ylim=(-0.02, 1.02))
    for a in ax:
        a.legend(loc="best")
    fig.tight_layout()
    p = out / f"learning_curves{('_' + tag) if tag else ''}.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def diagnostics(runs: list[str], labels: list[str], root: Path, out: Path, tag: str) -> Path:
    keys = [("entropy", "Entropy của π"), ("approx_kl", "KL xấp xỉ (cũ->mới)"), ("clip_frac", "Tỉ lệ bị clip"),
            ("value_loss", "Value loss"), ("explained_var", "Explained variance của V"), ("fps", "Tốc độ (bước/giây)")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for ax, (k, title) in zip(axes.ravel(), keys):
        for i, (r, lab) in enumerate(zip(runs, labels)):
            m = pd.read_csv(root / r / "metrics.csv")
            ax.plot(m["global_step"] / 1e6, smooth(m[k].to_numpy(), 10), color=PALETTE[i % len(PALETTE)], label=lab)
        ax.set(title=title, xlabel="Bước huấn luyện (triệu)")
        if k == "value_loss":
            ax.set_yscale("log")
        ax.legend(loc="best")
    fig.tight_layout()
    p = out / f"diagnostics{('_' + tag) if tag else ''}.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def episode_scatter(run: str, root: Path, out: Path) -> Path:
    e = pd.read_csv(root / run / "episodes.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    sc = ax.scatter(e["global_step"] / 1e6, e["max_x"], c=e["flag"], cmap="coolwarm", s=4, alpha=0.5)
    ax.set(title=f"Từng ván chơi ({run}) - đỏ = chạm cờ", xlabel="Bước huấn luyện (triệu)", ylabel="x lớn nhất (pixel)")
    fig.tight_layout()
    p = out / f"episodes_{run}.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--root", default="runs")
    ap.add_argument("--out", default="results/figs")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    labels = a.labels or a.runs
    for p in (learning_curves(a.runs, labels, Path(a.root), out, a.tag),
              diagnostics(a.runs, labels, Path(a.root), out, a.tag),
              *([episode_scatter(a.runs[0], Path(a.root), out)] if not a.tag else [])):
        print("đã vẽ", p)


if __name__ == '__main__':
    main()
