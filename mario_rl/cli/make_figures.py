"""Tạo các hình minh hoạ cho báo cáo (không phụ thuộc kết quả huấn luyện).

    python make_figures.py            ->  results/figs/fig_*.png
"""
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path("results/figs")
plt.rcParams.update({"font.size": 10, "figure.dpi": 140})


# --------------------------------------------------------------------------- 1. tiền xử lý ảnh
def fig_preprocess() -> None:
    from mario_rl.envs import MarioEnv

    env = MarioEnv(stages="1-1", seed=0)
    env.reset()
    for _ in range(28):  # right + A + B; dừng sớm trước khi Mario chạm Goomba đầu tiên
        obs, _, done, _ = env.step(4)
        assert not done
    raw, stack = env.render_rgb(), obs
    gray = cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY)

    fig = plt.figure(figsize=(12, 3.6))
    gs = fig.add_gridspec(1, 7, width_ratios=[3.2, 0.25, 2.2, 0.25, 1.4, 1.4, 1.4])
    ax = fig.add_subplot(gs[0, 0]); ax.imshow(raw); ax.set_title("1. Khung RGB gốc\n240×256×3 = 184.320 giá trị"); ax.axis("off")
    ax = fig.add_subplot(gs[0, 2]); ax.imshow(stack[-1], cmap="gray", vmin=0, vmax=255)
    ax.set_title("2. Xám + max 2 khung + 84×84\n= 7.056 giá trị"); ax.axis("off")
    for i, k in enumerate((0, 1, 3)):
        ax = fig.add_subplot(gs[0, 4 + i]); ax.imshow(stack[k], cmap="gray", vmin=0, vmax=255); ax.axis("off")
        ax.set_title(["khung t-3", "khung t-2", "khung t"][i], fontsize=9)
    fig.text(0.68, 0.97, "3. Xếp chồng 4 khung -> (4, 84, 84) = 28.224 giá trị  (để thấy vận tốc/hướng)",
             ha="center", va="top", fontsize=10)
    for src in fig.axes[:2]:  # mũi tên 1.->2. và 2.->3. vẽ theo toạ độ trục để không lệch
        src.annotate("", xy=(1.2, 0.5), xytext=(1.03, 0.5), xycoords="axes fraction",
                     arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#333"), annotation_clip=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "fig_preprocess.png", bbox_inches="tight")
    plt.close(fig)
    env.close()


# --------------------------------------------------------------------------- 2. hàm mục tiêu clip của PPO
def fig_ppo_clip(eps: float = 0.2) -> None:
    r = np.linspace(0, 2, 400)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
    for ax, A, title in zip(axes, (+1.0, -1.0), ("Lợi thế A > 0 (hành động tốt hơn kỳ vọng)", "Lợi thế A < 0 (hành động tệ hơn kỳ vọng)")):
        unclipped = r * A
        clipped = np.clip(r, 1 - eps, 1 + eps) * A
        ax.plot(r, unclipped, "--", color="gray", label=r"$r\,A$ (không clip)")
        ax.plot(r, np.minimum(unclipped, clipped), color="#d62728", lw=2.2, label=r"$L^{CLIP}=\min(rA,\ \mathrm{clip}(r)A)$")
        ax.axvline(1, color="k", lw=0.6)
        ax.axvspan(1 - eps, 1 + eps, color="#1f77b4", alpha=0.10, label=f"vùng tin cậy [1-ε, 1+ε], ε={eps}")
        ax.set(title=title, xlabel=r"tỉ số xác suất $r=\pi_\theta(a|s)/\pi_{\theta_{old}}(a|s)$", ylabel=r"$L^{CLIP}$")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_ppo_clip.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- 3. sơ đồ pipeline huấn luyện
def _box(ax, xy, w, h, text, color):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec="#333", lw=1.2))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=9)


def _arrow(ax, p, q, text="", rad=0.0, dy=0.12):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16, lw=1.4, color="#333", connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + dy, text, ha="center", fontsize=8, color="#444")


def fig_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4.6); ax.axis("off")
    _box(ax, (0.2, 2.9), 2.6, 1.3, "16 tiến trình\n(mỗi tiến trình = 1 emulator NES\n+ MarioEnv: frame-skip 4,\nxám 84×84, stack 4, reward)", "#dcebf7")
    _box(ax, (4.0, 2.9), 2.6, 1.3, "Tiến trình chính (GPU)\nmạng Actor-Critic\nπ(a|s) , V(s)", "#fde9d3")
    _box(ax, (7.7, 2.9), 2.2, 1.3, "Rollout buffer\n128 bước × 16 env\n= 2048 mẫu", "#e4f2e1")
    _box(ax, (10.1, 2.9), 1.7, 1.3, "GAE(γ,λ)\nlợi thế Â_t,\nreturn mục tiêu", "#f3e2f3")
    _box(ax, (7.7, 0.4), 4.1, 1.3, "Tối ưu hoá PPO: 4 lượt × 8 minibatch (256 mẫu)\nL = L_CLIP - c₁·L_V + c₂·Entropy\nAdam, lr 2.5e-4 giảm tuyến tính", "#fff3c4")
    _box(ax, (0.2, 0.4), 6.4, 1.3, "Ghi log (CSV) · checkpoint · đánh giá · quay video", "#eeeeee")
    _arrow(ax, (2.8, 3.85), (4.0, 3.85), "s, r, done", dy=0.1)
    _arrow(ax, (4.0, 3.25), (2.8, 3.25), "hành động a", dy=-0.28)
    _arrow(ax, (6.6, 3.55), (7.7, 3.55), "(s,a,r,log π,V)")
    _arrow(ax, (9.9, 3.55), (10.1, 3.55))
    _arrow(ax, (10.95, 2.9), (10.0, 1.7), "")
    _arrow(ax, (7.7, 1.05), (5.4, 2.9), "cập nhật θ", rad=-0.25, dy=0.0)
    _arrow(ax, (7.7, 0.9), (6.6, 0.9))
    fig.tight_layout()
    fig.savefig(OUT / "fig_pipeline.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- 4. kiến trúc mạng
def fig_network() -> None:
    import torch

    from mario_rl.models import ActorCritic

    net = ActorCritic(7)
    n_params = sum(p.numel() for p in net.parameters())
    with torch.no_grad():
        shapes, h = [], torch.zeros(1, 4, 84, 84)
        for layer in net.body:
            h = layer(h)
            if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
                shapes.append(tuple(h.shape[1:]))
    print("tham số:", n_params, "| kích thước tầng:", shapes)

    fig, ax = plt.subplots(figsize=(12, 3.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 3.4); ax.axis("off")
    items = [("Đầu vào\n4×84×84", "#dcebf7", 0.1), (f"Conv 8×8, s4\n32×{shapes[0][1]}×{shapes[0][2]}", "#e4f2e1", 2.0),
             (f"Conv 4×4, s2\n64×{shapes[1][1]}×{shapes[1][2]}", "#e4f2e1", 3.9), (f"Conv 3×3, s1\n64×{shapes[2][1]}×{shapes[2][2]}", "#e4f2e1", 5.8),
             ("Flatten\n3136", "#eeeeee", 7.5), ("FC + ReLU\n512", "#fde9d3", 9.0)]
    for text, c, x in items:
        _box(ax, (x, 1.2), 1.5, 1.1, text, c)
    for x0, x1 in [(1.6, 2.0), (3.5, 3.9), (5.4, 5.8), (7.3, 7.5), (9.0, 9.0)]:
        if x1 > x0:
            _arrow(ax, (x0, 1.75), (x1, 1.75))
    _box(ax, (10.8, 2.2), 1.1, 0.9, "Actor\n7 logits -> π", "#fbd5d5")
    _box(ax, (10.8, 0.3), 1.1, 0.9, "Critic\n1 giá trị V", "#d5e4fb")
    _arrow(ax, (10.5, 1.95), (10.8, 2.6)); _arrow(ax, (10.5, 1.55), (10.8, 0.75))
    ax.text(6, 3.1, f"Tổng cộng {n_params:,} tham số".replace(",", "."), ha="center", fontsize=11, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_network.png", bbox_inches="tight")
    plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    fig_ppo_clip(); print("fig_ppo_clip")
    fig_pipeline(); print("fig_pipeline")
    fig_network(); print("fig_network")
    fig_preprocess(); print("fig_preprocess")


if __name__ == '__main__':
    main()
