"""Đánh giá một checkpoint: thống kê nhiều episode + quay video/GIF."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from .envs import ACTION_SETS, MarioEnv, SubprocVecEnv
from .models import ActorCritic

X_START = 40  # toạ độ x ban đầu của Mario trong RAM (đo thực tế khi reset)


def load_model(ckpt_path: str | Path, device: torch.device) -> tuple[ActorCritic, dict]:
    ck = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = ck["config"]
    model = ActorCritic(len(ACTION_SETS[cfg["actions"]])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg


def run_eval(
    model: ActorCritic,
    cfg: dict,
    stage: str = "1-1",
    episodes: int = 20,
    deterministic: bool = True,
    noop_max: int = 0,
    seed: int = 12345,
    device: torch.device | None = None,
    video_path: str | Path | None = None,
    gif_path: str | Path | None = None,
    gif_every: int = 4,
    max_steps: int = 6000,
) -> list[dict]:
    """Chạy `episodes` ván và trả về thống kê từng ván.

    video_path : ghi mp4 60 fps (ván đầu tiên); gif_path: ghi GIF (lấy mỗi `gif_every` khung).
    """
    device = device or next(model.parameters()).device
    env = MarioEnv(stages=stage, actions=cfg["actions"], frame_skip=cfg["frame_skip"],
                   reward_scale=cfg["reward_scale"], flag_bonus=cfg["flag_bonus"],
                   death_penalty=cfg.get("death_penalty", 0.0),
                   score_weight=cfg.get('score_weight', 0.0), coin_bonus=cfg.get('coin_bonus', 0.0),
                   noop_max=noop_max, seed=seed)
    torch.manual_seed(seed)

    if episodes <= 0 or max_steps <= 0 or gif_every <= 0:
        raise ValueError('episodes, max_steps and gif_every must be positive')
    writer, gif_frames, counter = None, [], [0]
    results = []
    try:
        if video_path is not None or gif_path is not None:
            if video_path is not None:
                Path(video_path).parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 60.0, (512, 480))
                if not writer.isOpened():
                    raise OSError(f'Không mở được file video để ghi: {video_path}')

            def hook(screen: np.ndarray) -> None:
                if writer is not None:
                    big = cv2.resize(screen, (512, 480), interpolation=cv2.INTER_NEAREST)
                    writer.write(cv2.cvtColor(big, cv2.COLOR_RGB2BGR))
                if gif_path is not None and counter[0] % gif_every == 0:
                    gif_frames.append(screen.copy())
                counter[0] += 1

            env.frame_hook = hook

        for ep in range(episodes):
            if ep == 1:           # chỉ quay ván đầu tiên
                env.frame_hook = None
            obs, info, done, steps = env.reset(), {}, False, 0
            while not done and steps < max_steps:
                with torch.no_grad():
                    a = model.act(torch.as_tensor(obs, device=device).unsqueeze(0), deterministic)
                obs, _, done, info = env.step(int(a.item()))
                steps += 1
            stats = info.get("episode") or {
                "return": env._ep_return, "length": steps, "max_x": env._max_x, "flag": False,
                "time_left": int(info.get("time", 0)), "stage": stage,
                "score": env._score, "score_gain": env.score_reward.score_gain,
                "coins_collected": env.score_reward.coins_collected,
                "base_return": env._base_return, "score_reward_raw": env.score_reward.reward_total,
                "truncated": True,
            }
            if "time_start" not in stats and env._time_start is not None:
                stats["time_start"] = env._time_start
            results.append(stats)
    finally:
        if writer is not None:
            writer.release()
        env.close()

    if gif_path is not None and gif_frames:
        from PIL import Image

        Path(gif_path).parent.mkdir(parents=True, exist_ok=True)
        imgs = [Image.fromarray(f).quantize(colors=64, method=Image.Quantize.FASTOCTREE) for f in gif_frames]
        imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], loop=0,
                     duration=int(1000 * gif_every / 60), optimize=True, disposal=1)
    return results


@torch.no_grad()
def run_eval_vec(
    model: ActorCritic,
    cfg: dict,
    stage: str = "1-1",
    episodes: int = 100,
    deterministic: bool = True,
    noop_max: int = 0,
    seed: int = 12345,
    device: torch.device | None = None,
    num_envs: int = 16,
    max_steps: int = 20000,
) -> list[dict]:
    """Đánh giá song song: mỗi môi trường chạy đúng hạn ngạch ván của mình (tránh thiên lệch
    về các ván ngắn kết thúc sớm), suy luận gộp một lần cho tất cả môi trường.
    max_steps là giới hạn mỗi ván; vượt giới hạn sẽ báo lỗi thay vì trả thiếu kết quả.
    """
    if episodes <= 0 or num_envs <= 0 or max_steps <= 0:
        raise ValueError('episodes, num_envs and max_steps must be positive')
    device = device or next(model.parameters()).device
    n = min(num_envs, episodes)
    quota = [episodes // n + (1 if i < episodes % n else 0) for i in range(n)]
    kwargs = [dict(stages=stage, actions=cfg["actions"], frame_skip=cfg["frame_skip"],
                   reward_scale=cfg["reward_scale"], flag_bonus=cfg["flag_bonus"],
                   death_penalty=cfg.get("death_penalty", 0.0), score_weight=cfg.get('score_weight', 0.0),
                   coin_bonus=cfg.get('coin_bonus', 0.0), noop_max=noop_max, seed=seed + i)
              for i in range(n)]
    venv = SubprocVecEnv(kwargs)
    torch.manual_seed(seed)
    results: list[list[dict]] = [[] for _ in range(n)]
    episode_steps = [0] * n
    try:
        obs = venv.reset()
        while any(len(results[i]) < quota[i] for i in range(n)):
            a = model.act(torch.as_tensor(obs, device=device), deterministic).cpu().numpy()
            obs, _, _, eps = venv.step(a)
            for i, e in enumerate(eps):
                if len(results[i]) >= quota[i]:
                    continue
                episode_steps[i] += 1
                if e is not None:
                    results[i].append(e)
                    episode_steps[i] = 0
                elif episode_steps[i] >= max_steps:
                    completed = sum(len(r) for r in results)
                    raise RuntimeError(f'Evaluation incomplete: {completed}/{episodes} episodes; '
                                       f'worker {i} exceeded {max_steps} steps in one episode')
    finally:
        venv.close()
    return [e for r in results for e in r]


def summarize(results: list[dict]) -> dict:
    """Tóm tắt: tỉ lệ chạm cờ, x trung bình, thời gian còn lại khi thắng..."""
    if not results:
        raise ValueError('Cannot summarize zero episodes')
    flags = np.array([r["flag"] for r in results], dtype=bool)
    # Đồng hồ đầu ván là 400 ở đa số màn nhưng 300 ở một số màn (ví dụ 1-3, 1-4).
    # Log cũ không có time_start nên giữ giả định 400 như trước.
    win_time = [r.get("time_start", 400) - r["time_left"] for r in results if r["flag"]]
    summary = {
        "episodes": len(results),
        "flag_rate": float(flags.mean()),
        "max_x_mean": float(np.mean([r["max_x"] for r in results])),
        "max_x_min": int(np.min([r["max_x"] for r in results])),
        "return_mean": float(np.mean([r["return"] for r in results])),
        "return_std": float(np.std([r["return"] for r in results])),
        "length_mean": float(np.mean([r["length"] for r in results])),
        "clear_clock_ticks_mean": float(np.mean(win_time)) if win_time else None,
    }
    # Missing legacy telemetry is unknown, never silently reported as zero.
    for key in ('score', 'score_gain', 'coins_collected', 'base_return', 'score_reward_raw'):
        values = [r[key] for r in results if key in r]
        summary[key + '_mean'] = float(np.mean(values)) if len(values) == len(results) else None
    summary['truncated_episodes'] = sum(bool(r.get('truncated', False)) for r in results)
    return summary
