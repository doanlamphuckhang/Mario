"""Môi trường Super Mario Bros dùng cho RL.

Gồm hai phần:
  * `MarioEnv`      - một môi trường đã bọc đầy đủ (tiền xử lý ảnh, frame-skip,
                      frame-stack, reward shaping, thống kê episode).
  * `SubprocVecEnv` - chạy nhiều `MarioEnv` song song trên nhiều tiến trình.

File này cố ý KHÔNG import torch để các tiến trình con khởi động nhanh.
"""
from __future__ import annotations

import contextlib
import io
import multiprocessing as mp
import os
import warnings
from typing import Any, Sequence

import cv2
import numpy as np

from .score_reward import ScoreReward

cv2.setNumThreads(0)  # mỗi tiến trình con chỉ dùng 1 luồng cho OpenCV (tránh tranh chấp CPU)
warnings.filterwarnings("ignore")

# `gym` in cảnh báo "unmaintained" khi import; ta chỉ dùng nes_py/gym_super_mario_bros nên bỏ qua.
with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
    from gym_super_mario_bros import SuperMarioBrosEnv
    from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, RIGHT_ONLY, SIMPLE_MOVEMENT
    from nes_py.wrappers import JoypadSpace

ACTION_SETS = {
    "right": RIGHT_ONLY,        # 5 hành động
    "simple": SIMPLE_MOVEMENT,  # 7 hành động
    "complex": COMPLEX_MOVEMENT,  # 12 hành động
}


def parse_stages(spec: str | Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """'1-1,1-2' -> [(1, 1), (1, 2)]."""
    out = []
    if isinstance(spec, str):
        for token in spec.split(","):
            w, s = token.strip().split("-")
            out.append((int(w), int(s)))
    else:
        out = [tuple(s) for s in spec]
    if not out or len(set(out)) != len(out) or any(not (1 <= w <= 8 and 1 <= s <= 4) for w, s in out):
        raise ValueError('Stages must be unique world-stage pairs, worlds 1..8 and stages 1..4')
    return out


class MarioEnv:
    """Môi trường Mario cho agent học tăng cường.

    Quan sát : mảng uint8 shape (frame_stack, 84, 84) - ảnh xám của `frame_stack` bước gần nhất.
    Hành động: số nguyên trong [0, n_actions), ánh xạ sang tổ hợp phím của tay cầm NES.
    Phần thưởng mỗi bước của agent (đã scale):
        r = reward_scale * ( Σ_{khung trong frame-skip} [Δx + phạt thời gian + phạt chết]
                             + score_weight * Δscore_trước_cờ + coin_bonus * Δcoins
                             + flag_bonus * 1[chạm cờ] - death_penalty * 1[kết thúc không chạm cờ] )
    """

    def __init__(
        self,
        stages: str | Sequence[tuple[int, int]] = "1-1",
        actions: str = "simple",
        frame_skip: int = 4,
        frame_stack: int = 4,
        img_size: int = 84,
        reward_scale: float = 0.01,
        flag_bonus: float = 50.0,
        death_penalty: float = 0.0,
        score_weight: float = 0.0,
        coin_bonus: float = 0.0,
        noop_max: int = 0,
        seed: int | None = None,
    ) -> None:
        self.stages = parse_stages(stages)
        self.action_meanings = ACTION_SETS[actions]
        self.n_actions = len(self.action_meanings)
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.img_size = img_size
        self.reward_scale = reward_scale
        self.flag_bonus = flag_bonus
        self.death_penalty = death_penalty
        self.score_reward = ScoreReward(score_weight, coin_bonus)
        self.noop_max = noop_max
        self.rng = np.random.default_rng(seed)

        self._envs: dict[tuple[int, int], JoypadSpace] = {}
        self._env: JoypadSpace | None = None
        self._pool = [np.zeros((240, 256, 3), np.uint8) for _ in range(2)]  # 2 khung cuối để max-pool
        self._stack = np.zeros((frame_stack, img_size, img_size), np.uint8)
        self._last_rgb = np.zeros((240, 256, 3), np.uint8)
        self._reset_stats()
        self.stage: tuple[int, int] = self.stages[0]
        self.frame_hook = None  # hàm gọi với từng khung RGB gốc của emulator (dùng để quay video)

    # ------------------------------------------------------------------ nội bộ
    def _get_env(self, stage: tuple[int, int]) -> "JoypadSpace":
        if stage not in self._envs:
            raw = SuperMarioBrosEnv(rom_mode="vanilla", lost_levels=False, target=stage)
            self._envs[stage] = JoypadSpace(raw, self.action_meanings)
        return self._envs[stage]

    def _reset_stats(self) -> None:
        self._ep_return = 0.0
        self._ep_len = 0
        self._max_x = 0
        self._base_return = 0.0
        self._score = 0
        self._time_start: int | None = None
        self.score_reward.reset()

    def _preprocess(self, frame_rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

    # --------------------------------------------------------------------- API
    def reset(self, stage: tuple[int, int] | None = None) -> np.ndarray:
        self.stage = stage if stage is not None else self.stages[int(self.rng.integers(len(self.stages)))]
        self._env = self._get_env(self.stage)
        screen = self._env.reset()
        self._reset_stats()
        if self.noop_max > 0:  # điểm xuất phát ngẫu nhiên (dùng khi đánh giá độ bền)
            for _ in range(int(self.rng.integers(0, self.noop_max + 1))):
                screen, _, done, _ = self._env.step(0)
                if done:
                    screen = self._env.reset()
                    break
        # Read the installed gym-super-mario-bros API after startup NOOPs so
        # their score is not credited to the policy. No extra emulator step.
        raw = self._env.unwrapped
        self._score = int(raw._score)
        self.score_reward.reset(self._score, int(raw._coins))
        # Đồng hồ đầu ván khác nhau theo màn (400 hoặc 300); đọc sau NOOP để chỉ tính thời gian của policy.
        start_time = getattr(raw, '_time', None)
        self._time_start = None if start_time is None else int(start_time)
        np.copyto(self._last_rgb, screen)
        frame = self._preprocess(screen)
        self._stack[:] = frame  # lúc đầu lặp khung đầu tiên cho đủ frame_stack
        return self._stack.copy()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if self._env is None:
            raise RuntimeError("phải gọi reset() trước step()")
        total, score_bonus, done, info = 0.0, 0.0, False, {}
        for i in range(self.frame_skip):
            screen, r, done, info = self._env.step(int(action))
            total += r
            score_bonus += self.score_reward.step(info)
            if self.frame_hook is not None:
                self.frame_hook(screen)
            # chỉ cần giữ 2 khung cuối để lấy max (khử nhấp nháy sprite của NES)
            if i >= self.frame_skip - 2:
                np.copyto(self._pool[i - (self.frame_skip - 2)], screen)
            if done:
                np.copyto(self._pool[1], screen)
                np.copyto(self._pool[0], screen)
                break
        frame_rgb = np.maximum(self._pool[0], self._pool[1])
        np.copyto(self._last_rgb, frame_rgb)

        self._stack[:-1] = self._stack[1:]
        self._stack[-1] = self._preprocess(frame_rgb)

        flag = bool(info.get("flag_get", False))
        bonus = self.flag_bonus if flag else (-self.death_penalty if done else 0.0)
        reward = (total + bonus + score_bonus) * self.reward_scale
        self._base_return += total
        self._score = int(info.get('score', self._score))
        self._ep_return += reward
        self._ep_len += 1
        self._max_x = max(self._max_x, int(info.get("x_pos", 0)))

        if done:
            info = dict(info)
            info["episode"] = {
                "return": self._ep_return,
                "length": self._ep_len,
                "max_x": self._max_x,
                "flag": flag,
                "time_left": int(info.get("time", 0)),
                "stage": f"{self.stage[0]}-{self.stage[1]}",
                "score": self._score,
                "score_gain": self.score_reward.score_gain,
                "coins_collected": self.score_reward.coins_collected,
                "base_return": self._base_return,
                "score_reward_raw": self.score_reward.reward_total,
            }
            if self._time_start is not None:
                info["episode"]["time_start"] = self._time_start
        return self._stack.copy(), reward, bool(done), info

    def render_rgb(self) -> np.ndarray:
        """Khung hình RGB gốc (240x256x3) của bước gần nhất, dùng để quay video."""
        return self._last_rgb.copy()

    def close(self) -> None:
        for env in self._envs.values():
            env.close()
        self._envs.clear()


# ---------------------------------------------------------------------------
# Vec-env đa tiến trình
# ---------------------------------------------------------------------------
def _worker(remote, env_kwargs: dict) -> None:
    env = MarioEnv(**env_kwargs)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                obs, reward, done, info = env.step(data)
                if done:  # tự động reset khi hết episode (info vẫn giữ thống kê episode vừa xong)
                    obs = env.reset()
                remote.send((obs, reward, done, info.get("episode")))
            elif cmd == "reset":
                remote.send(env.reset())
            elif cmd == "close":
                break
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        env.close()
        remote.close()


class SubprocVecEnv:
    """Chạy N môi trường Mario trên N tiến trình; `step` gửi lệnh đồng thời rồi gom kết quả."""

    def __init__(self, env_kwargs_list: list[dict]) -> None:
        # Mỗi tiến trình con chỉ dùng 1 luồng BLAS/OpenMP để không tranh chấp với tiến trình huấn luyện.
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ.setdefault(var, "1")
        ctx = mp.get_context("spawn")
        self.num_envs = len(env_kwargs_list)
        self.remotes, self.procs = [], []
        for kwargs in env_kwargs_list:
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child, kwargs), daemon=True)
            p.start()
            child.close()
            self.remotes.append(parent)
            self.procs.append(p)

    def reset(self) -> np.ndarray:
        for r in self.remotes:
            r.send(("reset", None))
        return np.stack([r.recv() for r in self.remotes])

    def step(self, actions: np.ndarray):
        for r, a in zip(self.remotes, actions):
            r.send(("step", int(a)))
        results = [r.recv() for r in self.remotes]
        obs, rewards, dones, episodes = zip(*results)
        # episodes[i] là dict thống kê nếu môi trường i vừa kết thúc ván, ngược lại None
        return (np.stack(obs), np.asarray(rewards, np.float32), np.asarray(dones, np.bool_), list(episodes))

    def close(self) -> None:
        for r in self.remotes:
            try:
                r.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
