"""Tìm một chuỗi hành động về đích để viewer 3D có sẵn bản chạy mẫu khi chưa có checkpoint.

KHÔNG phải huấn luyện: đây là tìm kiếm ngẫu nhiên có quay lui trên emulator (tất định), dùng
savestate của nes-py để thử nhiều đoạn phím rồi giữ đoạn sống sót đi xa nhất. Chạy vài chục giây
trên 1 nhân CPU.

    python scripts/make_demo_replay.py --stage 1-1        ->  viewer3d/demos/1-1.json
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
    from gym_super_mario_bros import SuperMarioBrosEnv
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    from nes_py.wrappers import JoypadSpace

FRAME_SKIP = 4
HORIZON, COMMIT, TRIES = 30, 15, 16     # thử 30 bước, giữ 15 bước đầu của đoạn tốt nhất
WEIGHTS = np.array([0.04, 0.05, 0.05, 0.35, 0.45, 0.03, 0.03])   # ưu tiên chạy phải (+ nhảy)


class Sim:
    """Emulator + savestate một ô nhớ; bước giống hệt `MarioEnv.step` (lặp hành động FRAME_SKIP khung)."""

    def __init__(self, stage: tuple[int, int]) -> None:
        self.stage = stage
        self._new()

    def _new(self) -> None:
        self.raw = SuperMarioBrosEnv(rom_mode="vanilla", lost_levels=False, target=self.stage)
        self.env = JoypadSpace(self.raw, SIMPLE_MOVEMENT)
        self.env.reset()

    def save(self) -> None:
        self.raw._backup()

    def load(self) -> None:
        self.raw._restore()
        self.raw.done = False
        self.raw._x_position_last, self.raw._time_last = self.raw._x_position, self.raw._time

    def run(self, actions) -> tuple[bool, bool, int, int]:
        """Trả về (còn sống, chạm cờ, x cuối, số bước đã chạy)."""
        info = {"x_pos": self.raw._x_position}
        for n, a in enumerate(actions, 1):
            for _ in range(FRAME_SKIP):
                _, _, done, info = self.env.step(int(a))
                if done:
                    return False, bool(info["flag_get"]), int(info["x_pos"]), n
        return True, False, int(info["x_pos"]), len(actions)

    def rebuild(self, prefix: list[int]) -> None:
        """Về lại trạng thái sau `prefix` bằng cách chạy lại từ đầu (emulator tất định)."""
        self.raw.close()
        self._new()
        alive, _, _, _ = self.run(prefix)
        assert alive or not prefix
        self.save()


def random_segment(rng: np.random.Generator, n: int) -> list[int]:
    out: list[int] = []
    while len(out) < n:
        out += [int(rng.choice(len(WEIGHTS), p=WEIGHTS))] * int(rng.integers(1, 11))
    return out[:n]


def search(stage: tuple[int, int], seed: int, budget_s: float) -> tuple[list[int], bool, int]:
    rng = np.random.default_rng(seed)
    sim = Sim(stage)
    sim.save()
    prefix: list[int] = []
    x0, fails, t0 = int(sim.raw._x_position), 0, time.time()
    while time.time() - t0 < budget_s:
        best: tuple[int, list[int]] | None = None
        for _ in range(TRIES * (1 + fails)):
            sim.load()
            seg = random_segment(rng, HORIZON)
            alive, flag, x, used = sim.run(seg)
            if flag:
                return prefix + seg[:used], True, x
            if alive and (best is None or x > best[0]):
                best = (x, seg)
                if x - x0 > 2.0 * FRAME_SKIP * HORIZON:      # gần tốc độ chạy tối đa: nhận luôn
                    break
        if best is None or best[0] < x0 + 8:                 # chết hết hoặc kẹt: lùi lại và thử khác đi
            fails += 1
            prefix = prefix[:max(0, len(prefix) - 20 * fails)]
            sim.rebuild(prefix)
            x0 = int(sim.raw._x_position)
            print(f"  lùi về bước {len(prefix)} (x={x0})", flush=True)
            continue
        fails = 0
        sim.load()
        _, _, x0, _ = sim.run(best[1][:COMMIT])
        sim.save()
        prefix += best[1][:COMMIT]
        print(f"bước {len(prefix):4d}  x={x0}", flush=True)
    return prefix, False, x0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--stage", default="1-1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=float, default=180.0, help="giới hạn thời gian tìm kiếm (giây)")
    a = ap.parse_args()

    w, s = (int(v) for v in a.stage.split("-"))
    steps, flag, x = search((w, s), a.seed, a.budget)
    print(f"{'VỀ ĐÍCH' if flag else 'chưa về đích'}: {len(steps)} bước, x={x}")
    out = ROOT / "viewer3d" / "demos" / f"{a.stage}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"stage": a.stage, "actions": "simple", "frame_skip": FRAME_SKIP,
                               "flag": flag, "max_x": x, "steps": steps}), encoding="utf-8")
    print("đã ghi", out)
