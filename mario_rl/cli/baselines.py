"""Các baseline không học: agent ngẫu nhiên và các chính sách cố định.

Dùng để có mốc so sánh cho agent PPO (mục "Kết quả" trong báo cáo).

    python baselines.py --stage 1-1 --episodes 50 --json results/baselines_1-1.json
"""
import argparse
import json
import sys

import numpy as np

from mario_rl.envs import ACTION_SETS, MarioEnv
from mario_rl.evaluation import summarize


def rollout(policy, stage: str, episodes: int, seed: int, actions: str = "simple") -> list[dict]:
    env = MarioEnv(stages=stage, actions=actions, seed=seed)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(episodes):
        env.reset()
        done, info, steps = False, {}, 0
        while not done and steps < 6000:
            _, _, done, info = env.step(policy(rng, env.n_actions, steps))
            steps += 1
        out.append(info["episode"])
    env.close()
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--stage", default="1-1")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    names = [m for m in ACTION_SETS["simple"]]
    idx = {tuple(m): i for i, m in enumerate(names)}
    right = idx[("right",)]
    right_a = idx[("right", "A")]
    right_ab = idx[("right", "A", "B")]
    policies = {
        "Ngẫu nhiên (đều)": lambda rng, n, t: int(rng.integers(n)),
        "Luôn đi phải": lambda rng, n, t: right,
        "Luôn phải + nhảy (A)": lambda rng, n, t: right_a,
        "Luôn phải + chạy + nhảy (A+B)": lambda rng, n, t: right_ab,
    }
    report = {}
    for name, pol in policies.items():
        # chính sách cố định + môi trường tất định => mọi ván giống hệt nhau, chỉ cần chạy 1 ván
        res = rollout(pol, a.stage, a.episodes if name.startswith("Ngẫu nhiên") else 1, a.seed)
        report[name] = summarize(res)
        s = report[name]
        print(f"{name:32s} x_max_tb={s['max_x_mean']:7.1f}  chạm cờ={s['flag_rate']:.2f}  return_tb={s['return_mean']:.2f}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"stage": a.stage, "episodes": a.episodes, "baselines": report}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
