"""Kiểm tra nhanh môi trường Mario chạy được và đo tốc độ giả lập thô."""
import sys
import time
import warnings

sys.stdout.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore")

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", apply_api_compatibility=True,
                                disable_env_checker=True)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
obs, info = env.reset()
print("obs:", obs.shape, obs.dtype, "| info:", info)
t, n = time.time(), 0
for _ in range(3000):
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    n += 1
    if term or trunc:
        env.reset()
print(f"Tốc độ: {n / (time.time() - t):.0f} khung hình/giây (1 luồng, chưa frame-skip)")
print("info cuối:", info)
