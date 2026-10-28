"""Kiểm thử các phép tính tay trong báo cáo: chạy `python tests/test_core.py` hoặc `pytest tests`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from mario_rl.ppo import clipped_policy_loss, compute_gae


def test_gae_hand_example():
    """Ví dụ minh hoạ trong báo cáo: 3 bước, bước cuối Mario chết (episode kết thúc)."""
    gamma, lam = 0.99, 0.95
    rewards = torch.tensor([[1.0], [1.0], [-1.5]])
    values = torch.tensor([[10.0], [10.2], [9.0]])
    dones = torch.zeros(3, 1)               # không có ván mới bắt đầu trong đoạn này
    next_value, next_done = torch.tensor([7.0]), torch.tensor([1.0])  # V(next) bị che vì ván đã kết thúc
    adv, ret = compute_gae(rewards, values, dones, next_value, next_done, gamma, lam)
    # tính tay
    d2 = -1.5 - 9.0                          # = -10.5
    d1 = 1.0 + 0.99 * 9.0 - 10.2             # = -0.29
    d0 = 1.0 + 0.99 * 10.2 - 10.0            # = 1.098
    a2 = d2
    a1 = d1 + gamma * lam * a2
    a0 = d0 + gamma * lam * a1
    assert abs(adv[2].item() - a2) < 1e-5 and abs(adv[1].item() - a1) < 1e-5 and abs(adv[0].item() - a0) < 1e-5
    assert abs(a2 + 10.5) < 1e-9 and abs(a1 - (-10.16525)) < 1e-9 and abs(a0 - (-8.4624)) < 1e-3
    assert torch.allclose(ret, adv + values)


def test_gae_lambda_limits():
    """λ=0 cho TD(0) một bước; λ=1 cho return Monte-Carlo trừ baseline."""
    gamma = 0.9
    r = torch.tensor([[1.0], [2.0], [3.0]])
    v = torch.tensor([[0.5], [0.4], [0.3]])
    z, nv, nd = torch.zeros(3, 1), torch.tensor([0.2]), torch.tensor([0.0])
    a0, _ = compute_gae(r, v, z, nv, nd, gamma, 0.0)
    td0 = r[:, 0] + gamma * torch.cat([v[1:, 0], nv]) - v[:, 0]
    assert torch.allclose(a0[:, 0], td0)
    a1, _ = compute_gae(r, v, z, nv, nd, gamma, 1.0)
    mc = torch.tensor([1 + gamma * 2 + gamma**2 * 3 + gamma**3 * 0.2, 2 + gamma * 3 + gamma**2 * 0.2, 3 + gamma * 0.2])
    assert torch.allclose(a1[:, 0], mc - v[:, 0], atol=1e-5)


def test_gae_episode_boundary_blocks_bootstrap():
    """Nếu bước t+1 là khung đầu ván mới (done[t+1]=1) thì δ_t không được dùng V(s_{t+1})."""
    r = torch.tensor([[1.0], [5.0]])
    v = torch.tensor([[2.0], [100.0]])
    dones = torch.tensor([[0.0], [1.0]])
    adv, _ = compute_gae(r, v, dones, torch.tensor([0.0]), torch.tensor([0.0]), 0.99, 0.95)
    assert abs(adv[0].item() - (1.0 - 2.0)) < 1e-6   # δ_0 = r_0 - V_0, không bootstrap qua ranh giới ván


def test_clip_objective_hand_examples():
    eps = 0.2
    cases = [  # (ratio, advantage, L^CLIP kỳ vọng, có gradient theo ratio?)
        (1.5, 2.0, 2.4, False), (0.9, 2.0, 1.8, True),
        (0.7, -2.0, -1.6, False), (1.3, -2.0, -2.6, True),
    ]
    for ratio, adv, expected, has_grad in cases:
        r = torch.tensor([ratio], requires_grad=True)
        loss = clipped_policy_loss(r, torch.tensor([adv]), eps)   # = -L^CLIP
        assert abs(-loss.item() - expected) < 1e-6, (ratio, adv, loss.item())
        loss.backward()
        assert (abs(r.grad.item()) > 0) == has_grad, (ratio, adv, r.grad.item())


def test_reward_always_right_matches_hand_calculation():
    """Chính sách 'luôn đi phải' chết ở Goomba đầu tiên: return thô = (254 - 8) - 15 = 231."""
    from mario_rl.envs import MarioEnv

    env = MarioEnv(stages="1-1", seed=0, reward_scale=1.0)   # thang 1.0 để so với return thô
    env.reset()
    total, done, steps, info = 0.0, False, 0, {}
    while not done:
        _, r, done, info = env.step(1)       # index 1 = ('right',) trong SIMPLE_MOVEMENT
        total += r
        steps += 1
    ep = info["episode"]
    assert abs(total - 231.0) < 1e-6 and ep["max_x"] == 296 and ep["time_left"] == 392 and steps == 40
    env.close()
    # với thang c = 0,1 (Run 1 cũ) return là 23,10; với c = 0,01 mặc định là 2,31
    assert abs(total * 0.1 - 23.10) < 1e-6 and abs(total * 0.01 - 2.31) < 1e-9


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("OK  ", name)
