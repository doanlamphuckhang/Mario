"""Regression specifications for score shaping, transfer and evaluation.

Prepared for later execution; no real emulator is used by these tests.
"""
import dataclasses
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from mario_rl.score_reward import ScoreReward
from mario_rl.envs import ACTION_SETS, MarioEnv, parse_stages
from mario_rl.evaluation import summarize
from mario_rl.transfer import initialize_policy
from mario_rl.cli.train import parse_config
from mario_rl.ppo import PPOConfig, validate_config


class MiniPolicy(torch.nn.Module):
    def __init__(self, actions):
        super().__init__()
        self.body = torch.nn.Linear(2, 3)
        self.actor = torch.nn.Linear(3, len(ACTION_SETS[actions]))
        self.critic = torch.nn.Linear(3, 1)


class FakeGame:
    """A coin, then a non-coin score event, then the flag in one action repeat."""
    def __init__(self):
        self.unwrapped = self
        self._score, self._coins, self.index, self._time = 0, 0, 0, 300

    def reset(self):
        self._score, self._coins, self.index, self._time = 0, 0, 0, 300
        return np.zeros((240, 256, 3), dtype=np.uint8)

    def step(self, action):
        self.index += 1
        self._score = [200, 300, 5300][self.index - 1]
        self._coins = 1
        info = dict(score=self._score, coins=1, x_pos=40 + self.index,
                    flag_get=self.index == 3, time=399)
        return np.full((240, 256, 3), self.index, dtype=np.uint8), 1.0, self.index == 3, info


class UpgradeTests(unittest.TestCase):
    def test_score_is_incremental_and_coin_rollover(self):
        reward = ScoreReward(.025, 5)
        reward.reset(100, 99)
        self.assertEqual(reward.step(dict(score=300, coins=0)), 10)
        self.assertEqual(reward.step(dict(score=300, coins=0)), 0)
        self.assertEqual(reward.coins_collected, 1)
        self.assertEqual(reward.score_gain, 200)

    def test_flag_score_not_counted_and_reset_clears_history(self):
        reward = ScoreReward(.1)
        self.assertEqual(reward.step(dict(score=100)), 10)
        self.assertEqual(reward.step(dict(score=5100, flag_get=True)), 0)
        self.assertEqual(reward.step(dict(score=10000, flag_get=False)), 0)
        self.assertEqual(reward.score_gain, 100)
        reward.reset(1000, 3)
        self.assertEqual(reward.step(dict(score=1100, coins=3)), 10)
        self.assertEqual(reward.score_gain, 100)

    def test_defaults_preserve_old_reward(self):
        reward = ScoreReward()
        self.assertEqual(reward.step(dict(score=1000, coins=5)), 0)
        self.assertEqual(reward.score_gain, 1000)

    def test_frame_skip_counts_events_once_before_flag(self):
        env = MarioEnv(frame_skip=4, score_weight=.1, reward_scale=.01, flag_bonus=50)
        fake = FakeGame()
        with patch.object(env, '_get_env', return_value=fake):
            env.reset()
            obs, reward, done, info = env.step(1)
            self.assertTrue(done)
            self.assertEqual(obs.shape, (4, 84, 84))
            self.assertAlmostEqual(reward, .83)  # (3 base + 30 score + 50 flag) * .01
            self.assertEqual(info['episode']['score_gain'], 300)
            self.assertEqual(info['episode']['score'], 5300)
            self.assertEqual(info['episode']['coins_collected'], 1)
            self.assertEqual(info['episode']['time_start'], 300)
            env.reset()
            self.assertEqual(env.score_reward.score_gain, 0)

    def test_warm_start_copies_actions_but_not_critic(self):
        old, new = MiniPolicy('simple'), MiniPolicy('complex')
        original_critic = new.critic.weight.detach().clone()
        ck = dict(model=old.state_dict(), config=dict(actions='simple', frame_skip=4), global_step=123)
        report = initialize_policy(new, ck, 'complex', 4)
        self.assertEqual(len(report['copied_actions']), 7)
        self.assertTrue(torch.equal(new.body.weight, old.body.weight))
        self.assertTrue(torch.equal(new.critic.weight, original_critic))
        for i, buttons in enumerate(ACTION_SETS['complex']):
            if buttons in ACTION_SETS['simple']:
                j = ACTION_SETS['simple'].index(buttons)
                self.assertTrue(torch.equal(new.actor.weight[i], old.actor.weight[j]))
            else:
                self.assertEqual(new.actor.bias[i].item(), -4)
        with self.assertRaises(ValueError):
            initialize_policy(new, ck, 'complex', 2)

    def test_score_telemetry_missing_is_not_zero(self):
        ep = dict(flag=False, max_x=100, length=10, time_left=390, **{'return': 1})
        self.assertIsNone(summarize([ep])['score_gain_mean'])
        ep.update(score=500, score_gain=300, coins_collected=1, base_return=100, score_reward_raw=30)
        self.assertEqual(summarize([ep])['score_gain_mean'], 300)
        with self.assertRaises(ValueError):
            summarize([])

    def test_clear_time_uses_stage_clock(self):
        legacy = dict(flag=True, max_x=3000, length=10, time_left=350, **{'return': 1})
        self.assertEqual(summarize([legacy])['clear_clock_ticks_mean'], 50)
        castle = dict(legacy, time_left=250, time_start=300)
        self.assertEqual(summarize([castle])['clear_clock_ticks_mean'], 50)

    def test_step_before_reset_is_explicit(self):
        with self.assertRaises(RuntimeError):
            MarioEnv().step(0)

    def test_profile_override_and_validation(self):
        cfg = parse_config(['--config', str(ROOT / 'configs/score_1-1.json'), '--seed', '3'])
        self.assertEqual(cfg.seed, 3)
        self.assertEqual(cfg.score_weight, .1)
        validate_config(cfg)
        validate_config(parse_config(['--config', str(ROOT / 'configs/multistage.json')]))
        with self.assertRaises(ValueError):
            validate_config(dataclasses.replace(cfg, resume=True))
        for stages in ('', '1-1,1-1', '9-1', '1-5', []):
            with self.assertRaises(ValueError):
                parse_stages(stages)

    def test_suite_covers_each_case_without_overwriting(self):
        from mario_rl.cli import evaluate_suite as suite
        with tempfile.TemporaryDirectory(dir=ROOT / 'results') as tmp:
            root = Path(tmp)
            ckpt = root / 'dummy.pt'
            ckpt.write_bytes(b'synthetic-not-a-model')
            args = suite.build_parser().parse_args([
                '--ckpts', str(ckpt), '--out', str(root / 'eval'),
                '--stages', '1-1,2-2', '--seeds', '100,200', '--episodes', '1'])
            ep = dict(flag=True, max_x=100, length=10, time_left=390, score=500,
                      score_gain=300, coins_collected=1, **{'return': 2})
            with patch.object(suite, 'load_model', return_value=(object(), {'stages': '1-1'})), \
                    patch.object(suite, 'run_eval_vec', return_value=[ep]) as evaluate:
                suite.run_suite(args)
                self.assertEqual(evaluate.call_count, 8)
                details = [json.loads(line) for line in (root / 'eval/episodes.jsonl').read_text().splitlines()]
                self.assertEqual(len(details), 8)
                self.assertEqual(sum(r['trained_stage'] for r in details), 4)
                with self.assertRaises(FileExistsError):
                    suite.run_suite(args)


if __name__ == '__main__':
    unittest.main(verbosity=2)
