"""CPU-only regression checks; synthetic environments and a two-parameter model."""
import csv
import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch
from mario_rl import ppo, evaluation
from mario_rl.run_io import run_lock, reconcile_logs, restore_best_score, atomic_save

ROOT = Path(__file__).resolve().parents[1]


def workspace_temp():
    return tempfile.TemporaryDirectory(dir=ROOT / 'results')


class TinyModel(torch.nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.logit = torch.nn.Parameter(torch.zeros(()))
        self.value = torch.nn.Parameter(torch.zeros(()))
    def get_action_and_value(self, obs, action=None):
        n = len(obs)
        action = torch.zeros(n, dtype=torch.long) if action is None else action
        return action, (self.logit + 0).expand(n), (self.logit * 0 + 1).expand(n), (self.value + 0).expand(n)
    def get_value(self, obs):
        return (self.value + 0).expand(len(obs))
    def act(self, obs, deterministic):
        return torch.zeros(len(obs), dtype=torch.long)


class FakeVec:
    period = 1
    reward = 1.0
    instances = []
    def __init__(self, kwargs):
        self.n, self.tick, self.closed = len(kwargs), 0, False
        self.instances.append(self)
    def reset(self):
        return np.zeros((self.n, 4, 84, 84), np.uint8)
    def step(self, actions):
        self.tick += 1
        done = self.tick % self.period == 0
        ep = dict(stage='1-1', length=self.period, max_x=40, flag=False, time_left=0,
                  **{'return': self.reward})
        return self.reset(), np.full(self.n, self.reward, np.float32), np.full(self.n, done), [ep.copy() if done else None for _ in range(self.n)]
    def close(self):
        self.closed = True


class RegressionTests(unittest.TestCase):
    def test_invalid_batch_rejected_before_output(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'results') as tmp:
            for changes in (dict(num_envs=1, rollout_len=9, minibatches=4),
                            dict(minibatches=99999), dict(snapshot_every=0), dict(total_steps=1)):
                cfg = ppo.PPOConfig(out_dir=tmp, **changes)
                with self.assertRaises(ValueError), patch.object(ppo, 'SubprocVecEnv') as env:
                    ppo.train(cfg)
                env.assert_not_called()
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_missing_resume_preserves_files(self):
        with workspace_temp() as tmp:
            run = Path(tmp) / 'run'
            run.mkdir()
            for name in ('config.json', 'metrics.csv', 'episodes.csv'):
                (run / name).write_text('original', encoding='utf-8')
            with self.assertRaises(FileNotFoundError):
                ppo.train(ppo.PPOConfig(out_dir=tmp, run_name='run', resume=True))
            for name in ('config.json', 'metrics.csv', 'episodes.csv'):
                self.assertEqual((run / name).read_text(), 'original')
            with self.assertRaises(FileExistsError):
                ppo.train(ppo.PPOConfig(out_dir=tmp, run_name='run'))

    def test_lock_rejects_second_process_and_releases(self):
        with workspace_temp() as tmp:
            code = ('from pathlib import Path; from mario_rl.run_io import run_lock; '
                    'import sys;\nwith run_lock(Path(sys.argv[1])): print("acquired")')
            with run_lock(Path(tmp)):
                child = subprocess.run([sys.executable, '-c', code, tmp], capture_output=True, text=True)
                self.assertNotEqual(child.returncode, 0)
                self.assertIn('already owned', child.stderr)
            with run_lock(Path(tmp)):
                pass

    def test_resume_log_backup_and_trim(self):
        with workspace_temp() as tmp:
            path = Path(tmp) / 'metrics.csv'
            original = 'global_step,ep_return\n4,1\n8,2\n'
            path.write_text(original)
            reconcile_logs(Path(tmp), 4)
            with path.open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 1)
            backup = list(Path(tmp).glob('resume-backup-*/metrics.csv'))
            self.assertEqual(len(backup), 1)
            self.assertEqual(backup[0].read_text(), original)

    def test_legacy_best_score(self):
        with workspace_temp() as tmp:
            run = Path(tmp)
            atomic_save(dict(global_step=8), run / 'ckpt_best.pt')
            (run / 'metrics.csv').write_text('global_step,ep_return\n4,1\n8,10\n')
            self.assertEqual(restore_best_score(run, {}), 10)

    def test_fake_training_resume_keeps_best_and_closes(self):
        with workspace_temp() as tmp, patch.dict(os.environ, {'TORCHINDUCTOR_CACHE_DIR': str(Path(tmp) / 'torch-cache')}), patch.object(ppo, 'ActorCritic', TinyModel), \
                patch.object(ppo, 'SubprocVecEnv', FakeVec), patch.object(torch.cuda, 'is_available', return_value=False):
            cfg = ppo.PPOConfig(out_dir=tmp, run_name='tiny', num_envs=2, rollout_len=2,
                                minibatches=1, epochs=1, total_steps=52, snapshot_every=100,
                                drive_backup_dir=str(Path(tmp) / 'drive'))
            run = ppo.train(cfg)
            self.assertTrue(FakeVec.instances[-1].closed)
            best_before = (run / 'ckpt_best.pt').read_bytes()
            ck = torch.load(run / 'ckpt_latest.pt', weights_only=True)
            self.assertEqual(ck['best_return'], 1.0)
            self.assertTrue(all(torch.isfinite(v).all() for v in ck['model'].values()))
            original_config = (run / 'config.json').read_bytes()
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                ppo.train(dataclasses.replace(cfg, resume=True, total_steps=56, reward_scale=.1))
            self.assertEqual((run / 'config.json').read_bytes(), original_config)
            with patch.object(FakeVec, 'reward', -1.0):
                ppo.train(dataclasses.replace(cfg, resume=True, total_steps=56))
            self.assertEqual((run / 'ckpt_best.pt').read_bytes(), best_before)
            final = torch.load(run / 'ckpt_final.pt', weights_only=True)
            self.assertEqual(final['global_step'], 56)
            self.assertEqual(final['best_return'], 1.0)
            from mario_rl.drive_backup import restore_run
            restored_dir = Path(tmp) / 'restored'
            restore_run(Path(tmp) / 'drive', restored_dir)
            restored = torch.load(restored_dir / 'ckpt_latest.pt', weights_only=True)
            self.assertEqual(restored['global_step'], 56)
            self.assertEqual(restored['best_return'], 1.0)

    def test_eval_completes_quota_across_episodes(self):
        cfg = dict(actions='simple', frame_skip=4, reward_scale=.01, flag_bonus=50)
        with patch.object(evaluation, 'SubprocVecEnv', FakeVec), patch.object(FakeVec, 'period', 4):
            results = evaluation.run_eval_vec(TinyModel(7), cfg, episodes=100, num_envs=16,
                                              max_steps=4, device=torch.device('cpu'))
        self.assertEqual(len(results), 100)
        self.assertTrue(FakeVec.instances[-1].closed)

    def test_eval_timeout_is_explicit_and_closes(self):
        cfg = dict(actions='simple', frame_skip=4, reward_scale=.01, flag_bonus=50)
        with patch.object(evaluation, 'SubprocVecEnv', FakeVec), patch.object(FakeVec, 'period', 10):
            with self.assertRaisesRegex(RuntimeError, 'incomplete'):
                evaluation.run_eval_vec(TinyModel(7), cfg, episodes=2, max_steps=3, device=torch.device('cpu'))
        self.assertTrue(FakeVec.instances[-1].closed)

    def test_notebook_rerun_does_not_launch(self):
        root = Path(__file__).resolve().parents[1]
        nb = json.loads((root / 'colab_train.ipynb').read_text(encoding='utf-8'))
        source = next(''.join(c['source']) for c in nb['cells'] if 'proc = subprocess.Popen' in ''.join(c['source']))
        proc = Mock()
        proc.poll.return_value = None
        with patch('subprocess.Popen') as launch, self.assertRaisesRegex(RuntimeError, 'PID'):
            exec(compile(source, '<launch-cell>', 'exec'), {'proc': proc})
        launch.assert_not_called()
        self.assertFalse(any('%%writefile' in ''.join(c['source']) for c in nb['cells']))
        self.assertTrue(any("drive.mount('/content/drive')" in ''.join(c['source']) for c in nb['cells']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
