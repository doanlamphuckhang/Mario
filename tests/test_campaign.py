"""Train theo từng màn: lập kế hoạch new/resume/done, dừng sớm, dọn backup, quay demo nhiều màn.

Không train mô hình thật: checkpoint là dict nhỏ, trainer chạy với môi trường/mô hình giả.
"""
import dataclasses
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

import torch

from mario_rl import campaign, ppo
from mario_rl.drive_backup import backup_run
from mario_rl.run_io import atomic_save
from test_pretrain import FakeVec, TinyModel

PROFILE = ROOT / 'configs' / 'stage.json'


def workspace():
    return tempfile.TemporaryDirectory(dir=ROOT / 'results')


def fake_run(work, stage, *, update, stopped=False, final=True, flags=(True,) * 100):
    """Thư mục run giống trainer ghi ra: config.json + ckpt_latest/final."""
    run_dir = Path(work) / 'runs' / campaign.run_name(stage)
    run_dir.mkdir(parents=True)
    cfg = json.loads(PROFILE.read_text(encoding='utf-8'))
    cfg.update(run_name=campaign.run_name(stage), stages=stage, num_envs=4, rollout_len=512)
    cfg = dataclasses.asdict(ppo.PPOConfig(**cfg))
    (run_dir / 'config.json').write_text(json.dumps(cfg), encoding='utf-8')
    state = dict(model={}, global_step=update * 2048, update=update, config=cfg, stopped_early=stopped,
                 stage_recent={stage: [{'flag': f} for f in flags]})
    atomic_save(state, run_dir / 'ckpt_latest.pt')
    if final:
        atomic_save(state, run_dir / 'ckpt_final.pt')
    return run_dir


def plan(stage, work, drive=None, total=1_000_000, **kw):
    return campaign.prepare_stage(stage, Path(work), drive, base_profile=PROFILE, total_steps=total,
                                  device='cpu', num_envs=8, rollout_len=256, **kw)


class CampaignTests(unittest.TestCase):
    def test_stage_list(self):
        all_stages = campaign.parse_stage_list('all')
        self.assertEqual(len(all_stages), 32)
        self.assertEqual((all_stages[0], all_stages[4], all_stages[-1]), ('1-1', '2-1', '8-4'))
        self.assertEqual(campaign.parse_stage_list('2-1, 1-1'), ['2-1', '1-1'])
        with self.assertRaises(ValueError):
            campaign.parse_stage_list('9-1')

    def test_new_stage_from_scratch_and_warm_start(self):
        with workspace() as work:
            p = plan('1-1', work)
            self.assertEqual(p.status, 'new')
            cfg = json.loads(p.config_path.read_text(encoding='utf-8'))
            self.assertEqual((cfg['stages'], cfg['run_name'], cfg['init_from']), ('1-1', 'ppo_1-1', ''))
            self.assertEqual((cfg['num_envs'], cfg['rollout_len'], cfg['actions']), (8, 256, 'complex'))
            warm = plan('1-2', work, warm_from=Path(work) / 'runs/ppo_1-1/ckpt_final.pt')
            self.assertIn('ppo_1-1', json.loads(warm.config_path.read_text(encoding='utf-8'))['init_from'])
            self.assertIn('ppo_1-1', warm.describe())

    def test_done_resume_and_interrupted_final(self):
        with workspace() as work:
            fake_run(work, '1-1', update=100, stopped=True)
            done = plan('1-1', work)
            self.assertEqual((done.status, done.solved, done.flag_rate), ('done', True, 1.0))
            self.assertIsNone(done.config_path)

            fake_run(work, '1-2', update=100, final=False, flags=(False,) * 30)
            resume = plan('1-2', work)
            cfg = json.loads(resume.config_path.read_text(encoding='utf-8'))
            self.assertEqual(resume.status, 'resume')
            self.assertTrue(cfg['resume'])
            self.assertEqual((cfg['num_envs'], cfg['init_from']), (4, ''))   # giữ cấu hình đã lưu

            run_dir = fake_run(work, '1-3', update=488, final=False)          # 488 * 2048 >= 1M bước
            self.assertEqual(plan('1-3', work).status, 'done')
            self.assertTrue((run_dir / 'ckpt_final.pt').is_file())

            fake_run(work, '1-4', update=488, flags=(False,) * 100)           # hết bước, chưa vượt
            self.assertEqual(plan('1-4', work).status, 'done')
            self.assertEqual(plan('1-4', work, total=2_000_000).status, 'resume')

    def test_restore_from_drive_and_park_incomplete_local(self):
        with workspace() as work:
            source = fake_run(Path(work) / 'old', '2-1', update=100, stopped=True)
            drive = Path(work) / 'drive'
            backup_run(source, drive / 'ppo_2-1', 204800)
            partial = Path(work) / 'runs' / 'ppo_2-1'
            partial.mkdir(parents=True)
            (partial / 'metrics.csv').write_text('global_step\n', encoding='utf-8')
            p = plan('2-1', work, drive)
            self.assertEqual(p.status, 'done')
            self.assertTrue((partial / 'ckpt_final.pt').is_file())
            self.assertEqual(len(list(partial.parent.glob('ppo_2-1-incomplete-*'))), 1)
            self.assertIsNone(campaign.final_checkpoint('3-1', work, drive))

    def test_status_rows(self):
        with workspace() as work:
            fake_run(work, '1-1', update=100, stopped=True)
            fake_run(work, '1-2', update=10, final=False, flags=(True, False))
            rows = campaign.status_rows(['1-1', '1-2', '1-3'], work)
            self.assertEqual([r['trạng thái'] for r in rows], ['đã vượt màn', 'đang dở', 'chưa train'])
            self.assertEqual(rows[1]['về đích (train)'], .5)


class TrainerTests(unittest.TestCase):
    def test_early_stop_when_stage_solved(self):
        flag_vec = type('FlagVec', (FakeVec,), {'step': lambda self, a: self._flag(FakeVec.step(self, a))})

        def _flag(self, out):
            for ep in out[3]:
                if ep is not None:
                    ep['flag'] = True
            return out
        flag_vec._flag = _flag
        with workspace() as tmp, patch.dict(os.environ, {'TORCHINDUCTOR_CACHE_DIR': str(Path(tmp) / 'c')}), \
                patch.object(ppo, 'ActorCritic', TinyModel), patch.object(ppo, 'SubprocVecEnv', flag_vec), \
                patch.object(torch.cuda, 'is_available', return_value=False):
            cfg = ppo.PPOConfig(out_dir=tmp, run_name='solve', num_envs=2, rollout_len=2, minibatches=1, epochs=1,
                                total_steps=4000, snapshot_every=10**9, stop_flag_rate=.9, stop_min_episodes=10)
            run = ppo.train(cfg)
            final = torch.load(run / 'ckpt_final.pt', weights_only=True)
            latest = torch.load(run / 'ckpt_latest.pt', weights_only=True)
            self.assertTrue(final['stopped_early'])
            self.assertLess(final['global_step'], 4000)
            self.assertEqual(final['global_step'], latest['global_step'])

    def test_backup_keep_prunes_old_versions(self):
        with workspace() as tmp:
            run = Path(tmp) / 'run'
            run.mkdir()
            (run / 'ckpt_latest.pt').write_bytes(b'x')
            drive = Path(tmp) / 'drive'
            for step in range(4):
                backup_run(run, drive, step, keep=2)
            self.assertEqual(len(list(drive.glob('backup-*.zip'))), 2)
            self.assertEqual(len(list(drive.glob('backup-*.zip.sha256'))), 2)
            self.assertEqual(sorted(p.name.split('-step')[1] for p in drive.glob('*.zip')), ['2.zip', '3.zip'])


class DemoTests(unittest.TestCase):
    def test_records_each_episode_and_combined_video(self):
        from mario_rl import demo
        from mario_rl.envs import ACTION_SETS

        class RightRunner(torch.nn.Module):          # chính sách cố định, không phải mô hình đã học
            def act(self, obs, deterministic):
                return torch.full((len(obs),), ACTION_SETS['simple'].index(['right', 'A', 'B']))
        cfg = dict(actions='simple', frame_skip=4, reward_scale=.01, flag_bonus=50)
        with workspace() as tmp, patch.object(demo, 'load_model', return_value=(RightRunner(), cfg)):
            out = Path(tmp) / 'demo'
            results = demo.record([('1-1', Path('a.pt')), ('1-2', Path('a.pt'))], out, episodes=2,
                                  noop_max=0, max_steps=25)
            self.assertEqual([(r['stage'], r['episode']) for r in results],
                             [('1-1', 1), ('1-1', 2), ('1-2', 1), ('1-2', 2)])
            self.assertEqual([r['policy'] for r in results], ['argmax', 'sample'] * 2)
            for r in results:
                self.assertGreater((out / r['file']).stat().st_size, 1000)
            self.assertGreater((out / 'all_stages.mp4').stat().st_size, 1000)
            summary = json.loads((out / 'demo.json').read_text(encoding='utf-8'))
            self.assertEqual(len(summary['results']), 4)
            with self.assertRaises(FileExistsError):              # không ghi đè demo cũ
                demo.record([('1-1', Path('a.pt'))], out)


if __name__ == '__main__':
    unittest.main(verbosity=2)
