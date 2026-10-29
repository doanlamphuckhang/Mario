"""Train lần lượt từng màn, mỗi màn một mô hình, bền qua nhiều phiên Colab.

Mỗi màn `W-S` là một run `ppo_W-S` (config gốc `configs/stage.json`). Gọi `prepare_stage` cho
từng màn theo thứ tự rồi `run_training` nếu cần:

    new     chưa có gì (cả ở máy lẫn Drive): run mới, có thể khởi tạo từ mô hình màn trước
    resume  có checkpoint dở (ở máy, hoặc khôi phục từ backup Drive): train tiếp
    done    đã dừng sớm vì vượt màn ổn định, hoặc đã chạy hết số bước

Chạy lại cả vòng sau khi Colab ngắt kết nối là an toàn: màn xong được bỏ qua, màn dở được tiếp tục.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .drive_backup import restore_run
from .envs import parse_stages

ALL_STAGES = [f'{w}-{s}' for w in range(1, 9) for s in range(1, 5)]


def parse_stage_list(spec: str) -> list[str]:
    """'all' -> 32 màn theo thứ tự; '1-1,1-2' -> đúng thứ tự đã ghi."""
    if spec.strip().lower() == 'all':
        return list(ALL_STAGES)
    return [f'{w}-{s}' for w, s in parse_stages(spec)]


def run_name(stage: str) -> str:
    return f'ppo_{stage}'


def _load(path: Path) -> dict:
    import torch  # import muộn: module dùng được để lập kế hoạch mà chưa cần torch

    return torch.load(path, map_location='cpu', weights_only=True)


def checkpoint_summary(path: Path) -> dict:
    """Bước đã train, có dừng sớm không, tỉ lệ về đích của tối đa 100 ván train gần nhất mỗi màn."""
    ck = _load(path)
    flags = {stage: [bool(e['flag']) for e in episodes][-100:]
             for stage, episodes in ck.get('stage_recent', {}).items()}
    return {
        'global_step': int(ck['global_step']),
        'update': int(ck['update']),
        'stopped_early': bool(ck.get('stopped_early', False)),
        'flag_rate': {s: (sum(f) / len(f) if f else None) for s, f in flags.items()},
        'episodes': {s: len(f) for s, f in flags.items()},
        'config': ck['config'],
    }


def _has_backup(drive_dir: Path) -> bool:
    return drive_dir.is_dir() and any(drive_dir.glob('backup-*.zip.sha256'))


def local_run(stage: str, work_dir: Path, drive_runs: Path | None) -> Path | None:
    """Thư mục run cục bộ có checkpoint; khôi phục từ Drive khi cần. None nếu chưa có gì."""
    run_dir = Path(work_dir) / 'runs' / run_name(stage)
    if (run_dir / 'ckpt_latest.pt').is_file():
        return run_dir
    if run_dir.exists():
        # Run bị ngắt trước checkpoint đầu tiên: cất sang bên, không xoá, để run mới bắt đầu sạch.
        run_dir.rename(run_dir.with_name(f'{run_dir.name}-incomplete-{time.time_ns()}'))
    if drive_runs is not None and _has_backup(Path(drive_runs) / run_name(stage)):
        restore_run(Path(drive_runs) / run_name(stage), run_dir)
        return run_dir
    return None


def final_checkpoint(stage: str, work_dir: Path, drive_runs: Path | None = None) -> Path | None:
    run_dir = local_run(stage, work_dir, drive_runs)
    path = None if run_dir is None else run_dir / 'ckpt_final.pt'
    return path if path is not None and path.is_file() else None


@dataclass
class StagePlan:
    stage: str
    status: str                  # new | resume | done
    run_dir: Path
    config_path: Path | None = None
    steps_done: int = 0
    flag_rate: float | None = None
    solved: bool = False
    init_from: str = ''

    def describe(self) -> str:
        rate = '–' if self.flag_rate is None else f'{self.flag_rate:.0%}'
        if self.status == 'done':
            result = 'đã vượt màn' if self.solved else 'hết số bước, chưa đạt ngưỡng'
            return f'[{self.stage}] xong ({result}): {self.steps_done:,} bước, về đích {rate} ván train gần nhất'
        if self.status == 'resume':
            return f'[{self.stage}] train tiếp từ bước {self.steps_done:,} (về đích {rate})'
        source = f'khởi tạo từ {Path(self.init_from).parent.name}' if self.init_from else 'từ đầu'
        return f'[{self.stage}] train mới, {source}'


def prepare_stage(stage: str, work_dir: Path, drive_runs: Path | None, *, base_profile: Path,
                  total_steps: int, device: str, num_envs: int, rollout_len: int, seed: int = 1,
                  warm_from: Path | None = None, stop_flag_rate: float | None = None) -> StagePlan:
    """Quyết định new/resume/done cho một màn và ghi config cho trainer (nếu cần train)."""
    from .ppo import PPOConfig, validate_config

    work_dir = Path(work_dir)
    name = run_name(stage)
    out_dir = work_dir / 'runs'
    drive_dir = '' if drive_runs is None else str(Path(drive_runs) / name)
    run_dir = local_run(stage, work_dir, drive_runs)
    exp_dir = work_dir / 'experiments'
    exp_dir.mkdir(parents=True, exist_ok=True)

    if run_dir is not None:
        latest = checkpoint_summary(run_dir / 'ckpt_latest.pt')
        cfg = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
        batch = cfg['num_envs'] * cfg['rollout_len']
        target_updates = total_steps // batch
        plan = StagePlan(stage, 'done', run_dir, steps_done=latest['global_step'],
                         flag_rate=latest['flag_rate'].get(stage), solved=latest['stopped_early'])
        final = run_dir / 'ckpt_final.pt'
        if latest['stopped_early'] or latest['update'] >= target_updates:
            if not final.is_file():
                # Bị ngắt ngay sau checkpoint cuối, trước khi ghi final: hai trạng thái là một.
                shutil.copyfile(run_dir / 'ckpt_latest.pt', final)
            return plan
        cfg.update(resume=True, init_from='', total_steps=total_steps, device=device,
                   out_dir=str(out_dir), drive_backup_dir=drive_dir)
        if stop_flag_rate is not None:
            cfg['stop_flag_rate'] = stop_flag_rate
        plan.status = 'resume'
    else:
        cfg = json.loads(Path(base_profile).read_text(encoding='utf-8-sig'))
        cfg.update(run_name=name, stages=stage, out_dir=str(out_dir), drive_backup_dir=drive_dir,
                   total_steps=total_steps, device=device, num_envs=num_envs, rollout_len=rollout_len,
                   seed=seed, init_from=str(warm_from) if warm_from else '', resume=False)
        if stop_flag_rate is not None:
            cfg['stop_flag_rate'] = stop_flag_rate
        plan = StagePlan(stage, 'new', out_dir / name, init_from=cfg['init_from'])
    validate_config(PPOConfig(**cfg))
    plan.config_path = exp_dir / f'{name}.json'
    plan.config_path.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    return plan


def run_training(config_path: Path, log_path: Path, echo_every: int = 50) -> None:
    """Chạy trainer trong tiến trình con; ghi đủ log ra file, chỉ in mỗi `echo_every` dòng tiến độ."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, '-u', '-m', 'mario_rl.cli.train', '--config', str(config_path)]
    tail: list[str] = []
    progress = 0
    with log_path.open('a', encoding='utf-8') as log, \
            subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             encoding='utf-8', errors='replace') as proc:
        try:
            for line in proc.stdout:
                log.write(line)
                tail = (tail + [line])[-30:]
                if line.startswith('upd '):
                    progress += 1
                    if progress % echo_every != 1:
                        continue
                print(line, end='', flush=True)
            proc.wait()
        except KeyboardInterrupt:
            # Dừng ô notebook thì dừng luôn trainer; bản sao Drive gần nhất vẫn dùng để train tiếp.
            proc.terminate()
            proc.wait(timeout=30)
            raise
    if proc.returncode != 0:
        raise RuntimeError(f'Trainer lỗi (exit {proc.returncode}); log đầy đủ: {log_path}\n' + ''.join(tail))


def status_rows(stages: list[str], work_dir: Path) -> list[dict]:
    """Bảng tình trạng các màn từ checkpoint cục bộ (sau vòng train, mọi màn đã xong đều có ở máy)."""
    rows = []
    for stage in stages:
        run_dir = Path(work_dir) / 'runs' / run_name(stage)
        latest = run_dir / 'ckpt_latest.pt'
        if not latest.is_file():
            rows.append({'màn': stage, 'trạng thái': 'chưa train', 'bước': 0, 'về đích (train)': None})
            continue
        s = checkpoint_summary(latest)
        finished = (run_dir / 'ckpt_final.pt').is_file()
        state = 'đã vượt màn' if s['stopped_early'] else ('hết số bước' if finished else 'đang dở')
        rows.append({'màn': stage, 'trạng thái': state, 'bước': s['global_step'],
                     'về đích (train)': s['flag_rate'].get(stage)})
    return rows
