"""Exclusive run ownership and recoverable checkpoint/log writes."""
from contextlib import contextmanager
import csv
import math
import os
from pathlib import Path
import shutil
import time

import torch


@contextmanager
def run_lock(run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    # Keep the inode: unlinking a lock file can allow two independent owners.
    with (run_dir / '.train.lock').open('a+b') as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f'Run already owned by another trainer: {run_dir}') from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def atomic_save(state, path: Path):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(state, temporary)
    os.replace(temporary, path)


def restore_best_score(run_dir: Path, checkpoint: dict) -> float:
    score = float(checkpoint.get('best_return', float('-inf')))
    best_path = run_dir / 'ckpt_best.pt'
    if best_path.exists():
        best = torch.load(best_path, map_location='cpu', weights_only=True)
        if 'best_return' in best:
            score = max(score, float(best['best_return']))
        else:
            # Older checkpoints did not store their selection score.
            metrics = run_dir / 'metrics.csv'
            with metrics.open(newline='', encoding='utf-8') as handle:
                matches = [float(row['ep_return']) for row in csv.DictReader(handle)
                           if int(row['global_step']) == int(best['global_step'])]
            if not matches or not all(math.isfinite(x) for x in matches):
                raise ValueError('Cannot recover legacy best score from metrics.csv; preserve this run and use a new run name.')
            score = max(score, max(matches))
    return score


def reconcile_logs(run_dir: Path, global_step: int):
    """Preserve original CSVs before removing rows newer than the loaded checkpoint."""
    pending = []
    for name in ('metrics.csv', 'episodes.csv'):
        path = run_dir / name
        if not path.exists() or path.stat().st_size == 0:
            continue
        with path.open(newline='', encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            fields, rows = reader.fieldnames, list(reader)
        kept = [row for row in rows if int(row['global_step']) <= global_step]
        if len(kept) != len(rows):
            pending.append((path, fields, kept))
    if pending:
        backup = run_dir / f'resume-backup-{time.time_ns()}'
        backup.mkdir()
        for path, fields, kept in pending:
            shutil.copy2(path, backup / path.name)
            temporary = path.with_suffix('.csv.tmp')
            with temporary.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fields)
                writer.writeheader()
                writer.writerows(kept)
            os.replace(temporary, path)
