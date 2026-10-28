"""Versioned filesystem backups; works with a mounted Drive without a cloud SDK."""
import hashlib
from pathlib import Path
import shutil
import time
import zipfile

FILES = {'ckpt_latest.pt', 'ckpt_best.pt', 'ckpt_final.pt', 'config.json', 'metrics.csv', 'episodes.csv',
         'transfer.json', 'stage_metrics.json'}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def backup_run(run_dir, destination, step):
    """Called synchronously by the trainer, after complete checkpoint/log writes."""
    run_dir, destination = Path(run_dir), Path(destination)
    if not (run_dir / 'ckpt_latest.pt').is_file():
        raise FileNotFoundError('No latest checkpoint to back up')
    destination.mkdir(parents=True, exist_ok=True)
    name = f'backup-{time.time_ns()}-step{step}.zip'
    # Build locally; a Drive .sha256 marker is published only after verification.
    local = run_dir / 'drive-backup.pending.zip'
    try:
        with zipfile.ZipFile(local, 'w', zipfile.ZIP_STORED) as archive:
            for filename in sorted(FILES):
                path = run_dir / filename
                if path.is_file():
                    archive.write(path, filename)
        checksum = digest(local)
        target = destination / name
        shutil.copyfile(local, target)
        if digest(target) != checksum:
            raise OSError(f'Drive backup checksum mismatch: {target}')
        target.with_suffix('.zip.sha256').write_text(checksum, encoding='ascii')
    finally:
        # The local archive holds several checkpoints; do not leave it on the runtime disk.
        local.unlink(missing_ok=True)
    return target


def restore_run(destination, run_dir):
    """Restore the newest complete backup into a fresh local run directory."""
    destination, run_dir = Path(destination), Path(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f'Restore requires an empty local directory: {run_dir}')
    errors = []
    for marker in sorted(destination.glob('backup-*.zip.sha256'), reverse=True):
        archive_path = marker.with_suffix('')
        try:
            if digest(archive_path) != marker.read_text(encoding='ascii').strip():
                raise ValueError('checksum mismatch')
            with zipfile.ZipFile(archive_path) as archive:
                names = archive.namelist()
                if len(names) != len(set(names)) or not set(names) <= FILES:
                    raise ValueError('unexpected backup member')
                if not {'ckpt_latest.pt', 'config.json'} <= set(names) or archive.testzip():
                    raise ValueError('incomplete backup')
                run_dir.mkdir(parents=True, exist_ok=True)
                archive.extractall(run_dir)
            return archive_path
        except (ValueError, OSError, zipfile.BadZipFile) as exc:
            # Do not conceal a local write failure after extraction has started.
            if run_dir.exists() and any(run_dir.iterdir()):
                raise
            errors.append(f'{archive_path.name}: {exc}')
    raise FileNotFoundError(f'No verified backup in {destination}. ' + '; '.join(errors))
