"""Small standard-library-only tests for backup integrity and notebook wiring."""
import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mario_rl.drive_backup import backup_run, restore_run


class DriveTests(unittest.TestCase):
    def test_backup_restore_and_partial_fallback(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'results') as tmp:
            base = Path(tmp)
            run, drive = base / 'run', base / 'drive'
            run.mkdir()
            (run / 'ckpt_latest.pt').write_bytes(b'synthetic checkpoint')
            (run / 'config.json').write_text('{}')
            (run / 'metrics.csv').write_text('global_step\n4\n')
            good = backup_run(run, drive, 4)
            self.assertFalse((run / 'drive-backup.pending.zip').exists())
            bad = backup_run(run, drive, 8)
            bad.write_bytes(b'incomplete upload')
            (drive / 'backup-9999999999999999999-step9.zip').write_bytes(b'no marker')
            restored = restore_run(drive, base / 'restored')
            self.assertEqual(restored, good)
            self.assertEqual((base / 'restored/ckpt_latest.pt').read_bytes(), b'synthetic checkpoint')
            with self.assertRaises(FileExistsError):
                restore_run(drive, base / 'restored')

    def test_failed_backup_never_publishes_marker(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'results') as tmp:
            base = Path(tmp)
            run = base / 'run'
            run.mkdir()
            (run / 'ckpt_latest.pt').write_bytes(b'checkpoint')
            with patch('mario_rl.drive_backup.shutil.copyfile', side_effect=OSError('offline')):
                with self.assertRaises(OSError):
                    backup_run(run, base / 'drive', 4)
            self.assertFalse(list((base / 'drive').glob('*.sha256')))
            self.assertFalse((run / 'drive-backup.pending.zip').exists())

    def test_notebook_cells_and_double_launch(self):
        nb = json.loads((ROOT / 'colab_train.ipynb').read_text(encoding='utf-8'))
        sources = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
        self.assertIn("drive.mount('/content/drive')", sources[0])
        self.assertFalse(any('%%writefile' in s for s in sources))
        self.assertTrue(any('archive.extractall(WORK_DIR)' in s for s in sources))
        self.assertTrue(any('restore_run(DRIVE_RUN_DIR' in s for s in sources))
        for source in sources:
            if source.startswith('%%bash'):
                continue
            python = '\n'.join(line[:len(line) - len(line.lstrip())] + 'pass'
                               if line.lstrip().startswith('!') else line for line in source.splitlines())
            ast.parse(python)
        launch = next(s for s in sources if 'proc = subprocess.Popen' in s)
        proc = Mock()
        proc.poll.return_value = None
        with patch('subprocess.Popen') as popen, self.assertRaises(RuntimeError):
            exec(launch, {'proc': proc})
        popen.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
