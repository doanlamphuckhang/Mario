"""Package the Drive workflow without importing ML libraries."""
from pathlib import Path
import ast
import hashlib
import json
import zipfile

root = Path(__file__).resolve().parents[1]
names = ['colab_train.ipynb', 'train.py', 'evaluate.py', 'baselines.py', 'plot_results.py',
         'requirements.txt', 'DRIVE_START_HERE.md', 'scripts/patch_nes_py_numpy2.py',
         'scripts/build_colab_notebook.py', 'scripts/package_drive.py',
         'tests/test_core.py', 'tests/test_pretrain.py', 'tests/test_drive_workflow.py']
names += ['THIRD_PARTY_NOTICES.md', 'docs/DRIVE_START_HERE.md']
names += [p.relative_to(root).as_posix() for p in sorted((root / 'mario_rl').rglob('*.py'))]
payload = {name: (root / name).read_bytes() for name in names}
for name, data in payload.items():
    if name.endswith('.py'):
        ast.parse(data.decode('utf-8-sig'), filename=name)
nb = json.loads(payload['colab_train.ipynb'])
assert not any('%%writefile' in ''.join(c['source']) for c in nb['cells'])
manifest = {name: {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
            for name, data in payload.items()}
payload['MANIFEST.json'] = json.dumps(manifest, indent=2).encode()
out = root / 'mario_train_drive.zip'
with zipfile.ZipFile(out, 'x', zipfile.ZIP_DEFLATED) as archive:
    for name, data in payload.items():
        archive.writestr(name, data)
with zipfile.ZipFile(out) as archive:
    assert archive.testzip() is None
    for name, data in payload.items():
        assert archive.read(name) == data
print(json.dumps({'path': str(out), 'files': len(payload), 'bytes': out.stat().st_size,
                  'verified': True}))
