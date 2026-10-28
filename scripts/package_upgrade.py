"""Create a small source-only upgrade archive without importing ML libraries.

Does not run tests, install dependencies or include checkpoints/ROMs/data.
An existing archive is never overwritten.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def package(output):
    names = ['train.py', 'evaluate.py', 'evaluate_suite.py', 'baselines.py',
             'plot_results.py', 'make_figures.py', 'play3d.py', 'README.md',
             'THIRD_PARTY_NOTICES.md', 'requirements.txt', 'notebooks/train_upgrade.ipynb',
             'scripts/patch_nes_py_numpy2.py', 'scripts/package_upgrade.py']
    for folder, pattern in [('mario_rl', '*.py'), ('configs', '*.json'),
                            ('tests', '*.py'), ('docs', '*.md')]:
        names.extend(p.relative_to(ROOT).as_posix() for p in (ROOT / folder).rglob(pattern))
    for path in (ROOT / 'viewer3d').rglob('*'):
        if path.is_file() and path.suffix in ('.py', '.js', '.css', '.html', '.json') and '__pycache__' not in path.parts:
            names.append(path.relative_to(ROOT).as_posix())
    payload = {name: (ROOT / name).read_bytes() for name in sorted(set(names))}
    for name, data in payload.items():
        if name.endswith('.py'):
            ast.parse(data.decode('utf-8-sig'), filename=name)
    notebook = json.loads(payload['notebooks/train_upgrade.ipynb'])
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))
    manifest = {name: {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                for name, data in payload.items()}
    payload['MANIFEST.json'] = json.dumps(manifest, indent=2).encode('utf-8')
    with zipfile.ZipFile(output, 'x', zipfile.ZIP_DEFLATED) as archive:
        for name, data in payload.items():
            archive.writestr(name, data)
    with zipfile.ZipFile(output) as archive:
        for name, data in payload.items():
            if archive.read(name) != data:
                raise ValueError(f'Archive verification failed: {name}')
    print(json.dumps({'output': str(output), 'files': len(payload), 'tests_run': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'mario_upgrade_v2.zip')
    package(parser.parse_args().out)
