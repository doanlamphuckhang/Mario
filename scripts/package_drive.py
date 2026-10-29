"""Đóng gói mã nguồn thành mario_train_drive.zip cho cả hai notebook Colab.

    python scripts/package_drive.py

Không import thư viện ML, không kèm checkpoint/ROM/dữ liệu. Không ghi đè gói có sẵn:
đổi tên hoặc xóa gói cũ trước khi đóng gói lại.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = ['colab_train.ipynb', 'notebooks/train_all_stages.ipynb',
         'train.py', 'evaluate.py', 'evaluate_suite.py', 'baselines.py', 'plot_results.py',
         'make_figures.py', 'play3d.py', 'README.md', 'DRIVE_START_HERE.md',
         'THIRD_PARTY_NOTICES.md', 'requirements.txt',
         'scripts/patch_nes_py_numpy2.py', 'scripts/build_colab_notebook.py',
         'scripts/build_stage_notebook.py', 'scripts/package_drive.py']
FOLDERS = [('mario_rl', '*.py'), ('configs', '*.json'), ('tests', '*.py'), ('docs', '*.md')]
VIEWER_SUFFIXES = ('.py', '.js', '.css', '.html', '.json')


def collect() -> dict[str, bytes]:
    names = list(FILES)
    for folder, pattern in FOLDERS:
        names += [p.relative_to(ROOT).as_posix() for p in (ROOT / folder).rglob(pattern)
                  if '__pycache__' not in p.parts]
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT / 'viewer3d').rglob('*')
              if p.is_file() and p.suffix in VIEWER_SUFFIXES and '__pycache__' not in p.parts]
    return {name: (ROOT / name).read_bytes() for name in sorted(set(names))}


def check(payload: dict[str, bytes]) -> None:
    for name, data in payload.items():
        if name.endswith('.py'):
            ast.parse(data.decode('utf-8-sig'), filename=name)
        elif name.endswith('.json'):
            json.loads(data.decode('utf-8-sig'))
    colab = json.loads(payload['colab_train.ipynb'])
    if any('%%writefile' in ''.join(c['source']) for c in colab['cells']):
        raise ValueError('colab_train.ipynb must not embed source with %%writefile')
    stages = json.loads(payload['notebooks/train_all_stages.ipynb'])
    for cell in stages['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))


def package(output: Path) -> None:
    payload = collect()
    check(payload)
    manifest = {name: {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                for name, data in payload.items()}
    payload['MANIFEST.json'] = json.dumps(manifest, indent=2).encode('utf-8')
    with zipfile.ZipFile(output, 'x', zipfile.ZIP_DEFLATED) as archive:
        for name, data in payload.items():
            archive.writestr(name, data)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError('Archive verification failed')
        for name, data in payload.items():
            if archive.read(name) != data:
                raise ValueError(f'Archive verification failed: {name}')
    print(json.dumps({'path': str(output), 'files': len(payload), 'bytes': output.stat().st_size,
                      'verified': True}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--out', type=Path, default=ROOT / 'mario_train_drive.zip')
    package(parser.parse_args().out)
