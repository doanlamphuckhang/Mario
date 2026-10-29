"""Sinh notebooks/train_all_stages.ipynb: train từng màn từ đầu, đánh giá, quay demo nhiều màn.

    python scripts/build_stage_notebook.py

Notebook chỉ điều phối; logic nằm trong mario_rl.campaign, mario_rl.cli.train/evaluate/demo.
"""
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'notebooks' / 'train_all_stages.ipynb'
cells: list[dict] = []


def _lines(text: str) -> list[str]:
    parts = text.strip('\n').split('\n')
    return [p + '\n' for p in parts[:-1]] + [parts[-1]]


def md(text: str) -> None:
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': _lines(text)})


def code(text: str) -> None:
    ast.parse(text)
    cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
                  'source': _lines(text)})


md("""
# Mario PPO · train từng màn từ đầu

1. Upload `mario_train_drive.zip` vào My Drive (hoặc một thư mục con).
2. Runtime > Change runtime type > GPU; loại có nhiều CPU sẽ mô phỏng nhanh hơn.
3. Chạy lần lượt các ô. Mỗi màn có một mô hình riêng, lưu ở `runs/ppo_<màn>/` cạnh file ZIP trên Drive.

Bị ngắt kết nối hoặc hết phiên: Runtime > Restart session rồi chạy lại từ ô đầu. Màn đã xong
được bỏ qua, màn đang dở train tiếp từ bản sao gần nhất trên Drive (sao lưu mỗi 20 phút).
""")

code("""
from google.colab import drive
from pathlib import Path
import os, sys, json, hashlib, tempfile, zipfile, subprocess

drive.mount('/content/drive')
MY_DRIVE = Path('/content/drive/MyDrive')


def find_source_zip():
    \"\"\"Tìm gói nguồn trong MyDrive/Mario, MyDrive, rồi thư mục con tới 2 cấp; nhiều bản thì lấy bản mới nhất.\"\"\"
    for pattern in ('Mario/mario_train_drive*.zip', 'mario_train_drive*.zip',
                    '*/mario_train_drive*.zip', '*/*/mario_train_drive*.zip'):
        found = [p for p in MY_DRIVE.glob(pattern) if p.is_file()]
        if found:
            return max(found, key=lambda p: p.stat().st_mtime)
    raise FileNotFoundError('Không thấy mario_train_drive.zip trong Google Drive. Upload file vào MyDrive '
                            '(hoặc một thư mục con như MyDrive/Mario) rồi chạy lại ô này; '
                            'nếu vừa upload, đợi Drive đồng bộ vài giây.')


SOURCE_ZIP = find_source_zip()
DRIVE_ROOT = SOURCE_ZIP.parent      # runs/, evaluations/, demos/ nằm cạnh gói nguồn
DRIVE_RUNS = DRIVE_ROOT / 'runs'
print('Gói nguồn:', SOURCE_ZIP)
print('Thư mục lưu kết quả:', DRIVE_ROOT)
if any(name == 'mario_rl' or name.startswith('mario_rl.') for name in sys.modules):
    raise RuntimeError('Restart runtime trước khi đổi gói mã nguồn.')
with zipfile.ZipFile(SOURCE_ZIP) as archive:
    manifest = json.loads(archive.read('MANIFEST.json'))
    names = archive.namelist()
    if len(names) != len(set(names)) or set(names) != set(manifest) | {'MANIFEST.json'}:
        raise ValueError('ZIP không khớp manifest')
    if 'mario_rl/campaign.py' not in manifest:
        raise ValueError(f'{SOURCE_ZIP.name} là gói cũ. Tạo lại bằng scripts/package_drive.py rồi upload.')
    for name, item in manifest.items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or '\\\\' in name or ':' in name:
            raise ValueError(f'Đường dẫn không hợp lệ: {name}')
        data = archive.read(name)
        if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError(f'Checksum sai: {name}')
    WORK_DIR = Path(tempfile.mkdtemp(prefix='mario_', dir='/content'))
    archive.extractall(WORK_DIR)
os.chdir(WORK_DIR)
sys.path.insert(0, str(WORK_DIR))
print('Mã nguồn tại:', WORK_DIR)
""")

code("""
from importlib.metadata import version, PackageNotFoundError

INSTALL_DEPS = True   # False: chỉ kiểm tra, không tự cài
required = {'gym': '0.26.2', 'nes-py': '8.2.1', 'gym-super-mario-bros': '7.4.0'}


def installed(package):
    try:
        return version(package)
    except PackageNotFoundError:
        return None


mismatched = {p: installed(p) for p, v in required.items() if installed(p) != v}
if mismatched and INSTALL_DEPS:
    loaded = [p for p in ('gym', 'nes_py', 'gym_super_mario_bros') if p in sys.modules]
    if loaded:
        raise RuntimeError(f'Đã import {loaded} với phiên bản khác; Runtime > Restart session rồi chạy lại từ ô đầu.')
    # Colab có sẵn gym 0.25.x: cài đè đúng bản đã chốt. nes-py và gym-super-mario-bros khai báo
    # phụ thuộc cũ nên cài --no-deps; nes-py biên dịch C++ khi cài, mất khoảng 1-2 phút.
    for package, current in mismatched.items():
        print(f'Cài {package}=={required[package]} (hiện có: {current or "chưa cài"})', flush=True)
        extra = [] if package == 'gym' else ['--no-deps']
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                        f'{package}=={required[package]}', *extra], check=True)
for package, expected in required.items():
    actual = installed(package)
    if actual != expected:
        hint = 'xem log pip ở trên' if INSTALL_DEPS else 'đặt INSTALL_DEPS = True để tự cài'
        raise RuntimeError(f'{package}: cần {expected}, hiện có {actual or "chưa cài"}; {hint}')
# Vá nes-py cho NumPy 2; chạy lại nhiều lần không sao.
subprocess.run([sys.executable, 'scripts/patch_nes_py_numpy2.py'], check=True)
for package in ('torch', 'numpy', *required):
    print(package, version(package))
""")

md("## Cài đặt")
code("""
import torch
from mario_rl import campaign

STAGES = 'all'                   # 32 màn 1-1 ... 8-4 theo thứ tự; thử ít trước: '1-1,1-2,1-3,1-4'
MAX_STEPS_PER_STAGE = 5_000_000  # trần mỗi màn; màn dễ dừng sớm khi đạt ngưỡng dưới đây
TARGET_FLAG_RATE = 0.9           # dừng một màn khi 100 ván train gần nhất về đích >= 90%
WARM_START = True                # màn sau khởi tạo từ mô hình màn trước (học nhanh hơn)
SEED = 1
EVAL_EPISODES = 20               # số ván đánh giá mỗi màn
DEMO_EPISODES = 3                # số video mỗi màn

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
cpu = os.cpu_count() or 1
NUM_ENVS = 16 if cpu >= 10 else 8 if cpu >= 5 else 4 if cpu >= 3 else 2
ROLLOUT_LEN = 2048 // NUM_ENVS   # giữ lô 2048 mẫu như lần train 1-1 trước
BASE_PROFILE = WORK_DIR / 'configs' / 'stage.json'
stages = campaign.parse_stage_list(STAGES)
print(f'{len(stages)} màn: {", ".join(stages)}')
print(f'Thiết bị {DEVICE}; {cpu} CPU -> {NUM_ENVS} môi trường x {ROLLOUT_LEN} bước')
if DEVICE == 'cpu':
    print('Không có GPU: Runtime > Change runtime type > GPU để train nhanh hơn nhiều.')
print('Tham khảo: lần train 1-1 trước chạy ~416 bước/giây với 16 môi trường, '
      'cần ~1,3 triệu bước để về đích 90% số ván.')
""")

md("""
## Train

Chạy được nhiều giờ. Log đầy đủ của từng màn ở `logs/ppo_<màn>.log` trong thư mục làm việc;
ô này chỉ in tiến độ thưa. Dừng ô (nút Stop) sẽ dừng trainer; chạy lại ô để train tiếp.
""")
code("""
prev_final = None
for stage in stages:
    plan = campaign.prepare_stage(stage, WORK_DIR, DRIVE_RUNS, base_profile=BASE_PROFILE,
                                  total_steps=MAX_STEPS_PER_STAGE, device=DEVICE, num_envs=NUM_ENVS,
                                  rollout_len=ROLLOUT_LEN, seed=SEED,
                                  warm_from=prev_final if WARM_START else None,
                                  stop_flag_rate=TARGET_FLAG_RATE)
    print(plan.describe(), flush=True)
    if plan.status != 'done':
        campaign.run_training(plan.config_path, WORK_DIR / 'logs' / f'{campaign.run_name(stage)}.log')
        row = campaign.status_rows([stage], WORK_DIR)[0]
        rate = row['về đích (train)']
        print(f"[{stage}] {row['trạng thái']}: {row['bước']:,} bước, về đích "
              f"{'–' if rate is None else f'{rate:.0%}'}", flush=True)
    prev_final = campaign.final_checkpoint(stage, WORK_DIR)
print('Đã đi hết danh sách màn.')
""")

md("## Tình trạng các màn")
code("""
import pandas as pd

status = pd.DataFrame(campaign.status_rows(stages, WORK_DIR))
status.to_csv(DRIVE_ROOT / 'mario_stage_status.csv', index=False)
status
""")

md("""
## Đánh giá

Mỗi màn được đánh giá bằng chính mô hình của màn đó: chọn hành động argmax, xuất phát lệch
ngẫu nhiên tới 30 bước để các ván khác nhau. Kết quả lưu ở `evaluations/<thời điểm>/`.
""")
code("""
from datetime import datetime

EVAL_DIR = DRIVE_ROOT / 'evaluations' / datetime.now().strftime('%Y%m%d-%H%M%S')
EVAL_DIR.mkdir(parents=True)
rows = []
for stage in stages:
    ckpt = campaign.final_checkpoint(stage, WORK_DIR, DRIVE_RUNS)
    if ckpt is None:
        print(f'[{stage}] chưa có mô hình, bỏ qua')
        continue
    out = EVAL_DIR / f'{stage}.json'
    subprocess.run([sys.executable, '-m', 'mario_rl.cli.evaluate', '--ckpt', str(ckpt), '--stage', stage,
                    '--episodes', str(EVAL_EPISODES), '--noop_max', '30', '--parallel', str(min(NUM_ENVS, 8)),
                    '--device', DEVICE, '--json', str(out)], check=True, stdout=subprocess.DEVNULL)
    s = json.loads(out.read_text(encoding='utf-8'))['summary']
    rows.append({'màn': stage, 'về đích': s['flag_rate'], 'x trung bình': round(s['max_x_mean']),
                 'điểm trung bình': s['score_mean'], 'số ván': s['episodes']})
    print(f"[{stage}] về đích {s['flag_rate']:.0%}", flush=True)
evaluation = pd.DataFrame(rows)
evaluation.to_csv(EVAL_DIR / 'summary.csv', index=False)
evaluation
""")

md("""
## Video demo

Mỗi màn quay `DEMO_EPISODES` ván, mỗi ván một video: ván 1 chọn hành động tốt nhất (argmax), các ván
sau lấy mẫu từ chính sách nên mỗi video một khác. Thêm một video tổng `all_stages.mp4` ghép
tất cả (tua nhanh x2, có tên màn và kết quả từng ván). Video lưu ở `demos/<thời điểm>/` trên Drive.
""")
code("""
from IPython.display import Video, display

items = [(s, campaign.final_checkpoint(s, WORK_DIR, DRIVE_RUNS)) for s in stages]
items = [(s, c) for s, c in items if c is not None]
if not items:
    raise RuntimeError('Chưa có màn nào train xong; chạy ô Train trước.')
DEMO_DIR = DRIVE_ROOT / 'demos' / datetime.now().strftime('%Y%m%d-%H%M%S')
cmd = [sys.executable, '-m', 'mario_rl.cli.demo', '--out', str(DEMO_DIR),
       '--episodes', str(DEMO_EPISODES), '--device', DEVICE]
for stage, ckpt in items:
    cmd += ['--item', f'{stage}={ckpt}']
subprocess.run(cmd, check=True)
demo = json.loads((DEMO_DIR / 'demo.json').read_text(encoding='utf-8'))
display(pd.DataFrame(demo['results'])[['stage', 'episode', 'policy', 'flag', 'max_x', 'steps', 'file']])
print('Tất cả video trên Drive:', DEMO_DIR)
combined = DEMO_DIR / 'all_stages.mp4'
if combined.stat().st_size < 80e6:
    display(Video(str(combined), embed=True, width=512))
else:   # video tổng quá lớn để nhúng: hiện ván đầu của tối đa 8 màn
    for stage, _ in items[:8]:
        print(stage)
        display(Video(str(DEMO_DIR / f'{stage}_ep1.mp4'), embed=True, width=384))
""")

nb = {'cells': cells,
      'metadata': {'accelerator': 'GPU', 'colab': {'provenance': []},
                   'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                   'language_info': {'name': 'python'}},
      'nbformat': 4, 'nbformat_minor': 5}
OUT.write_bytes((json.dumps(nb, ensure_ascii=False, indent=1) + '\n').encode('utf-8'))
sys.stdout.reconfigure(encoding='utf-8')
print(f'Đã ghi {OUT} ({len(cells)} ô)')
