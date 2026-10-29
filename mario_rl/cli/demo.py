"""Quay video demo nhiều màn: mỗi màn vài ván, kèm một video tổng ghép tất cả.

Ví dụ:
    python -m mario_rl.cli.demo --item 1-1=runs/ppo_1-1/ckpt_final.pt --item 1-2=runs/ppo_1-2/ckpt_final.pt \\
        --episodes 3 --out results/demo_all
"""
import argparse
import json
import sys
from pathlib import Path

import torch

from mario_rl.demo import MODES, record
from mario_rl.envs import parse_stages


def parse_item(text: str) -> tuple[str, Path]:
    stage, sep, ckpt = text.partition('=')
    if not sep or not ckpt:
        raise argparse.ArgumentTypeError('Dạng đúng: MÀN=CHECKPOINT, ví dụ 1-2=runs/ppo_1-2/ckpt_final.pt')
    (w, s), = parse_stages(stage)
    return f'{w}-{s}', Path(ckpt)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--item', type=parse_item, action='append', required=True, help='MÀN=CHECKPOINT, lặp lại được')
    p.add_argument('--episodes', type=int, default=2, help='số ván (video) mỗi màn')
    p.add_argument('--out', type=Path, required=True, help='thư mục mới; không ghi đè demo cũ')
    p.add_argument('--mode', choices=MODES, default='mix',
                   help='mix: ván 1 argmax, các ván sau lấy mẫu từ π (mỗi video một khác)')
    p.add_argument('--noop_max', type=int, default=30, help='xuất phát lệch ngẫu nhiên để các ván khác nhau')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max_steps', type=int, default=3000, help='giới hạn bước mỗi ván')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--no_combined', action='store_true', help='không ghép video tổng')
    p.add_argument('--combined_every', type=int, default=2, help='video tổng tua nhanh: giữ 1 khung mỗi N khung')
    return p


def main(argv=None):
    sys.stdout.reconfigure(encoding='utf-8')
    a = build_parser().parse_args(argv)
    for _, ckpt in a.item:
        if not ckpt.is_file():
            sys.exit(f'Không thấy checkpoint: {ckpt}')
    results = record(a.item, a.out, a.episodes, a.mode, a.noop_max, a.seed, a.max_steps,
                     torch.device(a.device), not a.no_combined, a.combined_every)
    for r in results:
        verdict = 'về đích' if r['flag'] else f"dừng ở x={r['max_x']}"
        print(f"{r['stage']} ván {r['episode']} ({r['policy']}): {verdict}, {r['steps']} bước -> {r['file']}")
    print(json.dumps({'out': str(a.out), 'videos': len(results)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
