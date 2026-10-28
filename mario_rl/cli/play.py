"""Viewer 3D: xem agent (hoặc tự chơi) Super Mario Bros dưới dạng diorama voxel trong trình duyệt.

Chỉ suy luận + chạy emulator, KHÔNG huấn luyện. Ví dụ:
    python play3d.py                                             # tự tìm checkpoint trong runs/, không có thì phát bản ghi mẫu
    python play3d.py --ckpt runs/ppo_1-1/ckpt_final.pt           # xem agent PPO chơi
    python play3d.py --stage 1-2 --mode human                    # tự chơi bằng bàn phím
"""
import argparse
import sys
import threading
import webbrowser
from pathlib import Path

from viewer3d.server import serve
from viewer3d.session import MODES, SPEEDS, Session

ROOT = Path(__file__).resolve().parents[2]


def find_checkpoints(explicit: list[str]) -> list[Path]:
    """Checkpoint chỉ định trước, sau đó là mọi ckpt_*.pt trong runs/ (mới nhất trước)."""
    found = [Path(p) for p in explicit]
    for p in found:
        if not p.is_file():
            sys.exit(f"Không thấy checkpoint: {p}")
    auto = sorted((p for d in ("runs", "checkpoints", "results") for p in (ROOT / d).glob("**/*.pt")),
                  key=lambda p: (p.name != "ckpt_final.pt", p.name != "ckpt_best.pt", -p.stat().st_mtime))
    return found + [p for p in auto if p.resolve() not in {f.resolve() for f in found}]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ckpt", action="append", default=[], help="checkpoint .pt (có thể lặp lại cờ này)")
    ap.add_argument("--stage", default="1-1", help="ví dụ 1-1, 1-2, 4-1")
    ap.add_argument("--mode", choices=MODES, default=None, help="mặc định: agent nếu có checkpoint, không thì demo")
    ap.add_argument("--stochastic", action="store_true", help="lấy mẫu từ π thay vì argmax")
    ap.add_argument("--speed", type=float, choices=SPEEDS, default=1.0)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="không tự mở trình duyệt")
    a = ap.parse_args()

    session = Session(a.stage, find_checkpoints(a.ckpt), a.mode, a.speed, not a.stochastic)
    httpd = serve(session, a.host, a.port)
    session.start()
    url = f"http://{'localhost' if a.host in ('127.0.0.1', '0.0.0.0') else a.host}:{a.port}/"
    print(f"Viewer 3D đang chạy tại {url}  (chế độ: {session.mode}, màn {session.stage}). Ctrl+C để thoát.", flush=True)
    if not a.no_browser:
        threading.Timer(0.6, webbrowser.open, args=(url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        session.stop()
        httpd.server_close()


if __name__ == '__main__':
    main()
