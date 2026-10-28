"""Cài `nes-py` + `gym-super-mario-bros` trên Windows KHÔNG cần Visual Studio.

Vấn đề: nes-py chỉ phát hành mã nguồn (sdist); lõi giả lập NES viết bằng C++ nên
`pip install nes-py` cần trình biên dịch C++ (MSVC). Máy không có MSVC.

Giải pháp (script này tự động hoá):
  1. Tải sdist nes-py 8.2.1 và wheel gym-super-mario-bros 7.4.0 vào thư mục tạm.
  2. Biên dịch lõi C++ thành `lib_nes_env.dll` bằng `ziglang` (trình biên dịch C++
     cài được qua pip, không cần quyền admin).
  3. Chép phần Python của nes_py + DLL vào site-packages của môi trường hiện tại.
  4. Vá hai vấn đề tương thích với NumPy >= 2 (phép toán số học trên uint8 bị tràn).
  5. Cài gym-super-mario-bros (--no-deps).

Chạy:  .venv\\Scripts\\python.exe scripts\\setup_nes_py_windows.py
"""
from __future__ import annotations

import glob
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

NES_PY_VERSION = "8.2.1"
SMB_VERSION = "7.4.0"


def run(cmd: list[str], **kw) -> None:
    print("$", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, **kw)


def site_packages() -> Path:
    import sysconfig

    return Path(sysconfig.get_paths()["purelib"])


def main() -> None:
    py = sys.executable
    work = Path(tempfile.mkdtemp(prefix="nespy_build_"))
    dl_nes, dl_smb, src = work / "dl_nes", work / "dl_smb", work / "src"
    for d in (dl_nes, dl_smb, src):
        d.mkdir()

    # 0. công cụ biên dịch + phụ thuộc thuần Python
    run([py, "-m", "pip", "install", "-q", "ziglang", "gym==0.26.2", "pyglet", "tqdm"])

    # 1. tải mã nguồn (không thực thi setup.py)
    run([py, "-m", "pip", "download", "-q", f"nes-py=={NES_PY_VERSION}", "--no-deps",
         "--no-binary=:all:", "-d", str(dl_nes)])
    run([py, "-m", "pip", "download", "-q", f"gym-super-mario-bros=={SMB_VERSION}",
         "--no-deps", "-d", str(dl_smb)])
    with tarfile.open(next(dl_nes.glob("*.tar.gz"))) as tf:
        tf.extractall(src)  # noqa: S202 - gói từ PyPI chính thức, giải nén vào thư mục riêng
    pkg_root = next(src.glob("nes_py-*"))

    # 2. biên dịch lõi C++ -> lib_nes_env.dll
    sources = glob.glob(str(pkg_root / "nes_py/nes/src/*.cpp")) + \
        glob.glob(str(pkg_root / "nes_py/nes/src/mappers/*.cpp"))
    out_dll = work / "lib_nes_env.dll"
    run([py, "-m", "ziglang", "c++", "-target", "x86_64-windows-gnu", "-shared", "-O3",
         "-std=c++14", "-w", f"-I{pkg_root / 'nes_py/nes/include'}", *sources, "-o", str(out_dll)])

    # 3. cài phần Python + DLL
    sp = site_packages()
    dest = sp / "nes_py"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir()
    for name in ("__init__.py", "_image_viewer.py", "_rom.py", "nes_env.py"):
        shutil.copy(pkg_root / "nes_py" / name, dest / name)
    for sub in ("wrappers", "app"):
        shutil.copytree(pkg_root / "nes_py" / sub, dest / sub)
    shutil.copy(out_dll, dest / "lib_nes_env.dll")

    # 4. vá tương thích NumPy 2 (dùng chung với Colab)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from patch_nes_py_numpy2 import patch_nes_py

    patch_nes_py(dest)

    # 5. cài gym-super-mario-bros
    run([py, "-m", "pip", "install", "-q", "--no-deps", str(next(dl_smb.glob("*.whl")))])

    shutil.rmtree(work, ignore_errors=True)
    print("\nXong. Thử: python scripts/smoke_test_env.py")


if __name__ == "__main__":
    main()
