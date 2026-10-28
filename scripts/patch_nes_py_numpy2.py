"""Vá `nes-py` 8.2.1 để chạy được với NumPy >= 2 (dùng chung cho Windows và Colab/Linux).

`nes-py` viết cho NumPy 1.x. Với NumPy 2, phép toán `uint8 * int` không còn tự nới kiểu nên
ném `OverflowError: Python integer 1024 out of bounds for uint8`. Hai chỗ cần vá:
  1. `_rom.py`   : ép `int` khi tính kích thước PRG/CHR ROM từ header.
  2. `nes_env.py`: mảng RAM trả về `int` của Python khi đọc một phần tử (gym-super-mario-bros
                   có ~23 chỗ kiểu `ram[0x6d] * 0x100 + ram[0x86]`).

Chạy:  python scripts/patch_nes_py_numpy2.py     (idempotent: chạy lại không sao)
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

RAM_CLASS = '''

class _IntRam(np.ndarray):
    """Mảng RAM: đọc 1 phần tử trả về int của Python (tránh tràn uint8 trong NumPy 2)."""

    def __getitem__(self, idx):
        v = super().__getitem__(idx)
        return int(v) if isinstance(v, np.integer) else v
'''


def find_nes_py_dir() -> Path:
    spec = importlib.util.find_spec("nes_py")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("Chưa cài nes_py. Hãy cài nes-py==8.2.1 trước.")
    return Path(list(spec.submodule_search_locations)[0])


def patch_nes_py(pkg_dir: Path | None = None) -> list[str]:
    """Áp dụng các bản vá; trả về danh sách mô tả những gì đã sửa (rỗng nếu đã vá từ trước)."""
    pkg_dir = Path(pkg_dir) if pkg_dir else find_nes_py_dir()
    changes: list[str] = []

    rom = pkg_dir / "_rom.py"
    s = rom.read_text(encoding="utf-8")
    new = s.replace("return 16 * self.header[4]", "return 16 * int(self.header[4])")
    new = new.replace("return 8 * self.header[5]", "return 8 * int(self.header[5])")
    if new != s:
        rom.write_text(new, encoding="utf-8")
        changes.append("_rom.py: ép int khi đọc kích thước ROM")

    env = pkg_dir / "nes_env.py"
    s = env.read_text(encoding="utf-8")
    if "_IntRam" not in s:
        marker = "\n\n# the path to the directory this file is in"
        assert marker in s, "nes_env.py khác phiên bản 8.2.1, không tìm thấy vị trí chèn"
        s = s.replace(marker, RAM_CLASS + marker, 1)
        old_ret = "        return np.frombuffer(buffer_, dtype='uint8')\n\n    def _controller_buffer"
        assert old_ret in s, "nes_env.py khác phiên bản 8.2.1, không tìm thấy _ram_buffer"
        s = s.replace(old_ret, "        return np.frombuffer(buffer_, dtype='uint8').view(_IntRam)\n\n    def _controller_buffer", 1)
        env.write_text(s, encoding="utf-8")
        changes.append("nes_env.py: RAM trả về int (lớp _IntRam)")
    return changes


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    done = patch_nes_py()
    print("Đã vá:" if done else "Đã vá từ trước, không có gì để sửa.", *done, sep="\n  - ")
