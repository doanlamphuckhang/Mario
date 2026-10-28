"""Tách khung hình NES thành các lớp có chiều sâu (cảnh nền / khối / nhân vật) cho viewer 3D.

nes-py chỉ cho ta ảnh RGB 240x256 và 2 KB RAM của CPU, không cho đọc PPU. Chiều sâu được dựng lại từ:
  * $0200-$02FF  bản sao OAM của game (64 sprite: y, tile, thuộc tính, x). Hình 8x8 của từng tile lấy
                 từ CHR-ROM nên mặt nạ sprite chính xác tới từng điểm ảnh.
  * $0500-$069F  "block buffer": 2 trang x 13 hàng x 16 cột metatile mà Mario va chạm được. Hình dạng
                 từng metatile lấy từ bảng metatile trong PRG-ROM + CHR nền.
  * $071A/$071C  toạ độ cuộn của mép trái màn hình.
Điểm ảnh còn lại khác màu nền là cảnh trang trí (mây, đồi, bụi cây, lâu đài).

Game chạy toàn bộ logic trong ngắt NMI rồi mới vẽ ở khung kế tiếp, nên RAM luôn đi TRƯỚC ảnh đúng
một khung: `extract` phải nhận RAM của khung trước (xem `RAM_LAG`).

File này không import torch.
"""
from __future__ import annotations

import cv2
import numpy as np

HUD_ROWS = 32            # 32 dòng trên cùng là bảng điểm, viewer vẽ lại bằng HTML
WIDTH, HEIGHT = 256, 208  # vùng chơi: 16 x 13 ô 16x16
RAM_LAG = 1              # số khung RAM đi trước ảnh

# lớp của mỗi điểm ảnh (3 bit thấp của byte meta); 5 bit cao là tham số riêng của lớp
EMPTY, SCENERY, SOLID, GLOW, COIN, THIN, PIPE, SPRITE = range(8)

_PRG_SIZE, _CHR_SIZE = 0x8000, 0x2000
# 3 metatile đầu của Palette0_MTiles (trống, đen, bụi cây trái); ngay trước nó là bảng con trỏ 4 nhóm
_MTILE_SIGNATURE = bytes([0x24, 0x24, 0x24, 0x24, 0x27, 0x27, 0x27, 0x27, 0x24, 0x24, 0x24, 0x35])
_BLOCK_BUFFER = (0x0500, 0x05D0)   # trang chẵn / trang lẻ


def _decode_chr(chr_rom: np.ndarray) -> np.ndarray:
    """CHR-ROM 8 KB -> (512, 8, 8) chỉ số màu 0..3 (0 = trong suốt)."""
    tiles = chr_rom.reshape(512, 16)
    lo = np.unpackbits(tiles[:, :8, None], axis=2)
    hi = np.unpackbits(tiles[:, 8:, None], axis=2)
    return lo | (hi << 1)


def _metatile_tables(prg: np.ndarray, bg_opaque: np.ndarray) -> tuple[np.ndarray, bool]:
    """Mặt nạ đục 16x16 của 256 metatile. Trả về (mask, found)."""
    mask = np.zeros((256, 16, 16), bool)
    base = prg.tobytes().find(_MTILE_SIGNATURE)
    if base < 8:
        mask[1:] = True     # ROM lạ: coi mọi ô khác 0 là khối đặc, màu nền sẽ lọc bớt
        return mask, False
    ptr = [int(prg[base - 4 + g]) << 8 | int(prg[base - 8 + g]) for g in range(4)]
    counts = [(ptr[g + 1] - ptr[g]) // 4 for g in range(3)] + [6]
    for g in range(4):
        off = ptr[g] - 0x8000
        for k in range(min(counts[g], 64)):
            tl, bl, tr, br = (int(v) for v in prg[off + 4 * k: off + 4 * k + 4])
            m = mask[g * 64 + k]
            m[:8, :8], m[8:, :8], m[:8, 8:], m[8:, 8:] = bg_opaque[tl], bg_opaque[bl], bg_opaque[tr], bg_opaque[br]
    return mask, True


def _metatile_kinds() -> np.ndarray:
    kind = np.full(256, SOLID, np.uint8)
    kind[0] = EMPTY
    kind[[0xC0, 0xC1, 0xC5]] = GLOW               # khối "?", rìu cuối lâu đài
    kind[[0xC2, 0xC3]] = COIN
    kind[[0x24, 0x25, 0x26, 0x6D]] = THIN         # cột cờ, dây leo
    kind[0x10:0x16] = PIPE
    kind[0x1C:0x22] = PIPE
    kind[[0x6B, 0x6C]] = PIPE
    return kind


class SceneExtractor:
    """Biến (ảnh, RAM) của emulator thành 2 mặt phẳng 208x256: chỉ số màu và byte meta (lớp + tham số)."""

    def __init__(self, rom_path: str) -> None:
        rom = np.fromfile(rom_path, dtype=np.uint8)
        prg = rom[16:16 + _PRG_SIZE]
        chr_px = _decode_chr(rom[16 + _PRG_SIZE:16 + _PRG_SIZE + _CHR_SIZE])
        opaque = chr_px != 0
        spr = opaque[:256]                                   # game dùng bảng mẫu 0 cho sprite, 1 cho nền
        # chỉ số = bit lật của thuộc tính OAM (bit 6 lật ngang, bit 7 lật dọc)
        self._spr_mask = np.stack([spr, spr[:, :, ::-1], spr[:, ::-1, :], spr[:, ::-1, ::-1]])
        self._mt_mask, self.exact_tiles = _metatile_tables(prg, opaque[256:])
        self._mt_kind = _metatile_kinds()

        self.palette = np.zeros((64, 3), np.uint8)           # bảng màu toàn cục, lớn dần khi gặp màu mới
        self.n_colors = 0
        self._lut = np.full(1 << 24, 255, np.uint8)
        self.backdrop = 0

        self._rows = (np.arange(13) * 16)[:, None]
        self._cols = np.arange(17)
        self._canvas = np.zeros((256 + 8, 256 + 8), bool)    # có lề để sprite tràn mép không cần cắt
        self._solid_pad = np.zeros_like(self._canvas)
        self._lines = np.zeros(256 + 8, np.uint8)

    # ------------------------------------------------------------------ màu
    def _color_index(self, rgb: np.ndarray) -> np.ndarray:
        key = (rgb[..., 0].astype(np.uint32) << 16) | (rgb[..., 1].astype(np.uint32) << 8) | rgb[..., 2]
        idx = self._lut[key]
        if (idx == 255).any():
            for k in np.unique(key[idx == 255]):
                n = min(self.n_colors, 63)
                self.palette[n] = (k >> 16) & 255, (k >> 8) & 255, k & 255
                self._lut[k] = n
                self.n_colors = n + 1
            idx = self._lut[key]
        return idx

    # --------------------------------------------------------------- sprite
    def _sprite_mask(self, ram: np.ndarray, solid: np.ndarray) -> np.ndarray:
        canvas, lines = self._canvas, self._lines
        canvas[:] = False
        lines[:] = 0
        solid_ready = False
        oam = ram[0x200:0x300]
        for i in range(0, 256, 4):
            y = int(oam[i])
            if y >= 0xEF:                                    # game đẩy sprite không dùng xuống dưới màn hình
                continue
            tile, attr, x = int(oam[i + 1]), int(oam[i + 2]), int(oam[i + 3])
            top = y + 1                                      # PPU vẽ sprite trễ 1 dòng quét
            m = self._spr_mask[attr >> 6, tile]
            visible = lines[top:top + 8] < 8                 # PPU chỉ vẽ 8 sprite đầu trên mỗi dòng quét
            lines[top:top + 8] += 1
            m = m & visible[:, None]
            if attr & 0x20:                                  # sprite nằm sau nền (chui ống, nấm trồi lên)
                if not solid_ready:
                    self._solid_pad[HUD_ROWS:240, :WIDTH] = solid
                    solid_ready = True
                m = m & ~self._solid_pad[top:top + 8, x:x + 8]
            canvas[top:top + 8, x:x + 8] |= m
        return canvas[HUD_ROWS:240, :WIDTH]

    # ------------------------------------------------------------------ API
    def extract(self, screen: np.ndarray, ram: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        """screen: ảnh RGB 240x256 của khung t; ram: RAM của khung t - RAM_LAG.

        Trả về (color, meta, state): color/meta là uint8 (208, 256); state gồm toạ độ cuộn, màu nền,
        vị trí Mario trên màn hình.
        """
        color = self._color_index(screen[HUD_ROWS:])

        scroll = int(ram[0x71A]) * 256 + int(ram[0x71C])
        fine = scroll & 15
        cols = (scroll >> 4) + self._cols
        base = np.where((cols >> 4) & 1, _BLOCK_BUFFER[1], _BLOCK_BUFFER[0])
        cells = np.asarray(ram[base[None, :] + self._rows + (cols & 15)[None, :]])          # (13, 17)
        solid = self._mt_mask[cells].transpose(0, 2, 1, 3).reshape(HEIGHT, 272)[:, fine:fine + WIDTH]
        kind = np.repeat(np.repeat(self._mt_kind[cells], 16, 0), 16, 1)[:, fine:fine + WIDTH]

        sprite = self._sprite_mask(ram, solid)

        # màu nền = màu phổ biến nhất ở các ô trống; chỉ đổi khi chiếm ưu thế rõ (tránh nhấp nháy)
        free = color[~sprite & (kind == EMPTY)]
        if free.size:
            counts = np.bincount(free, minlength=64)
            top = int(counts.argmax())
            if top != self.backdrop and counts[top] > 0.3 * free.size:
                self.backdrop = top
        is_backdrop = color == self.backdrop
        dark = int(self.palette[self.backdrop].sum()) < 60
        if not dark:   # trên nền sáng, màu nền không bao giờ là màu đục của tile hay sprite
            sprite = sprite & ~is_backdrop
            solid = solid & ~is_backdrop
        solid = solid & ~sprite

        meta = np.where(is_backdrop, EMPTY, SCENERY).astype(np.uint8)
        meta[solid] = kind[solid]
        # sprite: tham số = độ phồng theo khoảng cách tới viền (0..3) để nhân vật có khối
        depth = cv2.distanceTransform(np.ascontiguousarray(sprite, dtype=np.uint8), cv2.DIST_L1, 3)
        meta[sprite] = SPRITE | ((np.clip(depth[sprite], 1, 4).astype(np.uint8) - 1) << 3)

        state = {
            "scroll": scroll,
            "backdrop": self.backdrop,
            "dark": dark,
            "area": int(ram[0x750]),
            "mx": int(ram[0x6D]) * 256 + int(ram[0x86]) - scroll,
            "my": int(ram[0xCE]) - HUD_ROWS if int(ram[0xB5]) == 1 else (-64 if int(ram[0xB5]) == 0 else 240),
        }
        return color, meta, state
