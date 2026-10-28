// Thế giới voxel: ghép các khung 256x208 đang cuộn thành một dải màn chơi liên tục trong toạ độ thế giới.
//
// Nền (cảnh trang trí + khối) được nhớ theo cột thế giới nên phần đã đi qua vẫn còn đó; mỗi mảnh 64 cột
// chỉ dựng lại lưới khi nội dung đổi. Sprite thì dựng lại mỗi khung. Mỗi instance là một dải điểm ảnh
// cùng màu trên một hàng (nén RLE) được đùn theo trục z.
import * as THREE from 'three';
import { FAR_INDEX_COUNT, STRIDE, makeBoxGeometry, makeVoxelDepthMaterial, makeVoxelMaterial } from './materials.js';
import { H, W } from './net.js';

const CW = 64;                        // bề rộng một mảnh (cột)
export const FOUNDATION = 24;         // phần móng kéo dài dưới hàng đất cuối cùng
const SETTLE = 0.45;                  // sau từng này giây hiệu ứng "mọc" đã xong
const BORN = -1000;                   // thời điểm sinh của voxel đã ổn định
const UNSEEN = 255;                   // chỉ số màu của điểm ảnh nền chưa từng nhìn thấy
const FAR = 430;                      // mảnh xa hơn từng này so với điểm nhìn: chỉ vẽ mặt trước/trên/hông
const REMESH_BUDGET = 8;              // số mảnh dựng lại tối đa mỗi khung khi đổi hướng nhìn
const [EMPTY, SCENERY, SOLID, GLOW, COIN, THIN, PIPE, SPRITE] = [0, 1, 2, 3, 4, 5, 6, 7];
const GROUND_ROW = H - 32;            // hai hàng ô cuối là mặt đất: kéo sâu ra sau làm thềm cho cảnh nền

const srgb = (v) => (v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4));
const smooth = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };

// cảnh nền thấp (đồi, bụi, lâu đài) đứng ngay sau làn chạy; mây ở tít phía sau để có thị sai
const SCENERY_Z = new Float32Array(H);
for (let y = 0; y < H; y++) SCENERY_Z[y] = -14.5 - 44 * smooth(118, 142, H - 1 - y);

class Batch {
  constructor(world, culled) {
    this.world = world;
    this.cap = 0;
    this.mesh = new THREE.Mesh(new THREE.BufferGeometry(), world.material);
    this.mesh.customDepthMaterial = world.depthMaterial;
    this.mesh.castShadow = this.mesh.receiveShadow = true;
    this.mesh.frustumCulled = culled;
    this.mesh.visible = false;
    world.group.add(this.mesh);
  }

  upload(src, n) {
    if (n === 0) { this.mesh.visible = false; if (this.cap) this.mesh.geometry.instanceCount = 0; return; }
    if (n > this.cap) {
      this.cap = Math.max(256, 1 << Math.ceil(Math.log2(n * 1.3)));
      const base = this.world.box;
      const geo = new THREE.InstancedBufferGeometry();
      geo.index = this.world.side > 0 ? base.indexRight : base.indexLeft;
      geo.setAttribute('position', base.position);
      geo.setAttribute('normal', base.normal);
      this.array = new Float32Array(this.cap * STRIDE);
      this.buffer = new THREE.InstancedInterleavedBuffer(this.array, STRIDE).setUsage(THREE.DynamicDrawUsage);
      geo.setAttribute('aBox', new THREE.InterleavedBufferAttribute(this.buffer, 4, 0));
      geo.setAttribute('aCol', new THREE.InterleavedBufferAttribute(this.buffer, 4, 4));
      geo.setAttribute('aFx', new THREE.InterleavedBufferAttribute(this.buffer, 4, 8));
      geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(CW / 2, 90, -16), 170);
      this.mesh.geometry.dispose();
      this.mesh.geometry = geo;
    }
    this.array.set(src.subarray(0, n * STRIDE));
    this.buffer.clearUpdateRanges();
    this.buffer.addUpdateRange(0, n * STRIDE);
    this.buffer.needsUpdate = true;
    this.mesh.geometry.instanceCount = n;
    this.mesh.visible = n > 0;
  }

  dispose() {
    this.world.group.remove(this.mesh);
    this.mesh.geometry.dispose();
  }
}

class Chunk {
  constructor(world, index) {
    this.color = new Uint8Array(CW * H).fill(UNSEEN);
    this.meta = new Uint8Array(CW * H);
    this.birth = new Float32Array(CW).fill(-1);
    this.dirty = false;
    this.side = world.side;                // hướng nhìn lúc dựng lưới (quyết định thứ tự vẽ các dải)
    this.settleAt = Infinity;
    this.batch = new Batch(world, true);
    this.batch.mesh.position.x = index * CW;
  }
}

export class World {
  constructor(scene) {
    this.group = new THREE.Group();
    scene.add(this.group);
    this.box = makeBoxGeometry();
    this.material = makeVoxelMaterial();
    this.depthMaterial = makeVoxelDepthMaterial();
    this.chunks = new Map();
    this.side = 1;                         // +1: máy quay lệch phải (thấy mặt +x của khối), -1: lệch trái
    this.sprites = new Batch(this, false);
    this.epoch = -1;
    this.frame = null;
    this.spritesDirty = false;
    this.voxels = 0;

    this.palette = new Uint8Array(192);
    this.lin = new Float32Array(192);           // bảng màu ở không gian tuyến tính
    this.relief = new Float32Array(64);         // viền tối lõm vào, màu sáng nhô ra: tạo chạm nổi
    this.scratch = new Float32Array(CW * (H + FOUNDATION) * STRIDE);
    this.spriteScratch = new Float32Array(W * H * STRIDE);
    this.eff = new Float32Array(CW);
    this.runX = new Uint8Array(CW);
    this.runW = new Uint8Array(CW);
  }

  clear() {
    for (const ch of this.chunks.values()) ch.batch.dispose();
    this.chunks.clear();
  }

  _palette(pal) {
    let same = true;
    for (let i = 0; i < 192; i++) if (pal[i] !== this.palette[i]) { same = false; break; }
    if (same) return;
    this.palette.set(pal);
    for (let i = 0; i < 64; i++) {
      const r = srgb(pal[i * 3] / 255), g = srgb(pal[i * 3 + 1] / 255), b = srgb(pal[i * 3 + 2] / 255);
      this.lin[i * 3] = r; this.lin[i * 3 + 1] = g; this.lin[i * 3 + 2] = b;
      const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      this.relief[i] = lum < 0.02 ? -1.3 : (lum > 0.72 ? 0.7 : 0);
    }
    for (const ch of this.chunks.values()) ch.dirty = true;
  }

  // ------------------------------------------------------------------ nhận một khung từ emulator
  ingest(frame, now) {
    const { head, color, meta } = frame;
    const fresh = head.epoch !== this.epoch;
    if (fresh) { this.clear(); this.epoch = head.epoch; }
    this._palette(frame.palette);
    const scroll = head.scroll;
    for (let sx = 0; sx < W;) {
      const wx = scroll + sx, ci = Math.floor(wx / CW), cx = wx - ci * CW, n = Math.min(CW - cx, W - sx);
      let ch = this.chunks.get(ci);
      if (!ch) this.chunks.set(ci, ch = new Chunk(this, ci));
      let dirty = false;
      const cm = ch.meta, cc = ch.color, left = cx === 0 ? this.chunks.get(ci - 1) : null;
      for (let y = 0; y < H; y++) {
        let fo = y * W + sx, co = y * CW + cx;
        for (let i = 0; i < n; i++, fo++, co++) {
          const m = meta[fo];
          if ((m & 7) === SPRITE) {
            // nền sau lưng sprite không nhìn thấy được: giữ giá trị cũ; nếu chưa từng thấy thì kéo dài cảnh
            // trang trí ở ngay bên trái (đồi, bụi) để không thủng một lỗ hình nhân vật
            if (cc[co] === UNSEEN) {
              const lm = i || cx ? cm[co - 1] : (left ? left.meta[co + CW - 1] : 0);
              if ((lm & 7) === SCENERY) {
                cm[co] = lm; cc[co] = i || cx ? cc[co - 1] : left.color[co + CW - 1]; dirty = true;
              }
            }
            continue;
          }
          const c = color[fo];
          if (cm[co] !== m || cc[co] !== c) { cm[co] = m; cc[co] = c; dirty = true; }
        }
      }
      for (let i = 0; i < n; i++) {
        if (ch.birth[cx + i] < 0) {
          // màn hình đầu ván quét từ trái sang phải; về sau cột mới lộ ra ở mép phải thì mọc ngay
          ch.birth[cx + i] = now + (fresh ? (sx + i) * 0.0026 : 0);
          ch.settleAt = Math.max(ch.settleAt === Infinity ? 0 : ch.settleAt, ch.birth[cx + i] + SETTLE);
          dirty = true;
        }
      }
      if (dirty) ch.dirty = true;
      sx += n;
    }
    this.frame = frame;
    this.spritesDirty = true;
  }

  // ------------------------------------------------------------------ dựng lưới
  /** side: dấu phương vị máy quay; focusX: hoành độ điểm nhìn (để chọn mức chi tiết theo khoảng cách). */
  update(now, side = 1, focusX = 0) {
    this.side = side;
    const index = side > 0 ? this.box.indexRight : this.box.indexLeft;
    let total = 0, budget = REMESH_BUDGET;
    for (const [ci, ch] of this.chunks) {
      // các dải phải được vẽ từ phía gần máy quay ra xa để GPU loại sớm những mặt hông bị che
      const stale = ch.side !== side && budget > 0;
      if (ch.dirty || now >= ch.settleAt || stale) { this._mesh(ch, now); if (stale) budget--; }
      const geo = ch.batch.mesh.geometry;
      if (geo.index && ch.side === side) {
        if (geo.index !== index) geo.index = index;
        const far = Math.abs(ci * CW + CW / 2 - focusX) > FAR;
        geo.setDrawRange(0, far ? FAR_INDEX_COUNT : Infinity);
      }
      total += geo.instanceCount || 0;
    }
    if (this.spritesDirty && this.frame) { this._meshSprites(this.frame); this.spritesDirty = false; }
    this.voxels = total + (this.sprites.mesh.geometry.instanceCount || 0);
  }

  _put(a, k, x, y, zb, zf, w, ci, shade, birth, glow, spin, side = 0) {
    const lin = this.lin, o = k * STRIDE, p = ci * 3;
    a[o] = x; a[o + 1] = y; a[o + 2] = (zb + zf) * 0.5; a[o + 3] = w;
    a[o + 4] = lin[p] * shade; a[o + 5] = lin[p + 1] * shade; a[o + 6] = lin[p + 2] * shade; a[o + 7] = zf - zb;
    a[o + 8] = birth; a[o + 9] = glow; a[o + 10] = spin; a[o + 11] = side;
  }

  /** Màu mặt hông cho một dải viền tối: lấy màu điểm ảnh sáng kề bên (trái trước, rồi phải) cùng lớp. */
  _side(meta, color, row, x, w, lo, hi) {
    const cls = meta[row + x] & 7, relief = this.relief, pal = this.palette;
    for (const n of [x - 1, x + w]) {
      if (n < lo || n >= hi || (meta[row + n] & 7) !== cls) continue;
      const c = color[row + n];
      if (relief[c] < 0) continue;
      return pal[c * 3] + pal[c * 3 + 1] * 256 + pal[c * 3 + 2] * 65536 || 1;
    }
    return 0;
  }

  _mesh(ch, now) {
    const a = this.scratch, cm = ch.meta, cc = ch.color, eff = this.eff, relief = this.relief;
    let pending = false;
    for (let x = 0; x < CW; x++) {
      const b = ch.birth[x];
      if (b >= 0 && now - b < SETTLE) { eff[x] = b; pending = true; } else eff[x] = BORN;
    }
    const runX = this.runX, runW = this.runW, rtl = this.side > 0;
    let k = 0;
    for (let y = 0; y < H; y++) {
      const row = y * CW, yUp = H - 1 - y;
      let n = 0;
      for (let x = 0; x < CW;) {
        const m = cm[row + x], cls = m & 7;
        if (cls === EMPTY) { x++; continue; }
        const c = cc[row + x], b = eff[x];
        const end = cls === COIN ? (x & ~15) + 16 : CW;   // đồng xu quay quanh tâm ô nên không gộp qua ô khác
        let w = 1;
        while (x + w < end && cm[row + x + w] === m && cc[row + x + w] === c && eff[x + w] === b) w++;
        runX[n] = x; runW[n++] = w;
        x += w;
      }
      for (let j = 0; j < n; j++) {
        const i = rtl ? n - 1 - j : j, x = runX[i], w = runW[i];
        const cls = cm[row + x] & 7, c = cc[row + x], b = eff[x], r = relief[c];
        const side = r < 0 ? this._side(cm, cc, row, x, w, 0, CW) : 0;
        if (cls === SCENERY) {
          const z = SCENERY_Z[y];
          this._put(a, k++, x, yUp, z - 1.5, z + 1.5 + r * 0.6, w, c, 1, b, 0, 0, side);
        } else if (cls === COIN) {
          this._put(a, k++, x, yUp, -1.6, 1.6, w, c, 1, b, 0.9, 1, side);
        } else if (cls === THIN) {
          this._put(a, k++, x, yUp, -4, -1.5 + r * 0.4, w, c, 1, b, 0, 0, side);
        } else if (cls === PIPE) {
          this._put(a, k++, x, yUp, -10, 10 + r, w, c, 1, b, 0, 0, side);
        } else {
          this._put(a, k++, x, yUp, y >= GROUND_ROW ? -22 : -8, 8 + r, w, c, 1, b, cls === GLOW ? 0.2 : 0, 0, side);
        }
      }
    }
    // móng: lặp lại ô đất cuối cùng xuống dưới, tối dần, để dải màn chơi có bề dày như một lát cắt
    const last = (H - 1) * CW;
    for (let d = 0; d < FOUNDATION; d++) {
      const row = (H - 16 + (d & 15)) * CW, shade = 0.78 - 0.5 * (d / FOUNDATION);
      let n = 0;
      for (let x = 0; x < CW;) {
        const m = cm[row + x], cls = m & 7, base = cm[last + x] & 7;
        if (!(base === SOLID || base === PIPE || base === GLOW) || cls === EMPTY || cls === SCENERY) { x++; continue; }
        const c = cc[row + x], b = eff[x];
        let w = 1;
        while (x + w < CW && cm[row + x + w] === m && cc[row + x + w] === c && eff[x + w] === b &&
               (cm[last + x + w] & 7) === base) w++;
        runX[n] = x; runW[n++] = w;
        x += w;
      }
      for (let j = 0; j < n; j++) {
        const i = rtl ? n - 1 - j : j, x = runX[i], c = cc[row + x];
        this._put(a, k++, x, -1 - d, -22, 8 + relief[c], runW[i], c, shade, eff[x], 0, 0);
      }
    }
    ch.side = this.side;
    ch.batch.upload(a, k);
    ch.dirty = false;
    ch.settleAt = pending ? now + 0.12 : Infinity;      // dựng lại lần nữa khi mọc xong để gộp dải cho gọn
  }

  _spriteSide(meta, color, row, x, w) {
    const relief = this.relief, pal = this.palette;
    for (const n of [x - 1, x + w]) {
      if (n < 0 || n >= W || (meta[row + n] & 7) !== SPRITE) continue;
      const c = color[row + n];
      if (relief[c] < 0) continue;
      return pal[c * 3] + pal[c * 3 + 1] * 256 + pal[c * 3 + 2] * 65536 || 1;
    }
    return 0;
  }

  _meshSprites(frame) {
    const a = this.spriteScratch, { color, meta } = frame, relief = this.relief;
    let k = 0;
    for (let y = 0; y < H; y++) {
      const row = y * W, yUp = H - 1 - y;
      for (let x = 0; x < W;) {
        const m = meta[row + x];
        if ((m & 7) !== SPRITE) { x++; continue; }
        const c = color[row + x];
        let w = 1;
        while (x + w < W && meta[row + x + w] === m && color[row + x + w] === c) w++;
        const half = 2.2 + (m >> 3) * 1.15;             // càng xa viền càng dày: nhân vật phồng như đồ chơi
        const side = relief[c] < 0 ? this._spriteSide(meta, color, row, x, w) : 0;
        this._put(a, k++, x, yUp, -half, half + relief[c] * 0.7, w, c, 1, BORN, 0, 0, side);
        x += w;
      }
    }
    this.sprites.mesh.position.x = frame.head.scroll;
    this.sprites.upload(a, k);
  }
}
