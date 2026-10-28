// Lớp giao diện HTML phủ trên cảnh 3D: bảng điểm, bảng "bộ não agent", điều khiển, thanh tiến độ.
import { H, OBS, W } from './net.js';

const $ = (id) => document.getElementById(id);
const pad = (n, w) => String(n).padStart(w, '0');
const ESC = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ESC[c]);   // tên file/thư mục checkpoint đi vào innerHTML
const MODE_LABEL = { agent: 'AGENT PPO', demo: 'BẢN GHI MẪU', human: 'BẠN CHƠI', random: 'NGẪU NHIÊN', right: 'LUÔN PHẢI' };
const BUTTON_GLYPH = { right: '→', left: '←', down: '↓', up: '↑', NOOP: 'đứng' };
const KEY_BITS = {                      // bàn phím -> bit tay cầm NES
  ArrowRight: 128, KeyD: 128, ArrowLeft: 64, KeyA: 64, ArrowDown: 32, KeyS: 32, ArrowUp: 16, KeyW: 16,
  KeyX: 2, ShiftLeft: 2, ShiftRight: 2, KeyJ: 2, KeyZ: 1, Space: 1, KeyK: 1,
};
const HISTORY = 260;

function sparkline(canvas, values, { bars = false, color = '#8fd0ff' } = {}) {
  const ctx = canvas.getContext('2d'), w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (values.length < 2) return;
  let lo = Math.min(...values), hi = Math.max(...values);
  if (bars) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
  if (hi - lo < 1e-6) { hi += 0.5; lo -= 0.5; }
  const px = (i) => (i / (HISTORY - 1)) * w, py = (v) => h - 5 - ((v - lo) / (hi - lo)) * (h - 10);
  const x0 = HISTORY - values.length;
  if (bars) {
    const zero = py(0), bw = Math.max(1, w / HISTORY - 0.5);
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      ctx.fillStyle = v >= 0 ? '#7ee787' : '#ff7b72';
      ctx.fillRect(px(x0 + i), Math.min(zero, py(v)), bw, Math.max(1, Math.abs(py(v) - zero)));
    }
    return;
  }
  ctx.beginPath();
  values.forEach((v, i) => (i ? ctx.lineTo(px(x0 + i), py(v)) : ctx.moveTo(px(x0 + i), py(v))));
  ctx.lineWidth = 3; ctx.strokeStyle = color; ctx.lineJoin = 'round'; ctx.stroke();
  ctx.lineTo(w, h); ctx.lineTo(px(x0), h); ctx.closePath();
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, 'rgba(143, 208, 255, .38)'); grad.addColorStop(1, 'rgba(143, 208, 255, 0)');
  ctx.fillStyle = grad; ctx.fill();
}

class Confetti {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.parts = [];
  }

  burst() {
    const w = this.canvas.width = innerWidth, h = this.canvas.height = innerHeight;
    const colors = ['#e5352b', '#ffcf3f', '#4fc14d', '#6aa6ff', '#ffffff', '#ff9f43'];
    for (let i = 0; i < 220; i++) {
      const side = i % 2 ? 0.15 : 0.85, a = (side < 0.5 ? -0.25 : -0.75) * Math.PI + (Math.random() - 0.5) * 0.9;
      const v = 9 + Math.random() * 13;
      this.parts.push({ x: w * side, y: h * 0.72, vx: Math.cos(a) * v, vy: Math.sin(a) * v, s: 5 + Math.random() * 7,
        r: Math.random() * 6.28, vr: (Math.random() - 0.5) * 0.5, c: colors[i % colors.length], life: 1 });
    }
    if (!this.running) { this.running = true; requestAnimationFrame(() => this.tick()); }
  }

  tick() {
    const { ctx, canvas } = this;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    this.parts = this.parts.filter((p) => p.life > 0 && p.y < canvas.height + 40);
    for (const p of this.parts) {
      p.vy += 0.32; p.vx *= 0.992; p.x += p.vx; p.y += p.vy; p.r += p.vr; p.life -= 0.0045;
      ctx.save(); ctx.translate(p.x, p.y); ctx.rotate(p.r); ctx.globalAlpha = Math.min(1, p.life * 3);
      ctx.fillStyle = p.c; ctx.fillRect(-p.s / 2, -p.s / 4, p.s, p.s / 2); ctx.restore();
    }
    if (this.parts.length) requestAnimationFrame(() => this.tick());
    else { this.running = false; ctx.clearRect(0, 0, canvas.width, canvas.height); }
  }
}

export class UI {
  constructor({ send, stage }) {
    this.send = send;
    this.stage3d = stage;
    this.state = null;
    this.values = [];
    this.rewards = [];
    this.keys = 0;
    this.best = 0;
    this.confetti = new Confetti($('fx'));
    this.nes = $('nes').getContext('2d');
    this.nesImage = this.nes.createImageData(W, H);
    this.obs = $('obs').getContext('2d');
    this.obsImage = this.obs.createImageData(OBS, OBS);
    this.pal32 = new Uint32Array(64);
    this._bind();
  }

  // ------------------------------------------------------------------ sự kiện từ máy chủ
  hello(msg) {
    $('stage').innerHTML = msg.stages.map((s) => `<option>${s}</option>`).join('');
    $('speeds').innerHTML = msg.speeds.map((s) => `<button data-speed="${s}">${s}×</button>`).join('');
    $('ckpt').innerHTML = msg.ckpts.map((c, i) => `<option value="${i}">${esc(c.dir)}/${esc(c.name)}</option>`).join('');
    this.ckpts = msg.ckpts;
    this.setState(msg.state);
    $('overlay').classList.add('hidden');
  }

  setState(s) {
    const changed = !this.state || this.state.mode !== s.mode || this.state.stage !== s.stage;
    this.state = s;
    for (const b of document.querySelectorAll('#modes button')) {
      b.classList.toggle('on', b.dataset.mode === s.mode);
      b.disabled = (b.dataset.mode === 'agent' && !s.has_agent) || (b.dataset.mode === 'demo' && !s.has_demo);
      if (b.dataset.mode === 'agent') b.title = s.has_agent ? '' : 'Chưa có checkpoint: python play3d.py --ckpt <file .pt>';
      if (b.dataset.mode === 'demo') b.title = s.has_demo ? '' : `Chưa có bản ghi cho màn ${s.stage}: python scripts/make_demo_replay.py --stage ${s.stage}`;
    }
    for (const b of document.querySelectorAll('#speeds button')) b.classList.toggle('on', Number(b.dataset.speed) === s.speed);
    $('stage').value = s.stage;
    $('mode-badge').textContent = MODE_LABEL[s.mode] || s.mode;
    $('mode-badge').className = `badge ${s.mode}`;
    const agent = s.mode === 'agent';
    $('ckpt-row').hidden = !(agent && this.ckpts.length > 1);
    $('det-row').hidden = !agent;
    $('keys-help').hidden = s.mode !== 'human';
    $('policy-box').hidden = $('value-box').hidden = !agent;
    if (s.ckpt != null) $('ckpt').value = s.ckpt;
    $('det').checked = !s.det;
    $('brain-title').textContent = agent ? 'Bộ não agent' : 'Tín hiệu điều khiển';
    $('brain-sub').textContent = agent && s.ckpt != null ? this.ckpts[s.ckpt].name : `frame-skip ${s.frame_skip}`;
    $('btn-pause').classList.toggle('on', s.paused);
    $('btn-pause').textContent = s.paused ? '▶ Chạy tiếp' : '❚❚ Tạm dừng';
    $('btn-step').disabled = !s.paused;
    $('h-world').textContent = s.stage;

    $('policy').innerHTML = s.actions.map((a, i) => {
      const name = a.split(' ').map((b) => BUTTON_GLYPH[b] || b).join(' ');
      return `<div class="pi" data-i="${i}"><span class="name">${name}</span><span class="bar"><i></i></span><span class="pct">–</span></div>`;
    }).join('');
    this.rows = [...document.querySelectorAll('#policy .pi')];
    if (changed) { this.values = []; this.rewards = []; this.best = 0; }
    this._marks();
  }

  toast(text, error) {
    const el = document.createElement('div');
    el.className = `toast${error ? ' error' : ''}`;
    el.textContent = text;
    $('toasts').append(el);
    setTimeout(() => el.remove(), error ? 5200 : 2600);
  }

  status(online) {
    if (online) return;
    $('overlay').classList.remove('hidden');
    $('overlay-text').textContent = 'Mất kết nối với emulator. Hãy chạy lại: python play3d.py';
  }

  // ------------------------------------------------------------------ mỗi khung hình
  frame(f) {
    const h = f.head, [score, coins, time] = h.hud;
    $('h-score').textContent = pad(score, 6);
    $('h-coins').textContent = `×${pad(coins, 2)}`;
    $('h-time').textContent = pad(time, 3);
    $('h-time').classList.toggle('low', time < 100 && time > 0);
    if (h.btn !== this.btn) {
      this.btn = h.btn;
      for (const el of document.querySelectorAll('#pad i[data-b]')) el.classList.toggle('on', (h.btn & Number(el.dataset.b)) !== 0);
    }
    this.x = h.x;
    this.best = Math.max(this.best, h.x);
    this._rail();
    $('e-num').textContent = h.ep;
    if (h.step) this._step(h.step);
    if (f.obs) this._obs(f.obs);
    if (h.done) this._done(h.done);
  }

  _step(s) {
    $('s-steps').textContent = s.n;
    $('s-ret').textContent = s.ret.toFixed(2);
    $('r-now').textContent = (s.r >= 0 ? '+' : '') + s.r.toFixed(3);
    this.rewards.push(s.r);
    if (this.rewards.length > HISTORY) this.rewards.shift();
    if (s.n === 1) { this.values = []; this.rewards = [s.r]; }
    if (s.p && this.rows.length === s.p.length) {
      s.p.forEach((p, i) => {
        const row = this.rows[i];
        row.classList.toggle('on', i === s.a);
        row.children[1].firstChild.style.width = `${(p * 100).toFixed(1)}%`;
        row.children[2].textContent = `${(p * 100).toFixed(0)}%`;
      });
      this.values.push(s.v);
      if (this.values.length > HISTORY) this.values.shift();
      $('v-now').textContent = s.v.toFixed(2);
      $('s-ent').textContent = s.h.toFixed(2);
    } else $('s-ent').textContent = '–';
    this.sparkDirty = true;
  }

  _obs(obs) {
    const d = new Uint32Array(this.obsImage.data.buffer);
    for (let i = 0; i < obs.length; i++) { const v = obs[i]; d[i] = 0xff000000 | (v << 16) | (v << 8) | v; }
    this.obs.putImageData(this.obsImage, 0, 0);
  }

  _done(d) {
    const s = this.state;
    s.history.push({ x: d.max_x, flag: d.flag });
    if (d.flag && !s.goal_x) s.goal_x = d.max_x;
    this._marks();
    const banner = $('banner');
    banner.className = `show ${d.flag ? 'win' : 'lose'}`;
    banner.innerHTML = d.flag
      ? `COURSE CLEAR!<small>Về đích sau ${d.n} bước · còn ${d.time} giây · return ${d.ret.toFixed(2)}</small>`
      : `GAME OVER<small>Mất mạng ở x = ${d.max_x} · ${d.n} bước · return ${d.ret.toFixed(2)}</small>`;
    clearTimeout(this.bannerTimer);
    this.bannerTimer = setTimeout(() => { banner.className = ''; }, d.flag ? 2900 : 1150);
    if (d.flag) this.confetti.burst();
    else {
      const flash = $('flash');
      flash.className = 'lose on';
      setTimeout(() => { flash.className = 'lose'; }, 90);
      this.stage3d.kick(5);
    }
  }

  _scale() {
    const s = this.state;
    const seen = Math.max(this.best, ...s.history.map((e) => e.x), 0);
    return s.goal_x || Math.max(1200, seen * 1.12);
  }

  _rail() {
    const scale = this._scale(), f = Math.min(1, this.x / scale) * 100;
    $('rail-fill').style.width = `${f}%`;
    $('rail-mario').style.left = `${f}%`;
    $('rail-x').textContent = `x = ${this.x}`;
    $('e-best').textContent = Math.max(this.best, ...this.state.history.map((e) => e.x), 0);
  }

  _marks() {
    const s = this.state, scale = this._scale();
    $('rail-marks').innerHTML = s.history.map((e) =>
      `<i class="${e.flag ? 'win' : ''}" style="left:${Math.min(100, (e.x / scale) * 100).toFixed(2)}%"></i>`).join('');
    $('e-wins').textContent = `${s.history.filter((e) => e.flag).length}/${s.history.length}`;
    $('rail-goal').style.visibility = s.goal_x ? 'visible' : 'hidden';
    $('rail-end').textContent = s.goal_x ? `cột cờ: x = ${s.goal_x}` : 'chưa biết vị trí cột cờ';
  }

  /** Vẽ lại các canvas 2D, gọi một lần mỗi khung hiển thị. */
  draw(f, fps, voxels) {
    if (f) {
      const pal = f.palette, p32 = this.pal32, d = new Uint32Array(this.nesImage.data.buffer), c = f.color;
      for (let i = 0; i < 64; i++) p32[i] = 0xff000000 | (pal[i * 3 + 2] << 16) | (pal[i * 3 + 1] << 8) | pal[i * 3];
      for (let i = 0; i < c.length; i++) d[i] = p32[c[i]];
      this.nes.putImageData(this.nesImage, 0, 0);
    }
    if (this.sparkDirty) {
      sparkline($('spark-v'), this.values);
      sparkline($('spark-r'), this.rewards, { bars: true });
      this.sparkDirty = false;
    }
    if (fps) $('fps').textContent = `${fps} fps · ${(voxels / 1000).toFixed(1)}k khối`;
  }

  // ------------------------------------------------------------------ thao tác của người dùng
  _bind() {
    const send = this.send;
    $('modes').addEventListener('click', (e) => {
      const b = e.target.closest('button');
      if (b && !b.disabled) send('mode', b.dataset.mode);
    });
    $('speeds').addEventListener('click', (e) => {
      const b = e.target.closest('button');
      if (b) send('speed', Number(b.dataset.speed));
    });
    $('stage').addEventListener('change', (e) => { send('stage', e.target.value); e.target.blur(); });
    $('ckpt').addEventListener('change', (e) => { send('ckpt', Number(e.target.value)); e.target.blur(); });
    $('det').addEventListener('change', (e) => send('det', !e.target.checked));
    $('btn-pause').addEventListener('click', () => send('pause', !this.state.paused));
    $('btn-step').addEventListener('click', () => send('step'));
    $('btn-reset').addEventListener('click', () => send('reset'));
    $('cams').addEventListener('click', (e) => {
      const b = e.target.closest('button');
      if (b) this.camera(b.dataset.cam);
    });
    for (const key of ['shadow', 'bloom', 'tilt', 'grid', 'hi']) {
      $(`q-${key}`).addEventListener('change', (e) => this.stage3d.setOption(key, e.target.checked));
    }
    // nút/công tắc vừa bấm không được giữ tiêu điểm, kẻo phím Space (nhảy) lại kích hoạt chúng
    document.addEventListener('click', (e) => { const el = e.target.closest('button, .switch'); if (el) document.activeElement.blur(); });
    $('btn-ui').addEventListener('click', () => document.body.classList.toggle('clean'));
    $('btn-full').addEventListener('click', () => this.fullscreen());
    this.camera('diorama');

    const cams = ['diorama', 'runner', 'classic', 'orbit'];
    addEventListener('keydown', (e) => {
      const t = e.target;
      if ((t instanceof Element && t.matches('select, textarea')) || e.ctrlKey || e.metaKey || e.altKey || !this.state) return;
      const bit = KEY_BITS[e.code];
      if (bit && this.state.mode === 'human') {
        e.preventDefault();
        if (!(this.keys & bit)) { this.keys |= bit; send('keys', this.keys); }
        return;
      }
      if (e.repeat) return;
      if (e.code === 'KeyP') send('pause', !this.state.paused);
      else if (e.code === 'KeyR') send('reset');
      else if (e.code === 'Period') send('step');
      else if (e.code === 'KeyH') document.body.classList.toggle('clean');
      else if (e.code === 'KeyF') this.fullscreen();
      else if (/^Digit[1-4]$/.test(e.code)) this.camera(cams[Number(e.code.slice(5)) - 1]);
    });
    addEventListener('keyup', (e) => {
      const bit = KEY_BITS[e.code];
      if (bit && (this.keys & bit)) { this.keys &= ~bit; send('keys', this.keys); }
    });
    addEventListener('blur', () => { if (this.keys) { this.keys = 0; send('keys', 0); } });
  }

  camera(name) {
    this.stage3d.setCamera(name);
    for (const b of document.querySelectorAll('#cams button')) b.classList.toggle('on', b.dataset.cam === name);
  }

  fullscreen() {
    if (document.fullscreenElement) document.exitFullscreen();
    else document.documentElement.requestFullscreen().catch(() => {});
  }
}
