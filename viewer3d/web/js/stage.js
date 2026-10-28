// Sân khấu: renderer, ánh sáng, bầu trời, mặt bàn và máy quay. Hậu kỳ nằm ở post.js.
import * as THREE from 'three';
import { voxelUniforms } from './materials.js';
import { Post } from './post.js';
import { FOUNDATION } from './world.js';

const DEG = Math.PI / 180;
const lerp = (a, b, t) => a + (b - a) * t;
const damp = (a, b, rate, dt) => lerp(a, b, 1 - Math.exp(-rate * dt));

// góc máy: phương vị/độ cao (độ), hệ số zoom (1 = vừa khít vùng chơi), tiêu cự, điểm nhìn, độ đón đầu
export const CAMERAS = {
  diorama: { az: 24, el: 13, zoom: 1.0, fov: 26, ty: 76, lead: 14 },
  runner: { az: -50, el: 9, zoom: 0.72, fov: 34, ty: 66, lead: 46 },
  classic: { az: 0, el: 1.5, zoom: 0.98, fov: 9, ty: 80, lead: 0 },
  orbit: { az: 0, el: 13, zoom: 1.08, fov: 26, ty: 76, lead: 8 },
};

const SKY_VERT = /* glsl */`
  varying vec3 vDir;
  void main() {
    vDir = position;
    gl_Position = (projectionMatrix * modelViewMatrix * vec4(position, 1.0)).xyww;   // luôn ở mặt phẳng xa
  }`;
const SKY_FRAG = /* glsl */`
  uniform vec3 uZenith, uHorizon; uniform float uGlow; varying vec3 vDir;
  void main() {
    float h = normalize(vDir).y;
    vec3 c = mix(uHorizon, uZenith, pow(smoothstep(-0.02, 0.5, h), 0.7));
    c += uHorizon * uGlow * exp(-abs(h) * 11.0);              // quầng sáng mảnh ở đường chân trời
    gl_FragColor = vec4(c, 0.0);                              // alpha 0 = không bloom
  }`;

// lưới 16px mờ trên mặt bàn, như tấm lót cắt mô hình; cũng giúp mắt thấy chuyển động khi máy quay trượt
const FLOOR_GRID = /* glsl */`#include <color_fragment>
  {
    vec2 g = vFloor.xz / 16.0;
    vec2 w = fwidth(g);
    vec2 a = 1.0 - smoothstep(vec2(0.0), max(w * 1.5, vec2(0.02)), 0.5 - abs(fract(g) - 0.5));
    vec2 b = 1.0 - smoothstep(vec2(0.0), max(w * 0.5, vec2(0.012)), 0.5 - abs(fract(g / 4.0) - 0.5));
    float fade = 1.0 - smoothstep(0.02, 0.2, max(w.x, w.y));
    diffuseColor.rgb *= 1.0 - (max(a.x, a.y) * 0.06 + max(b.x, b.y) * 0.09) * fade;
  }`;

export class Stage {
  constructor(canvas) {
    this.canvas = canvas;
    const renderer = this.renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance' });
    renderer.setClearColor(0x000000, 0);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.BasicShadowMap;
    renderer.shadowMap.autoUpdate = false;      // tự cập nhật cách khung, xem update()

    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.Fog(0x000000, 900, 4200);
    this.camera = new THREE.PerspectiveCamera(26, 1, 20, 12000);
    this.options = { shadow: true, bloom: true, tilt: true, grid: true, hi: false };

    // trạng thái máy quay (giá trị hiện tại bám dần theo giá trị đích)
    this.rig = { name: 'diorama', ...CAMERAS.diorama, userAz: 0, userEl: 0, userZoom: 1 };
    this.cur = { az: 40, el: 28, dist: 1500, fov: 26, x: 128, y: 80, lead: 0 };
    this.focus = { x: 128, mx: 40, my: 40 };
    this.shake = 0;
    this.night = 0;
    this.nightGoal = 0;
    this.scale = 1;            // hệ số độ phân giải động (tự hạ khi GPU đuối)
    this.ema = 8.5;
    this.cooldown = 120;
    this.tilt = 1;
    this._tmp = new THREE.Vector3();

    this.post = new Post(renderer);
    this._buildSet();
    this._bindPointer();
    this.resize();
    addEventListener('resize', () => this.resize());
  }

  // ------------------------------------------------------------------ bối cảnh
  _buildSet() {
    const scene = this.scene;
    this.sky = new THREE.Mesh(new THREE.SphereGeometry(1, 24, 12), new THREE.ShaderMaterial({
      vertexShader: SKY_VERT, fragmentShader: SKY_FRAG, side: THREE.BackSide, depthWrite: false, fog: false,
      uniforms: { uZenith: { value: new THREE.Color() }, uHorizon: { value: new THREE.Color() }, uGlow: { value: 0 } },
    }));
    this.sky.frustumCulled = false;
    this.sky.renderOrder = 10;            // vẽ sau cùng: chỉ tô những điểm ảnh chưa bị che
    scene.add(this.sky);

    const floorMat = this.floorMat = new THREE.MeshLambertMaterial();
    floorMat.onBeforeCompile = (s) => {
      s.vertexShader = s.vertexShader
        .replace('#include <common>', '#include <common>\nvarying vec3 vFloor;')
        .replace('#include <begin_vertex>', '#include <begin_vertex>\nvFloor = (modelMatrix * vec4(position, 1.0)).xyz;');
      s.fragmentShader = s.fragmentShader
        .replace('#include <common>', '#include <common>\nvarying vec3 vFloor;')
        .replace('#include <color_fragment>', FLOOR_GRID)
        .replace('#include <dithering_fragment>', '#include <dithering_fragment>\ngl_FragColor.a = 0.0;');
    };
    this.floor = new THREE.Mesh(new THREE.PlaneGeometry(60000, 24000), floorMat);
    this.floor.rotation.x = -Math.PI / 2;
    this.floor.position.y = -FOUNDATION;
    this.floor.receiveShadow = true;
    this.floor.renderOrder = 5;           // sau voxel để phần bị dải màn chơi che khỏi phải tô
    scene.add(this.floor);

    this.hemi = new THREE.HemisphereLight(0xffffff, 0xffffff, 1);
    scene.add(this.hemi);
    const sun = this.sun = new THREE.DirectionalLight(0xffffff, 3);
    sun.castShadow = true;
    sun.shadow.bias = -0.0012;
    sun.shadow.normalBias = 0.45;
    Object.assign(sun.shadow.camera, { left: -280, right: 280, top: 230, bottom: -190, near: 20, far: 1400 });
    scene.add(sun, sun.target);
    // đèn phụ mát từ phía phải-trước, không đổ bóng: nâng sáng các mặt khuất nắng
    this.fill = new THREE.DirectionalLight(0xbfd4ff, 0.7);
    this.fill.position.set(300, 120, 260);
    scene.add(this.fill);
    // đèn ấm đi theo Mario ở màn tối (hang, lâu đài)
    this.torch = new THREE.PointLight(0xffc27a, 0, 300, 1.6);
    scene.add(this.torch);

    this.mood = {
      day: { zenith: new THREE.Color(), horizon: new THREE.Color(), floor: new THREE.Color() },
      night: { zenith: new THREE.Color(0x04060e), horizon: new THREE.Color(0x121a36), floor: new THREE.Color(0x232a45) },
    };
    this.setBackdrop(92, 148, 252, false);
    this._shadowQuality();
  }

  /** Bầu trời, sương, mặt bàn lấy tông theo màu nền của màn chơi. */
  setBackdrop(r, g, b, dark) {
    this.nightGoal = dark ? 1 : 0;
    if (dark) return;
    const base = this._tmpColor || (this._tmpColor = new THREE.Color());
    base.setRGB(r / 255, g / 255, b / 255, THREE.SRGBColorSpace);
    const white = new THREE.Color(1, 1, 1), day = this.mood.day;
    day.zenith.copy(base).multiplyScalar(0.6);
    day.horizon.copy(base).lerp(white, 0.3);
    day.floor.copy(base).lerp(white, 0.1);       // mặt bàn cùng tông trời: nền sau màn chơi vẫn xanh như game gốc
  }

  _applyMood(dt) {
    const n = this.night = damp(this.night, this.nightGoal, 3.5, dt);
    const { day, night } = this.mood, sky = this.sky.material.uniforms, p = this.post.params;
    sky.uZenith.value.lerpColors(day.zenith, night.zenith, n);
    sky.uHorizon.value.lerpColors(day.horizon, night.horizon, n);
    sky.uGlow.value = lerp(0.1, 0.45, n);
    this.scene.fog.color.copy(sky.uHorizon.value);
    this.floorMat.color.lerpColors(day.floor, night.floor, n);
    this.hemi.color.setRGB(lerp(0.82, 0.3, n), lerp(0.9, 0.42, n), lerp(1.0, 0.8, n));
    this.hemi.groundColor.setRGB(lerp(0.95, 0.1, n), lerp(0.84, 0.1, n), lerp(0.7, 0.16, n));
    this.hemi.intensity = lerp(1.3, 0.6, n);
    this.sun.color.setRGB(lerp(1.0, 0.6, n), lerp(0.96, 0.72, n), lerp(0.88, 1.0, n));
    this.sun.intensity = lerp(2.6, 1.2, n);
    this.torch.intensity = 950 * n;
    this.fill.intensity = lerp(0.75, 0.3, n);
    voxelUniforms.uNight.value = n;
    p.uBloom.value = lerp(0.6, 1.25, n);
    p.uVignette.value = lerp(0.24, 0.5, n);
    p.uGain.value = lerp(1.0, 1.12, n);
  }

  // ------------------------------------------------------------------ chất lượng
  _shadowQuality() {
    const hi = this.options.hi, r = this.renderer;
    const type = hi ? THREE.PCFSoftShadowMap : THREE.BasicShadowMap;
    if (r.shadowMap.type !== type) {
      r.shadowMap.type = type;
      this.scene.traverse((o) => { if (o.material) o.material.needsUpdate = true; });
    }
    const size = hi ? 3072 : 1536;
    if (this.sun.shadow.mapSize.x !== size) {
      this.sun.shadow.mapSize.set(size, size);
      if (this.sun.shadow.map) { this.sun.shadow.map.dispose(); this.sun.shadow.map = null; }
    }
  }

  setOption(key, on) {
    this.options[key] = on;
    this.sun.castShadow = this.options.shadow;
    voxelUniforms.uEdge.value = this.options.grid ? 0.2 : 0;
    if (key === 'hi') { this.scale = 1; this.cooldown = 120; this._shadowQuality(); this.resize(); }
  }

  resize() {
    const w = innerWidth, h = innerHeight, hi = this.options.hi;
    const ratio = Math.min(devicePixelRatio || 1, hi ? 2 : 1) * this.scale;
    this.renderer.setPixelRatio(ratio);
    this.renderer.setSize(w, h, false);
    this.post.setSize(Math.round(w * ratio), Math.round(h * ratio), hi ? 4 : 0);
    this.camera.aspect = w / h;
  }

  /** Độ phân giải động: giữ thời gian dựng hình trong ngân sách; GPU đuối thì giảm, dư sức thì tăng lại. */
  _adapt(dt) {
    if (document.hidden || --this.cooldown > 0) return;
    const gpu = this.post.gpuMs;
    let want = this.scale;
    if (gpu != null) {                                   // đo trực tiếp thời gian GPU: chi phí gần tỉ lệ với số điểm ảnh
      this.ema = lerp(this.ema, gpu, 0.1);
      if (this.ema > 11 || (this.ema < 6.5 && this.scale < 1)) want = this.scale * Math.sqrt(8.5 / this.ema);
    } else if (dt < 0.07) {                              // không đo được GPU: suy từ nhịp khung (bỏ qua khi tab bị bóp rAF)
      this.ema = lerp(this.ema, dt * 1000, 0.06);
      if (this.ema > 22.5) want = this.scale * 0.86;
    }
    want = Math.min(1, Math.max(0.6, want));
    if (Math.abs(want - this.scale) < 0.06) return;
    this.scale = want;
    this.cooldown = 75;
    this.ema = gpu != null ? 8.5 : 1000 / 60;
    this.resize();
  }

  // ------------------------------------------------------------------ máy quay
  setCamera(name) {
    if (!CAMERAS[name]) return;
    Object.assign(this.rig, CAMERAS[name], { name, userAz: 0, userEl: 0, userZoom: 1 });
  }

  _bindPointer() {
    const el = this.canvas;
    let drag = null;
    el.addEventListener('pointerdown', (e) => { drag = { x: e.clientX, y: e.clientY }; el.setPointerCapture(e.pointerId); });
    el.addEventListener('pointermove', (e) => {
      if (!drag) return;
      const r = this.rig;
      r.userAz = THREE.MathUtils.clamp(r.userAz - (e.clientX - drag.x) * 0.22, -62 - r.az, 62 - r.az);
      r.userEl = THREE.MathUtils.clamp(r.userEl + (e.clientY - drag.y) * 0.18, 0.5 - r.el, 58 - r.el);
      drag = { x: e.clientX, y: e.clientY };
    });
    const end = () => { drag = null; };
    el.addEventListener('pointerup', end);
    el.addEventListener('pointercancel', end);
    el.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.rig.userZoom = THREE.MathUtils.clamp(this.rig.userZoom * Math.exp(e.deltaY * 0.0011), 0.32, 2.6);
    }, { passive: false });
    el.addEventListener('dblclick', () => this.setCamera(this.rig.name));
  }

  /** Điểm máy quay bám theo: tâm màn hình NES (toạ độ thế giới) và vị trí Mario. */
  follow(scroll, mx, my, snap) {
    this.focus.x = scroll + 128;
    this.focus.mx = scroll + mx + 8;
    this.focus.my = Math.min(208, Math.max(0, 208 - my - 12));
    if (snap) { this.cur.x = this.focus.x - 70; this.cur.dist *= 1.3; this.cur.el += 7; }   // lia máy vào đầu ván
  }

  kick(amount) { this.shake = Math.max(this.shake, amount); }

  update(dt, time) {
    const rig = this.rig, cur = this.cur, cam = this.camera;
    this._adapt(dt);
    this._applyMood(dt);

    let az = rig.az + rig.userAz, el = rig.el + rig.userEl;
    if (rig.name === 'orbit') { az += 30 * Math.sin(time * 0.21); el += 5 * Math.sin(time * 0.13); }
    cur.az = damp(cur.az, az, 5, dt);
    cur.el = damp(cur.el, el, 5, dt);
    cur.fov = damp(cur.fov, rig.fov, 4, dt);
    cur.lead = damp(cur.lead, rig.lead, 3, dt);
    cur.x = damp(cur.x, this.focus.x + cur.lead + (this.focus.mx - this.focus.x) * 0.22, 6.5, dt);
    cur.y = damp(cur.y, rig.ty + (this.focus.my - 70) * 0.1, 3, dt);

    // khoảng cách để vùng chơi 256x208 (thêm lề) vừa khung hình ở mọi tỉ lệ màn hình
    const t = Math.tan(cur.fov * DEG / 2);
    const fit = Math.max(258 / (2 * t), 318 / (2 * t * cam.aspect));
    cur.dist = damp(cur.dist, fit * rig.zoom * rig.userZoom, 4.5, dt);

    this.shake = damp(this.shake, 0, 7, dt);
    const sx = Math.sin(time * 71) * this.shake, sy = Math.cos(time * 53) * this.shake;
    const a = cur.az * DEG, e = cur.el * DEG;
    cam.position.set(cur.x + cur.dist * Math.sin(a) * Math.cos(e) + sx, cur.y + cur.dist * Math.sin(e) + sy,
      cur.dist * Math.cos(a) * Math.cos(e));
    cam.fov = cur.fov;
    cam.near = Math.max(10, cur.dist * 0.08);
    cam.far = cur.dist + 11000;
    cam.updateProjectionMatrix();
    cam.lookAt(cur.x + sx, cur.y + sy, -6);

    this.sky.position.copy(cam.position);
    this.floor.position.x = cur.x;
    this.scene.fog.near = cur.dist + 260;
    this.scene.fog.far = cur.dist + 2300;

    // bóng đổ: khung chiếu của mặt trời bám theo máy quay, nhảy từng 8 đơn vị để mép bóng không rung.
    // Bản đồ bóng chỉ vẽ lại cách một khung (chất lượng thường); đèn chỉ dời vào đúng khung đó để bóng không lệch.
    if (this.options.hi || (this.tick = (this.tick || 0) + 1) % 2 === 0) {
      const lx = Math.round(cur.x / 8) * 8;
      this.sun.target.position.set(lx, 70, 0);
      this.sun.position.set(lx - 250, 70 + 520, 380);
      this.renderer.shadowMap.needsUpdate = true;
    }
    this.torch.position.set(this.focus.mx, this.focus.my + 14, 34);

    // tiêu điểm tilt-shift nằm ở độ cao làn chạy trên màn hình; góc máy tele (2D cổ điển) thì gần như không mờ
    const p = this._tmp.set(cur.x, 66, 0).project(cam), post = this.post.params;
    post.uFocus.value = p.y * 0.5 + 0.5;
    post.uBand.value = 0.24;
    this.tilt = THREE.MathUtils.clamp((cur.fov - 12) / 12, 0, 1) * THREE.MathUtils.clamp(cur.dist / fit * 1.7 - 0.75, 0.12, 1);
    post.uTime.value = time;
    voxelUniforms.uTime.value = time;
  }

  render() {
    this.post.render(this.scene, this.camera, { tilt: this.options.tilt ? this.tilt : 0, bloom: this.options.bloom });
  }
}
