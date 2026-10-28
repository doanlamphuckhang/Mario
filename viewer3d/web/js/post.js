// Hậu kỳ gọn nhẹ cho GPU tích hợp: chỉ 2 lượt vẽ ở độ phân giải đầy đủ (cảnh + ghép), mọi lượt làm mờ
// chạy ở 1/4 .. 1/16 độ phân giải bằng bộ lọc dual-Kawase.
//
//   cảnh (RGBA8 sRGB, alpha = mặt nạ phát sáng do vật liệu voxel ghi)
//     └─ Q = 1/4 (dùng chung)
//          ├─ tilt-shift: Q -> 1/8 -> phóng ngược lên 1/4 (mờ rộng, mịn)
//          └─ bloom:      Q.rgb * Q.a -> 1/8 -> 1/16 -> cộng dồn ngược lên 1/8
//   ghép: trộn nét / Q / mờ rộng theo dải tiêu điểm + bloom + chỉnh màu + tối góc + hạt -> màn hình
import * as THREE from 'three';
import { FullScreenQuad } from 'three/addons/postprocessing/Pass.js';

const VERT = /* glsl */`varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`;

const DOWN = /* glsl */`
uniform sampler2D tSrc; uniform vec2 uHalf; uniform float uPremult; varying vec2 vUv;
vec4 tap(vec2 uv) { vec4 c = texture2D(tSrc, uv); return mix(c, vec4(c.rgb * c.a, c.a), uPremult); }
void main() {
  vec4 s = tap(vUv) * 4.0;
  s += tap(vUv - uHalf) + tap(vUv + uHalf) + tap(vUv + vec2(uHalf.x, -uHalf.y)) + tap(vUv - vec2(uHalf.x, -uHalf.y));
  gl_FragColor = s / 8.0;
}`;

const UP = /* glsl */`
uniform sampler2D tSrc; uniform sampler2D tAdd; uniform vec2 uHalf; uniform float uAdd; varying vec2 vUv;
void main() {
  vec2 h = uHalf;
  vec4 s = texture2D(tSrc, vUv + vec2(-h.x * 2.0, 0.0)) + texture2D(tSrc, vUv + vec2(h.x * 2.0, 0.0))
         + texture2D(tSrc, vUv + vec2(0.0, h.y * 2.0)) + texture2D(tSrc, vUv + vec2(0.0, -h.y * 2.0))
         + (texture2D(tSrc, vUv + vec2(-h.x, h.y)) + texture2D(tSrc, vUv + vec2(h.x, h.y))
         +  texture2D(tSrc, vUv + vec2(h.x, -h.y)) + texture2D(tSrc, vUv + vec2(-h.x, -h.y))) * 2.0;
  gl_FragColor = s / 12.0 + texture2D(tAdd, vUv) * uAdd;
}`;

const COMPOSITE = /* glsl */`
uniform sampler2D tScene, tSoft, tBlur, tBloom;
uniform vec2 uPixel;
uniform float uFocus, uBand, uTilt, uBloom, uVignette, uTime, uSat, uGain, uFxaa;
varying vec2 vUv;
float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
float luma(vec3 c) { return dot(sqrt(c), vec3(0.299, 0.587, 0.114)); }

// FXAA rút gọn (Lottes): khử răng cưa theo hướng cạnh, bỏ qua sớm ở vùng phẳng. Rẻ hơn MSAA nhiều trên GPU tích hợp.
vec3 sceneColor(vec2 uv) {
  vec3 m = texture2D(tScene, uv).rgb;
  if (uFxaa < 0.5) return m;
  vec3 nw = texture2D(tScene, uv + vec2(-1.0, -1.0) * uPixel).rgb, ne = texture2D(tScene, uv + vec2(1.0, -1.0) * uPixel).rgb;
  vec3 sw = texture2D(tScene, uv + vec2(-1.0, 1.0) * uPixel).rgb, se = texture2D(tScene, uv + vec2(1.0, 1.0) * uPixel).rgb;
  float lM = luma(m), lNW = luma(nw), lNE = luma(ne), lSW = luma(sw), lSE = luma(se);
  float lMin = min(lM, min(min(lNW, lNE), min(lSW, lSE))), lMax = max(lM, max(max(lNW, lNE), max(lSW, lSE)));
  if (lMax - lMin < max(0.035, lMax * 0.11)) return m;
  vec2 dir = vec2(-((lNW + lNE) - (lSW + lSE)), (lNW + lSW) - (lNE + lSE));
  float reduce = max((lNW + lNE + lSW + lSE) * 0.03125, 0.0078125);
  dir = clamp(dir / (min(abs(dir.x), abs(dir.y)) + reduce), -8.0, 8.0) * uPixel;
  vec3 a = 0.5 * (texture2D(tScene, uv + dir * (1.0 / 3.0 - 0.5)).rgb + texture2D(tScene, uv + dir * (2.0 / 3.0 - 0.5)).rgb);
  vec3 b = a * 0.5 + 0.25 * (texture2D(tScene, uv - dir * 0.5).rgb + texture2D(tScene, uv + dir * 0.5).rgb);
  float lB = luma(b);
  return (lB < lMin || lB > lMax) ? a : b;
}

void main() {
  vec3 c = sceneColor(vUv);
  float a = smoothstep(uBand, uBand + 0.34, abs(vUv.y - uFocus)) * uTilt;       // 0 = nét, 1 = mờ nhất
  if (a > 0.003) {
    vec3 soft = texture2D(tSoft, vUv).rgb;
    c = a < 0.5 ? mix(c, soft, a * 2.0) : mix(soft, texture2D(tBlur, vUv).rgb, a * 2.0 - 1.0);
  }
  c += texture2D(tBloom, vUv).rgb * uBloom;
  c *= uGain;
  c = mix(vec3(dot(c, vec3(0.2126, 0.7152, 0.0722))), c, uSat);
  c = mix(c, c * c * (3.0 - 2.0 * c), 0.22);                                    // đường cong chữ S nhẹ
  vec2 q = (vUv - 0.5) * vec2(1.0, 0.86);
  c *= 1.0 - uVignette * smoothstep(0.32, 0.88, length(q));
  c += (hash(vUv * 917.0 + fract(uTime) * 61.0) - 0.5) * 0.018;                 // hạt phim, cũng khử dải màu
  gl_FragColor = vec4(clamp(c, 0.0, 1.0), 1.0);
  #include <colorspace_fragment>
}`;

function target(depth, samples = 0) {
  return new THREE.WebGLRenderTarget(4, 4, { depthBuffer: depth, samples, colorSpace: THREE.SRGBColorSpace,
    minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, generateMipmaps: false });
}

const shader = (fragmentShader, uniforms) =>
  new THREE.ShaderMaterial({ vertexShader: VERT, fragmentShader, uniforms, depthTest: false, depthWrite: false });

export class Post {
  constructor(renderer) {
    this.r = renderer;
    this.quad = new FullScreenQuad();
    this.samples = 0;
    this.scene = target(true, this.samples);
    this.quarter = target(false);                                      // 1/4, dùng chung
    this.eighth = target(false);                                       // 1/8 của cảnh
    this.blur = target(false);                                         // 1/4, mờ rộng cho tilt-shift
    this.glow = [target(false), target(false), target(false)];         // 1/8 (rgb*alpha), 1/16, 1/8 (kết quả bloom)
    this.black = new THREE.DataTexture(new Uint8Array(4), 1, 1);
    this.black.needsUpdate = true;
    this.down = shader(DOWN, { tSrc: { value: null }, uHalf: { value: new THREE.Vector2() }, uPremult: { value: 0 } });
    this.up = shader(UP, { tSrc: { value: null }, tAdd: { value: this.black }, uHalf: { value: new THREE.Vector2() }, uAdd: { value: 0 } });
    this.composite = shader(COMPOSITE, {
      tScene: { value: null }, tSoft: { value: this.black }, tBlur: { value: this.black }, tBloom: { value: this.black },
      uFocus: { value: 0.5 }, uBand: { value: 0.2 }, uTilt: { value: 1 }, uBloom: { value: 1 }, uVignette: { value: 0.3 },
      uTime: { value: 0 }, uSat: { value: 1.06 }, uGain: { value: 1 }, uFxaa: { value: 1 }, uPixel: { value: new THREE.Vector2() },
    });
    this.params = this.composite.uniforms;
    this.gpuMs = null;
    this.query = null;
  }

  setSize(w, h, samples) {
    if (samples !== this.samples) {                   // số mẫu MSAA cố định theo target: phải tạo lại
      this.scene.dispose();
      this.scene = target(true, this.samples = samples);
    }
    const s = (k) => [Math.max(2, Math.round(w / k)), Math.max(2, Math.round(h / k))];
    this.scene.setSize(w, h);
    this.params.uPixel.value.set(1 / w, 1 / h);
    this.params.uFxaa.value = samples > 0 ? 0 : 1;          // đã có MSAA thì thôi FXAA
    this.quarter.setSize(...s(4));
    this.eighth.setSize(...s(8));
    this.blur.setSize(...s(4));
    this.glow[0].setSize(...s(8));
    this.glow[1].setSize(...s(16));
    this.glow[2].setSize(...s(8));
  }

  _pass(material, src, dst) {
    const u = material.uniforms;
    u.tSrc.value = src.texture;
    u.uHalf.value.set(0.5 / dst.width, 0.5 / dst.height);
    this.quad.material = material;
    this.r.setRenderTarget(dst);
    this.quad.render(this.r);
  }

  /** Bắt đầu đo thời gian GPU của khung này (nếu trình duyệt cho phép); kết quả về trễ vài khung, lưu ở gpuMs. */
  _timerBegin() {
    const gl = this.r.getContext();
    if (this.timer === undefined) this.timer = gl.getExtension('EXT_disjoint_timer_query_webgl2') || null;
    if (!this.timer) return false;
    if (this.query) {
      if (!gl.getQueryParameter(this.query, gl.QUERY_RESULT_AVAILABLE)) return false;
      if (!gl.getParameter(this.timer.GPU_DISJOINT_EXT)) this.gpuMs = gl.getQueryParameter(this.query, gl.QUERY_RESULT) / 1e6;
      gl.deleteQuery(this.query);
    }
    this.query = gl.createQuery();
    gl.beginQuery(this.timer.TIME_ELAPSED_EXT, this.query);
    return true;
  }

  /** tilt: cường độ làm mờ 0..1 (0 = bỏ qua cả chuỗi làm mờ); bloom: bật/tắt. */
  render(scene, camera, { tilt, bloom }) {
    const r = this.r, u = this.params, { quarter, eighth, blur, glow, down, up } = this;
    const useTilt = tilt > 0.02, timing = this._timerBegin();
    r.setRenderTarget(this.scene);
    r.render(scene, camera);
    down.uniforms.uPremult.value = 0;
    up.uniforms.uAdd.value = 0;
    up.uniforms.tAdd.value = this.black;
    if (useTilt || bloom) this._pass(down, this.scene, quarter);
    if (useTilt) {
      this._pass(down, quarter, eighth);
      this._pass(up, eighth, blur);
    }
    if (bloom) {
      down.uniforms.uPremult.value = 1;
      this._pass(down, quarter, glow[0]);
      down.uniforms.uPremult.value = 0;
      this._pass(down, glow[0], glow[1]);
      up.uniforms.uAdd.value = 1;
      up.uniforms.tAdd.value = glow[0].texture;
      this._pass(up, glow[1], glow[2]);
    }
    u.uTilt.value = useTilt ? tilt : 0;
    u.tSoft.value = useTilt ? quarter.texture : this.black;
    u.tBlur.value = useTilt ? blur.texture : this.black;
    u.tBloom.value = bloom ? glow[2].texture : this.black;
    u.tScene.value = this.scene.texture;
    this.quad.material = this.composite;
    r.setRenderTarget(null);
    this.quad.render(r);
    if (timing) r.getContext().endQuery(this.timer.TIME_ELAPSED_EXT);
  }
}
