// Vật liệu voxel: MeshLambertMaterial được vá để mỗi instance là một "thanh" điểm ảnh NES đùn khối.
// Kênh alpha của ảnh ra là mặt nạ phát sáng, lượt bloom trong post.js đọc lại nó.
//
// Thuộc tính mỗi instance (12 float):
//   aBox = (x, y, z tâm, bề rộng theo x)      aCol = (r, g, b tuyến tính, bề dày theo z)
//   aFx  = (thời điểm sinh, độ phát sáng, 1 nếu là đồng xu quay, màu mặt hông đóng gói r + 256 g + 65536 b)
// Điểm ảnh viền đen của NES khi đùn khối sẽ thành cả một mặt hông đen kịt; mesher gửi kèm màu của điểm
// ảnh kề bên để mặt hông/mặt trên của viền mang tông màu của chính vật thể.
import * as THREE from 'three';

export const STRIDE = 12;
export const voxelUniforms = {
  uTime: { value: 0 },
  uEdge: { value: 0.2 },      // độ đậm của vân lưới voxel
  uNight: { value: 0 },       // 0 = màn ngoài trời, 1 = hang/lâu đài: màu sáng tự phát sáng
};

const PLACE = /* glsl */`
attribute vec4 aBox;
attribute vec4 aCol;
attribute vec4 aFx;
uniform float uTime;

float voxAngle() {
  return uTime * 3.6 + (floor(aBox.x / 16.0) + modelMatrix[3].x / 16.0) * 0.55;
}

vec3 voxPlace(vec3 p) {
  vec3 q = vec3(p.x * aBox.w, p.y, p.z * aCol.w) + aBox.xyz;
  if (aFx.z > 0.5) {                                   // đồng xu: quay quanh trục dọc giữa ô 16px
    float cx = floor(aBox.x / 16.0) * 16.0 + 8.0;
    float a = voxAngle();
    float s = sin(a), c = cos(a), dx = q.x - cx;
    q.x = cx + dx * c + q.z * s;
    q.z = q.z * c - dx * s;
  }
  // hiệu ứng "mọc" khi cột vừa lộ ra ở mép màn hình: từng dải điểm ảnh phồng lên tại chỗ và nhô nhẹ lên
  float t = clamp((uTime - aFx.x) / 0.4, 0.0, 1.0);
  float e = 1.0 - pow(1.0 - t, 3.0);
  q = mix(aBox.xyz + vec3(aBox.w * 0.5, 0.5, 0.0), q, e);
  q.y -= (1.0 - e) * 7.0;
  return q;
}
`;

const VERT_PARS = PLACE + /* glsl */`
varying vec3 vICol;
varying vec3 vGrid;
varying vec3 vFace;
varying float vGlow;
`;

const VERT_NORMAL = /* glsl */`
vec3 objectNormal = vec3(normal);
if (aFx.z > 0.5) {
  float na = voxAngle();
  float ns = sin(na), nc = cos(na);
  objectNormal = vec3(objectNormal.x * nc + objectNormal.z * ns, objectNormal.y, objectNormal.z * nc - objectNormal.x * ns);
}
`;

const VERT_BEGIN = /* glsl */`
vec3 transformed = voxPlace(position);
vICol = aCol.rgb;
if (aFx.w > 0.5 && abs(normal.z) < 0.5) {
  vec3 side = vec3(mod(aFx.w, 256.0), mod(floor(aFx.w / 256.0), 256.0), floor(aFx.w / 65536.0)) / 255.0;
  vICol = side * side * 0.42;                       // xấp xỉ sRGB -> tuyến tính, rồi làm tối như mặt khuất sáng
}
vGlow = aFx.y;
vFace = normal;
vGrid = vec3(position.x * aBox.w, position.y, (position.z + 0.5) * aCol.w);
`;

const FRAG_PARS = /* glsl */`
uniform float uEdge;
uniform float uNight;
varying vec3 vICol;
varying vec3 vGrid;
varying vec3 vFace;
varying float vGlow;
`;

const FRAG_COLOR = /* glsl */`
#include <color_fragment>
diffuseColor.rgb *= vICol;
{
  // vân lưới: mỗi điểm ảnh NES là một khối 1x1x1, kẻ rãnh mảnh giữa các khối (tự mờ khi nhìn xa)
  vec3 an = abs(vFace);
  vec2 gc = an.z > 0.5 ? vGrid.xy : (an.y > 0.5 ? vGrid.xz : vGrid.zy);
  vec2 gd = 0.5 - abs(fract(gc) - 0.5);
  vec2 gw = fwidth(gc);
  vec2 gl = 1.0 - smoothstep(vec2(0.0), max(gw * 1.25, vec2(0.07)), gd);
  float fade = 1.0 - smoothstep(0.16, 0.42, max(gw.x, gw.y));
  diffuseColor.rgb *= 1.0 - uEdge * max(gl.x, gl.y) * fade;
}
`;

const FRAG_EMISSIVE = /* glsl */`
#include <emissivemap_fragment>
float voxGlow = vGlow * (1.0 + uNight) + uNight * smoothstep(0.3, 0.85, dot(vICol, vec3(0.2126, 0.7152, 0.0722))) * 0.16;
totalEmissiveRadiance += vICol * voxGlow;
`;

const FRAG_END = /* glsl */`
#include <dithering_fragment>
gl_FragColor.a = clamp(voxGlow, 0.0, 1.0);
`;

function bind(shader) {
  shader.uniforms.uTime = voxelUniforms.uTime;
  shader.uniforms.uEdge = voxelUniforms.uEdge;
  shader.uniforms.uNight = voxelUniforms.uNight;
}

export function makeVoxelMaterial() {
  // hộp không có mặt sau nên bản đồ bóng phải vẽ các mặt hướng về đèn (trước/trên/hông) mới đủ hình bóng
  const mat = new THREE.MeshLambertMaterial({ shadowSide: THREE.FrontSide });
  mat.onBeforeCompile = (shader) => {
    bind(shader);
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\n' + VERT_PARS)
      .replace('#include <beginnormal_vertex>', VERT_NORMAL)
      .replace('#include <begin_vertex>', VERT_BEGIN);
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', '#include <common>\n' + FRAG_PARS)
      .replace('#include <color_fragment>', FRAG_COLOR)
      .replace('#include <emissivemap_fragment>', FRAG_EMISSIVE)
      .replace('#include <dithering_fragment>', FRAG_END);
  };
  mat.customProgramCacheKey = () => 'voxel-main';
  return mat;
}

// vật liệu ghi bản đồ bóng: phải đặt đỉnh giống hệt vật liệu chính thì bóng mới khớp
export function makeVoxelDepthMaterial() {
  const mat = new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking });
  mat.onBeforeCompile = (shader) => {
    bind(shader);
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\n' + PLACE)
      .replace('#include <begin_vertex>', 'vec3 transformed = voxPlace(position);');
  };
  mat.customProgramCacheKey = () => 'voxel-depth';
  return mat;
}

// hộp đơn vị x,y thuộc [0,1], z thuộc [-0.5,0.5]; bỏ mặt sau vì máy quay không bao giờ vòng ra sau.
// Chỉ số xếp theo thứ tự [trước, trên, hông phía máy quay, hông kia, dưới] để mảnh ở xa chỉ cần vẽ 18 chỉ số đầu.
export const FAR_INDEX_COUNT = 18;
export function makeBoxGeometry() {
  const box = new THREE.BoxGeometry(1, 1, 1);          // nhóm mặt của three: +x, -x, +y, -y, +z, -z
  box.translate(0.5, 0.5, 0);
  const idx = box.index.array, face = (i) => Array.from(idx.slice(i * 6, i * 6 + 6));
  const order = (side) => new THREE.BufferAttribute(new Uint16Array([...face(4), ...face(2), ...face(side), ...face(1 - side), ...face(3)]), 1);
  return {
    position: box.attributes.position,
    normal: box.attributes.normal,
    indexRight: order(0),                               // máy quay lệch về +x: thấy mặt +x
    indexLeft: order(1),
  };
}
