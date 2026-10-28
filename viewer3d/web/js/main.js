// Điểm vào: nối kênh emulator -> thế giới voxel -> sân khấu 3D -> giao diện.
import { Link } from './net.js';
import { Stage } from './stage.js';
import { UI } from './ui.js';
import { World } from './world.js';

const stage = new Stage(document.getElementById('gl'));
const world = new World(stage.scene);
const queue = [];
let last = null;

const link = new Link({
  onFrame(f) {
    queue.push(f);
    if (document.hidden || queue.length > 8) while (queue.length) consume(queue.shift());   // tab nền: không dồn khung
  },
  onEvent(msg) {
    if (msg.type === 'hello') ui.hello(msg);
    else if (msg.type === 'state') ui.setState(msg);
    else if (msg.type === 'toast') ui.toast(msg.text, msg.error);
  },
  onStatus: (online) => ui.status(online),
});
const ui = new UI({ send: (cmd, value) => link.send(cmd, value), stage });

function consume(f) {
  const h = f.head, fresh = h.epoch !== world.epoch;
  world.ingest(f, performance.now() / 1000);
  const b = h.bd * 3;
  stage.setBackdrop(f.palette[b], f.palette[b + 1], f.palette[b + 2], h.dark);
  stage.follow(h.scroll, h.mx, h.my, fresh);
  ui.frame(f);
  last = f;
}

let prev = performance.now() / 1000, frames = 0, fpsAt = prev, fps = 0;
function tick(ms) {
  requestAnimationFrame(tick);
  const now = ms / 1000, dt = Math.min(0.1, Math.max(0.001, now - prev));
  prev = now;
  // giữ lại 1 khung làm đệm để chuyển động sprite đều dù gói tin đến lệch nhịp
  for (let n = queue.length > 2 ? queue.length - 1 : Math.min(1, queue.length); n > 0; n--) consume(queue.shift());
  stage.update(dt, now);
  world.update(now, stage.cur.az >= 0 ? 1 : -1, stage.cur.x);
  stage.render();
  frames++;
  let shown = 0;
  if (now - fpsAt >= 0.5) { fps = Math.round(frames / (now - fpsAt)); frames = 0; fpsAt = now; shown = fps; }
  ui.draw(last, shown, world.voxels);
}
requestAnimationFrame(tick);

window.__viewer = { stage, world, ui, link, queue };   // tiện gỡ lỗi từ console
