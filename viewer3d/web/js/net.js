// Kênh WebSocket tới emulator: nhận khung nhị phân + sự kiện JSON, gửi lệnh điều khiển.
export const W = 256, H = 208, OBS = 84;
const PIXELS = W * H;
const decoder = new TextDecoder();

// u32 LE độ dài header | header JSON | bảng màu 64x3 | chỉ số màu | meta | [quan sát 84x84]
function parseFrame(buf) {
  const hl = new DataView(buf).getUint32(0, true);
  const head = JSON.parse(decoder.decode(new Uint8Array(buf, 4, hl)));
  let o = 4 + hl;
  const palette = new Uint8Array(buf, o, 192); o += 192;
  const color = new Uint8Array(buf, o, PIXELS); o += PIXELS;
  const meta = new Uint8Array(buf, o, PIXELS); o += PIXELS;
  const obs = head.obs ? new Uint8Array(buf, o, OBS * OBS) : null;
  return { head, palette, color, meta, obs };
}

export class Link {
  constructor({ onFrame, onEvent, onStatus }) {
    this.onFrame = onFrame;
    this.onEvent = onEvent;
    this.onStatus = onStatus;
    this.ws = null;
    this.retry = 0;
    this.connect();
  }

  connect() {
    const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
    ws.binaryType = 'arraybuffer';
    ws.onopen = () => { this.retry = 0; this.onStatus(true); };
    ws.onmessage = (e) => {
      if (typeof e.data === 'string') this.onEvent(JSON.parse(e.data));
      else this.onFrame(parseFrame(e.data));
    };
    ws.onclose = () => {
      this.onStatus(false);
      this.retry = Math.min(this.retry + 1, 6);
      setTimeout(() => this.connect(), 300 * this.retry);
    };
    this.ws = ws;
  }

  send(cmd, value) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify({ cmd, value }));
  }
}
