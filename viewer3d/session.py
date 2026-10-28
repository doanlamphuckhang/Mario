"""Vòng mô phỏng của viewer 3D: chạy emulator theo thời gian thực và phát từng khung cho trình duyệt.

Một luồng duy nhất giữ emulator + mô hình. Mỗi khung gốc 60 Hz của NES được `SceneExtractor` tách lớp
rồi đóng gói nhị phân:

    u32 LE độ dài header | header JSON | bảng màu 64x3 | chỉ số màu 208x256 | meta 208x256 | [quan sát 84x84]

Chế độ điều khiển:
    agent   checkpoint PPO (cùng `MarioEnv` và tiền xử lý như lúc đánh giá)
    demo    phát lại chuỗi hành động trong viewer3d/demos/<màn>.json
    human   người chơi bằng bàn phím (bước emulator từng khung)
    random  hành động ngẫu nhiên đều;  right: luôn phải + chạy + nhảy  (hai baseline trong báo cáo)
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import queue
import struct
import threading
import time
from pathlib import Path

import numpy as np

from mario_rl.envs import ACTION_SETS, MarioEnv

with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
    from gym_super_mario_bros._roms import rom_path
    from nes_py.wrappers import JoypadSpace

from .scene import HEIGHT, WIDTH, SceneExtractor

DEMO_DIR = Path(__file__).resolve().parent / "demos"
MODES = ("agent", "demo", "human", "random", "right")
SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0)
STAGES = [f"{w}-{s}" for w in range(1, 9) for s in range(1, 5)]
DEFAULT_ENV = dict(actions="simple", frame_skip=4, reward_scale=0.01, flag_bonus=50.0,
                   death_penalty=0.0, score_weight=0.0, coin_bonus=0.0)
HUMAN_KEYS = 0b11110011          # người chơi không được bấm Start/Select (tránh tạm dừng game)
FRAME_COUNTER = 0x09             # bộ đếm khung của game, dùng để biết emulator vừa nhảy cóc bao nhiêu khung


def button_mask(buttons) -> int:
    return sum(JoypadSpace._button_map[b] for b in buttons)


class Session(threading.Thread):
    def __init__(self, stage: str = "1-1", ckpts: list[Path] | None = None, mode: str | None = None,
                 speed: float = 1.0, deterministic: bool = True) -> None:
        super().__init__(daemon=True, name="mario-sim")
        self.stage = stage if stage in STAGES else "1-1"
        self.ckpts = list(ckpts or [])
        self.speed = speed if speed in SPEEDS else 1.0
        self.deterministic = deterministic
        self.paused = False
        self.keys = 0                               # phím người chơi, luồng WebSocket ghi trực tiếp

        self.cmds: queue.Queue[dict] = queue.Queue()
        self.clients: set = set()
        self._lock = threading.Lock()
        self._quit = threading.Event()

        self.extractor = SceneExtractor(rom_path(False, "vanilla"))
        self.model = None
        self.model_cfg: dict = {}
        self.ckpt_index: int | None = None
        self.env: MarioEnv | None = None
        self.rng = np.random.default_rng()

        self.epoch = 0            # tăng mỗi khi thế giới phải dựng lại (ván mới, đổi khu vực)
        self.episode = 0
        self.history: list[dict] = []
        self._frame = 0
        self._pending: list[tuple[dict, bytes, bytes, bytes]] = []
        self._next_t: float | None = None
        self._hold_until = 0.0
        self._step_once = False
        self._last_area, self._last_scroll = -1, 0
        self._last_data = b""

        if mode not in MODES or (mode == "agent" and not self.ckpts) or (mode == "demo" and self._demo() is None):
            mode = "agent" if self.ckpts else ("demo" if self._demo() is not None else "human")
        self.mode = mode

    # ------------------------------------------------------------------ cấu hình
    def _demo(self) -> dict | None:
        path = DEMO_DIR / f"{self.stage}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def _env_kwargs(self) -> dict:
        if self.mode == "agent" and self.model_cfg:
            return {k: self.model_cfg.get(k, v) for k, v in DEFAULT_ENV.items()}
        if self.mode == "demo" and self.demo:
            return {**DEFAULT_ENV, "actions": self.demo["actions"], "frame_skip": self.demo["frame_skip"]}
        return dict(DEFAULT_ENV)

    def _load_ckpt(self, index: int) -> bool:
        from mario_rl.evaluation import load_model   # import torch muộn: chế độ khác khởi động nhanh hơn
        import torch

        self.toast("Đang nạp checkpoint…")
        try:
            torch.set_num_threads(2)
            self.model, self.model_cfg = load_model(self.ckpts[index], torch.device("cpu"))
            self.ckpt_index = index
            return True
        except Exception as e:  # noqa: BLE001 - báo lỗi cho người dùng thay vì làm chết luồng mô phỏng
            self.toast(f"Không nạp được {self.ckpts[index].name}: {e}", error=True)
            return False

    def _build_env(self) -> None:
        """Tạo lại môi trường cho (màn, chế độ) hiện tại và bắt đầu ván mới."""
        self.demo = self._demo()
        if self.mode == "agent" and self.model is None and not (self.ckpts and self._load_ckpt(0)):
            self.mode = "demo" if self.demo else "human"
        if self.mode == "demo" and self.demo is None:
            self.mode = "human"
        if self.env is not None:
            self.env.close()
        self.kw = self._env_kwargs()
        self.env = MarioEnv(stages=self.stage, **self.kw)
        self.env.frame_hook = self._on_frame
        self.masks = [button_mask(a) for a in self.env.action_meanings]
        self.right_run_jump = self.masks.index(button_mask(("right", "A", "B"))) \
            if button_mask(("right", "A", "B")) in self.masks else 0
        self.history = []
        self.episode = 0
        self._reset_episode()
        self.broadcast_state()

    def _reset_episode(self) -> None:
        self.obs = self.env.reset()
        self.raw = self.env._env.unwrapped
        self._ram_prev = np.array(self.raw.ram, dtype=np.uint8)
        self.epoch += 1
        self.episode += 1
        self._ep = {"ret": 0.0, "n": 0, "max_x": 0, "frames": 0}
        self._demo_i = 0
        self._last_area = -1
        self._btn = 0
        self._hold_until = 0.0
        self._next_t = None
        self._pending.clear()
        self._on_frame(self.raw.screen)       # khung đầu tiên của ván
        self._flush()

    # ------------------------------------------------------------------ khung hình
    def _on_frame(self, screen: np.ndarray) -> None:
        raw = self.raw
        ram = np.array(raw.ram, dtype=np.uint8)
        # RAM đi trước ảnh 1 khung. Bộ đếm khung của game bình thường nhảy 0-2 mỗi bước emulator (ranh giới bước
        # nằm sát thời điểm game tăng bộ đếm); nhảy hơn thế nghĩa là gym vừa tua qua hoạt cảnh (chui ống, đổi khu
        # vực), RAM của khung trước không còn khớp nữa nên dùng luôn RAM hiện tại.
        skipped = (int(ram[FRAME_COUNTER]) - int(self._ram_prev[FRAME_COUNTER])) & 0xFF > 2
        lagged = ram if skipped else self._ram_prev
        self._ram_prev = ram
        color, meta, st = self.extractor.extract(screen, lagged)
        if st["area"] != self._last_area or st["scroll"] < self._last_scroll - 8:
            if self._last_area != -1:
                self.epoch += 1               # vào ống / đổi khu vực: toạ độ thế giới bắt đầu lại
            self._last_area = st["area"]
        self._last_scroll = st["scroll"]
        x = int(raw._x_position)
        self._ep["max_x"] = max(self._ep["max_x"], x)
        head = {
            "ep": self.episode, "epoch": self.epoch, "scroll": st["scroll"], "bd": st["backdrop"],
            "dark": st["dark"], "mx": st["mx"], "my": st["my"], "x": x, "btn": self._btn,
            "hud": [raw._score, raw._coins, raw._time, int(raw._world), int(raw._stage), int(raw._life)],
        }
        self._pending.append((head, color.tobytes(), meta.tobytes(), b""))

    def _flush(self) -> None:
        """Phát các khung đang chờ, giãn cách đúng nhịp 60 Hz x tốc độ."""
        keep = max(1, round(self.speed))
        palette = self.extractor.palette.tobytes()
        batch = len(self._pending)
        for i, (head, color, meta, obs) in enumerate(self._pending, 1):
            self._frame += 1
            now = time.perf_counter()
            if self._next_t is None or now - self._next_t > 0.25:
                self._next_t = now
            self._next_t += 1.0 / (60.0 * self.speed)
            # tua nhanh: chỉ gửi mỗi `keep` khung (đếm trong một bước của agent để khung mang sự kiện luôn trúng
            # nhịp), và luôn gửi khung mang sự kiện
            if (i if batch > 1 else self._frame) % keep == 0 or "step" in head or "done" in head:
                head["f"] = self._frame
                if obs:
                    head["obs"] = 1
                hb = json.dumps(head, separators=(",", ":")).encode()
                data = b"".join((struct.pack("<I", len(hb)), hb, palette, color, meta, obs))
                self._last_data = data
                with self._lock:
                    for c in self.clients:
                        c.offer(data, important="done" in head)
            delay = self._next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        self._pending.clear()

    # ------------------------------------------------------------------ một bước mô phỏng
    def _policy(self) -> tuple[int, dict]:
        n = self.env.n_actions
        if self.mode == "agent":
            import torch

            with torch.no_grad():
                logits, value = self.model(torch.as_tensor(self.obs).unsqueeze(0))
                p = torch.softmax(logits[0], -1)
                a = int(p.argmax()) if self.deterministic else int(torch.multinomial(p, 1))
            probs = p.tolist()
            ent = -sum(q * math.log(q) for q in probs if q > 0)
            return a, {"p": [round(q, 4) for q in probs], "v": round(float(value), 4), "h": round(ent, 4)}
        if self.mode == "demo":
            steps = self.demo["steps"]
            a = steps[self._demo_i] if self._demo_i < len(steps) else 0
            self._demo_i += 1
            return a, {}
        if self.mode == "random":
            return int(self.rng.integers(n)), {}
        return self.right_run_jump, {}

    def _advance(self) -> None:
        ep = self._ep
        if self.mode == "human":
            self._btn = self.keys & HUMAN_KEYS
            screen, r, done, info = self.raw.step(self._btn)
            self._on_frame(screen)
            ep["frames"] += 1
            flag = bool(info.get("flag_get", False))
            ep["ret"] += (r + (self.kw["flag_bonus"] if flag else 0.0)) * self.kw["reward_scale"]
            head = self._pending[-1][0]
            if ep["frames"] % 4 == 0 or done:
                ep["n"] += 1
                head["step"] = {"a": -1, "r": 0.0, "ret": round(ep["ret"], 3), "n": ep["n"]}
                self._pending[-1] = (*self._pending[-1][:3], self.env._preprocess(np.asarray(screen)).tobytes())
        else:
            a, extra = self._policy()
            self._btn = self.masks[a]
            self.obs, r, done, info = self.env.step(a)
            flag = bool(info.get("flag_get", False))
            ep["ret"] += r
            ep["n"] += 1
            head = self._pending[-1][0]
            head["step"] = {"a": a, "r": round(float(r), 4), "ret": round(ep["ret"], 3), "n": ep["n"], **extra}
            self._pending[-1] = (*self._pending[-1][:3], self.obs[-1].tobytes())
        if done:
            result = {"flag": flag, "max_x": ep["max_x"], "ret": round(ep["ret"], 3), "n": ep["n"],
                      "time": int(info.get("time", 0)), "ep": self.episode}
            head["done"] = result
            self.history.append({"x": ep["max_x"], "flag": flag})
            self._hold_until = time.perf_counter() + (3.2 if flag else 1.3)
        elif self.mode == "demo" and self._demo_i >= len(self.demo["steps"]):
            self._hold_until = time.perf_counter() + 1.0      # bản ghi chưa về đích đã hết: phát lại từ đầu
        self._flush()

    # ------------------------------------------------------------------ lệnh từ trình duyệt
    def _apply(self, cmd: dict) -> None:
        name, value = cmd.get("cmd"), cmd.get("value")
        if name == "mode" and value in MODES:
            if value == "agent" and not self.ckpts:
                return self.toast("Chưa có checkpoint. Chạy: python play3d.py --ckpt <file .pt>", error=True)
            if value == "demo" and self._demo() is None:
                return self.toast(f"Chưa có bản ghi mẫu cho màn {self.stage}", error=True)
            self.mode = value
            self._build_env()
        elif name == "stage" and value in STAGES:
            self.stage = value
            self._build_env()
        elif name == "ckpt" and isinstance(value, int) and 0 <= value < len(self.ckpts):
            if self._load_ckpt(value):
                self.mode = "agent"
                self._build_env()
        elif name == "speed" and value in SPEEDS:
            self.speed = float(value)
            self.broadcast_state()
        elif name == "pause":
            self.paused = bool(value)
            self.broadcast_state()
        elif name == "step":
            self._step_once = True
        elif name == "det":
            self.deterministic = bool(value)
            self.broadcast_state()
        elif name == "reset":
            self._reset_episode()

    # ------------------------------------------------------------------ giao tiếp
    def state(self) -> dict:
        demo = getattr(self, "demo", None)
        return {
            "type": "state", "mode": self.mode, "stage": self.stage, "speed": self.speed,
            "paused": self.paused, "det": self.deterministic, "ckpt": self.ckpt_index,
            "actions": [" ".join(a) for a in self.env.action_meanings] if self.env else [],
            "frame_skip": self.kw["frame_skip"] if self.env else 4,
            "has_demo": demo is not None, "has_agent": bool(self.ckpts),
            "goal_x": demo["max_x"] if demo and demo.get("flag") else None,
            "history": self.history[-200:],
        }

    def hello(self) -> dict:
        ckpts = [{"name": p.name, "dir": p.parent.name} for p in self.ckpts]
        return {"type": "hello", "size": [WIDTH, HEIGHT], "stages": STAGES, "speeds": SPEEDS,
                "ckpts": ckpts, "state": self.state()}

    def _send_all(self, msg: dict) -> None:
        text = json.dumps(msg, ensure_ascii=False)
        with self._lock:
            for c in self.clients:
                c.offer(text, important=True)

    def broadcast_state(self) -> None:
        self._send_all(self.state())

    def toast(self, text: str, error: bool = False) -> None:
        self._send_all({"type": "toast", "text": text, "error": error})

    def add_client(self, client) -> None:
        self.cmds.put({"cmd": "_join", "client": client})

    def remove_client(self, client) -> None:
        with self._lock:
            self.clients.discard(client)

    def stop(self) -> None:
        self._quit.set()

    # ------------------------------------------------------------------ vòng chính
    def run(self) -> None:
        self._build_env()
        while not self._quit.is_set():
            try:
                while True:
                    cmd = self.cmds.get_nowait()
                    if cmd.get("cmd") == "_join":      # khách mới: gửi cấu hình + khung gần nhất (để có hình cả khi đang dừng)
                        cmd["client"].offer(json.dumps(self.hello(), ensure_ascii=False), important=True)
                        if self._last_data:
                            cmd["client"].offer(self._last_data, important=True)
                        with self._lock:
                            self.clients.add(cmd["client"])
                    else:
                        self._apply(cmd)
            except queue.Empty:
                pass
            if not self.clients:                       # không ai xem thì không tốn CPU
                time.sleep(0.05)
                self._next_t = None
                continue
            if self._hold_until:                       # dừng hình một nhịp sau khi ván kết thúc
                if time.perf_counter() < self._hold_until:
                    time.sleep(0.01)
                    continue
                self._reset_episode()
                self.broadcast_state()
                continue
            if self.paused and not self._step_once:
                time.sleep(0.01)
                self._next_t = None
                continue
            self._step_once = False
            self._advance()
        if self.env is not None:
            self.env.close()
