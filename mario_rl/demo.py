"""Quay video demo: nhiều màn, nhiều ván mỗi màn, kèm một video tổng ghép tất cả.

Mỗi ván ra một file `<màn>_ep<k>.mp4` (60 fps, 512x480). Video tổng `all_stages.mp4` có dòng chữ
màn/ván ở trên cùng và một khung kết quả ngắn sau mỗi ván. Có ffmpeg (Colab có sẵn) thì video được
chuyển sang H.264 để phát được trong trình duyệt và notebook; không có thì giữ mã mp4v của OpenCV.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
import torch

from .envs import MarioEnv
from .evaluation import load_model

SIZE = (512, 480)
FPS = 60.0


class VideoFile:
    """Ghi khung BGR ra mp4; đóng lại thì chuyển sang H.264 nếu có ffmpeg."""

    def __init__(self, path: Path, ffmpeg: str | None) -> None:
        self.path, self.ffmpeg = Path(path), ffmpeg
        self.raw = self.path.with_name(self.path.stem + '.mp4v.mp4') if ffmpeg else self.path
        self.writer = cv2.VideoWriter(str(self.raw), cv2.VideoWriter_fourcc(*'mp4v'), FPS, SIZE)
        if not self.writer.isOpened():
            raise OSError(f'Không mở được file video để ghi: {self.raw}')
        self.frames = 0

    def write(self, frame_bgr: np.ndarray) -> None:
        self.writer.write(frame_bgr)
        self.frames += 1

    def close(self) -> Path:
        self.writer.release()
        if self.ffmpeg:
            subprocess.run([self.ffmpeg, '-y', '-loglevel', 'error', '-i', str(self.raw), '-c:v', 'libx264',
                            '-pix_fmt', 'yuv420p', '-crf', '23', '-movflags', '+faststart', str(self.path)],
                           check=True)
            self.raw.unlink()
        return self.path


def _banner(frame: np.ndarray, text: str, color=(255, 255, 255)) -> np.ndarray:
    """Dải chữ trên cùng (OpenCV chỉ vẽ ASCII nên nhãn trên video viết không dấu)."""
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (SIZE[0], 30), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    return out


MODES = ('mix', 'argmax', 'sample')


def record(items: list[tuple[str, Path]], out_dir: Path, episodes: int = 2, mode: str = 'mix',
           noop_max: int = 30, seed: int = 0, max_steps: int = 3000, device: torch.device | None = None,
           combined: bool = True, combined_every: int = 2) -> list[dict]:
    """items: danh sách (màn, checkpoint). Trả về kết quả từng ván và ghi demo.json vào out_dir.

    mode: 'mix' = ván 1 argmax (cách chơi tốt nhất), các ván sau lấy mẫu từ π để mỗi video một khác;
    'argmax' hoặc 'sample' cho mọi ván. Argmax trên emulator tất định thường lặp lại cùng một đường chơi.
    """
    if mode not in MODES:
        raise ValueError(f'mode must be one of {MODES}')
    if episodes <= 0 or max_steps <= 0 or combined_every <= 0 or not items:
        raise ValueError('Cần ít nhất một màn; episodes, max_steps, combined_every phải dương')
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)          # không ghi đè demo cũ
    device = device or torch.device('cpu')
    ffmpeg = shutil.which('ffmpeg')
    models: dict[Path, tuple] = {}
    results: list[dict] = []
    total = VideoFile(out_dir / 'all_stages.mp4', ffmpeg) if combined else None
    speed = f'  x{combined_every}' if combined_every > 1 else ''
    try:
        for stage, ckpt in items:
            ckpt = Path(ckpt)
            if ckpt not in models:
                models[ckpt] = load_model(ckpt, device)
            model, cfg = models[ckpt]
            env = MarioEnv(stages=stage, actions=cfg['actions'], frame_skip=cfg['frame_skip'],
                           reward_scale=cfg['reward_scale'], flag_bonus=cfg['flag_bonus'],
                           death_penalty=cfg.get('death_penalty', 0.0), score_weight=cfg.get('score_weight', 0.0),
                           coin_bonus=cfg.get('coin_bonus', 0.0), noop_max=noop_max, seed=seed)
            torch.manual_seed(seed)
            try:
                for ep in range(1, episodes + 1):
                    video = VideoFile(out_dir / f'{stage}_ep{ep}.mp4', ffmpeg)
                    deterministic = mode == 'argmax' or (mode == 'mix' and ep == 1)
                    policy = 'argmax' if deterministic else 'sample'
                    label = f'World {stage}   Ep {ep}/{episodes}   {policy}{speed}'
                    counter = [0]
                    last = [None]

                    def hook(screen: np.ndarray) -> None:
                        frame = cv2.cvtColor(cv2.resize(screen, SIZE, interpolation=cv2.INTER_NEAREST),
                                             cv2.COLOR_RGB2BGR)
                        video.write(frame)
                        last[0] = frame
                        if total is not None and counter[0] % combined_every == 0:
                            total.write(_banner(frame, label))
                        counter[0] += 1

                    env.frame_hook = hook
                    obs, done, steps, info = env.reset(), False, 0, {}
                    while not done and steps < max_steps:
                        with torch.no_grad():
                            action = model.act(torch.as_tensor(obs, device=device).unsqueeze(0), deterministic)
                        obs, _, done, info = env.step(int(action.item()))
                        steps += 1
                    ep_info = info.get('episode') or {}
                    flag = bool(ep_info.get('flag', False))
                    result = {'stage': stage, 'episode': ep, 'policy': policy, 'flag': flag, 'max_x': env._max_x, 'steps': steps,
                              'time_left': int(info.get('time', 0)), 'score': env._score,
                              'truncated': not done, 'checkpoint': str(ckpt),
                              'file': video.close().name}
                    results.append(result)
                    if total is not None and last[0] is not None:   # khung kết quả ~1 giây
                        text = (f'World {stage} Ep {ep}: CLEAR  ({steps} steps)' if flag else
                                f'World {stage} Ep {ep}: x = {env._max_x}' + ('  (time cap)' if not done else ''))
                        card = _banner(last[0], text, (80, 255, 80) if flag else (80, 80, 255))
                        for _ in range(int(FPS / combined_every)):
                            total.write(card)
            finally:
                env.close()
    finally:
        if total is not None:
            total.close()
    summary = {'episodes_per_stage': episodes, 'mode': mode, 'noop_max': noop_max,
               'seed': seed, 'max_steps': max_steps, 'h264': bool(ffmpeg),
               'combined': 'all_stages.mp4' if combined else None, 'results': results}
    (out_dir / 'demo.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    return results
