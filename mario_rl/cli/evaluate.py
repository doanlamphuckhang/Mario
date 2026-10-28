"""Đánh giá checkpoint và quay video.

Ví dụ:
    python evaluate.py --ckpt runs/ppo_1-1/ckpt_final.pt --stage 1-1 --episodes 20
    python evaluate.py --ckpt runs/ppo_1-1/ckpt_final.pt --stage 1-1 --episodes 1 --video results/demo_1-1.mp4 --gif results/demo_1-1.gif
    python evaluate.py --ckpt runs/ppo_1-1/ckpt_final.pt --stage 1-2 --stochastic     # kiểm tra tổng quát hoá
"""
import argparse
import json
import sys

import torch

from mario_rl.evaluation import load_model, run_eval, run_eval_vec, summarize

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stage", default="1-1", help="ví dụ 1-1, 1-2, 4-1")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--stochastic", action="store_true", help="lấy mẫu từ π thay vì argmax")
    ap.add_argument("--noop_max", type=int, default=0, help="số bước NOOP ngẫu nhiên tối đa ở đầu ván")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--video", default=None)
    ap.add_argument("--gif", default=None)
    ap.add_argument("--gif_every", type=int, default=4)
    ap.add_argument("--json", default=None, help="ghi tóm tắt ra file JSON")
    ap.add_argument("--parallel", type=int, default=4, help="số tiến trình đánh giá song song (bỏ qua nếu quay video)")
    ap.add_argument("--device", choices=['cpu', 'cuda'], default='cpu')
    a = ap.parse_args()

    device = torch.device(a.device)
    model, cfg = load_model(a.ckpt, device)
    if a.video or a.gif:
        res = run_eval(model, cfg, a.stage, a.episodes, not a.stochastic, a.noop_max, a.seed, device,
                       a.video, a.gif, a.gif_every)
    else:
        res = run_eval_vec(model, cfg, a.stage, a.episodes, not a.stochastic, a.noop_max, a.seed, device,
                           a.parallel)
    for i, r in enumerate(res):
        print(f"ván {i:2d}: x_max={r['max_x']:5d} cờ={int(r['flag'])} thời gian còn={r['time_left']:3d} "
              f"bước={r['length']:4d} return={r['return']:.2f}")
    summ = summarize(res)
    print(json.dumps(summ, indent=2, ensure_ascii=False))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            reward_cfg = {k: cfg.get(k) for k in ("reward_scale", "flag_bonus", "death_penalty", "score_weight", "coin_bonus", "frame_skip", "actions")}
            json.dump({"ckpt": a.ckpt, "stage": a.stage, "deterministic": not a.stochastic,
                       "noop_max": a.noop_max, "reward_cfg": reward_cfg,
                       "episodes_detail": res, "summary": summ},
                      f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
