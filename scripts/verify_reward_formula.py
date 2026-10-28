"""Kiểm chứng công thức phần thưởng ở mục 4.6 của báo cáo trên các ván đánh giá.

Với ván chạm cờ:
    R_dự_đoán = c · [ (x_cờ - x0) - (t_đầu - t_còn) + b_cờ + thưởng_điểm_thô ]
t_đầu là đồng hồ đầu ván (400 nếu log cũ không ghi), thưởng_điểm_thô = score_reward_raw
(bằng 0 với cấu hình cũ score_weight = coin_bonus = 0).
Đọc file JSON do `evaluate.py --json` ghi ra và so sánh với return đo được.
Một số ván lệch nhỏ (khoảng 0,06 với c = 0,01), đã thấy cả trong results/reward_formula_check.csv;
công thức dùng x lớn nhất nên không bắt được mọi chi tiết reward gốc của gym theo từng khung.

    python scripts/verify_reward_formula.py results/eval_main_1-1_det.json
"""
import json
import sys

X0 = 40


def predict(ep: dict, reward_scale: float, flag_bonus: float) -> float:
    time_used = ep.get("time_start", 400) - ep["time_left"]
    score_bonus = ep.get("score_reward_raw", 0.0)
    return reward_scale * ((ep["max_x"] - X0) - time_used + flag_bonus + score_bonus)


def main(path: str) -> None:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    cfg = data["reward_cfg"]
    c, b_flag = cfg["reward_scale"], cfg["flag_bonus"]
    rows, errs = [], []
    for i, ep in enumerate(data["episodes_detail"]):
        if not ep["flag"]:
            continue
        pred = predict(ep, c, b_flag)
        err = ep["return"] - pred
        errs.append(abs(err))
        rows.append((i, ep["max_x"], ep["time_left"], ep["return"], pred, err))
    print(f"{'ván':>4} {'x_cờ':>6} {'t_còn':>6} {'return đo':>12} {'dự đoán':>12} {'sai lệch':>10}")
    for r in rows[:15]:
        print(f"{r[0]:4d} {r[1]:6d} {r[2]:6d} {r[3]:12.4f} {r[4]:12.4f} {r[5]:10.5f}")
    if rows:
        print(f"\nSố ván chạm cờ: {len(rows)}; sai lệch tuyệt đối lớn nhất: {max(errs):.2e}")
    else:
        print("Không có ván nào chạm cờ trong file này.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
