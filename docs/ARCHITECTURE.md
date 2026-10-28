# Cấu trúc dự án

Tham khảo cách tách package, tests, docs và scripts của
[CleanRL](https://github.com/vwxyzjn/cleanrl) và
[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3).
Giữ flat package để dùng trên Colab và chạy từ repository mà không phải cài thêm package.

```text
Mario/
├── mario_rl/                 # package chính
│   ├── envs.py               # emulator, preprocessing, vector environments
│   ├── score_reward.py       # reward tăng điểm và thống kê gameplay
│   ├── models.py             # mạng actor/critic
│   ├── ppo.py                # vòng tối ưu và log huấn luyện
│   ├── transfer.py           # chuyển đặc trưng/hành động từ checkpoint
│   ├── evaluation.py         # đánh giá từng episode và tóm tắt
│   ├── run_io.py             # quyền sở hữu run, checkpoint nguyên tử
│   ├── drive_backup.py       # backup/restore Drive
│   └── cli/                  # train, evaluate, evaluate_suite, play, baselines, plots
├── configs/                  # profile JSON có thể so sánh và tái lập
├── notebooks/                # notebook điều phối thí nghiệm mới
├── tests/                    # kiểm thử công thức, workflow, regression
├── scripts/                  # đóng gói, tạo notebook, công cụ phát triển
├── docs/                     # báo cáo, kiến trúc, hướng dẫn và nguồn tham khảo
├── viewer3d/                 # trình xem, tách khỏi trainer
├── runs/                     # checkpoint, config đã dùng, CSV từng run
├── results/                  # đánh giá, video, hình và bản sao trước thay đổi
└── train.py, evaluate.py, …   # entry point tương thích với các lệnh cũ
```

Lệnh chuẩn, chạy từ thư mục gốc:

```text
python -m mario_rl.cli.train --config configs/score_1-1.json
python -m mario_rl.cli.evaluate --ckpt runs/ppo_1-1/ckpt_final.pt --device cpu
python -m mario_rl.cli.evaluate_suite --ckpts runs/ppo_1-1/ckpt_final.pt --out results/eval_v2
python -m mario_rl.cli.play --ckpt runs/ppo_1-1/ckpt_final.pt
```

Các lệnh trên là ví dụ cho người dùng, không tự khởi chạy. Agent trên Windows phải
dùng runner đã cài và chỉ chạy sau khi thỏa điều kiện tài nguyên/ủy quyền.

CLI đọc tham số; package xử lý nghiệp vụ. Notebook gọi CLI thay vì nhúng bản sao
trainer. Config JSON chỉ cung cấp giá trị mặc định; cờ CLI cụ thể được ưu tiên.
Đường dẫn dữ liệu trong config được tính từ thư mục làm việc, nên luôn chạy ở gốc dự án.
Checkpoint lưu toàn bộ config đã phân giải. Không di chuyển dữ liệu chạy cũ.

`colab_train.ipynb` ở gốc giữ nguyên output của thí nghiệm trước. Notebook mới là
`notebooks/train_upgrade.ipynb`. Hai ZIP nguồn tách tên để tránh nhầm phiên bản.
Tài liệu gốc `BAO_CAO.md`, `DRIVE_START_HERE.md` dẫn đến bản chính trong `docs/`.
