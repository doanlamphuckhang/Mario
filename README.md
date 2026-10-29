# Mario RL: PPO tự chơi Super Mario Bros

Tác tử PPO học chơi Super Mario Bros từ ảnh. Lần train đầu đã vượt màn 1-1; bản hiện tại
train lại từ đầu, mỗi màn một mô hình, lần lượt qua cả 32 màn và quay video demo nhiều màn.

**Trạng thái:** `pytest tests` (38 test, chỉ CPU) chạy qua; toàn bộ notebook đã được mô phỏng
trên máy (trừ bước train). Chưa có lần train thật nào theo cách làm mới, nên chưa có số liệu
về số màn vượt được.

Checkpoint (`*.pt`) và gói ZIP sinh ra không nằm trong Git vì dung lượng lớn; chỉ có cấu
hình và log CSV của các run. Viewer 3D có sẵn bản ghi mẫu màn 1-1 để xem khi chưa có checkpoint.

## Train tất cả các màn trên Colab

1. Tạo gói nguồn: `python scripts/package_drive.py` (không ghi đè gói có sẵn; xoá gói cũ trước).
2. Upload `mario_train_drive.zip` vào My Drive.
3. Mở [notebooks/train_all_stages.ipynb](notebooks/train_all_stages.ipynb) trên Colab, chọn GPU,
   chạy lần lượt các ô: train từng màn → bảng tình trạng → đánh giá → video demo.
4. Colab ngắt giữa chừng: Restart session, chạy lại từ ô đầu; màn xong được bỏ qua, màn dở train tiếp.

Chi tiết: [hướng dẫn](docs/UPGRADE_GUIDE.md), [cấu trúc mã](docs/ARCHITECTURE.md),
[profile mỗi màn](configs/stage.json). Lệnh tương đương từ thư mục gốc:

```bash
python -m mario_rl.cli.train --config configs/stage.json --stages 1-2 --run_name ppo_1-2
python -m mario_rl.cli.evaluate --ckpt runs/ppo_1-2/ckpt_final.pt --stage 1-2 --episodes 20
python -m mario_rl.cli.demo --item 1-1=runs/ppo_1-1/ckpt_final.pt --item 1-2=runs/ppo_1-2/ckpt_final.pt --episodes 3 --out results/demo
```

Mỗi lệnh train là một workload dài; không chạy đồng thời. Lệnh gốc `python train.py ...`
vẫn hoạt động qua lớp tương thích.

Chỉ một báo cáo Word tổng hợp được dự kiến xuất tại `docs/BAO_CAO_MARIO.docx`, không
chèn ảnh demo. Nội dung nằm trong `docs/report_content.json`, mã tạo là
`scripts/build_report.py`; **chưa xuất hoặc render Word**. Báo cáo Markdown ban đầu
được giữ trong [docs/BAO_CAO.md](docs/BAO_CAO.md) để bảo toàn tài liệu có sẵn.

## Luồng Colab ban đầu

Tạo gói nguồn bằng `python scripts/package_drive.py` (gói ZIP không nằm trong Git; script không ghi đè
gói có sẵn, nên đổi tên hoặc xóa gói cũ trước). Upload `mario_train_drive.zip` vào `MyDrive/Mario/` và mở
`colab_train.ipynb` mới trong Colab.
Cell đầu mount Drive; cell kế tiếp kiểm checksum và giải nén mã `.py` từ ZIP xuống ổ runtime.
Notebook không còn nhúng mã nguồn. Train chạy cục bộ và tự sao lưu checkpoint lên Drive sau update
đầu tiên, mỗi khoảng 5 phút và khi kết thúc. Xem `DRIVE_START_HERE.md` để biết cách resume/lưu kết quả.
Sinh notebook bằng `python scripts/build_colab_notebook.py`; đóng gói bằng `python scripts/package_drive.py`.

Mỗi lần train mới cần một `RUN_NAME` chưa có dữ liệu. Chạy lại ô khởi chạy khi tiến trình còn sống sẽ báo lỗi;
khóa thư mục run cũng ngăn hai trainer ghi chung checkpoint/log. File `.train.lock` có thể còn sau khi kết thúc;
khóa thực tế được hệ điều hành giải phóng khi tiến trình thoát, không cần xóa file này.

Để tiếp tục, giữ nguyên cấu hình của lần chạy cũ, đặt `RESUME = True` và đặt `TOTAL_STEPS` lớn hơn số bước
đã lưu. Notebook khôi phục bản sao Drive hợp lệ mới nhất nếu local run còn trống. Resume yêu cầu
`ckpt_latest.pt` hợp lệ; không có checkpoint thì chọn tên run mới. Log mới hơn checkpoint
được sao lưu vào `resume-backup-*` trước khi đồng bộ lại. Điểm chọn `ckpt_best.pt` được giữ qua resume.
Resume khôi phục model/optimizer và lịch sử đánh giá gần nhất; môi trường game được reset, không phải phát lại
chính xác trạng thái ngẫu nhiên và emulator tại thời điểm gián đoạn.

Batch `NUM_ENVS * ROLLOUT_LEN` phải chia hết cho `minibatches`, mỗi minibatch ít nhất 2 mẫu. Đánh giá song song
chạy đủ hạn ngạch ván; `max_steps` giới hạn từng ván và báo lỗi rõ ràng nếu vượt, thay vì trả kết quả thiếu.

## Cấu trúc

```
mario_rl/            mã nguồn chính
  envs.py            MarioEnv (frame-skip, xám 84x84, stack 4, reward shaping) + SubprocVecEnv (đa tiến trình)
  models.py          mạng Actor-Critic (CNN kiểu Nature-DQN)
  ppo.py             PPO: GAE, hàm mất mát clip, vòng huấn luyện, log, checkpoint
  evaluation.py      đánh giá nhiều ván (song song), quay mp4/GIF
train.py             huấn luyện (CLI)
evaluate.py          đánh giá checkpoint, quay video (CLI)
baselines.py         agent ngẫu nhiên / chính sách cố định để so sánh
play3d.py            viewer 3D: xem agent chơi (hoặc tự chơi) dưới dạng diorama voxel trong trình duyệt
viewer3d/            mã của viewer 3D
  scene.py           tách khung NES thành lớp cảnh nền / khối / nhân vật từ RAM + ROM
  session.py         vòng mô phỏng thời gian thực (agent, bản ghi mẫu, người chơi, baseline)
  server.py          máy chủ HTTP + WebSocket cục bộ (chỉ thư viện chuẩn)
  web/               giao diện Three.js (world.js dựng voxel, stage.js ánh sáng/máy quay, post.js hậu kỳ, ui.js)
  demos/             chuỗi hành động mẫu theo màn (sinh bằng scripts/make_demo_replay.py)
plot_results.py      vẽ đường cong học từ runs/<tên>/metrics.csv
make_figures.py      hình minh hoạ cho báo cáo
colab_train.ipynb    notebook train trên Colab
scripts/             cài nes-py (Windows / Colab), build notebook, smoke test, kiểm chứng công thức reward,
                     tìm bản ghi mẫu cho viewer
tests/               kiểm thử: phép tính tay trong báo cáo (test_core), resume/khóa run/đánh giá
                     (test_pretrain), reward điểm số/chuyển checkpoint (test_upgrade),
                     backup Drive (test_drive_workflow), máy chủ viewer (test_viewer)
```

## Chạy cục bộ (Windows, không cần Visual Studio)

```bash
python -m venv --system-site-packages .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe scripts/setup_nes_py_windows.py   # build nes-py bằng zig, vá cho NumPy 2
.venv/Scripts/python.exe -m pytest tests                   # kiểm thử (chỉ CPU, khoảng 10 giây)

python train.py --run_name ppo_main --stages 1-1 --total_steps 3000000
python evaluate.py --ckpt runs/ppo_main/ckpt_final.pt --stage 1-1 --episodes 20 --video results/demo_1-1.mp4
python plot_results.py --runs ppo_main --out results/figs
```

Mọi siêu tham số là cờ dòng lệnh (xem `python train.py --help`). `nes-py` chỉ có mã nguồn (cần trình biên dịch C++)
và viết cho NumPy 1.x; chi tiết ở mục 3.2 của báo cáo.

## Viewer 3D

Dựng lại màn chơi thành diorama voxel 3D chạy trong trình duyệt, cập nhật trực tiếp từ emulator. Viewer chỉ suy luận
và chạy emulator, không huấn luyện, nên chạy được trên máy không có GPU rời.

```bash
python play3d.py                                      # tự tìm checkpoint trong runs/; chưa có thì phát bản ghi mẫu màn 1-1
python play3d.py --ckpt runs/ppo_1-1/ckpt_final.pt    # xem agent PPO (giải nén mario_results.zip từ Colab vào thư mục gốc)
python play3d.py --stage 1-2 --mode human             # tự chơi: mũi tên di chuyển, Z/Space nhảy, X/Shift chạy
```

Trình duyệt mở `http://localhost:8765`. Trong giao diện có thể đổi người cầm tay cầm (agent PPO, bản ghi mẫu, tự chơi,
hai baseline), đổi màn, tốc độ 0,25× đến 4×, tạm dừng / tiến từng bước, và bốn góc máy (kéo chuột để xoay, lăn để zoom).
Bảng bên trái hiển thị khung NES gốc, ảnh 84×84 mà agent thật sự nhìn thấy, phân phối π(a|s), giá trị V(s) và phần
thưởng từng bước; thanh dưới đánh dấu vị trí kết thúc của các ván trước.

Cách dựng 3D: `nes-py` chỉ cho ảnh RGB và 2 KB RAM, nên `viewer3d/scene.py` đọc bản sao OAM (`$0200`) cùng CHR-ROM để
tách sprite chính xác từng điểm ảnh, đọc block buffer (`$0500`) cùng bảng metatile trong PRG-ROM để biết ô nào là khối,
phần còn lại khác màu nền là cảnh trang trí. Mỗi lớp được đùn khối với bề dày riêng; phần màn đã đi qua được giữ lại
theo toạ độ thế giới. RAM đi trước ảnh đúng một khung, viewer đã bù độ trễ này.

Ghi chú:
- Trang cần mạng để tải three.js từ CDN (jsdelivr) và phông chữ; emulator và mô hình chạy hoàn toàn trên máy.
- Mặc định tối ưu cho GPU tích hợp (FXAA, bóng cứng, độ phân giải tự hạ khi quá tải). Bật "Nét cao" nếu có GPU rời.
- Bản ghi mẫu màn 1-1 là kết quả tìm kiếm ngẫu nhiên có quay lui trên emulator (khoảng 20 giây CPU), không phải agent
  đã học. Thử cho màn khác: `python scripts/make_demo_replay.py --stage 1-2`; script mới chỉ được kiểm chứng với 1-1,
  màn khó hơn có thể hết giới hạn thời gian mà chưa về đích (khi đó bản ghi dừng ở chỗ xa nhất tìm được).
