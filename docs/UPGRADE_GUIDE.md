# Train lại từ đầu trên mọi màn

Trạng thái: mã, cấu hình và notebook đã sẵn sàng; kiểm thử đơn vị và mô phỏng toàn bộ
notebook (trừ bước train) đã chạy qua. **Chưa có lần train thật nào với cách làm này**,
nên chưa có số liệu về việc vượt được bao nhiêu màn.

## Đọc gì và tận dụng gì

| Repo đã đọc mã | Điểm liên quan | Áp dụng ở đây |
|---|---|---|
| [Viet Nguyen / PPO Mario](https://github.com/vietnh1009/Super-mario-bros-PPO-pytorch/blob/ab4248d715346c6adc33c2157455e2b98c130bcc/src/env.py) | `CustomReward` cộng phần tăng score / 40, thưởng cờ, phạt chết | Adapt score-delta shaping, giữ attribution và MIT trong `THIRD_PARTY_NOTICES.md`; hệ số cấu hình được |
| [SugarmanZhu / Mario](https://github.com/SugarmanZhu/Mario/blob/main/wrappers.py) | Thưởng tăng điểm, coin bonus, bộ hành động đầy đủ | Tham khảo thiết kế; không chép toàn wrapper và không đổi sang RGB/IMPALA gây mất tương thích |
| [SugarmanZhu / multi-level trainer](https://github.com/SugarmanZhu/Mario/blob/main/train_ppo.py) | Phân màn cho worker và đánh giá theo màn | Tận dụng bộ chọn màn khi reset sẵn có của dự án; thêm profile đa màn và thống kê từng màn |
| [mgelsinger / mario-ai-trainer](https://github.com/mgelsinger/mario-ai-trainer/blob/main/env_wrappers.py) | Tách wrapper, bridge Gym/Gymnasium, recorder | Đã đối chiếu; reward chủ yếu tiến độ nên không giải quyết trực tiếp mục tiêu điểm cao. Không thay stack thư viện chỉ để có dashboard |
| [CleanRL](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari.py), [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) | Tách code thuật toán, công cụ, test, docs | Tổ chức lại thư mục, giữ PPO/CNN hiện có; không cài framework mới |

Đây là tham khảo cách triển khai, không coi demo hoặc README của repo khác là bằng
chứng mô hình này sẽ đạt cùng kết quả. Không tải checkpoint, ROM hay dataset từ các repo.

## Mục tiêu thưởng

Reward cũ khuyến khích tiến về bên phải và về đích. Bỏ qua địch là hành vi hợp lý
với mục tiêu đó. Reward mới bổ sung phần tăng điểm gameplay:

```text
reward = reward_scale * (
    tổng reward gốc qua frame-skip
    + score_weight * tổng phần tăng score trước cờ
    + coin_bonus * số xu thu được trước cờ
    + thưởng cờ hoặc phạt kết thúc thất bại
)
```

Mặc định của thư viện vẫn `score_weight=0`, `coin_bonus=0` để checkpoint và cấu hình
cũ không âm thầm đổi ý nghĩa. Profile `configs/stage.json` dùng `score_weight=0.025`
(bằng hệ số upstream 1/40), `flag_bonus=500`, `death_penalty=50`, `reward_scale=0.01`.
Một sự kiện +100 điểm được thêm 0,025 reward; về đích thêm 5; chết bị trừ 0,5. Mục tiêu
chính là vượt màn, điểm số chỉ là phần thưởng phụ. `coin_bonus=0` vì score đã thưởng xu.
Đây là giả thuyết khởi đầu, chưa tối ưu.

Theo dõi `score`, `score_gain`, `coins_collected`, `base_return` và `score_reward_raw`.
`score_gain` loại khung chạm cờ và các khung sau đó để tránh thưởng cuối màn lấn át
gameplay. Bộ đếm xu xử lý vòng 99 → 0. Mỗi emulator step được tính một lần.
`score_gain` bao gồm diệt địch, lấy vật phẩm, phá block và xu: **không phải số địch
đã diệt**. Muốn kết luận về chiến đấu cần xem video và ghi nhận sự kiện thực tế;
không suy ra kill count bằng cách chia score cho 100.

## Mỗi màn một mô hình

`notebooks/train_all_stages.ipynb` train lần lượt 32 màn 1-1 → 8-4 (hoặc danh sách bạn
chọn trong `STAGES`). Mỗi màn là một run `ppo_<màn>` dùng `configs/stage.json`:

- Bộ 12 hành động (`complex`, có đi/nhảy sang trái) cho mọi màn, để mô hình màn trước
  khởi tạo trọn vẹn cho màn sau (`WARM_START = True`). Màn đầu danh sách train từ đầu.
  Khởi tạo chỉ lấy phần trích đặc trưng và actor; critic và optimizer làm lại từ đầu.
- Dừng sớm khi 100 ván train gần nhất về đích ít nhất `TARGET_FLAG_RATE` (90%); nếu
  không, dừng ở `MAX_STEPS_PER_STAGE` (5 triệu bước) rồi chuyển sang màn kế. Màn lâu đài
  x-4 kết thúc bằng rìu và được thư viện tính là về đích như cờ.
- Sao lưu Drive mỗi 20 phút vào `runs/ppo_<màn>/`, chỉ giữ 2 bản mới nhất mỗi màn để
  32 màn không làm đầy Drive. File bị xoá nằm trong Thùng rác Drive 30 ngày và vẫn tính
  vào dung lượng cho tới khi Drive tự dọn.
- Chạy lại notebook sau khi Colab ngắt: màn đã xong được khôi phục và bỏ qua, màn dở train
  tiếp từ bản sao gần nhất (mất tối đa khoảng 20 phút tiến độ). Tăng `MAX_STEPS_PER_STAGE`
  thì các màn đã hết bước mà chưa vượt sẽ được train tiếp.

Thời gian tham khảo: lần train 1-1 trước chạy khoảng 416 bước/giây với 16 môi trường và
cần khoảng 1,3 triệu bước (gần 1 giờ) để về đích 90%. Màn khó hơn có thể cần đủ 5 triệu
bước (khoảng 3,5 giờ ở tốc độ đó). Cả 32 màn là hàng chục giờ, tức nhiều phiên Colab;
runtime ít CPU sẽ chậm hơn nhiều. Không có gì bảo đảm mọi màn đều vượt được: các màn mê
cung như 4-4, 7-4, 8-4 thường khó nhất với PPO.

Logic nằm trong `mario_rl/campaign.py` (lập kế hoạch new/resume/done, chạy trainer, bảng
tình trạng); notebook sinh bằng `python scripts/build_stage_notebook.py`.

## Đánh giá và video demo

Ô đánh giá chạy mô hình của từng màn trên chính màn đó (20 ván, argmax, xuất phát lệch
ngẫu nhiên tới 30 bước) và lưu vào `evaluations/<thời điểm>/`. `evaluate_suite` vẫn dùng
được để đánh giá chéo nhiều checkpoint trên cùng ma trận màn/seed.

`python -m mario_rl.cli.demo --item 1-1=<ckpt> --item 1-2=<ckpt> --episodes 3 --out <thư mục>`
quay mỗi màn vài ván, mỗi ván một video (`<màn>_ep<k>.mp4`), cộng video tổng
`all_stages.mp4` ghép tất cả (tua nhanh x2, có tên màn và kết quả). Ván 1 chơi argmax,
các ván sau lấy mẫu từ chính sách (`--mode mix`); mô hình đã học chắc thì các ván vẫn có
thể gần giống nhau. Có ffmpeg (Colab có sẵn) thì video được chuyển sang H.264.

## Colab và Drive

Upload `mario_train_drive.zip` (tạo bằng `python scripts/package_drive.py`) vào My Drive
hoặc thư mục con tới 2 cấp. Notebook tự tìm gói, lấy bản mới nhất nếu có nhiều, và lưu
`runs/`, `evaluations/`, `demos/` cạnh gói. Nếu runtime thiếu hoặc có phiên bản khác của
`gym`, `nes-py`, `gym-super-mario-bros` (Colab có sẵn `gym 0.25.x`), notebook cài đúng phiên
bản đã chốt rồi vá `nes-py` cho NumPy 2. Notebook tự dùng GPU nếu có và chọn số môi trường
theo số CPU (16 môi trường khi có từ 10 CPU).

## Kiểm tra

`pytest tests` (38 test, chỉ CPU) chạy qua trên Windows, gồm `tests/test_campaign.py`:
lập kế hoạch new/resume/done, khôi phục từ Drive, dừng sớm khi vượt màn, dọn bản sao cũ,
quay video trên emulator thật. Toàn bộ notebook đã được mô phỏng trên máy với Drive giả
qua hai phiên (phiên sau khôi phục và bỏ qua màn đã xong), chạy thật giải nén, đánh giá và
quay video; riêng bước train được thay bằng checkpoint không tối ưu. **Chưa** chạy train
thật, nên số màn vượt được và thời gian thực tế trên Colab chưa được đo.
