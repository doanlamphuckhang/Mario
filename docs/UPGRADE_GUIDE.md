# Nâng cấp mục tiêu và thí nghiệm

Trạng thái: đã chuẩn bị mã, cấu hình và notebook; kiểm thử đơn vị đã chạy qua,
**chưa fine-tuning hoặc train mới**. Chưa có kết quả chứng minh điểm tăng hoặc khả
năng chơi nhiều màn được cải thiện. Checkpoint cũ giữ nguyên.

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
cũ không âm thầm đổi ý nghĩa. Profile mới thử `score_weight=0.1`, `flag_bonus=500`,
`death_penalty=50`, `reward_scale=0.01`. Đây là giả thuyết khởi đầu, chưa tối ưu.
Một sự kiện +100 điểm được thêm 0,1 reward; +1.000 điểm thêm 1 reward; về đích thêm 5.
`coin_bonus=0` vì score đã thưởng xu; thêm coin bonus có thể làm agent chỉ tìm xu.
Hệ số upstream là 0,025; nên đối chứng 0 / 0,025 / 0,1 trong các run riêng với cùng
các tham số khác và nhiều seed huấn luyện khi có ngân sách.

Theo dõi `score`, `score_gain`, `coins_collected`, `base_return` và `score_reward_raw`.
`score_gain` loại khung chạm cờ và các khung sau đó để tránh thưởng cuối màn lấn át
gameplay. Bộ đếm xu xử lý vòng 99 → 0. Mỗi emulator step được tính một lần.
`score_gain` bao gồm diệt địch, lấy vật phẩm, phá block và xu: **không phải số địch
đã diệt**. Muốn kết luận về chiến đấu cần xem video và ghi nhận sự kiện thực tế;
không suy ra kill count bằng cách chia score cho 100.

## Hai giai đoạn đề xuất

1. `configs/score_1-1.json`: dùng checkpoint đã có, giữ 7 hành động, học mục tiêu
   điểm trên 1-1 trong ngân sách đề xuất 1 triệu bước. Khởi tạo lại critic/optimizer
   để thích nghi reward, lấy lại CNN/actor. NOOP đầu ván tối đa 30 khung.
2. Sau khi đánh giá: `configs/multistage.json`, học đồng thời `1-1,1-2,1-3,1-4,2-1`,
   12 hành động (thêm nhảy/chạy trái, lên/xuống), ngân sách đề xuất 5 triệu bước.
   Khi đổi bộ hành động, ánh xạ theo tổ hợp nút; hành động mới có prior thấp nhưng
   vẫn được học. Màn 1-1 vẫn xuất hiện để luyện lại kỹ năng cũ.

Đây là hai profile thực nghiệm, không phải một curriculum tự động đã được chứng minh.
Mỗi episode chọn đều một màn trong danh sách; số bước trên từng màn có thể lệch vì
độ dài episode khác nhau. `stage_metrics.json` theo dõi 100 ván gần nhất **mỗi màn**;
không dùng tỷ lệ thắng trộn chung để kết luận mọi màn đều học tốt.

`init_from` tạo thí nghiệm mới, không tiếp tục optimizer cũ. `resume` tiếp tục đúng
thí nghiệm đang dở; đặt `init_from=""` khi resume. Hai chế độ không bật cùng lúc.
Đổi mục tiêu reward/màn/action dùng tên run mới. `ckpt_best.pt` vẫn được chọn theo
return train; phải đánh giá các checkpoint trước khi chọn bản demo, đặc biệt khi đa màn.

## Đánh giá công bằng

`python -m mario_rl.cli.evaluate_suite` đánh giá checkpoint cũ/mới trên cùng danh
sách màn, seed, argmax/lấy mẫu, mức NOOP và số ván. Mặc định 8 màn (World 1 và 2),
3 seed đánh giá, 2 chế độ, 20 ván mỗi tổ hợp = 960 ván/checkpoint. Có thể chạy ma trận
nhỏ trước khi mở rộng; ma trận đầy đủ chưa được chạy.

Ghi `summary.csv`, `episodes.jsonl` (từng ván và config), `protocol.json`, SHA-256
checkpoint và cờ màn đã train. Vượt giới hạn bước thì báo lỗi và giữ các case đã
hoàn thành; không tính kết quả dở thành 0%. Không ghi đè thư mục đánh giá cũ.

Tiêu chí chấp nhận cần đo:

- Trên 1-1: tỷ lệ thắng vẫn ít nhất 95% và score gameplay trung bình tăng so với
  checkpoint cũ trong cùng giao thức; xem video để xác nhận địch thực sự bị diệt.
- Từng màn train mới: báo tỷ lệ thắng/score/xu riêng, tránh trung bình che màn thất bại.
- `2-2,2-3,2-4` giữ riêng cho kiểm tra cuối nếu train đúng profile; không chọn
  checkpoint hay tuning dựa trên điểm của những màn giữ riêng này.
- Thử nhiều seed **huấn luyện**. Nhiều seed đánh giá của một checkpoint không thay
  thế nhiều lần train độc lập. Khi so các seed train, dùng cùng checkpoint nguồn nếu
  kiểm tra fine-tuning; kết luận vẫn phụ thuộc checkpoint nguồn đó.

Không so return trực tiếp giữa reward cũ/mới. Các số score/xu chưa có trong CSV cũ
không thể khôi phục từ return; phải đánh giá lại checkpoint cũ khi được phép chạy.

## Colab và Drive

Notebook ngắn: `notebooks/train_upgrade.ipynb`. Gói dự kiến: `mario_upgrade_v2.zip`,
chưa tạo; dùng `scripts/package_upgrade.py` sau khi được phép chạy workload.
Upload gói nguồn vào `MyDrive/Mario/`; checkpoint hiện có upload riêng vào
`MyDrive/Mario/checkpoints/ppo_1-1/ckpt_final.pt` (~20,3 MB). Sửa `DRIVE_ROOT` nếu
bạn dùng vị trí khác. Không cần upload lại dữ liệu cũ vào gói nguồn.

Notebook kiểm manifest, tạo workspace mới, đọc profile JSON và gọi CLI. Không
nhúng trainer, không tải model. Nếu runtime thiếu `gym`, `nes-py` hoặc
`gym-super-mario-bros`, notebook cài đúng phiên bản đã chốt (không nâng cấp gói có sẵn)
rồi vá `nes-py` cho NumPy 2; đặt `INSTALL_MISSING = False` nếu chỉ muốn kiểm tra.
Biến `DEVICE` mặc định `cpu`; chỉ chọn `cuda` khi chủ động muốn dùng GPU.
Huấn luyện chạy foreground; chạy đánh giá sau khi trainer đã thoát. Backup Drive
vẫn theo chu kỳ và cuối run, gồm cả telemetry theo màn và thông tin chuyển checkpoint.

Đổi `PROFILE` sang `multistage` sau khi chọn được checkpoint giai đoạn 1. Notebook
có thể restore bản backup hợp lệ của run nguồn từ `DRIVE_RUNS`, hoặc dùng file tại
`checkpoints/<run>/ckpt_final.pt`. Khi muốn resume, giữ tên run, bật `RESUME`, tăng
`TOTAL_STEPS`; config cũ được lấy lại từ backup và `init_from` được xóa khỏi lệnh.

## Kiểm tra

`tests/test_upgrade.py` gồm các trường hợp: score delta không cộng lặp, coin rollover,
bỏ thưởng cờ, reset, tích lũy frame-skip, ánh xạ action và giữ critic mới, config
override, telemetry thiếu, đồng hồ đầu màn, đủ ma trận đánh giá và không ghi đè kết quả.
Toàn bộ `pytest tests` (30 test, chỉ CPU) đã chạy qua trên Windows. Đã chạy thử đánh
giá ngắn `ppo_1-1/ckpt_final.pt` trên emulator thật. **Chưa** chạy train mới, ma trận
`evaluate_suite` đầy đủ hay fine-tuning với hai profile ở trên.

Dự án đã chuyển sang Git; lịch sử thay đổi nằm trong commit. Các bản sao thủ công cũ
trong `results/*backup*/` vẫn để ở máy nhưng không đưa lên repository.
