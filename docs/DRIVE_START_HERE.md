# Chạy Mario PPO bằng mã nguồn trên Google Drive

1. Tạo thư mục `MyDrive/Mario`, upload **mario_train_drive.zip** vào đó.
2. Upload bản **colab_train.ipynb mới** lên Colab, chọn runtime GPU rồi chạy từ đầu.
3. Cell **1. Mount Google Drive**: cấp quyền mount. Nếu lưu ZIP nơi khác, đổi `DRIVE_ROOT`.
4. Cell **2. Lấy mã nguồn từ Drive**: kiểm checksum, giải nén ZIP vào `/content/mario_*`.
   Notebook chạy các file `.py` từ gói này, không còn code nguồn nhúng `%%writefile`.
5. Chạy các cell cài thư viện, kiểm tra, cấu hình và train. Đặt tên `RUN_NAME` mới cho run mới.

## Checkpoint và kết quả

Train trên ổ cục bộ Colab. Chính tiến trình train sao lưu đồng bộ lên
`MyDrive/Mario/runs/<RUN_NAME>/` sau update đầu tiên, mỗi khoảng 300 giây
(ở cuối update) và khi train hoàn tất. Dừng cell theo dõi không dừng trainer hoặc sao lưu.
Log `[Drive backup]` xác nhận một bản sao đã ghi và kiểm SHA-256 xong.

Mỗi bản sao là ZIP có phiên bản riêng và file `.sha256` xác nhận hoàn tất. Bản sao có
model, optimizer, config, CSV và checkpoint tốt nhất; không kèm checkpoint mốc/video.
Không tự xóa phiên bản cũ. Cần dự trù dung lượng Drive; lỗi sao lưu làm trainer dừng
và hiện lỗi, không tiếp tục chạy mà âm thầm bỏ qua backup.

Cell **Lưu toàn bộ kết quả lên Drive** cuối notebook lưu cả checkpoint mốc, CSV và video
sau khi train/đánh giá kết thúc. Các mục đối chứng/tổng quát hóa là tùy chọn; train đối chứng
không tự backup định kỳ, nhưng được đưa vào ZIP kết quả cuối.

## Resume trên runtime mới

Chạy lại từ cell mount/source, đặt cùng `RUN_NAME`, `RESUME=True`, tăng `TOTAL_STEPS`
nếu cần. Notebook chọn bản sao hoàn tất mới nhất, xác minh checksum và khôi phục xuống
ổ Colab. Cấu hình tương thích (kể cả số env/rollout cũ) được giữ nguyên.
Nếu runtime hiện tại đã có dữ liệu run, notebook dùng checkpoint cục bộ, không ghi đè nó.
Không chạy cùng RUN_NAME từ hai runtime Colab đồng thời: khóa trainer chỉ bảo vệ
workspace cục bộ, không phải khóa phân tán trên Drive.

Có thể mất phần tiến độ sau bản sao hoàn tất gần nhất nếu runtime bị thu hồi giữa chừng.
Resume không khôi phục chính xác trạng thái emulator/RNG. Các thao tác Drive đã kiểm
tra bằng filesystem mô phỏng cục bộ; chưa xác minh mount thật trên Colab.

## Thời gian

Mặc định 3 triệu bước. Dùng sps trong log: thời gian còn lại = bước còn lại / sps.
500 sps ~100 phút; 1.000 sps ~50 phút; 2.000 sps ~25 phút cho phần train.
Đây là kịch bản tốc độ, không phải benchmark GPU. Upload ZIP không bao gồm môi trường
ảo, ROM riêng, weights, kết quả cũ hoặc thông tin đăng nhập.
