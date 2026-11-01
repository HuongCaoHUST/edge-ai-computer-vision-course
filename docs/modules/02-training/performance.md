# FPS và độ trễ

## Mục tiêu

Đo tốc độ suy luận của SSD MobileNet trên từng thiết bị và xác định bước chiếm nhiều thời gian nhất.

## 1. Quy trình đo

Trước khi đo, chạy warm-up để loại ảnh hưởng của khởi tạo model, cấp phát bộ nhớ và cache. Sau đó giữ cố định số lần lặp, kích thước input, số thread và nguồn dữ liệu.

```bash
python lab4_benchmark.py --source voc --tag laptop
python lab4_benchmark.py --source voc --tag edge-device --threads 4
```

Đo riêng:

- `preprocess`: đọc, đổi màu, resize và chuẩn hóa;
- `inference`: thời gian model chạy;
- `postprocess`: confidence filtering, NMS và vẽ kết quả;
- `pipeline`: tổng thời gian từ frame đầu vào đến kết quả.

## 2. Các chỉ số

Báo cáo tối thiểu:

- mean và standard deviation;
- p50, p95, p99;
- latency của từng bước;
- FPS của pipeline.

Nếu latency tính bằng mili giây:

```text
FPS = 1000 / latency_ms
```

P95 cho biết phần lớn request có latency không vượt quá giá trị đó. Với ứng dụng thời gian thực, p95/p99 giúp phát hiện các lần chậm bất thường mà mean có thể che giấu.

## 3. Phân tích nút thắt

So sánh FPS của pipeline đầy đủ với FPS chỉ tính inference. Nếu inference nhanh nhưng pipeline chậm, cần tối ưu đọc frame, chuyển đổi màu, resize, NMS hoặc thao tác vẽ.

Thử `--threads` bằng `1`, `2` và `4`. FPS không nhất thiết tăng tuyến tính vì còn phụ thuộc số lõi CPU, bộ nhớ, contention và backend model.

## 4. So sánh thiết bị

Khi so sánh laptop, Raspberry Pi hoặc Jetson, ghi rõ:

- model thiết bị và chế độ nguồn;
- phiên bản Python, PyTorch, OpenCV;
- backend và số thread;
- kích thước input;
- số lần warm-up và số lần đo.

Chỉ kết luận thiết bị đạt thời gian thực khi nêu rõ ngưỡng mục tiêu, ví dụ `15 FPS` hoặc `30 FPS`.
