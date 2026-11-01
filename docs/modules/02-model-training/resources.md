# CPU, RAM và mức sử dụng hệ thống

## Mục tiêu

Đánh giá chi phí tài nguyên khi chạy inference SSD MobileNet trên thiết bị biên, gồm CPU, RAM và nhiệt độ nếu thiết bị cung cấp cảm biến.

## 1. Thu thập dữ liệu

Đo một khoảng idle trước, sau đó chạy inference trong thời gian cố định:

```bash
python lab5_resources.py \
  --source voc \
  --tag edge-device \
  --duration 60
```

Lưu chuỗi thời gian vào `results/resources_edge-device.csv`. Nên ghi mỗi khoảng 0.5 giây các trường:

- CPU của tiến trình;
- CPU toàn hệ thống;
- RSS hoặc RAM của tiến trình;
- phần trăm RAM toàn hệ thống;
- nhiệt độ và xung nhịp nếu thiết bị hỗ trợ.

Trên Jetson có thể ghi thêm:

```bash
tegrastats --interval 500 --logfile results/tegra.log
```

## 2. Phân tích CPU

So sánh CPU lúc idle và lúc inference. CPU tiến trình có thể vượt `100%` khi chương trình dùng nhiều lõi, vì vậy cần ghi rõ công cụ và cách tính phần trăm.

Quan sát CPU theo từng giai đoạn để xác định bước nào tạo tải lớn. Thử thay đổi số thread và kiểm tra FPS có tăng tương ứng hay không.

## 3. Phân tích RAM

Đo ba mốc:

1. RAM khi chương trình idle;
2. RAM ngay sau khi nạp model;
3. RAM đỉnh trong lúc chạy liên tục.

Nếu RAM tăng liên tục theo thời gian, kiểm tra việc lưu frame, prediction hoặc tensor mà không giải phóng. Dùng `eval()` và `torch.no_grad()` trong inference để tránh lưu graph không cần thiết.

## 4. Nhiệt độ và ổn định hệ thống

Vẽ CPU, nhiệt độ và FPS theo thời gian. Nếu nhiệt độ tăng rồi FPS giảm, có thể thiết bị đang thermal throttling. Khi đó cần kiểm tra tản nhiệt, chế độ nguồn, số thread và kích thước input.

## 5. Kết luận đánh đổi

Kết hợp số liệu tài nguyên với mAP và FPS để trả lời:

- thiết bị nào đạt yêu cầu thời gian thực;
- model có dùng quá nhiều CPU hoặc RAM không;
- giảm kích thước input có thể tăng FPS bao nhiêu và ảnh hưởng chất lượng thế nào;
- lượng tử hóa, cắt tỉa hoặc thay backend có phải hướng tối ưu phù hợp không.
