# Lộ trình khóa học

## Mục tiêu tổng quát

Sau khóa học, học viên có thể xây dựng một pipeline object detection hoàn chỉnh và phân tích khả năng chạy trên thiết bị biên:

- Chuẩn bị môi trường và dữ liệu.
- Huấn luyện SSD MobileNet bằng Python.
- Chạy inference ảnh, video và camera với OpenCV.
- Đánh giá mAP, Precision, Recall và AP.
- Đo FPS, độ trễ, CPU, RAM và nhiệt độ.
- Viết báo cáo có số liệu tái lập được.

## Bản đồ nội dung

| Phần | Trọng tâm | Sản phẩm chính |
| --- | --- | --- |
| [Lab 1](modules/01-environment/index.md) | Môi trường | Phiên bản công cụ, `pip list` |
| [Lab 2](modules/02-training/index.md) | Huấn luyện | `runs/log.csv`, `curves.png`, `best.pt` |
| [Lab 3](modules/03-inference/index.md) | Inference OpenCV | Ảnh/video kết quả, thời gian từng bước |
| [Lab 4](modules/04-evaluation/index.md) | Đánh giá chất lượng | `results/quality.json`, AP từng lớp |
| [Lab 5](modules/05-benchmark/index.md) | FPS và độ trễ | Benchmark theo thiết bị |
| [Lab 6](modules/06-resources/index.md) | Tài nguyên và báo cáo | CSV tài nguyên, báo cáo tổng kết |
| [Lab 7](modules/07-topic/index.md) | Chưa đặt tên | Sẽ bổ sung |
| [Lab 8](modules/08-topic/index.md) | Chưa đặt tên | Sẽ bổ sung |

## Cấu trúc mỗi lab

Mỗi lab sẽ được phát triển theo cùng một khung:

1. **Tổng quan** — mục tiêu, yêu cầu đầu vào, thời lượng.
2. **Hướng dẫn thực hành** — kiến thức, lệnh chạy và các bước kiểm chứng.
3. **Bài tập & thí nghiệm** — câu hỏi, biến cần thay đổi và kết quả cần ghi nhận.
4. **Nộp bài** — checklist, sản phẩm, rubric và xử lý sự cố.

!!! note "Trạng thái khung nội dung"
    Các trang đã được dựng sẵn để điền nội dung chi tiết. Nội dung 7–8 đang giữ placeholder vì workspace hiện chưa có tên và nội dung tương ứng trong `course.md`.
