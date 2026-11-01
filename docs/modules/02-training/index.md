# Lab 2 · Huấn luyện

## Huấn luyện, triển khai và đánh giá hiệu năng mô hình SSD MobileNet trên thiết bị biên

Trong bài thực hành này, bạn xây dựng một pipeline object detection hoàn chỉnh:

`dữ liệu → huấn luyện → lưu checkpoint → triển khai → inference → đánh giá chất lượng → benchmark → đo tài nguyên`

## Nội dung chính

1. Huấn luyện mô hình SSD MobileNet bằng Python và bộ dữ liệu có sẵn.
2. Xây dựng chương trình inference sử dụng OpenCV.
3. Đánh giá chất lượng mô hình qua các chỉ số phù hợp.
4. Đo hiệu năng suy luận: FPS và độ trễ xử lý.
5. Đánh giá tài nguyên: CPU, RAM và mức sử dụng hệ thống.

## Mô hình và dữ liệu

- **Mô hình:** SSDLite320 với backbone MobileNetV3-Large.
- **Bộ dữ liệu:** Pascal VOC 2007, gồm 20 lớp đối tượng và 1 lớp nền (`num_classes = 21`).
- **Kích thước đầu vào:** `320×320`.
- **Đầu ra:** bounding box, nhãn lớp và confidence score.

Nên huấn luyện trên máy có GPU hoặc Google Colab, sau đó sao chép checkpoint sang Raspberry Pi, Jetson hoặc thiết bị edge cần đánh giá.

## Sản phẩm đầu ra

| Sản phẩm | Nội dung |
| --- | --- |
| `runs/log.csv` | loss train/validation theo epoch |
| `curves.png` | đường cong loss và các chỉ số theo dõi |
| `best.pt` | checkpoint tốt nhất |
| `results/out.jpg` hoặc video | kết quả inference có bounding box |
| `results/quality.json` | Precision, Recall, AP và mAP |
| `results/bench_<thiet-bi>.json` | FPS, mean, p50, p95, p99 và độ trễ |
| `results/resources_<thiet-bi>.csv` | CPU, RAM và nhiệt độ theo thời gian |
| `results/report.md` | phân tích và kết luận |

## Yêu cầu đầu vào

- Python 3.10 trở lên;
- PyTorch, torchvision, OpenCV, NumPy, Pillow và Matplotlib;
- GPU được khuyến nghị cho bước huấn luyện;
- một thiết bị edge để đo inference nếu có.

Các nội dung được chia theo từng bước của pipeline. Bắt đầu từ [Huấn luyện SSD MobileNet](training.md), sau đó thực hiện [Inference với OpenCV](inference.md), [Đánh giá chất lượng](evaluation.md), [FPS và độ trễ](performance.md), rồi [CPU, RAM và hệ thống](resources.md).
