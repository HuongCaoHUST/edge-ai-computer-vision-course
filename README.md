# Tối ưu AI trên thiết bị biên

Course thực hành về tối ưu, triển khai và đánh giá mô hình AI trên thiết bị
biên. Nội dung xuyên suốt sử dụng **SSDLite320 MobileNetV3-Large** với
**Pascal VOC 2007**: 20 lớp đối tượng và 1 lớp nền (`num_classes=21`).

## Mục tiêu

Sau khóa học, học viên có thể xây dựng pipeline:

```text
bài toán → dữ liệu → huấn luyện → inference → tối ưu → export → triển khai
→ benchmark → phân tích tài nguyên
```

Học viên thực hành với PyTorch, OpenCV, ONNX Runtime, TensorRT, C++,
GStreamer và NVIDIA DeepStream tùy theo nền tảng thiết bị.

## Lộ trình course

| Lab | Nội dung chính | Kết quả mong đợi |
| --- | --- | --- |
| 1 | Giới thiệu Edge AI và quy trình phát triển ứng dụng AI | Đặc tả bài toán, pipeline và tiêu chí đánh giá |
| 2 | Huấn luyện, triển khai và đánh giá SSD MobileNet | Checkpoint, inference, mAP, FPS và tài nguyên |
| 3 | Lượng tử hóa mô hình | So sánh FP32, FP16, INT8; PTQ, calibration và QAT |
| 4 | Cắt tỉa mô hình và đánh giá trên thiết bị biên | Sparsity, mAP, kích thước, latency, FPS và trade-off |
| 5 | Chuyển đổi, kết xuất và triển khai mô hình | ONNX, TensorRT, TensorFlow Lite và runtime tương ứng |
| 6 | ONNX Runtime C++ trên Raspberry Pi | Inference C++, FPS, latency, CPU và RAM |
| 7 | TensorRT và C++ trên NVIDIA Jetson | Engine FP32/FP16/INT8, GPU, latency và công suất |
| 8 | GStreamer và DeepStream trên Jetson | Pipeline AI thời gian thực, tracking và benchmark |

## Cấu trúc tài liệu

```text
docs/
├── modules/
│   ├── 01-edge-ai-introduction/
│   ├── 02-model-training/
│   ├── 03-model-quantization/
│   ├── 04-model-pruning/
│   ├── 05-benchmark/
│   ├── 06-resources/
│   ├── 07-topic/
│   └── 08-topic/
└── lab/index.md
```

Trang bắt đầu: [Lộ trình course](docs/course.md) và [Danh sách Lab](docs/lab/index.md).

## Môi trường cơ bản

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision opencv-python numpy pillow matplotlib psutil
```

Kiểm tra thư viện:

```bash
python -c "import torch, torchvision, cv2; print(torch.__version__, torchvision.__version__, cv2.__version__)"
```

Huấn luyện nên thực hiện trên GPU hoặc Google Colab. Sau đó, chép checkpoint
`runs/best.pt` sang Raspberry Pi, Jetson hoặc thiết bị edge cần benchmark.

Để xem tài liệu cục bộ:

```bash
mkdocs serve
```

## Lab 4 — Pruning

Lab 4 có hai phần thực hành:

1. Notebook nền tảng [Lab1.ipynb](docs/modules/04-model-pruning/Lab1.ipynb):
   VGG/CIFAR-10, fine-grained pruning, channel pruning, sensitivity scan,
   fine-tune, MACs và latency.
2. Mở rộng trên SSD MobileNet bằng script:

```bash
python docs/modules/04-model-pruning/pruning_baseline.py \
  --checkpoint runs/best.pt \
  --method unstructured \
  --amount 0.30 \
  --out results/prune_unstructured_30.pt \
  --report results/prune_unstructured_30.json \
  --device cpu
```

Structured pruning:

```bash
python docs/modules/04-model-pruning/pruning_baseline.py \
  --checkpoint runs/best.pt \
  --method structured \
  --amount 0.20 \
  --out results/prune_structured_20.pt \
  --report results/prune_structured_20.json \
  --device cpu
```

Khi so sánh các phiên bản, giữ cố định checkpoint, tập validation, input size,
preprocessing, số thread, warm-up và số lần benchmark. Không kết luận pruning
đã tăng tốc nếu runtime chưa khai thác sparse kernel hoặc model chưa được
compact topology.

## Kết quả cần báo cáo

- chất lượng: Precision, Recall, AP và mAP;
- tối ưu: sparsity, số tham số khác 0, MACs và kích thước model;
- hiệu năng: latency mean/p50/p95, FPS và speedup;
- tài nguyên: CPU, RAM, nhiệt độ, GPU hoặc công suất nếu thiết bị hỗ trợ;
- phân tích đánh đổi và lựa chọn model đáp ứng ngưỡng ứng dụng.

## Ghi chú triển khai

- Raspberry Pi dùng Raspberry Pi OS 64-bit; có thể dùng
  `opencv-python-headless` nếu không cần hiển thị.
- Jetson nên dùng PyTorch/TensorRT theo phiên bản JetPack tương ứng.
- Kết quả benchmark phải ghi rõ model thiết bị, runtime, backend, precision,
  input shape, số thread và chế độ nguồn.
