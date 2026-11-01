# Lab 4 · Cắt tỉa mô hình

## Cắt tỉa mô hình và đánh giá hiệu năng trên thiết bị biên

Lab 4 tiếp tục dùng checkpoint **SSDLite320 MobileNetV3-Large** và pipeline
đánh giá của Lab 2–3. Mục tiêu không phải chỉ làm cho nhiều weight bằng 0,
mà là kiểm tra pruning có tạo ra lợi ích thực tế trên thiết bị hay không.

## Mục tiêu

Sau khi hoàn thành lab, học viên có thể:

- giải thích nguyên lý, mục tiêu và các tiêu chí đánh giá pruning;
- thực hành **unstructured pruning** và **structured pruning** bằng PyTorch;
- tính sparsity, tỷ lệ tham số bằng 0 và tỷ lệ filter bị loại bỏ;
- so sánh mAP/AP, kích thước model, latency, FPS, CPU và RAM trước/sau pruning;
- phân tích đánh đổi giữa độ chính xác, mức pruning, dung lượng và hiệu năng;
- nhận biết khi nào mask pruning chỉ làm weight bằng 0 nhưng chưa làm inference nhanh hơn.

## Đầu vào và sản phẩm

- `runs/best.pt` từ Lab 2;
- cùng validation/test set và evaluator từ Lab 3;
- cùng input `320×320`, preprocessing, NMS, device và số thread khi so sánh.

Sản phẩm tối thiểu:

| Sản phẩm | Nội dung |
| --- | --- |
| `results/prune_*.pt` | checkpoint sau pruning |
| `results/prune_*.json` | sparsity, kích thước, latency, CPU và RSS |
| `results/quality_pruning.json` | mAP/AP của FP32 và các mức pruning |
| `results/report.md` | bảng so sánh và phân tích đánh đổi |

## Lộ trình

1. [Nguyên lý và tiêu chí](guide.md)
2. [Notebook thực hành](practice.md)
3. [Unstructured pruning](unstructured.md)
4. [Structured pruning](structured.md)
5. [Đánh giá chất lượng và hiệu năng](evaluation.md)
6. [Bài tập và nộp bài](assignment.md)

Script thực hành: [pruning_baseline.py](pruning_baseline.py).
