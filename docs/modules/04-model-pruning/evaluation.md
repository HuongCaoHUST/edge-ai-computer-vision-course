# Đánh giá chất lượng và hiệu năng sau pruning

## 1. Đánh giá mAP bằng pipeline Lab 3

Dùng đúng evaluator của Lab 3 cho từng checkpoint. Không thay đổi confidence
threshold, NMS hoặc tập ảnh giữa các phiên bản. Một bảng kết quả nên có dạng:

| Model | Method | Amount | mAP@0.5 | mAP@0.5:0.95 | ΔmAP | AP thấp nhất |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FP32 | — | 0 |  |  | 0 |  |
| Pruned | Unstructured | 0.30 |  |  |  |  |
| Pruned | Structured | 0.30 |  |  |  |  |

Nếu có fine-tune, thêm một dòng riêng, ví dụ `Structured 30% + FT 3 epochs`.
Không ghi đè kết quả trước fine-tune.

## 2. Tính compression và speedup

```text
compression = size_FP32 / size_variant
speedup_mean = latency_FP32_mean / latency_variant_mean
speedup_p95 = latency_FP32_p95 / latency_variant_p95
ΔmAP = mAP_variant - mAP_FP32
```

Nếu `compression` gần `1×` dù sparsity cao, checkpoint vẫn đang được lưu ở
dạng dense. Nếu speedup gần `1×`, backend hiện tại chưa khai thác sparsity.
Đây là kết luận hợp lệ và phải được nêu trong báo cáo.

## 3. Benchmark trên thiết bị biên

Chạy script với `--warmup 20 --iterations 200` cho kết quả ổn định hơn khi
đo chính thức. Bản chạy thử có thể dùng mặc định `10/30`.

Ngoài JSON của script, dùng collector tài nguyên từ Lab 2 để ghi CPU, RAM và
nhiệt độ trong một phiên inference liên tục. Ghi rõ:

- model thiết bị, chế độ nguồn và nhiệt độ môi trường;
- phiên bản Python, PyTorch, torchvision;
- số thread và backend;
- input size, batch size, warm-up và iterations;
- latency inference riêng và latency pipeline đầy đủ.

Khi chạy trên CPU, có thể cố định số thread bằng `--threads 1`, `--threads 2`
hoặc `--threads 4`; không so sánh hai model với số thread khác nhau.

Không so sánh số đo laptop CPU với Jetson GPU như thể cùng một runtime.

## 4. Phân tích đánh đổi

Vẽ hoặc lập bảng quan hệ giữa `sparsity → mAP → p95 → FPS → RAM`. Kết luận
phải trả lời:

1. Mức pruning nào còn giữ mAP trên ngưỡng yêu cầu?
2. Unstructured hay structured ảnh hưởng AP lớp nào nhiều hơn?
3. Mức sparsity nào bắt đầu làm chất lượng giảm rõ rệt?
4. Có speedup thực tế hay chỉ có thêm số 0 trong dense tensor?
5. Phiên bản nào thỏa đồng thời:

```text
mAP ≥ mAP_min
latency_p95 ≤ latency_max
size_mb ≤ storage_budget
ram_mb ≤ memory_budget
```

Model được chọn để triển khai là model thỏa các ràng buộc trên, không nhất
thiết model có sparsity hoặc mAP cao nhất riêng lẻ.
