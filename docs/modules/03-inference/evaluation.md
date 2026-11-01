# Đánh giá chất lượng và hiệu năng

## 1. Đánh giá chất lượng sau lượng tử hóa

Đối với từng phiên bản FP32, FP16, INT8-PTQ và INT8-QAT, dùng cùng một tập test để tính:

- Precision và Recall;
- AP theo từng lớp;
- mAP tại ngưỡng IoU đã chọn;
- số false positive và false negative;
- tỷ lệ thay đổi prediction so với FP32.

Nên báo cáo cả độ thay đổi tuyệt đối và tương đối:

```text
ΔmAP = mAP_quantized - mAP_FP32
relative_drop = (mAP_FP32 - mAP_quantized) / mAP_FP32 × 100%
speedup = latency_FP32 / latency_quantized
```

Ví dụ, latency giảm từ `20 ms` xuống `10 ms` tương ứng speedup `2×`, không phải tăng `10 FPS`. FPS cần được tính lại từ latency của cùng một pipeline.

Với object detection, cần giữ nguyên cách ghép box, ngưỡng IoU, confidence và NMS khi so sánh. Nếu thay đổi các điều kiện này, chênh lệch có thể không còn phản ánh riêng tác động của lượng tử hóa.

## 2. Đo kích thước và tốc độ

Với baseline static PTQ mô phỏng bằng PyTorch, cần tách hai kết luận: weights/activation đã được lượng tử hóa, nhưng latency có thể không giảm vì forward vẫn dequantize và dùng phép `Conv2d` FP32. Chỉ gọi là tăng tốc INT8 khi runtime và phần cứng thực sự thực hiện kernel integer.

Đo trên cùng một thiết bị và cùng số lần lặp:

```text
warm-up → đo model load → đo preprocess → đo inference → đo postprocess
```

Ghi các trường tối thiểu:

```text
format, method, model_size_mb, load_ms,
latency_mean_ms, latency_p50_ms, latency_p95_ms,
fps, cpu_percent, ram_mb
```

Sau warm-up, nên đo nhiều lần và báo cáo mean cùng p50/p95. Nếu chỉ chạy một lần, kết quả bị ảnh hưởng bởi cache, CPU scheduling hoặc lần cấp phát bộ nhớ đầu tiên. Khi so sánh hai model, dùng cùng số lần lặp và cùng trạng thái nguồn/nhiệt độ của thiết bị.

Không chỉ đo thời gian `inference` nếu ứng dụng thực tế còn phải resize, copy tensor, NMS và vẽ/ghi frame. Báo cáo cả latency của inference riêng và toàn pipeline.

## 3. Mức tiêu thụ tài nguyên

So sánh CPU, RAM và accelerator utilization giữa các định dạng. Một model INT8 nhỏ hơn chưa chắc nhanh hơn nếu runtime phải chuyển đổi dtype hoặc phần cứng không có kernel INT8 tối ưu.

Với thiết bị chạy pin hoặc giới hạn nhiệt, ghi thêm công suất và nhiệt độ nếu có thể. Quan sát FPS ở đầu và cuối phiên để phát hiện throttling.

## 4. Phân tích đánh đổi

Lập bảng kết luận theo ba trục:

| Trục | Câu hỏi |
| --- | --- |
| Độ chính xác | mAP giảm bao nhiêu so với FP32? Có lớp nào giảm mạnh không? |
| Tốc độ | latency và FPS có đạt yêu cầu thời gian thực không? |
| Dung lượng/tài nguyên | model nhỏ hơn bao nhiêu? CPU/RAM có giảm không? |

Một lựa chọn triển khai hợp lý là lựa chọn thỏa ngưỡng chất lượng và latency của ứng dụng, không nhất thiết là model có file nhỏ nhất.

Có thể biểu diễn quyết định bằng ràng buộc:

```text
chọn model có mAP ≥ mAP_min,
latency_p95 ≤ latency_max,
size_mb ≤ storage_budget,
ram_mb ≤ memory_budget
```

Nếu có nhiều model cùng thỏa điều kiện, ưu tiên model có biên an toàn lớn hơn về latency và RAM thay vì chỉ chọn model có mean FPS cao nhất.

## 5. Kết luận

Kết luận cần chỉ ra:

1. PTQ hay QAT giữ chất lượng tốt hơn trong thí nghiệm;
2. FP16 hay INT8 phù hợp hơn với thiết bị đã đo;
3. mức giảm kích thước và mức tăng tốc thực tế;
4. nguyên nhân nếu INT8 không cải thiện tốc độ;
5. phiên bản model được chọn và đánh đổi chấp nhận được.
