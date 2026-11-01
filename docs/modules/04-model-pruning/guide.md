# Nguyên lý và tiêu chí đánh giá pruning

## 1. Pruning là gì?

Pruning loại bỏ hoặc làm không còn tác dụng một phần tham số, channel hoặc
filter ít quan trọng trong mạng. Mục tiêu là giảm chi phí lưu trữ, bandwidth
bộ nhớ, FLOPs, latency và năng lượng, trong khi vẫn giữ chất lượng đủ cho ứng
dụng.

Quy trình chuẩn của lab là:

```text
checkpoint FP32 → đo baseline → pruning → (fine-tune tùy chọn)
→ đánh giá chất lượng → benchmark trên cùng thiết bị
```

Không được dùng model sau pruning làm baseline mới. Mọi `ΔmAP`, speedup và
compression đều phải tính từ checkpoint FP32 ban đầu.

## 2. Tiêu chí chọn tham số để cắt

Một tiêu chí phổ biến là magnitude pruning: tham số có `|w|` nhỏ được xem là
ít quan trọng hơn. Với structured pruning, dùng chuẩn của một filter:

```text
score(filter) = ||W_filter||₂
```

Các tiêu chí khác có thể dùng trong nghiên cứu mở rộng là sensitivity theo
layer, gradient, Taylor score hoặc regularization. Trong lab này dùng
`L1Unstructured` cho unstructured và `L2-norm` theo output channel cho
structured để kết quả dễ tái lập.

## 3. Hai dạng pruning

| Dạng | Đơn vị bị cắt | Ưu điểm | Hạn chế |
| --- | --- | --- | --- |
| **Unstructured** | từng weight riêng lẻ | dễ đạt sparsity cao, ít thay đổi kiến trúc | dense kernel thường vẫn tính cả số 0 |
| **Structured** | filter/output channel hoặc block | có thể giảm kích thước và FLOPs sau khi compact | ảnh hưởng lớn hơn đến feature và cần giữ topology hợp lệ |

`torch.nn.utils.prune` tạo mask trên weight. Trong ví dụ của lab, sau khi
`prune.remove()` các phần tử bị cắt trở thành số 0 nhưng số channel của mạng
không tự giảm. Đây là điểm cần kiểm chứng bằng benchmark, không được suy ra
rằng sparsity cao luôn đồng nghĩa FPS cao.

Trong [notebook thực hành](practice.md), fine-grained pruning minh họa dạng
unstructured còn channel pruning thực hiện compact Conv2d/BatchNorm thật sự.
Vì vậy notebook cho thấy rõ sự khác nhau giữa “weight bằng 0” và “tensor có
ít channel hơn”.

## 4. Các đại lượng phải báo cáo

### Sparsity và tỷ lệ tham số bị loại bỏ

```text
sparsity = số weight bằng 0 / tổng số weight
removed_ratio = (N_before - N_after) / N_before
```

Với mask pruning không compact, `N_after` về mặt tensor vẫn bằng
`N_before`; vì vậy nên báo cáo rõ cả `zero_parameters` và `nonzero_parameters`.

Với structured pruning, bổ sung:

```text
filter_sparsity = số filter có toàn bộ weight bằng 0 / tổng số filter
```

### Chất lượng

Giữ nguyên evaluator của Lab 3 và báo cáo:

- Precision, Recall;
- AP từng lớp;
- mAP@0.5 và mAP@0.5:0.95;
- `ΔmAP = mAP_pruned - mAP_FP32`;
- số false positive/false negative nếu evaluator có hỗ trợ.

### Hiệu năng và tài nguyên

- kích thước file checkpoint trên đĩa;
- RSS sau khi load model và RSS trong lúc benchmark;
- latency mean, p50, p95 và FPS;
- CPU/RAM trung bình và đỉnh nếu chạy thêm collector của Lab 2;
- FLOPs hoặc số MACs nếu có công cụ đo, nhưng không dùng FLOPs lý thuyết thay
  cho latency thực tế.

## 5. Các điều kiện phải cố định

Khi so sánh FP32, unstructured và structured, cố định:

- checkpoint và class mapping;
- validation/test set;
- input size, preprocessing, batch size và postprocessing;
- warm-up, số iteration, số thread và chế độ nguồn của thiết bị;
- phiên bản PyTorch/torchvision, backend và device.

Nếu pruning làm mAP giảm quá ngưỡng cho phép, fine-tune vài epoch với learning
rate nhỏ rồi đánh giá lại. Ghi rõ đây là model **pruned + fine-tuned**, không
gộp nhầm với kết quả pruning ngay sau mask.
