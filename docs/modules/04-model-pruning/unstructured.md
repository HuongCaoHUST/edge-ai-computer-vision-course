# Unstructured pruning

## 1. Ý tưởng

Unstructured pruning loại bỏ từng weight độc lập. Với global magnitude pruning,
các weight có trị tuyệt đối nhỏ nhất trên toàn bộ layer được đặt về 0 cho đến
khi đạt tỷ lệ `--amount` yêu cầu.

Ưu điểm là có thể đạt sparsity cao mà thay đổi ít topology. Nhược điểm là
ma trận vẫn có cùng kích thước; nếu CPU/GPU dùng dense kernel, phép nhân với
số 0 vẫn được thực hiện.

Phần fine-grained pruning trong [Lab1.ipynb](Lab1.ipynb) là ví dụ đầy đủ của
quy trình này: chọn magnitude threshold, tạo mask, sensitivity scan và
fine-tune với mask được áp dụng lại sau mỗi bước cập nhật.

## 2. Chạy thực hành

Từ thư mục gốc repository:

```bash
python docs/modules/04-model-pruning/pruning_baseline.py \
  --checkpoint runs/best.pt \
  --method unstructured \
  --amount 0.30 \
  --out results/prune_unstructured_30.pt \
  --report results/prune_unstructured_30.json \
  --device cpu
```

Chạy thêm `0.50` và `0.70`. Nếu dùng Jetson hoặc GPU, đổi `--device cuda`,
nhưng phải dùng cùng device cho tất cả phiên bản.

Script thực hiện:

1. nạp checkpoint FP32;
2. tạo bản sao và áp dụng `global_unstructured` trên Conv2d/Linear;
3. gỡ re-parametrization để lưu weight đã có số 0;
4. đo parameter sparsity, kích thước checkpoint, latency p50/p95, FPS, CPU và RSS;
5. lưu JSON để dùng cho bảng báo cáo.

## 3. Quan sát cần giải thích

Thường sẽ thấy `prunable_sparsity` gần với `--amount`, nhưng kích thước file
không giảm tương ứng và latency có thể không tăng. Đây không phải lỗi; đó là
hệ quả của lưu tensor dense và dense convolution.

Để có lợi ích thật, cần một trong các hướng sau:

- lưu sparse format và dùng sparse kernel được phần cứng hỗ trợ;
- chuyển sang block-sparse pattern phù hợp backend;
- compact/rebuild model sau khi chọn weight hoặc channel;
- dùng runtime có kernel khai thác sparsity.

Không được gọi việc weight bằng 0 là “đã giảm 30% thời gian suy luận” nếu
benchmark không chứng minh điều đó.

## 4. Đánh giá chất lượng

Chạy evaluator Lab 3 trên checkpoint `prune_unstructured_30.pt`, giữ nguyên
validation set và các ngưỡng. Lưu AP từng lớp để xem pruning ảnh hưởng lớp
nào mạnh nhất. So sánh tối thiểu các mức 0%, 30%, 50% và 70%.
