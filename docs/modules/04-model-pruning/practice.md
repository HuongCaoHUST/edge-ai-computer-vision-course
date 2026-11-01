# Lab 4 · Notebook thực hành

Notebook chính của phần thực hành là [Lab1.ipynb](Lab1.ipynb). Notebook này
dùng VGG trên CIFAR-10 để minh họa rõ hai kỹ thuật pruning trước khi áp dụng
các tiêu chí tương tự cho SSD MobileNet của khóa học:

| Trong notebook | Tương ứng trong Lab 4 |
| --- | --- |
| Fine-grained pruning | Unstructured pruning, từng weight bằng 0 |
| Channel pruning | Structured pruning, giảm số output/input channel |
| `get_sparsity` | Mức độ thưa của tensor/model |
| `profile_macs` | Số phép tính lý thuyết |
| `measure_latency` | Latency và speedup thực tế |

## 1. Chuẩn bị và chạy notebook

Notebook gốc sử dụng GPU CUDA, CIFAR-10, VGG checkpoint và `torchprofile`.
Có thể mở trực tiếp bằng Jupyter hoặc tải lên Google Colab:

```bash
python -m pip install torchprofile
jupyter lab docs/modules/04-model-pruning/Lab1.ipynb
```

Nếu chạy CPU, cần thay các lệnh `.cuda()` bằng việc dùng một biến `device`
nhất quán và bỏ assertion yêu cầu CUDA. Thời gian sensitivity scan và
fine-tune sẽ dài hơn đáng kể.

## 2. Các phần cần hoàn thành

### Phần A — Fine-grained pruning

1. Đo accuracy, kích thước và histogram của model dense.
2. Hoàn thành `fine_grained_prune()` bằng magnitude `abs(weight)`.
3. Kiểm tra sparsity trên tensor mẫu và tạo `FineGrainedPruner`.
4. Chạy sensitivity scan theo từng layer.
5. Chọn `sparsity_dict` dựa trên độ nhạy và số tham số của layer.
6. Fine-tune nhưng phải áp dụng mask sau mỗi bước cập nhật để weight đã cắt
   không quay lại khác 0.

### Phần B — Channel pruning

1. Hoàn thành `get_num_channels_to_keep()` và `channel_prune()`.
2. Kiểm tra channel pruning làm giảm shape của Conv2d và BatchNorm đồng bộ.
3. Tính importance của input channel bằng Frobenius/L2 norm.
4. Sắp xếp channel trước khi cắt để giữ lại channel quan trọng hơn.
5. Fine-tune model đã compact và đo lại accuracy.

### Phần C — Đánh giá

Chạy cell đo cuối notebook và ghi lại:

- accuracy trước/sau pruning và sau fine-tune;
- sparsity và số weight khác 0;
- số tham số, kích thước model;
- MACs trước/sau channel pruning;
- latency CPU, speedup và tỷ lệ giảm computation.

Notebook yêu cầu trả lời 9 câu hỏi, trong đó câu 8 giải thích vì sao giữ lại
70% channel có thể làm MACs còn khoảng `0.7² ≈ 49%`, còn câu 9 so sánh
fine-grained với channel pruning.

## 3. Liên hệ với SSD MobileNet của khóa học

Notebook là bài thực hành nền tảng trên VGG/CIFAR-10 để học viên thấy rõ
thuật toán compact channel. Sau khi hoàn thành notebook, dùng
[pruning_baseline.py](pruning_baseline.py) với checkpoint `runs/best.pt` của
Lab 2 để kiểm tra hiện tượng trên SSD MobileNet.

Lưu ý: script SSD dùng mask-based pruning để an toàn với detection head và
depthwise convolution; nó không tự compact toàn bộ topology. Vì vậy kết quả
size/FPS có thể khác notebook. Đây chính là phần phân tích đánh đổi cần nêu
trong báo cáo.
