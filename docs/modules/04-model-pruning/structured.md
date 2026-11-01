# Structured pruning

## 1. Ý tưởng

Structured pruning loại bỏ các nhóm weight có cấu trúc, trong lab là toàn bộ
filter/output channel của Conv2d. Khi một filter bị đặt toàn bộ về 0, feature
do filter đó tạo ra không còn đóng góp.

Khác với unstructured pruning, structured pruning có cơ hội giảm FLOPs và
latency rõ ràng hơn sau khi **rebuild model** với ít channel hơn. Tuy nhiên,
việc giảm channel phải cập nhật đồng bộ layer kế tiếp, BatchNorm, residual
connection và detection head. Không thể chỉ xóa một filter khỏi tensor rồi
tiếp tục dùng kiến trúc cũ.

Phần channel pruning trong [Lab1.ipynb](Lab1.ipynb) minh họa việc cắt output
channel của Conv2d, input channel của layer kế tiếp và các vector BatchNorm
đồng bộ với nhau. Đây là structured pruning compact thật sự trên mạng VGG.

## 2. Chạy mask-based structured pruning

```bash
python docs/modules/04-model-pruning/pruning_baseline.py \
  --checkpoint runs/best.pt \
  --method structured \
  --amount 0.20 \
  --out results/prune_structured_20.pt \
  --report results/prune_structured_20.json \
  --device cpu
```

Chạy thêm `0.30` và `0.40`. Script dùng `ln_structured(..., n=2, dim=0)`
trên Conv2d, nghĩa là chọn filter có L2 norm nhỏ hơn trong từng layer. JSON
bao gồm `zero_filters`, `filter_count` và `filter_sparsity`.

## 3. Mask-based chưa phải compact model

Trong bài này, structured pruning được dùng để minh họa nguyên lý và đo tác
động chất lượng. Số output channel của layer vẫn giữ nguyên, nên:

- `filter_sparsity` có thể tăng;
- checkpoint dense thường không nhỏ đi;
- latency có thể gần bằng baseline;
- FLOPs của graph chưa tự giảm.

Đây là một kết quả thí nghiệm cần ghi nhận. Bước mở rộng là xây dựng channel
dependency graph, chọn channel giữ lại, khởi tạo model nhỏ hơn, copy weight
phù hợp và fine-tune. Với SSD MobileNet, detection head và các depthwise
convolution làm bước này phức tạp hơn CNN phân loại đơn giản, vì vậy không
được tự ý xóa channel mà không kiểm tra shape.

## 4. So sánh với unstructured

Lập bảng cho cùng mức `amount` gần nhau:

| Phương án | Sparsity weight | Sparsity filter | mAP | p95 (ms) | FPS | Size (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 | 0 | 0 |  |  |  |  |
| Unstructured 30% |  | — |  |  |  |  |
| Structured 30% |  |  |  |  |  |  |

Giải thích vì sao structured có ý nghĩa triển khai tốt hơn về lý thuyết,
nhưng phiên bản mask-based trong lab vẫn chưa chắc nhanh hơn. Nếu cần chứng
minh speedup, phải có compact model hoặc sparse-aware runtime.
