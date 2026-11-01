# Lab 4 · Bài tập và nộp bài

## Phần A — Notebook thực hành

- [ ] Chạy [Lab1.ipynb](Lab1.ipynb) từ đầu đến cuối.
- [ ] Hoàn thành và giải thích 9 câu hỏi trong notebook.
- [ ] Nộp kết quả fine-grained pruning và channel pruning trên VGG/CIFAR-10.
- [ ] Ghi accuracy, sparsity, số tham số, MACs, kích thước và latency.

## Phần B — Mở rộng với SSD MobileNet

- [ ] Đo baseline FP32 trên thiết bị được chọn.
- [ ] Chạy unstructured pruning ở mức 30%, 50%, 70%.
- [ ] Chạy structured pruning ở mức 20%, 30%, 40%.
- [ ] Đánh giá mAP/AP bằng evaluator của Lab 3.
- [ ] Đo kích thước checkpoint, sparsity weight/filter, latency p50/p95, FPS, CPU và RSS.
- [ ] Fine-tune một mức pruning nếu mAP giảm vượt ngưỡng do giảng viên đặt ra.

## Phần C — Phân tích

1. Vẽ đồ thị mAP theo sparsity cho hai phương pháp.
2. So sánh `filter_sparsity` và `prunable_sparsity`.
3. Giải thích tại sao 50% zero weight không nhất thiết làm model nhỏ hơn 50%.
4. Giải thích tại sao structured pruning có tiềm năng tăng tốc nhưng script
   mask-based có thể chưa tăng FPS.
5. Chọn một model triển khai cho thiết bị và nêu rõ các đánh đổi đã chấp nhận.

## Sản phẩm nộp

```text
results/prune_*.json
results/quality_pruning.json
results/plots_pruning.png hoặc bảng/đồ thị tương đương
results/report.md
```

`report.md` phải có cấu hình thí nghiệm, bảng số liệu, nhận xét về sai số và
kết luận dựa trên số đo thật. Không thay thế số đo bằng giá trị lý thuyết.

## Rubric gợi ý (100 điểm)

| Tiêu chí | Điểm |
| --- | ---: |
| Baseline và điều kiện so sánh tái lập được | 15 |
| Unstructured pruning và phân tích sparsity | 20 |
| Structured pruning và phân tích filter | 20 |
| mAP/AP trước và sau pruning | 20 |
| Benchmark latency/FPS/kích thước/RAM | 15 |
| Phân tích đánh đổi và đề xuất triển khai | 10 |

!!! warning "Diễn giải kết quả"
    Pruning bằng mask trong lab không tự compact topology. Nếu size hoặc FPS
    không cải thiện, đó là hiện tượng cần giải thích chứ không phải lý do để
    sửa số liệu.
