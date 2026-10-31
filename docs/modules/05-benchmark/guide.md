# Lab 5 · Nội dung 5 · Khung hướng dẫn

## Các phần nội dung

1. Warm-up và thiết lập số lần đo.
2. Đo pipeline đầy đủ và riêng bước inference.
3. Thu thập mean, std, p50, p95, p99.
4. So sánh `--threads` và nhiều thiết bị.

```bash
python lab4_benchmark.py --source voc --tag laptop
python lab4_benchmark.py --source voc --tag pi4 --threads 4
```

!!! warning "Sẽ bổ sung"
    Bổ sung cách đọc benchmark, kiểm soát điều kiện đo và mẫu bảng so sánh.
