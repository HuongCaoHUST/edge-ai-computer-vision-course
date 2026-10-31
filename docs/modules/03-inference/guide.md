# Lab 3 · Nội dung 3 · Khung hướng dẫn

## Các phần nội dung

1. Đọc checkpoint và danh sách nhãn.
2. Tiền xử lý BGR → RGB, resize `320×320`, chuẩn hóa.
3. Chạy inference và hậu xử lý NMS theo từng lớp.
4. Ghi kết quả ảnh/video và thời gian từng bước.

```bash
python lab2_infer.py --source anh.jpg
python lab2_infer.py --source video.mp4 --out results/out.mp4 --max-frames 300
```

!!! warning "Sẽ bổ sung"
    Bổ sung sơ đồ pipeline, giải thích API OpenCV và hướng dẫn chạy camera USB.
