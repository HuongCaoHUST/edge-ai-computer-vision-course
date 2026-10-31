# Lab 1 · Nội dung 1 · Nội dung & thực hành

## Khung nội dung

1. Tạo virtual environment.
2. Cài `requirements.txt`.
3. Kiểm tra phiên bản PyTorch, torchvision, OpenCV.
4. Ghi chú khác biệt giữa laptop, Raspberry Pi và Jetson.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import torch, torchvision, cv2; print(torch.__version__, torchvision.__version__, cv2.__version__)"
```

!!! warning "Sẽ bổ sung"
    Bổ sung nội dung dependency, phiên bản được hỗ trợ và hướng dẫn cài đặt riêng cho từng phần cứng.

Tiếp tục với [Bài tập & nộp bài](assignment.md).
