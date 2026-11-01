# Huấn luyện SSD MobileNet

## Mục tiêu

Huấn luyện mô hình SSDLite320 MobileNetV3-Large bằng Python trên bộ dữ liệu Pascal VOC 2007. Kết quả của phần này là checkpoint dùng cho các bước inference và đánh giá tiếp theo.

## 1. Chuẩn bị môi trường

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision opencv-python numpy pillow matplotlib psutil
```

Kiểm tra các thư viện:

```bash
python -c "import torch, torchvision, cv2; print(torch.__version__, torchvision.__version__, cv2.__version__)"
```

Nên huấn luyện trên máy có GPU hoặc Google Colab. Sau đó, checkpoint được chuyển sang thiết bị biên để chạy inference.

## 2. Dữ liệu và nhãn

Pascal VOC lưu ảnh trong `JPEGImages/` và nhãn trong các file XML ở `Annotations/`. Mỗi object trong XML được chuyển thành một bounding box và một class label.

SSD MobileNet dùng:

- 20 lớp đối tượng của Pascal VOC;
- 1 lớp `background`;
- `num_classes = 21`;
- box dạng `xmin, ymin, xmax, ymax`.

Target của một ảnh trong torchvision có dạng:

```python
target = {
    "boxes": boxes,          # FloatTensor [N, 4]
    "labels": labels,        # Int64Tensor [N]
    "image_id": image_id,
    "area": areas,
    "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
}
```

Kiểm tra `xmax > xmin`, `ymax > ymin` và mapping class cố định trước khi bắt đầu train.

## 3. Khởi tạo model

```python
import torch
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

NUM_CLASSES = 21

model = ssdlite320_mobilenet_v3_large(
    weights=None,
    weights_backbone=MobileNet_V3_Large_Weights.DEFAULT,
    num_classes=NUM_CLASSES,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

Vì mỗi ảnh có thể có số lượng object khác nhau, DataLoader cần hàm gom batch:

```python
def collate_fn(batch):
    return tuple(zip(*batch))
```

## 4. Vòng lặp huấn luyện

Ở chế độ train, model detection trả về nhiều thành phần loss. Tổng loss được dùng để cập nhật trọng số:

```python
optimizer = torch.optim.SGD(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-3,
    momentum=0.9,
    weight_decay=5e-4,
)

for epoch in range(epochs):
    model.train()
    for images, targets in trainloader:
        images = [image.to(device) for image in images]
        targets = [
            {key: value.to(device) if hasattr(value, "to") else value
             for key, value in target.items()}
            for target in targets
        ]

        optimizer.zero_grad(set_to_none=True)
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        loss.backward()
        optimizer.step()
```

Một bước train có thứ tự: `zero_grad → forward → loss → backward → optimizer.step`.

## 5. Chạy thử và lưu model

Chạy tập con nhỏ trước:

```bash
python lab1_train.py --n-train 500 --epochs 5 --device cpu
python plot_log.py
```

Sau khi pipeline chạy đúng, huấn luyện đầy đủ trên GPU:

```bash
python lab1_train.py \
  --n-train 5011 \
  --n-val 500 \
  --epochs 30 \
  --device cuda
python plot_log.py
```

Nếu bộ mã nguồn đặt tên script theo số thứ tự website, `lab1_train.py` có thể được đặt là `lab2_train.py`.

Lưu tối thiểu:

- `runs/log.csv`: loss theo epoch;
- `curves.png`: đường cong train/validation;
- `runs/best.pt`: checkpoint tốt nhất.

Khi đọc đường cong, `train_loss` giảm nhưng `val_loss` tăng là dấu hiệu có thể overfit. Nếu loss là `nan`, kiểm tra learning rate, ảnh lỗi và bounding box.
