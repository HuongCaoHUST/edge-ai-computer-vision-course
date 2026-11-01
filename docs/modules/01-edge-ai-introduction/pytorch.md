# Lab 1 · PyTorch

Phần này trình bày toàn bộ pipeline bằng PyTorch với cùng một bài toán mẫu: phân loại vector 10 chiều thành 3 lớp.

## 1. Tensor và layer

```python
import torch
from torch import nn

x = torch.randn(4, 10)       # 4 mẫu, mỗi mẫu có 10 đặc trưng
layer = nn.Linear(10, 3)
y = layer(x)
print(y.shape)               # torch.Size([4, 3])
```

Tensor PyTorch có `shape`, `dtype`, device (`cpu` hoặc `cuda`) và có thể tham gia cơ chế autograd để tính gradient.

## 2. Định nghĩa model

Model PyTorch kế thừa `nn.Module`. Các layer được khai báo trong `__init__`; đường đi của dữ liệu được mô tả trong `forward`.

```python
class Classifier(nn.Module):
    def __init__(self, num_features=10, num_classes=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.network(x)


model = Classifier()
logits = model(torch.randn(8, 10))
print(logits.shape)           # torch.Size([8, 3])
```

Output là logits. Khi dùng `CrossEntropyLoss`, không cần thêm `Softmax` vào cuối model.

## 3. Load dữ liệu

`Dataset` định nghĩa cách lấy một mẫu; `DataLoader` gom mẫu thành batch và tạo iterator cho training.

```python
from torch.utils.data import Dataset, DataLoader


class ToyDataset(Dataset):
    def __init__(self, n=1000):
        self.features = torch.randn(n, 10)
        self.labels = torch.randint(0, 3, (n,))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


train_loader = DataLoader(
    ToyDataset(), batch_size=32, shuffle=True
)
```

Với ảnh, `__getitem__` thường thực hiện đọc file, resize, normalize và trả về `(image, label)`. Với object detection, label có thể gồm nhiều box và class trong một ảnh.

## 4. Training loop

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Classifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    model.train()
    total_loss = 0.0

    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"epoch={epoch + 1}, loss={total_loss / len(train_loader):.4f}")
```

Một bước training gồm: `zero_grad → forward → loss → backward → optimizer.step`.

## 5. Validation và inference

```python
model.eval()
sample = torch.randn(4, 10).to(device)

with torch.no_grad():
    logits = model(sample)
    predicted_class = logits.argmax(dim=1)

print(predicted_class)
```

`model.eval()` chuyển các layer như BatchNorm và Dropout sang chế độ suy luận. `torch.no_grad()` tắt việc lưu graph gradient để tiết kiệm bộ nhớ.

## 6. Triển khai trên edge

Sau khi model chạy đúng, cần kiểm tra preprocessing, input shape, dtype, output và postprocessing. PyTorch có thể xuất model sang ONNX hoặc runtime khác tùy phần cứng. Kết quả cuối cùng phải được kiểm tra lại bằng dữ liệu thực tế trên thiết bị đích.
