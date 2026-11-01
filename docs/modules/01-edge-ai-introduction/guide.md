# Lab 1 · Nội dung

## 1. Edge AI là gì?

**Edge AI** là cách xây dựng hệ thống trong đó dữ liệu được xử lý và mô hình AI được chạy gần nơi dữ liệu phát sinh, ví dụ camera, cảm biến, điện thoại, Raspberry Pi hoặc NVIDIA Jetson. Thiết bị biên có thể gửi kết quả hoặc dữ liệu đã được rút gọn lên máy chủ thay vì luôn phải gửi toàn bộ dữ liệu thô lên đám mây.

### Kiến trúc tổng quát

| Thành phần | Vai trò | Ví dụ |
| --- | --- | --- |
| Nguồn dữ liệu | Tạo dữ liệu đầu vào liên tục hoặc theo sự kiện | Camera, microphone, cảm biến y tế |
| Thiết bị biên | Tiền xử lý, suy luận, ra quyết định nhanh | Raspberry Pi, Jetson, điện thoại |
| Kết nối | Gửi kết quả, cảnh báo hoặc dữ liệu cần thiết | Wi-Fi, 4G/5G, Ethernet |
| Cloud/server | Huấn luyện, lưu trữ, phân tích dài hạn, quản trị thiết bị | Máy chủ GPU, cloud |
| Ứng dụng | Hiển thị hoặc kích hoạt hành động | Cảnh báo, điều khiển đèn, dashboard |

Luồng điển hình là: **dữ liệu → tiền xử lý → suy luận → hậu xử lý → hành động/ghi log → đồng bộ tùy chọn**. Huấn luyện thường thực hiện trên máy mạnh hoặc cloud; thiết bị biên chủ yếu chạy suy luận.

## 2. Đặc điểm, lợi ích và thách thức

### Đặc điểm

- Độ trễ thấp vì dữ liệu không phải đi xa đến cloud.
- Hoạt động gần thời gian thực và có thể tiếp tục khi mạng chập chờn.
- Tài nguyên hạn chế: CPU/GPU/NPU, RAM, bộ nhớ lưu trữ và công suất.
- Cần quan tâm đồng thời đến chất lượng mô hình và hiệu năng hệ thống.

### Lợi ích

- **Nhanh hơn:** phản hồi tại chỗ cho các tác vụ cảnh báo và điều khiển.
- **Riêng tư hơn:** hạn chế gửi hình ảnh, âm thanh hoặc dữ liệu sức khỏe ra ngoài.
- **Tiết kiệm băng thông:** chỉ truyền sự kiện, metadata hoặc kết quả cần thiết.
- **Có khả năng mở rộng:** nhiều thiết bị xử lý phân tán thay vì dồn toàn bộ tải lên server.

### Thách thức

- Mô hình lớn có thể vượt giới hạn CPU, RAM, bộ nhớ và điện năng.
- Nhiệt độ, nguồn điện và điều kiện mạng làm hiệu năng thay đổi theo thời gian.
- Dữ liệu tại hiện trường có thể khác dữ liệu huấn luyện, gây suy giảm chất lượng.
- Cần cập nhật mô hình, bảo mật thiết bị, giám sát lỗi và quản lý nhiều phiên bản.
- Các ứng dụng y tế, giao thông và xe tự hành đòi hỏi kiểm thử, độ tin cậy và an toàn cao.

## 3. Ví dụ ứng dụng

| Lĩnh vực | Bài toán | Đầu vào | Đầu ra tại biên | Tiêu chí cần quan tâm |
| --- | --- | --- | --- | --- |
| Giao thông thông minh | Đếm xe, phát hiện ùn tắc hoặc vượt đèn đỏ | Video camera | Số xe, loại xe, cảnh báo và thời điểm | Precision/Recall, FPS, độ trễ, độ ổn định |
| Y tế | Phát hiện bất thường trên ảnh hoặc theo dõi bệnh nhân | Ảnh X-quang, tín hiệu cảm biến | Nhãn, điểm tin cậy hoặc cảnh báo | Recall, độ nhạy, độ đặc hiệu, riêng tư |
| Xe tự hành | Phát hiện người đi bộ, làn đường và vật cản | Camera, LiDAR, radar | Hộp phát hiện, đường làn, khoảng cách | Độ trễ đầu-cuối, Recall, an toàn, tính liên tục |

## 4. Đặc tả một bài toán AI

Trước khi chọn mô hình, hãy điền đủ năm nhóm thông tin sau:

1. **Mục tiêu:** hệ thống cần dự đoán hoặc tự động hóa điều gì?
2. **Đầu vào:** loại dữ liệu, kích thước, tần suất, điều kiện sáng/nhiễu và nguồn cảm biến.
3. **Đầu ra:** nhãn, bounding box, mask, số đo, điểm tin cậy hay hành động điều khiển.
4. **Ràng buộc triển khai:** thiết bị, RAM, công suất, kết nối, FPS và độ trễ tối đa.
5. **Tiêu chí đánh giá:** chỉ số chất lượng mô hình cùng chỉ số hệ thống và điều kiện đạt.

Ví dụ: với bài toán đếm xe từ camera, đầu vào là video 1280×720 ở 25 FPS; đầu ra là số xe theo từng khung hình; chất lượng dùng Precision/Recall; hệ thống yêu cầu ít nhất 15 FPS và độ trễ không quá 100 ms trên thiết bị đích.

## 5. PyTorch và TensorFlow là gì?

### PyTorch

PyTorch là framework học sâu dựa trên tensor, thường được dùng để nghiên cứu, huấn luyện và triển khai mô hình bằng Python. Mỗi tensor có `shape`, `dtype` và có thể nằm trên `cpu` hoặc `cuda`. PyTorch dùng cơ chế autograd để tự động tính gradient trong quá trình huấn luyện.

Các thành phần cơ bản:

```python
import torch
from torch import nn

x = torch.randn(4, 10)       # batch gồm 4 mẫu, mỗi mẫu có 10 đặc trưng
layer = nn.Linear(10, 3)     # ánh xạ 10 đặc trưng thành 3 đầu ra
y = layer(x)
print(y.shape)               # torch.Size([4, 3])
```

- `torch.Tensor`: dữ liệu và tham số dạng số.
- `nn.Module`: lớp cơ sở để định nghĩa model.
- `Dataset` và `DataLoader`: đọc dữ liệu và tạo mini-batch.
- `loss function`: đo sai khác giữa dự đoán và nhãn.
- `optimizer`: cập nhật trọng số theo gradient.
- `model.train()` và `model.eval()`: chuyển model giữa chế độ huấn luyện và suy luận.

### TensorFlow và Keras

TensorFlow là framework cho tensor và runtime tính toán; Keras là API cấp cao thường dùng để xây dựng model. Với TensorFlow hiện đại, Keras cung cấp cách viết model tuần tự, model dạng Functional API hoặc subclass `Model`.

```python
import tensorflow as tf

x = tf.random.normal((4, 10))
layer = tf.keras.layers.Dense(3)
y = layer(x)
print(y.shape)               # (4, 3)
```

Các thành phần tương ứng trong Keras gồm `Layer`, `Model`, `Dataset`, `compile`, `fit`, `evaluate` và `predict`. TensorFlow cũng cung cấp TensorFlow Lite để chuyển model sang nhiều thiết bị edge.

## 6. Cấu trúc một mạng neural

Một mạng neural thường gồm các lớp biến đổi dữ liệu theo chuỗi:

`input → layer → activation → layer → output`

Ví dụ với bài toán phân loại 3 lớp và vector đầu vào có 10 đặc trưng:

```text
10 đặc trưng → Linear/Dense(32) → ReLU → Linear/Dense(3) → logits
```

- **Input shape:** kích thước dữ liệu mỗi mẫu, không tính batch dimension.
- **Layer:** phép biến đổi có tham số cần học, ví dụ `Linear/Dense`, `Conv2D`, `BatchNorm`.
- **Activation:** tạo tính phi tuyến, ví dụ `ReLU`, `Sigmoid`, `Softmax`.
- **Output:** phụ thuộc bài toán; classification thường trả logits hoặc xác suất, detection trả box và class, regression trả giá trị liên tục.
- **Loss:** phải phù hợp với output và loại nhãn. Ví dụ `CrossEntropyLoss`/`SparseCategoricalCrossentropy` cho classification nhiều lớp.

Không cần đặt `Softmax` ở cuối khi dùng PyTorch `CrossEntropyLoss`, vì hàm loss đã xử lý bước tương ứng trên logits. Với Keras, có thể đặt `activation="softmax"` hoặc để output là logits và cấu hình `from_logits=True`.

### Định nghĩa model bằng PyTorch

```python
import torch
from torch import nn


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
logits = model(torch.randn(8, 10))  # batch 8 mẫu
print(logits.shape)                 # torch.Size([8, 3])
```

### Định nghĩa model bằng TensorFlow/Keras

```python
import tensorflow as tf


model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(10,)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(3),  # logits
])

logits = model(tf.random.normal((8, 10)))
print(logits.shape)              # (8, 3)
```

Với mạng phức tạp hơn, dùng `nn.Module` của PyTorch hoặc subclass `tf.keras.Model` để mô tả nhiều nhánh, nhiều đầu ra và logic riêng trong `forward`/`call`.

## 7. Pipeline load dữ liệu

Pipeline dữ liệu có cùng ý tưởng trong cả hai framework:

`file/camera → đọc mẫu → decode → resize → normalize → label → batch → model`

Các bước quan trọng:

1. Chia dữ liệu thành train, validation và test.
2. Chỉ áp dụng augmentation ngẫu nhiên cho tập train.
3. Giữ cùng quy ước kích thước, màu ảnh, range và normalization giữa training và inference.
4. Kiểm tra shape của batch và kiểu dữ liệu trước khi đưa vào model.
5. Với object detection, mỗi mẫu thường gồm ảnh và nhiều bounding box cùng nhãn.

### Dataset và DataLoader trong PyTorch

```python
from torch.utils.data import Dataset, DataLoader
import torch


class ToyDataset(Dataset):
    def __init__(self, n=1000):
        self.x = torch.randn(n, 10)
        self.y = torch.randint(0, 3, (n,))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


train_loader = DataLoader(ToyDataset(), batch_size=32, shuffle=True)
for features, labels in train_loader:
    print(features.shape, labels.shape)  # [32, 10], [32]
    break
```

Với dữ liệu ảnh, `Dataset` thường mở file trong `__getitem__`, áp dụng transform, đổi ảnh thành tensor và trả về `(image, label)`. `DataLoader` phụ trách batch, xáo trộn và có thể đọc song song bằng `num_workers`.

### `tf.data.Dataset` trong TensorFlow

```python
import numpy as np
import tensorflow as tf

features = np.random.randn(1000, 10).astype("float32")
labels = np.random.randint(0, 3, size=(1000,)).astype("int32")

train_ds = (
    tf.data.Dataset.from_tensor_slices((features, labels))
    .shuffle(1000)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)

for batch_features, batch_labels in train_ds.take(1):
    print(batch_features.shape, batch_labels.shape)  # (32, 10), (32,)
```

Với ảnh trong thư mục, có thể dùng `tf.keras.utils.image_dataset_from_directory`; với pipeline tùy biến, dùng `map` để decode, resize và normalize từng mẫu.

## 8. Pipeline training

### Training loop trong PyTorch

```python
import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Classifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    model.train()
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

    print(f"epoch={epoch + 1}, loss={loss.item():.4f}")
```

Ý nghĩa của một vòng lặp là: xóa gradient cũ → forward → tính loss → backward → cập nhật trọng số. Khi validation, dùng `model.eval()` và `torch.no_grad()` để không cập nhật gradient.

### Training bằng Keras

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
)
```

`compile` khai báo loss, optimizer và metric; `fit` thực hiện vòng lặp training; `evaluate` đo trên dữ liệu đánh giá; `predict` chạy suy luận. Khi cần logic đặc biệt, có thể viết custom training loop với `tf.GradientTape`.

## 9. Inference và chuyển sang Edge AI

Sau training, pipeline inference thường là:

`camera/file → preprocess → model forward → postprocess → kết quả`

PyTorch:

```python
model.eval()
with torch.no_grad():
    logits = model(features.to(device))
    predicted_class = logits.argmax(dim=1)
```

TensorFlow/Keras:

```python
logits = model.predict(features, verbose=0)
predicted_class = tf.argmax(logits, axis=1)
```

Khi đưa lên thiết bị biên, cần kiểm tra các bước tiếp theo: export model, runtime hỗ trợ, kích thước input, preprocessing/postprocessing, độ chính xác sau chuyển đổi, thời gian khởi tạo và độ trễ mỗi frame. Với TensorFlow có thể dùng TensorFlow Lite; với PyTorch có thể xuất sang ONNX hoặc runtime phù hợp với thiết bị.

## 10. Quy trình phát triển Edge AI

| Bước | Việc cần làm | Kết quả cần kiểm chứng |
| --- | --- | --- |
| 1. Xác định bài toán | Chọn mục tiêu, người dùng, ràng buộc và rủi ro | Bản đặc tả bài toán |
| 2. Thu thập dữ liệu | Lấy dữ liệu đại diện cho điều kiện thực tế | Tập train/validation/test có nhãn |
| 3. Chọn và huấn luyện mô hình | Chọn kiến trúc, loss, tham số và baseline | Checkpoint và log huấn luyện |
| 4. Đánh giá offline | Đo chỉ số chất lượng trên tập chưa thấy | Bảng metric, lỗi và phân tích theo nhóm |
| 5. Tối ưu cho edge | Giảm kích thước, lượng tử hóa, cắt tỉa hoặc đổi runtime | Mô hình/runtime triển khai được |
| 6. Tích hợp pipeline | Nối camera/cảm biến, tiền xử lý, suy luận, hậu xử lý | Ứng dụng chạy trên thiết bị |
| 7. Đánh giá trên thiết bị | Đo FPS, độ trễ, CPU, RAM, nhiệt độ và điện năng | Báo cáo benchmark thực tế |
| 8. Triển khai và giám sát | Quản lý phiên bản, cập nhật, log và phát hiện drift | Quy trình vận hành và rollback |

Chỉ số “đúng” trên máy huấn luyện chưa đủ để kết luận hệ thống dùng được. Một mô hình cần đồng thời đạt chất lượng yêu cầu và chạy ổn định trên đúng thiết bị đích.

## 11. Thực hành kiểm tra môi trường

Tạo môi trường Python và kiểm tra các thư viện sẽ dùng ở các lab sau:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import torch, torchvision, cv2; print(torch.__version__, torchvision.__version__, cv2.__version__)"
```

Nếu đã cài TensorFlow, kiểm tra thêm:

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

TensorFlow không bắt buộc cho các bài thực hành PyTorch tiếp theo; mục tiêu của bước này là xác nhận framework phù hợp với môi trường của bạn.

Nếu repository chưa có `requirements.txt`, hãy ghi lại lệnh cài đặt hoặc bỏ qua bước `pip install` và chỉ kiểm tra các package đã có. Với Raspberry Pi và Jetson, cần dùng hướng dẫn cài đặt phù hợp với hệ điều hành/JetPack; không coi phiên bản trên laptop là mặc định cho mọi thiết bị.

!!! tip "Câu hỏi định hướng"
    Một ứng dụng nhận diện người từ camera có thực sự cần gửi toàn bộ video lên cloud không? Hãy cân nhắc độ trễ, riêng tư, băng thông và khả năng hoạt động khi mất mạng.

## 12. So sánh PyTorch và TensorFlow/Keras

Hai framework có cùng pipeline học sâu nhưng khác cách tổ chức API. Bảng dưới đây liên kết các khái niệm tương ứng:

| Giai đoạn | PyTorch | TensorFlow/Keras |
| --- | --- | --- |
| Tensor | `torch.Tensor` | `tf.Tensor` |
| Layer tuyến tính | `nn.Linear(10, 3)` | `tf.keras.layers.Dense(3)` |
| Activation | `nn.ReLU()` | `activation="relu"` hoặc `tf.keras.layers.ReLU()` |
| Model tuần tự | `nn.Sequential` | `tf.keras.Sequential` |
| Model tùy biến | Kế thừa `nn.Module`, viết `forward()` | Kế thừa `tf.keras.Model`, viết `call()` |
| Dataset | `torch.utils.data.Dataset` | `tf.data.Dataset` |
| Batch | `DataLoader(batch_size=32)` | `.batch(32)` |
| Xáo trộn | `DataLoader(shuffle=True)` | `.shuffle(buffer_size)` |
| Loss nhiều lớp | `nn.CrossEntropyLoss()` | `SparseCategoricalCrossentropy(from_logits=True)` |
| Optimizer Adam | `torch.optim.Adam(...)` | `tf.keras.optimizers.Adam(...)` |
| Training | Tự viết loop | `model.compile()` và `model.fit()` |
| Inference | `model.eval()` + `torch.no_grad()` | `model.predict()` |
| Định dạng edge phổ biến | ONNX hoặc runtime phù hợp | TensorFlow Lite |

### Mạch xử lý tương ứng

```text
PyTorch:       Dataset → DataLoader → nn.Module → loss → backward → optimizer
TensorFlow:    tf.data   → batch     → Keras Model → loss → fit/GradientTape
```

PyTorch thường cho phép kiểm soát chi tiết training loop, phù hợp khi cần thử nghiệm hoặc tùy biến cách huấn luyện. Keras cung cấp API cấp cao hơn, phù hợp với pipeline chuẩn và các hướng triển khai trong hệ sinh thái TensorFlow.

### Nguyên tắc chọn framework

- Chọn PyTorch khi cần kiểm soát chi tiết vòng lặp training hoặc sử dụng hệ sinh thái `torchvision`.
- Chọn TensorFlow/Keras khi muốn pipeline `compile`/`fit` rõ ràng hoặc hướng đến TensorFlow Lite.
- Với dự án thực tế, cần ưu tiên model có sẵn, runtime của thiết bị, operator được hỗ trợ và hiệu năng benchmark thực tế.

Tiếp tục với [PyTorch](pytorch.md) để xem pipeline đầy đủ, sau đó thực hiện bài [Thực hành phân loại ảnh CIFAR-10](assignment.md).
