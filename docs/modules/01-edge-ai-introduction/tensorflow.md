# Lab 1 · TensorFlow/Keras

Phần này trình bày cùng pipeline với trang PyTorch: tensor → model → dữ liệu → training → inference → triển khai edge.

## 1. Tensor và layer

```python
import tensorflow as tf

x = tf.random.normal((4, 10))
layer = tf.keras.layers.Dense(3)
y = layer(x)
print(y.shape)               # (4, 3)
```

TensorFlow cung cấp tensor và runtime tính toán; Keras là API cấp cao để định nghĩa layer, model và quy trình training.

## 2. Định nghĩa model

### Sequential API

```python
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(10,)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(3),  # logits
])
```

`Sequential` phù hợp với model có các layer nối tiếp theo một đường thẳng.

### Functional API

```python
inputs = tf.keras.Input(shape=(10,))
x = tf.keras.layers.Dense(32, activation="relu")(inputs)
outputs = tf.keras.layers.Dense(3)(x)
model = tf.keras.Model(inputs, outputs)
```

Functional API phù hợp với model có nhiều nhánh, nhiều đầu vào hoặc nhiều đầu ra. Khi cần logic phức tạp hơn, có thể subclass `tf.keras.Model` và định nghĩa `call()`.

## 3. Load dữ liệu

`tf.data.Dataset` tạo pipeline đọc, xáo trộn, gom batch và chuẩn bị dữ liệu.

```python
import numpy as np

features = np.random.randn(1000, 10).astype("float32")
labels = np.random.randint(0, 3, size=(1000,)).astype("int32")

train_ds = (
    tf.data.Dataset.from_tensor_slices((features, labels))
    .shuffle(1000)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)
```

Với ảnh trong thư mục, có thể dùng `tf.keras.utils.image_dataset_from_directory`. Với pipeline tùy biến, dùng `.map()` để decode, resize, normalize và tạo label.

## 4. Training

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

history = model.fit(train_ds, epochs=5)
```

`compile()` kết nối model với loss, optimizer và metric. `fit()` thực hiện vòng lặp training. Khi cần tự kiểm soát từng bước, dùng `tf.GradientTape` để viết custom training loop.

## 5. Validation và inference

```python
predicted_logits = model.predict(features[:4], verbose=0)
predicted_class = tf.argmax(predicted_logits, axis=1)
print(predicted_class)
```

Có thể truyền `validation_data=val_ds` vào `fit()` để theo dõi validation sau mỗi epoch. `evaluate()` dùng để đo trên một dataset; `predict()` dùng để lấy output suy luận.

## 6. Triển khai trên edge

TensorFlow thường chuyển model sang TensorFlow Lite để chạy trên nhiều thiết bị edge. Sau khi chuyển đổi, cần kiểm tra operator được hỗ trợ, input shape, dtype, preprocessing, output và độ chính xác so với model gốc.
