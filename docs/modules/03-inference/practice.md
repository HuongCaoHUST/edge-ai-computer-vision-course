# Lab 3 · Thực hành PTQ và QAT bằng PyTorch

## Mục tiêu

Bài thực hành được chia thành hai phần độc lập nhưng dùng chung một model FP32:

```text
Model SSD MobileNet FP32 đã train
        ├── Phần A: PTQ + calibration → INT8
        └── Phần B: QAT + fine-tune    → INT8
```

PTQ dùng để đo mức suy giảm ban đầu sau lượng tử hóa. QAT chỉ thực hiện sau đó nếu PTQ làm giảm mAP vượt ngưỡng chấp nhận.

## 1. Chuẩn bị chung

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision jupyter numpy pillow matplotlib psutil
```

Dùng checkpoint và DataLoader của Lab 2. Cố định các yếu tố sau trong cả hai phần:

- kiến trúc SSD MobileNet và `num_classes=21`;
- class mapping;
- input size, resize, normalize và channel order;
- validation/test set;
- IoU threshold, confidence threshold và NMS;
- thiết bị, số thread, số lần warm-up và số lần benchmark.

## 2. Baseline FP32 dùng chung

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_fp32 = build_ssd_mobilenet(num_classes=21)
checkpoint = torch.load("runs/best.pt", map_location=device)
model_fp32.load_state_dict(checkpoint["model"])
model_fp32.to(device).eval()

metrics_fp32 = evaluate_detection(model_fp32, val_loader, device)
bench_fp32 = benchmark_detection(
    model_fp32, val_loader, device, warmup=20, iterations=200
)
```

Ghi mAP tổng, AP từng lớp, Precision, Recall, latency p50/p95, FPS, kích thước model và peak RAM. Đây là mốc để tính `ΔmAP` cho cả PTQ và QAT.

# Phần A · PTQ

## A1. Chuẩn bị calibration data

Calibration dùng dữ liệu đại diện cho dữ liệu thực tế, không cập nhật weights và không cần label:

```python
calibration_loader = torch.utils.data.DataLoader(
    calibration_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn,
)
```

Chọn khoảng 100–300 ảnh, bao phủ nhiều loại cảnh, kích thước object, điều kiện sáng và các lớp quan trọng. Không dùng augmentation ngẫu nhiên; preprocessing phải giống inference thật.

## A2. Prepare → Calibration → Convert

Sử dụng [pytorch_quantization_baseline.py](pytorch_quantization_baseline.py):

```python
from pytorch_quantization_baseline import (
    prepare_static_ptq,
    calibrate,
    convert_static_ptq,
)

model_ptq, layer_count = prepare_static_ptq(model_fp32)
model_ptq.to(device).eval()

seen = calibrate(
    model_ptq,
    calibration_loader,
    device,
    limit=300,
)
print({"conv_layers": layer_count, "calibration_images": seen})

convert_static_ptq(model_ptq)
model_ptq.to(device).eval()
```

Trong calibration:

- không gọi `optimizer`;
- không gọi `backward()`;
- không cập nhật weights;
- chỉ thu thập range activation.

Sau `convert_static_ptq`, weights có dtype `torch.int8`, scale và zero-point đã được cố định.

## A3. Đánh giá và benchmark PTQ

```python
metrics_ptq = evaluate_detection(model_ptq, val_loader, device)
bench_ptq = benchmark_detection(
    model_ptq, val_loader, device, warmup=20, iterations=200
)

print("mAP FP32:", metrics_fp32["map"])
print("mAP PTQ:", metrics_ptq["map"])
print("delta mAP:", metrics_ptq["map"] - metrics_fp32["map"])
```

Nếu mAP giảm mạnh, kiểm tra theo thứ tự:

1. calibration set có đại diện không;
2. preprocessing có giống FP32 không;
3. scale và zero-point có hợp lệ không;
4. lớp nào giảm AP nhiều nhất;
5. prediction thay đổi trước hay sau NMS.

## A4. Kết quả PTQ cần ghi

| Phiên bản | Calibration | mAP | ΔmAP | Size MB | p50 ms | p95 ms | FPS | Peak RAM MB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 | Không |  | 0 |  |  |  |  |  |
| INT8 PTQ | 100–300 ảnh |  |  |  |  |  |  |  |

# Phần B · QAT

## B1. QAT là gì?

QAT (Quantization-Aware Training) đưa fake quantizer vào forward để model nhìn thấy sai số lượng tử hóa trong lúc fine-tune:

```text
FP32 → fake quantizer → fine-tune có nhãn → đóng gói INT8
```

Khác với PTQ, QAT có cập nhật weights. Calibration PTQ chỉ thu thập thống kê; QAT dùng train set có nhãn để model học cách bù sai số.

## B2. Chuẩn bị model QAT

QAT bắt đầu từ checkpoint FP32 hoặc model đã chuẩn bị quantizer, không bắt đầu từ một model INT8 PTQ đã convert:

```python
from pytorch_quantization_baseline import prepare_qat

model_qat, layer_count = prepare_qat(model_fp32)
model_qat.to(device).train()
print("fake-quantized Conv2d layers:", layer_count)

optimizer = torch.optim.SGD(
    model_qat.parameters(),
    lr=1e-5,
    momentum=0.9,
    weight_decay=5e-4,
)
```

Learning rate QAT nhỏ hơn learning rate train FP32. Fine-tune vài epoch trước khi quyết định có cần tiếp tục hay không.

## B3. Fine-tune với loss detection

```python
best_map = -1.0
for epoch in range(3):
    model_qat.train()

    for images, targets in train_loader:
        images = [image.to(device) for image in images]
        targets = [
            {key: value.to(device) if hasattr(value, "to") else value
             for key, value in target.items()}
            for target in targets
        ]

        optimizer.zero_grad(set_to_none=True)
        loss_dict = model_qat(images, targets)
        loss = sum(loss_dict.values())
        loss.backward()
        optimizer.step()

    model_qat.eval()
    metrics = evaluate_detection(model_qat, val_loader, device)
    print({"epoch": epoch, "loss": float(loss), **metrics})
    best_map = max(best_map, metrics["map"])
```

Chọn checkpoint theo mAP validation tốt nhất, không chọn mặc định epoch cuối. Sau 1–2 epoch đầu, có thể freeze observer:

```python
from pytorch_quantization_baseline import freeze_qat_observers

freeze_qat_observers(model_qat)
```

## B4. Đóng gói và đánh giá QAT

```python
from pytorch_quantization_baseline import pack_qat_weights

model_qat.eval()
pack_qat_weights(model_qat)

metrics_qat = evaluate_detection(model_qat, val_loader, device)
bench_qat = benchmark_detection(
    model_qat, val_loader, device, warmup=20, iterations=200
)

print("mAP QAT:", metrics_qat["map"])
print("delta mAP:", metrics_qat["map"] - metrics_fp32["map"])
```

QAT baseline vẫn dequantize weights trước `Conv2d`, nên chỉ dùng để đánh giá tác động của fine-tune lên chất lượng. Không kết luận có tăng tốc hardware nếu chưa có backend integer thực sự.

## B5. Kết quả QAT cần ghi

| Phiên bản | Fine-tune | mAP | ΔmAP | Size MB | p50 ms | p95 ms | FPS | Peak RAM MB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 | Không |  | 0 |  |  |  |  |  |
| INT8 QAT | 3 epoch, lr `1e-5` |  |  |  |  |  |  |  |

# Tổng hợp kết quả

## So sánh PTQ và QAT

| Tiêu chí | PTQ | QAT |
| --- | --- | --- |
| Calibration | Bắt buộc trong baseline static | Observer/fake quant trong fine-tune |
| Cập nhật weights | Không | Có |
| Thời gian | Nhanh | Dài hơn |
| Chất lượng INT8 | Có thể giảm nhiều | Thường phục hồi tốt hơn |
| Mục đích | Baseline đầu tiên | Cứu chất lượng sau PTQ |

Tính:

```text
ΔmAP = mAP_variant - mAP_FP32
compression = size_FP32 / size_variant
speedup = latency_FP32 / latency_variant
```

Kết luận cần trả lời: QAT có phục hồi mAP đáng kể không, chi phí fine-tune là bao nhiêu, và phiên bản nào đáp ứng đồng thời yêu cầu chất lượng, latency, RAM và dung lượng.
