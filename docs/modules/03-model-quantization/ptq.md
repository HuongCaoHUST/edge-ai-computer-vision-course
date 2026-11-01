# PTQ native PyTorch và Calibration

## 1. Baseline chính

Baseline của Lab 3 đi theo đúng quy trình:

```text
train FP32 → calibration bằng dữ liệu đại diện → convert INT8 → benchmark
```

Baseline dùng code PyTorch trong [pytorch_quantization_baseline.py](pytorch_quantization_baseline.py). Nó lượng tử hóa per-tensor:

- weights `Conv2d` về INT8 signed;
- input/output activation theo range thu được từ calibration;
- bias giữ FP32;
- phép convolution vẫn thực thi bằng PyTorch với weights dequantize.

Vì vậy baseline dễ kiểm chứng mAP và sai số do calibration, nhưng không tuyên bố đã kích hoạt kernel INT8 chuyên dụng của phần cứng.

## 2. PTQ là gì?

Post-Training Quantization thực hiện sau khi model FP32 đã huấn luyện xong. Calibration không cập nhật weights; nó chỉ đưa dữ liệu đại diện qua model để thu thập `min/max` activation và tính:

```text
scale = (max - min) / (qmax - qmin)
zero_point = round(qmin - min / scale)
```

Với mỗi lớp, baseline lưu các tham số range sau calibration. Khi inference, tensor được lượng tử hóa rồi dequantize lại để mô phỏng sai số INT8.

## 3. Chuẩn bị calibration set

Calibration set nên có khoảng 100–300 ảnh đại diện cho dữ liệu triển khai:

- cùng loại cảnh, ánh sáng và kích thước object;
- có đủ các lớp quan trọng;
- dùng cùng resize, normalize và channel order;
- không dùng transform ngẫu nhiên;
- không cần label cho bước thu thập range.

Không dùng toàn bộ ảnh dễ hoặc chỉ một lớp. Nếu calibration không đại diện, activation có thể bị clipping và Recall giảm mạnh.

## 4. Prepare, calibrate và convert

```python
from pytorch_quantization_baseline import (
    prepare_static_ptq,
    calibrate,
    convert_static_ptq,
    count_int8_weights,
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
print("INT8 weights:", count_int8_weights(model_ptq))
```

Thứ tự rất quan trọng:

1. tạo bản sao từ model FP32;
2. thay `Conv2d` bằng module có observer;
3. chạy calibration với `torch.no_grad()`;
4. finalize scale/zero-point và pack weights INT8;
5. mới chạy evaluation và benchmark.

Không benchmark model đang ở trạng thái chuẩn bị calibration.

## 5. Kiểm tra sau convert

```python
metrics_fp32 = evaluate_detection(model_fp32, val_loader, device)
metrics_int8 = evaluate_detection(model_ptq, val_loader, device)

print("mAP FP32:", metrics_fp32["map"])
print("mAP INT8:", metrics_int8["map"])
print("delta mAP:", metrics_int8["map"] - metrics_fp32["map"])
```

Kiểm tra thêm:

- `qweight.dtype == torch.int8`;
- output vẫn có đủ box/class/score;
- prediction thay đổi ở ảnh nào;
- lớp nào giảm AP nhiều nhất;
- latency p50/p95 và RAM sau khi load model.

## 6. Giới hạn của baseline

Baseline này nhằm kiểm chứng quy trình PTQ và tác động chất lượng. Vì convolution dequantize weights trước khi chạy, tốc độ có thể không tăng. Sau khi baseline ổn định, mới mở rộng sang native static quantization có kernel INT8 hoặc QAT nếu thiết bị và operator hỗ trợ.
