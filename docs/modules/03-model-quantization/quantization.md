# Khái niệm và nguyên lý lượng tử hóa

## 1. Lượng tử hóa là gì?

Lượng tử hóa chuyển trọng số và/hoặc activation từ số thực có độ chính xác cao sang dạng biểu diễn ít bit hơn. Mục tiêu là:

- giảm kích thước model;
- giảm bandwidth bộ nhớ;
- tận dụng kernel INT8 hoặc FP16 trên phần cứng;
- giảm latency và mức tiêu thụ năng lượng.

Đổi lại, việc làm tròn và giới hạn miền biểu diễn có thể làm thay đổi output và giảm chất lượng model.

## 2. Ánh xạ số thực sang số nguyên

Với lượng tử hóa affine, giá trị thực `x` được ánh xạ thành giá trị nguyên `q`:

```text
q = clamp(round(x / scale) + zero_point, q_min, q_max)
x ≈ scale × (q - zero_point)
```

`scale` xác định độ phân giải của miền giá trị, còn `zero_point` giúp biểu diễn số 0. Với signed INT8, miền thường là `[-128, 127]`; với unsigned INT8, miền thường là `[0, 255]`.

## 3. Các thành phần có thể lượng tử hóa

- **Weights:** tham số của model, thường có thể lượng tử hóa theo từng tensor hoặc từng channel.
- **Activations:** output trung gian, cần biết miền giá trị qua calibration hoặc quá trình training.
- **Input/output:** cần thống nhất dtype và scale giữa chương trình tiền xử lý, runtime và hậu xử lý.

Nếu chỉ lượng tử hóa weights, model có thể nhỏ hơn nhưng tốc độ không nhất thiết tăng nhiều. Muốn tận dụng INT8 hiệu quả, runtime và phần cứng phải hỗ trợ cả các phép tính tương ứng.

## 4. Per-tensor và per-channel

Với **per-tensor quantization**, toàn bộ tensor dùng chung một `scale`. Cách này đơn giản và dễ được runtime hỗ trợ, nhưng một outlier lớn có thể làm giảm độ phân giải của phần lớn giá trị còn lại.

Với **per-channel quantization**, mỗi channel của weights có `scale` riêng. Các channel có miền giá trị khác nhau được biểu diễn tốt hơn, thường giúp giữ chất lượng INT8 tốt hơn với convolution. Đổi lại, backend phải hỗ trợ phép nhân và dequantization theo channel.

Đây là một trong các lý do hai model cùng là INT8 nhưng có thể cho mAP và latency khác nhau.

## 5. Symmetric và asymmetric quantization

- **Symmetric:** thường dùng `zero_point = 0`, miền số nguyên đối xứng quanh 0; phù hợp với weights có phân bố gần đối xứng.
- **Asymmetric:** cho phép `zero_point` khác 0; phù hợp với activation lệch về phía dương, chẳng hạn output sau ReLU.

Chọn symmetric hay asymmetric ảnh hưởng đến sai số lượng tử hóa và số phép tính mà backend phải thực hiện. Không nên chỉ nhìn vào số bit; cần kiểm tra cách runtime triển khai định dạng đó.

## 6. Static, dynamic và mixed precision

- **Dynamic quantization:** activation được lượng tử hóa khi chạy; dễ dùng nhưng ít phù hợp với mọi lớp convolution.
- **Static quantization:** scale của activation được xác định trước bằng calibration; thường phù hợp hơn cho inference edge.
- **Mixed precision:** giữ một số lớp ở FP16/FP32 và lượng tử hóa các lớp còn lại để bảo toàn chất lượng.

Với object detection, detection head hoặc các phép toán nhạy cảm có thể cần giữ precision cao hơn backbone.

## 7. Nguồn gây suy giảm chất lượng

Suy giảm sau lượng tử hóa thường đến từ:

1. **Rounding error:** giá trị thực bị làm tròn về các mức rời rạc;
2. **Clipping error:** giá trị ngoài range bị ép về `q_min` hoặc `q_max`;
3. **Accumulated error:** sai số của nhiều layer nối tiếp nhau tích lũy;
4. **Unsupported operator:** một phần model chạy ở dtype khác và phát sinh chuyển đổi;
5. **Pre/postprocess mismatch:** input scale, channel order hoặc output decode không giống baseline.

Vì vậy, khi INT8 giảm mAP, cần phân biệt lỗi do calibration, do model, hay do pipeline triển khai.

## 8. Khi nào nên lượng tử hóa?

Chỉ lượng tử hóa sau khi baseline FP32 đã chạy đúng và có số liệu validation. Nếu không có baseline, không thể biết chất lượng giảm là do model chưa học tốt hay do lượng tử hóa.

Quy trình chung:

`FP32 baseline → chọn runtime → PTQ/calibration hoặc QAT → kiểm tra output → đo chất lượng → đo hiệu năng → phân tích đánh đổi`

