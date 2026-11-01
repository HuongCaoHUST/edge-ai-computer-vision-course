# FP32, FP16 và INT8

## 1. Vai trò của từng định dạng

| Định dạng | Đặc điểm | Điểm mạnh | Điểm cần lưu ý |
| --- | --- | --- | --- |
| FP32 | 32-bit floating point | Baseline dễ kiểm tra, chất lượng thường cao | Model lớn và tốn tài nguyên hơn |
| FP16 | 16-bit floating point | Giảm dung lượng, thường tăng tốc trên GPU/NPU hỗ trợ | Không phải CPU nào cũng tăng tốc FP16 |
| INT8 | 8-bit integer | Model nhỏ, phù hợp accelerator và edge runtime | Cần calibration/QAT, có thể giảm chất lượng |

FP32 là mốc chuẩn để so sánh. Không dùng model FP16 hoặc INT8 làm baseline mới rồi bỏ qua kết quả FP32 ban đầu.

Trong baseline chính của Lab 3, INT8 là **static PTQ mô phỏng bằng PyTorch**: weights và activation được lượng tử hóa theo scale/zero-point thu được từ calibration, sau đó dequantize để thực thi `Conv2d` bằng PyTorch. Vì vậy baseline giúp kiểm chứng quy trình và chất lượng, nhưng không đại diện cho full INT8 kernel hoặc tăng tốc phần cứng.

Về lý thuyết, nếu toàn bộ weights được chuyển đổi và metadata không đáng kể:

```text
size(FP16) ≈ 1/2 × size(FP32)
size(INT8) ≈ 1/4 × size(FP32)
```

Đây chỉ là ước lượng. File thực tế còn chứa graph, scale, zero-point, tensor chưa lượng tử hóa và phần metadata của runtime.

## 2. Các phiên bản cần tạo

Từ cùng một checkpoint SSD MobileNet, tạo:

```text
model_fp32  → model_fp16 → model_int8_ptq
                         → model_int8_qat
```

Tên file, preprocessing, danh sách lớp và tập đánh giá phải được ghi rõ để tránh so sánh nhầm các model khác cấu hình.

## 3. So sánh công bằng

Với mỗi định dạng, chạy cùng một tập validation/test và thu thập:

- Precision, Recall, AP và mAP;
- số prediction và confidence distribution;
- kích thước file model;
- thời gian load model;
- latency và FPS;
- CPU, RAM và accelerator utilization nếu có.

Nên lưu kết quả thành một bảng với các cột `format`, `method`, `device`, `runtime`, `size_mb`, `map`, `latency_ms`, `fps` và `ram_mb`.

Đặc biệt cần phân biệt:

- **model size:** dung lượng file model trên đĩa;
- **memory footprint:** RAM/VRAM sau khi runtime load model;
- **peak memory:** mức RAM cao nhất trong một lần inference.

Model INT8 có file nhỏ nhưng runtime có thể giải nén, tạo workspace hoặc giữ một số tensor FP32, nên memory footprint không luôn giảm đúng theo tỷ lệ 4 lần.

## 4. Kiểm tra sai khác output

Chọn một tập ảnh cố định và đối chiếu model lượng tử hóa với FP32:

1. kiểm tra shape output;
2. đối chiếu box có IoU cao;
3. so sánh class và confidence;
4. ghi lại ảnh có thay đổi prediction lớn;
5. xác định sai khác do quantization hay do preprocessing/runtime.

Một vài confidence thay đổi nhỏ là bình thường. Cần chú ý các trường hợp object bị mất, nhầm lớp hoặc box lệch nhiều.

Khi so sánh confidence, không nên dùng sai số tuyệt đối của score làm metric duy nhất. Với detection, thay đổi nhỏ về score có thể không ảnh hưởng kết quả nếu thứ tự box và quyết định NMS giữ nguyên; ngược lại, một thay đổi gần ngưỡng có thể làm mất cả một detection.

## 5. Tiêu chí lựa chọn

Không có định dạng tốt nhất cho mọi thiết bị. FP16 thường là lựa chọn cân bằng khi phần cứng có GPU/NPU hỗ trợ; INT8 phù hợp khi cần giảm mạnh kích thước và runtime có kernel integer hiệu quả. Quyết định cuối cùng phải dựa trên chất lượng tối thiểu và latency mục tiêu của ứng dụng.
