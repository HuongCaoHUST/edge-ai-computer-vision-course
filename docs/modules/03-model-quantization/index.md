# Lab 3 · Lượng tử hóa mô hình

## Lượng tử hóa mô hình và đánh giá hiệu năng triển khai trên thiết bị biên

Lượng tử hóa thay thế các số thực có độ chính xác cao bằng biểu diễn nhỏ hơn, từ đó giảm kích thước model và chi phí suy luận. Trong Lab 3, mô hình SSD MobileNet từ Lab 2 được dùng làm baseline để so sánh các phương án FP32, FP16 và INT8.

Lab này sử dụng **PyTorch làm framework duy nhất**. Baseline chính là PTQ static: calibration activation bằng dữ liệu đại diện, lượng tử hóa weights/activation `Conv2d`, sau đó dùng cùng pipeline PyTorch để đánh giá chất lượng và hiệu năng.

## Nội dung chính

1. Tìm hiểu khái niệm, nguyên lý và mục tiêu của lượng tử hóa mô hình.
2. Thực hành lượng tử hóa trước huấn luyện (PTQ) và trong huấn luyện (QAT).
3. Thực hiện calibration cho PTQ static; với QAT, theo dõi observer và fake quantizer trong quá trình fine-tune.
4. So sánh độ chính xác giữa mô hình gốc và mô hình lượng tử hóa với FP32, FP16 và INT8.
5. Đánh giá sự thay đổi chất lượng sau lượng tử hóa.
6. Đo kích thước model, tốc độ suy luận và mức tiêu thụ tài nguyên.
7. Phân tích đánh đổi giữa độ chính xác, tốc độ và dung lượng model.

## PTQ và QAT trong cùng pipeline

Hai phương pháp bắt đầu từ model FP32 đã train nhưng có mục đích khác nhau:

| Phương pháp | Thời điểm | Dữ liệu | Có cập nhật weights? |
| --- | --- | --- | --- |
| **PTQ** | Sau khi train | Calibration set đại diện | Không |
| **QAT** | Sau khi chuẩn bị quantizer | Train set có nhãn | Có fine-tune |

Luồng chính của Lab là `FP32 → PTQ + calibration → benchmark`. Nếu PTQ làm giảm mAP quá mức, dùng QAT để fine-tune model với fake quantization, sau đó convert và benchmark lại.

## Đầu vào

- checkpoint SSD MobileNet từ [Lab 2](../02-model-training/index.md);
- tập calibration đại diện cho dữ liệu thực tế;
- tập validation/test cố định để so sánh công bằng;
- runtime tương ứng với định dạng model trên thiết bị biên.

## Kết quả cần quan sát

- kích thước model trước và sau lượng tử hóa;
- Precision, Recall, AP/mAP hoặc metric phù hợp của từng phiên bản;
- latency, FPS và mức sử dụng CPU/RAM;
- trường hợp chất lượng giảm mạnh sau INT8 và nguyên nhân có thể có.

Bắt đầu từ [Khái niệm và nguyên lý](quantization.md), sau đó thực hiện [PTQ và Calibration](ptq.md), [QAT](qat.md), so sánh [FP32, FP16 và INT8](formats.md), tổng hợp trong [Đánh giá chất lượng và hiệu năng](evaluation.md), rồi thực hiện [Thực hành](practice.md).
