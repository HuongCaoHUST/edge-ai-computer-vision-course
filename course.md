1
Giới thiệu Edge AI và quy trình phát triển ứng dụng AI
Tổng quan kiến trúc, đặc điểm, lợi ích và thách thức của Edge AI.
Ứng dụng Edge AI trong giao thông thông minh, y tế và xe tự hành.
Xác định bài toán AI, yêu cầu đầu vào, đầu ra và tiêu chí đánh giá.
Nắm được quy trình tổng thể để triển khai mô hình trên thiết bị thực tế.
Các Framework: PyTorch, TensorFlow.
2
Huấn luyện, triển khai và đánh giá hiệu năng mô hình SSD MobileNet trên thiết bị biên
Huấn luyện mô hình SSD MobileNet bằng Python và bộ dữ liệu có sẵn.
Xây dựng chương trình inference sử dụng OpenCV.
Đánh giá chất lượng mô hình qua các chỉ số phù hợp.
Đo hiệu năng suy luận: FPS và độ trễ xử lý.
Đánh giá tài nguyên: CPU, RAM và mức sử dụng hệ thống.
3
Lượng tử hóa mô hình (Quantization) và đánh giá hiệu năng triển khai trên thiết bị biên
Tìm hiểu khái niệm, nguyên lý và mục tiêu của lượng tử hóa mô hình.
Thực hành lượng tử hóa trước huấn luyện (PTQ) và trong huấn luyện (QAT).
Thực hiện Calibration bằng dữ liệu đại diện cho dữ liệu thực tế.
So sánh độ chính xác giữa mô hình gốc và mô hình lượng tử hóa với các định dạng khác nhau (FP32, FP16, INT8).
Đánh giá sự thay đổi chất lượng mô hình sau lượng tử hóa.
Đo lường kích thước mô hình, tốc độ suy luận và mức tiêu thụ tài nguyên.
Phân tích sự đánh đổi giữa độ chính xác, tốc độ và dung lượng mô hình.
4
Cắt tỉa mô hình (Pruning) và đánh giá hiệu năng trên thiết bị biên
Tìm hiểu nguyên lý, mục tiêu và các tiêu chí đánh giá cắt tỉa mô hình.
Thực hành cắt tỉa có cấu trúc (Structured Pruning) và phi cấu trúc (Unstructured Pruning).
Đánh giá mức độ thưa (Sparsity) và tỷ lệ tham số bị loại bỏ.
So sánh chất lượng mô hình trước và sau cắt tỉa.
Đo lường kích thước mô hình, tốc độ suy luận và mức tiêu thụ tài nguyên.
Phân tích sự đánh đổi giữa độ chính xác, mức độ cắt tỉa và hiệu năng suy luận.
5
Chuyển đổi, kết xuất và triển khai mô hình
Tìm hiểu mục đích, nguyên lý và quy trình kết xuất mô hình (Model Export).
Thực hành chuyển đổi mô hình sang ONNX, TensorRT và TensorFlow Lite.
Tìm hiểu các tham số export: opset, precision, input shape, batch size và dynamic shapes.
Thực hành kết xuất mô hình với kích thước đầu vào tĩnh (Static Shapes) và động (Dynamic Shapes).
Tìm hiểu các runtime suy luận phù hợp: ONNX Runtime, TensorRT Runtime và TensorFlow Lite Interpreter.
Đánh giá độ chính xác, kích thước mô hình và tốc độ suy luận sau chuyển đổi.
Lựa chọn định dạng mô hình và runtime phù hợp với từng nền tảng Edge AI.
6
Triển khai suy luận ONNX Runtime bằng C++ trên Raspberry Pi và đánh giá hiệu năng
Tìm hiểu kiến trúc ONNX Runtime và quy trình suy luận bằng C++.
Cài đặt, cấu hình ONNX Runtime và môi trường biên dịch trên Raspberry Pi.
Xây dựng chương trình inference bằng C++ với mô hình ONNX.
Tích hợp các bước tiền xử lý, suy luận và hậu xử lý.
Đánh giá tốc độ suy luận, FPS và độ trễ xử lý.
Đo lường mức sử dụng CPU, RAM và tài nguyên hệ thống.
So sánh hiệu năng suy luận giữa Python và C++ trên Raspberry Pi.
7
Triển khai mô hình AI với TensorRT và C++ trên NVIDIA Jetson
Tìm hiểu kiến trúc TensorRT, CUDA và vai trò của các thành phần trong suy luận GPU.
- Cài đặt và cấu hình môi trường TensorRT, CUDA, C++ trên NVIDIA Jetson.
- Chuyển đổi mô hình ONNX sang TensorRT Engine với các cấu hình phù hợp.
- Xây dựng chương trình inference bằng C++ sử dụng TensorRT API.
- Tối ưu cấu hình suy luận với FP32, FP16 và INT8 khi phần cứng hỗ trợ.
- Đánh giá độ chính xác, FPS, độ trễ và thời gian khởi tạo engine.
- Đo lường mức sử dụng GPU, CPU, RAM và công suất tiêu thụ.
- So sánh hiệu năng TensorRT C++ với ONNX Runtime C++ trên nền tảng phù hợp.
8
Xây dựng pipeline AI thời gian thực với GStreamer và NVIDIA DeepStream trên Jetson
Tìm hiểu kiến trúc pipeline GStreamer và cơ chế xử lý luồng dữ liệu đa phương tiện.
Xây dựng pipeline xử lý video từ camera hoặc file video bằng GStreamer.
Tích hợp mô hình AI vào pipeline để thực hiện suy luận thời gian thực.
Tìm hiểu kiến trúc NVIDIA DeepStream và các thành phần xử lý video AI.
Xây dựng pipeline DeepStream với các thành phần giải mã, suy luận, theo dõi đối tượng và hiển thị kết quả.
Tìm hiểu cơ chế tăng tốc GPU, quản lý bộ nhớ và xử lý video trên Jetson.
Sử dụng GST Shark đánh giá hiệu năng, FPS, độ trễ đầu-cuối, mức sử dụng CPU/GPU và bộ nhớ.

