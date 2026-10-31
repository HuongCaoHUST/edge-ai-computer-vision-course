# Bộ Lab: Huấn luyện, triển khai và đánh giá SSD MobileNet trên thiết bị biên

Mô hình xuyên suốt: **SSDLite320 MobileNetV3-Large** (torchvision), dữ liệu **Pascal VOC 2007** (20 lớp + 1 lớp nền = `num_classes=21`).

| Lab | Mục tiêu | Script | Sản phẩm nộp | Thời lượng |
|---|---|---|---|---|
| 0 | Cài môi trường | `requirements.txt` | Ảnh chụp `pip list` + phiên bản | 30 phút |
| 1 | Huấn luyện bằng Python | `lab1_train.py`, `plot_log.py`, `export_onnx.py` | `runs/log.csv`, `curves.png`, `best.pt` | 2 buổi |
| 2 | Inference với OpenCV | `lab2_infer.py`, `detector.py` | Ảnh/video kết quả + bảng thời gian từng bước | 1 buổi |
| 3 | Đánh giá chất lượng | `lab3_eval.py` | `results/quality.json` | ½ buổi |
| 4 | FPS và độ trễ | `lab4_benchmark.py` | `results/bench_<thiết bị>.json` | ½ buổi |
| 5 | CPU, RAM, nhiệt độ + báo cáo | `lab5_resources.py`, `make_report.py` | `results/report.md` | 1 buổi |

## Lab 0 - Môi trường

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import torch, torchvision, cv2; print(torch.__version__, torchvision.__version__, cv2.__version__)"
```

- **Raspberry Pi:** dùng Raspberry Pi OS 64-bit; thay `opencv-python` bằng `opencv-python-headless`.
- **Jetson:** cài PyTorch theo wheel của NVIDIA (JetPack), không dùng wheel PyPI.
- **Huấn luyện nên làm trên máy mạnh/Colab GPU**, sau đó chép `runs/best.pt` sang thiết bị biên.

## Lab 1 - Huấn luyện

```bash
python lab1_train.py --n-train 500 --epochs 5 --device cpu      # chạy thử nhanh
python lab1_train.py --n-train 5011 --n-val 500 --epochs 30 --device cuda   # chạy đầy đủ
python plot_log.py
```
Lần đầu script tự tải VOC2007 (~1 GB gồm trainval và test).

**Việc cần làm**
1. Đọc `common.py`: chỉ ra chỗ chuyển nhãn VOC XML về `boxes`/`labels` và chỗ tăng cường dữ liệu.
2. Chạy huấn luyện, nộp `curves.png`.
3. Thí nghiệm (chọn 2): `--lr 1e-3` so với `1e-2`; `--batch 8` so với `32`; bỏ augmentation; tăng `--n-train`.

**Câu hỏi**
- `val_loss` bắt đầu tăng ở epoch nào? Đó là dấu hiệu gì, nên làm gì?
- Vì sao `num_classes = 21`? Điều gì xảy ra nếu đặt 20?
- Vì sao `model.train()` phải gọi trước vòng lặp và vì sao hàm `val_loss` phải giữ BatchNorm ở chế độ eval?

> Với tập con nhỏ và ít epoch, mAP sẽ thấp. Đây là hiện tượng bình thường, hãy ghi lại số thật của bạn chứ không so với số trong slide (slide chỉ là giá trị minh họa).

**Tùy chọn - xuất ONNX:** `python export_onnx.py` (thử nghiệm; nếu lỗi, thử `--opset 11/17` hoặc dùng backend `torch` cho các lab sau).

## Lab 2 - Inference với OpenCV

```bash
python lab2_infer.py --source anh.jpg                       # ảnh -> results/out.jpg
python lab2_infer.py --source 0 --show                      # camera USB (cần màn hình)
python lab2_infer.py --source video.mp4 --out results/out.mp4 --max-frames 300   # không màn hình
```

**Đọc `detector.py`** và xác định 3 bước: `preprocess` → `infer` → `postprocess`. Tiền xử lý đổi BGR→RGB, resize 320×320, chia 255. NMS dùng `cv2.dnn.NMSBoxes` theo từng lớp.

**Thí nghiệm**
1. Bỏ dòng `cv2.cvtColor(..., BGR2RGB)` trong `preprocess`: kết quả thay đổi thế nào? Giải thích vì sao.
2. Thay đổi `--conf` (0.2, 0.5, 0.8) và `--nms` (0.2, 0.45, 0.7): chụp ảnh so sánh, nhận xét số hộp trùng và số đối tượng bị bỏ sót.
3. Ghi lại thời gian `pre/inf/post` do script in ra. Bước nào là nút thắt cổ chai?

**Mở rộng (đúng phong cách slide `cv2.dnn.DetectionModel`):** backend `cvdnn` chạy được các mô hình TensorFlow SSD MobileNet COCO của OpenCV (cần tự tải `.pb`, `.pbtxt` và file nhãn):
```bash
python lab2_infer.py --backend cvdnn --weights frozen_inference_graph.pb --pbtxt graph.pbtxt --labels coco.names --source anh.jpg
```
Backend này không dùng được cho Lab 3 vì nhãn là COCO, không phải VOC.

## Lab 3 - Đánh giá chất lượng

```bash
python lab3_eval.py --n 300 --crosscheck
```
Script tự cài đặt: IoU, ghép dự đoán với ground truth, đường Precision-Recall, AP (nội suy mọi điểm), mAP@0.5, mAP@0.5:0.95. Cờ `--crosscheck` đối chiếu với `torchmetrics`.

**Việc cần làm**
1. Nộp `results/quality.json` và bảng AP từng lớp. Lớp nào tốt nhất, kém nhất? Giả thuyết nguyên nhân?
2. Chạy lại với `--conf-op 0.3` và `0.7`: Precision/Recall thay đổi theo hướng nào? Vì sao mAP không phụ thuộc `--conf-op`?
3. So sánh số của bạn với `torchmetrics`. Vì sao có thể lệch nhẹ (gợi ý: cách nội suy, VOC `difficult`, ngưỡng điểm)?
4. **Tính tay:** 1 lớp có 4 đối tượng thật, mô hình trả 5 hộp (TP, FP, TP, TP, FP theo thứ tự điểm giảm dần). Vẽ bảng cumulative TP/FP, tính Precision, Recall tại từng hộp và AP.

## Lab 4 - FPS và độ trễ

Chạy trên **từng thiết bị**, đặt `--tag` khác nhau:
```bash
python lab4_benchmark.py --source voc --tag laptop
python lab4_benchmark.py --source voc --tag pi4 --threads 4
```
Quy trình chuẩn đã cài sẵn: warm-up 20 lần, đo 200 lần, báo cáo mean, std, p50, p95, p99, FPS = 1000/mean, thời gian từng bước.

**Câu hỏi**
- Vì sao p95 quan trọng hơn mean với ứng dụng thời gian thực?
- Bỏ warm-up (`--warmup 0`) thì kết quả đổi ra sao?
- Thay đổi `--threads` (1, 2, 4): FPS có tăng tuyến tính không? Vì sao?
- So sánh FPS của `pipeline đầy đủ` với FPS chỉ tính bước `inf`.

## Lab 5 - Tài nguyên và báo cáo

```bash
python lab5_resources.py --source voc --tag laptop --duration 60
python make_report.py
```
Script đo trong luồng nền mỗi 0.5 s: CPU tiến trình (có thể vượt 100% khi dùng nhiều lõi), CPU hệ thống, RAM (RSS), % RAM hệ thống, nhiệt độ; có giai đoạn nghỉ (idle) để so sánh. Dữ liệu chuỗi thời gian lưu ở `results/resources_<tag>.csv`.

Trên Jetson chạy thêm ở terminal khác: `tegrastats --interval 500 --logfile results/tegra.log`.

**Việc cần làm**
1. Vẽ CPU và nhiệt độ theo thời gian từ file CSV. Có hiện tượng nóng lên rồi giảm xung (thermal throttling) không? FPS đầu và cuối phiên có khác nhau không?
2. RAM tăng bao nhiêu sau khi nạp mô hình? Có rò rỉ bộ nhớ không (RAM đỉnh so với trung bình)?
3. Viết kết luận 1 trang dựa trên `results/report.md`: thiết bị nào đạt thời gian thực (≥ 15-30 FPS)? Đánh đổi giữa mAP, FPS, CPU/RAM? Đề xuất tối ưu (lượng tử hóa, cắt tỉa, giảm kích thước đầu vào), nối sang bài sau.

## Rubric gợi ý (100 điểm)

| Tiêu chí | Điểm |
|---|---:|
| Lab 1: huấn luyện chạy được, log và đường cong đúng, trả lời câu hỏi | 20 |
| Lab 2: inference ảnh/video/camera, thí nghiệm BGR/RGB, conf, NMS | 20 |
| Lab 3: số liệu chất lượng đầy đủ, tính tay AP đúng | 20 |
| Lab 4: đo đúng quy trình, có p95, phân tích từng bước | 15 |
| Lab 5: CPU/RAM/nhiệt độ, biểu đồ, báo cáo so sánh thiết bị | 20 |
| Chất lượng code, tái lập được, trình bày | 5 |

## Xử lý sự cố nhanh

| Hiện tượng | Cách xử lý |
|---|---|
| Tải VOC chậm/lỗi | Tải thủ công `VOCtrainval_06-Nov-2007.tar` và `VOCtest_06-Nov-2007.tar`, giải nén vào `data/` |
| Hết RAM khi huấn luyện | Giảm `--batch`, `--workers 0` |
| `loss` là `nan` | Giảm `--lr`, kiểm tra nhãn hộp có `x2 > x1` |
| Không có phát hiện nào | Hạ `--conf` xuống 0.2; kiểm tra đúng checkpoint; model chưa học đủ |
| `cv2.imshow` lỗi trên Pi | Dùng `opencv-python-headless` và chạy không có `--show`, ghi ra file bằng `--out` |
| ONNX export lỗi | Thử `--opset` khác hoặc dùng backend `torch` |

## Lưu ý về mã nguồn

Mã được kiểm tra cú pháp và phần tính AP được thử trên dữ liệu mẫu, nhưng **chưa chạy huấn luyện/suy luận đầy đủ trên phần cứng của bạn** (cần tải dữ liệu và trọng số). Hãy chạy thử toàn bộ chuỗi Lab 1→5 với `--n-train 50 --epochs 1` trước khi phát cho học viên. Phần `export_onnx.py` và backend `onnx`/`cvdnn` là tùy chọn, cần kiểm tra theo phiên bản PyTorch/OpenCV bạn dùng.
