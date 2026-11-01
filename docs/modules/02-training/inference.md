# Inference với OpenCV

## Mục tiêu

Xây dựng chương trình chạy checkpoint SSD MobileNet trên ảnh, video và camera bằng OpenCV.

## 1. Pipeline inference

Tách chương trình thành ba bước để có thể đo thời gian riêng:

1. **Preprocess:** đọc ảnh BGR, đổi sang RGB, resize về `320×320` và chuẩn hóa.
2. **Inference:** nạp checkpoint, chuyển model sang `eval()` và chạy với `torch.no_grad()`.
3. **Postprocess:** lọc confidence, NMS, đổi tọa độ box về ảnh gốc và vẽ kết quả.

OpenCV đọc ảnh theo BGR, trong khi model thường được huấn luyện với RGB:

```python
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```

Nếu bỏ bước này, model vẫn có thể chạy nhưng kết quả nhận diện thường thay đổi do các kênh màu bị đảo.

## 2. Chạy trên các nguồn dữ liệu

```bash
python lab2_infer.py --source anh.jpg --weights runs/best.pt
python lab2_infer.py --source video.mp4 --out results/out.mp4 --max-frames 300
python lab2_infer.py --source 0 --show
```

Với thiết bị không có màn hình, không dùng `--show`; hãy ghi kết quả ra file bằng `--out`.

## 3. Confidence và NMS

Thử các cấu hình sau trên cùng một ảnh:

```bash
python lab2_infer.py --source anh.jpg --conf 0.2 --nms 0.45
python lab2_infer.py --source anh.jpg --conf 0.5 --nms 0.45
python lab2_infer.py --source anh.jpg --conf 0.8 --nms 0.70
```

Confidence thấp giữ lại nhiều prediction hơn nhưng có thể làm tăng false positive. NMS loại các box chồng lấn; ngưỡng NMS thấp sẽ loại mạnh hơn.

Ghi lại số box, số object bị bỏ sót và các false positive trong từng cấu hình. Không thay đổi đồng thời quá nhiều tham số nếu muốn giải thích nguyên nhân kết quả thay đổi.

## 4. Kết quả cần quan sát

- ảnh/video có bounding box, class name và confidence;
- thời gian `preprocess`, `inference`, `postprocess`;
- thay đổi kết quả khi bỏ BGR → RGB;
- thay đổi kết quả khi thay `conf` và `nms`.

Kết quả của bước này được dùng tiếp cho đánh giá chất lượng và benchmark trên thiết bị biên.
