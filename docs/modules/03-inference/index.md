# Lab 3 · Inference với OpenCV

## Mục tiêu

- Tách pipeline thành `preprocess → infer → postprocess`.
- Chạy mô hình trên ảnh, video và camera.
- Đo thời gian từng bước và quan sát ảnh hưởng của ngưỡng `conf`/`nms`.

## Đầu vào

- Checkpoint `best.pt` từ Lab 2.
- Ảnh hoặc video kiểm thử.
- OpenCV và môi trường đã hoàn tất ở Lab 1.

## Sản phẩm đầu ra

Ảnh/video có bounding box và bảng thời gian `pre/inf/post`.

Tiếp tục với [Khung hướng dẫn](guide.md).
