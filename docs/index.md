# Computer Vision trên thiết bị biên

Course thực hành về **huấn luyện, triển khai và đánh giá SSDLite320 MobileNetV3-Large** với dữ liệu Pascal VOC 2007 trên thiết bị biên.

## Bắt đầu

1. Đọc [Lộ trình khóa học](course.md).
2. Hoàn thành [Lab 1 · Chuẩn bị môi trường](modules/01-environment/index.md).
3. Thực hiện 8 nội dung theo thứ tự trong thanh điều hướng.

## Mô hình và dữ liệu

- Mô hình: `SSDLite320 MobileNetV3-Large` từ `torchvision`.
- Dữ liệu: Pascal VOC 2007, 20 lớp đối tượng và 1 lớp nền (`num_classes=21`).
- Quy trình: huấn luyện → inference → đánh giá chất lượng → benchmark → báo cáo tài nguyên.

!!! tip "Cách sử dụng"
    Dùng ô tìm kiếm để tìm nhanh lệnh, thuật ngữ hoặc nội dung trong từng lab.
