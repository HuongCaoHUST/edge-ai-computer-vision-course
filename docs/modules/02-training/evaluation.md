# Đánh giá chất lượng mô hình

## Mục tiêu

Đánh giá SSD MobileNet trên tập validation/test bằng các chỉ số phù hợp cho object detection, thay vì chỉ nhìn vào một vài ảnh kết quả.

## 1. IoU và ghép prediction

Với prediction box và ground-truth box:

```text
IoU = diện tích phần giao / diện tích phần hợp
```

Một prediction thường được ghép với ground truth khi cùng lớp và đạt ngưỡng IoU, ví dụ `IoU ≥ 0.5`. Một ground-truth chỉ được ghép với một prediction có điểm tin cậy cao nhất phù hợp.

## 2. Các chỉ số

- **TP:** dự đoán đúng lớp và đúng vị trí;
- **FP:** dự đoán sai lớp, sai vị trí hoặc dự đoán trùng;
- **FN:** object thật chưa được phát hiện;
- **Precision:** `TP / (TP + FP)`;
- **Recall:** `TP / (TP + FN)`;
- **AP:** diện tích dưới đường Precision–Recall của một lớp;
- **mAP:** trung bình AP trên các lớp.

Precision cao cho thấy các box được giữ lại thường đúng. Recall cao cho thấy model bỏ sót ít object. Hai chỉ số cần được đọc cùng nhau.

## 3. Chạy đánh giá

Đánh giá trên tập không dùng để cập nhật trọng số:

```bash
python lab3_eval.py --n 300 --crosscheck
```

Lưu kết quả dạng JSON và lập bảng AP cho từng lớp. Có thể đối chiếu implementation tự viết với `torchmetrics` để phát hiện khác biệt trong cách nội suy Precision–Recall, xử lý object `difficult` hoặc ngưỡng score.

## 4. Thí nghiệm với confidence

Chạy lại với các ngưỡng hiển thị khác nhau, ví dụ `0.3` và `0.7`, rồi so sánh Precision và Recall. Confidence threshold làm thay đổi prediction được giữ lại; vì vậy nó thường ảnh hưởng Precision/Recall tại một operating point.

Khi tính AP/mAP, cần giữ quy trình sắp xếp prediction theo score và quét nhiều ngưỡng score. Không kết luận chất lượng model chỉ từ một confidence threshold.

## 5. Phân tích kết quả

Xác định:

- lớp có AP cao nhất và thấp nhất;
- các lỗi thường gặp: bỏ sót, nhầm lớp, box quá rộng hoặc quá hẹp;
- sự khác biệt giữa kết quả trên ảnh dễ và ảnh có nhiều object;
- đánh đổi giữa Precision và Recall khi thay đổi confidence.
