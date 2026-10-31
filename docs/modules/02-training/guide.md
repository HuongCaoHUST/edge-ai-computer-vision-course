# Hướng dẫn thực hành

## 1. Kiểm tra công cụ

```bash
python3 --version
git --version
```

Python cần có phiên bản 3.10 trở lên.

## 2. Tạo thư mục làm việc

```bash
mkdir lab1
cd lab1
mkdir src output
```

Tạo `src/main.py`:

```python
from datetime import datetime


def main() -> None:
    message = f"Lab 2 hoàn thành lúc {datetime.now():%Y-%m-%d %H:%M:%S}"
    print(message)


if __name__ == "__main__":
    main()
```

## 3. Chạy chương trình

```bash
python src/main.py | tee output/result.txt
```

Trên Windows PowerShell, dùng:

```powershell
python src/main.py | Tee-Object -FilePath output/result.txt
```

## 4. Tạo README

```markdown
# Lab 2

- Họ và tên:
- MSSV:
- Môi trường thực hiện:
- Cách chạy: `python src/main.py`
```

!!! success "Hoàn thành"
    Bài làm đạt yêu cầu cơ bản khi chương trình chạy không lỗi và có đủ `README.md`, `src/main.py`, `output/result.txt`.

Tiếp tục với [Kiểm tra & nộp bài](submission.md).
