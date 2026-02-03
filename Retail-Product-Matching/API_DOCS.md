# 📖 Retail Product Matching API Documentation

Tài liêu này cung cấp thông tin chi tiết về các API Endpoints được triển khai trong hệ thống nhận diện sản phẩm bán lẻ (RPM).

## 🌍 Tổng quan
- **Base URL**: `http://<your-server-ip>:8000`
- **Tài liệu tương tác (Swagger)**: `/docs`
- **Tài liệu thay thế (ReDoc)**: `/redoc`

---

## 🛠 Endpoints Details

### 1. [GET] Trang chủ & Chỉ mục
Trả về thông tin cơ bản và các đường dẫn chức năng.
- **URL**: `/`
- **Response**:
```json
{
  "message": "Welcome to Retail Product Matching API",
  "docs": "/docs",
  "health": "/health"
}
```

### 2. [GET] Kiểm tra sức khỏe hệ thống (Health Check)
Dùng để kiểm tra server đã sẵn sàng phục vụ chưa (đã load xong model và gallery chưa).
- **URL**: `/health`
- **Response**:
```json
{
  "status": "ok",
  "model_loaded": true,
  "device_info": {
    "yolo": "cuda",
    "dino": "cuda",
    "lg": "cuda"
  }
}
```

### 3. [POST] Nhận diện sản phẩm (Predict)
Endpoint chính để xử lý ảnh và khớp mã sản phẩm.
- **URL**: `/predict`
- **Content-Type**: `multipart/form-data`
- **Input**:
    - `file`: Ảnh cần xử lý (Format: JPG, PNG, JPEG).

#### Cấu trúc kết quả trả về (JSON):
| Trường           | Kiểu dữ liệu | Mô tả                                           |
| :--------------- | :----------- | :---------------------------------------------- |
| `matches`        | `Array`      | Danh sách các vật thể phát hiện được và khớp mã |
| `inference_time` | `Float`      | Tổng thời gian xử lý (giây)                     |
| `image_size`     | `Array`      | Kích thước ảnh đầu vào [Width, Height]          |

#### Chi tiết mỗi item trong `matches`:
| Trường       | Kiểu dữ liệu | Mô tả                                             |
| :----------- | :----------- | :------------------------------------------------ |
| `class_name` | `String`     | Tên mã sản phẩm khớp được (Gallery name)          |
| `score`      | `Float`      | Điểm tin cậy tổng hợp (0.0 - 1.0)                 |
| `box`        | `Array`      | Toạ độ [x1, y1, x2, y2] trong ảnh gốc             |
| `matched`    | `Boolean`    | `true` nếu vượt ngưỡng tin cậy, ngược lại `false` |
| `details`    | `Object`     | Chi tiết điểm DINO và số inliers từ LightGlue     |

---

## 💻 Ví dụ cách gọi API

### Sử dụng cURL:
```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@data/test_images/1.jpg;type=image/jpeg'
```

### Sử dụng Python (requests):
```python
import requests

url = "http://localhost:8000/predict"
with open("image.jpg", "rb") as f:
    files = {"file": ("image.jpg", f, "image/jpeg")}
    response = requests.post(url, files=files)

print(response.json())
```

---

## ⚠️ Mã lỗi thường gặp
- **400 Bad Request**: File gửi lên không phải là ảnh hoặc định dạng không hỗ trợ.
- **503 Service Unavailable**: Server đang trong quá trình load models (thường mất 3-5s lúc khởi động).
- **500 Internal Server Error**: Lỗi logic bên trong pipeline xử lý.

## 🚀 Mở rộng GPU cho Team
Nếu triển khai cho team trên server có GPU mạnh:
1. Sửa `configs/settings.yaml` đặt tất cả thiết bị thành `"cuda"`.
2. Sử dụng `docker-compose up -d --build` để chạy môi trường container hóa ổn định nhất.
