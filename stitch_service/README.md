# 🖼️ Stitch Service

> **Mô tả:** Microservice nhận nhiều ảnh chụp từ camera di động, ghép thành ảnh panorama, ánh xạ toàn bộ bounding box (sản phẩm + nhãn giá) vào hệ tọa độ panorama, sau đó chạy thuật toán **Cluster-based Voting** để gắn giá đúng cho từng nhóm sản phẩm.

---

## 🏗️ Kiến trúc tổng quan

```
Mobile App gửi N ảnh
        │
        ▼
[/upload-batch endpoint]
        │
        ├─► Gọi RPM API (song song) ─► lấy products + price_tags từng ảnh
        │
        │
        ▼
[Image Stitching]   ─── Ghép panorama bằng stitching library
        │
        ▼
[Coordinate Warping]
   ├─ Warp product boxes  ─┐
   └─ Warp tag boxes ──────┤─► Tất cả về hệ tọa độ panorama
                           │
        ▼
[NMS] ── Lọc sản phẩm trùng lặp (KHÔNG áp lên price tags)
        │
        ▼
[Cluster-based Voting]
   1. Local Mapping  ─ mỗi sản phẩm → tìm tag gần nhất bên dưới
   2. Clustering     ─ nhóm sản phẩm theo class_name
   3. Voting         ─ tag được vote nhiều nhất → assign cho cả nhóm
        │
        ▼
JSON Response (panorama base64 + mapped_products + price_tags)
```

---

## 📁 Cấu trúc thư mục

```text
stitch_service/
├── mapping_core.py            # Logic chính: stitching + warp + NMS + Voting
│
├── server/
│   ├── app.py                 # FastAPI app (endpoint /upload-batch, /stitch-with-mapping)
│   └── schemas.py             # Pydantic schemas (StitchResponse, MappedProduct...)
│
├── stitching/                 # Thư viện stitching (superpoint + lightglue)
│   ├── stitcher.py
│   ├── warper.py
│   ├── cropper.py
│   └── ...
│
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## ⚙️ Yêu cầu hệ thống

| Thành phần | Tối thiểu     | Mô tả                           |
| ---------- | ------------- | ------------------------------- |
| Python     | 3.10          |                                 |
| RAM        | 8 GB          | Stitching ảnh lớn cần nhiều RAM |
| Disk       | 2 GB          | Không cần GPU                   |
| OS         | Ubuntu 20.04+ |                                 |

> ℹ️ Service này **không cần GPU** — chỉ dùng CPU cho stitching và geometric warping.

---

## 🚀 Hướng dẫn triển khai

### Cách 1: Docker (Khuyến nghị)

```bash
# Bước 1: Vào thư mục service
cd stitch_service

# Bước 2: Cấu hình URL của RPM API
# Mặc định trỏ tới http://localhost:8000/predict
# Nếu RPM chạy trên server khác, đặt biến môi trường:
export RPM_API_URL=http://<ip-rpm-server>:8000/predict

# Bước 3: Build và chạy
docker-compose up -d --build

# Bước 4: Kiểm tra
curl http://localhost:8001/
```

---

### Cách 2: Local không Docker

```bash
cd stitch_service

# Bước 1: Tạo môi trường ảo
python3 -m venv venv
source venv/bin/activate

# Bước 2: Cài dependencies
pip install -r requirements.txt

# Bước 3: Đặt URL RPM API (nếu RPM chạy port khác)
export RPM_API_URL=http://localhost:8000/predict

# Bước 4: Khởi động
python3 server/app.py
```

---

## 🌍 Biến môi trường

| Biến          | Mặc định                        | Mô tả               |
| ------------- | ------------------------------- | ------------------- |
| `RPM_API_URL` | `http://localhost:8000/predict` | URL của RPM Service |

---

## 🌐 API Endpoints

**Base URL:** `http://localhost:8001`

---

### `GET /`
Kiểm tra service đang chạy.

```bash
curl http://localhost:8001/
# {"message": "Stitching & Mapping Service is running", "docs": "/docs"}
```

---

### `POST /upload-batch`
**Luồng chính cho Mobile App.** Tự động gọi RPM API để lấy detections rồi stitch + mapping.

**Yêu cầu:** RPM Service phải đang chạy và accessible qua `RPM_API_URL`.

**Request:** `multipart/form-data`
| Field   | Type   | Mô tả                                 |
| ------- | ------ | ------------------------------------- |
| `files` | File[] | Tối thiểu 2 ảnh, gửi theo thứ tự chụp |

**Ví dụ curl:**
```bash
curl -X POST http://localhost:8001/upload-batch \
  -F "files=@anh1.jpg" \
  -F "files=@anh2.jpg" \
  -F "files=@anh3.jpg" \
  | python3 -m json.tool
```

**Response:**
```json
{
  "panorama_width": 3840,
  "panorama_height": 1080,
  "mapped_products": [
    {
      "class_name": "coca_cola_330ml",
      "box": [145.0, 50.0, 330.0, 290.0],
      "score": 0.87,
      "original_image": "anh1.jpg",
      "price_tag": {
        "tag_id": 0,
        "price": "15,000",
        "box": [145.0, 295.0, 330.0, 340.0]
      }
    }
  ],
  "price_tags": [
    {"tag_id": 0, "price": "15,000", "box": [145.0, 295.0, 330.0, 340.0]},
    {"tag_id": 1, "price": "25,000", "box": [450.0, 295.0, 620.0, 340.0]}
  ],
  "panorama_url": "data:image/jpeg;base64,/9j/4AAQSkZ..."
}
```

---

### `POST /stitch-with-mapping`
Dùng khi caller **tự cung cấp detections** (không gọi RPM). Phù hợp để test logic warping/voting độc lập.

**Request:** `multipart/form-data`
| Field        | Type          | Mô tả                               |
| ------------ | ------------- | ----------------------------------- |
| `files`      | File[]        | Tối thiểu 2 ảnh                     |
| `detections` | String (JSON) | Detections map theo format dưới đây |

**Format JSON của `detections`:**
```json
{
  "anh1.jpg": {
    "products": [
      {"box": [100, 50, 300, 250], "class_name": "cola", "score": 0.92}
    ],
    "price_tags": [
      {"box": [100, 255, 300, 310], "price": "15,000", "tag_id": 0}
    ]
  },
  "anh2.jpg": {
    "products": [
      {"box": [80, 60, 280, 240], "class_name": "pepsi", "score": 0.88}
    ],
    "price_tags": [
      {"box": [80, 245, 280, 300], "price": "12,000", "tag_id": 1}
    ]
  }
}
```

**Ví dụ curl:**
```bash
curl -X POST http://localhost:8001/stitch-with-mapping \
  -F "files=@anh1.jpg" \
  -F "files=@anh2.jpg" \
  -F 'detections={"anh1.jpg": {"products": [{"box":[10,10,200,200],"class_name":"cola","score":0.9}], "price_tags": [{"box":[10,205,200,250],"price":"15000","tag_id":0}]}, "anh2.jpg": {"products": [], "price_tags": []}}' \
  | python3 -m json.tool
```

---

## 🔧 Vận hành & Bảo trì

### Điều chỉnh URL RPM API

Không cần rebuild image — chỉ cần đặt biến môi trường và restart:

```bash
# Cập nhật docker-compose.yml, thêm environment section:
# environment:
#   - RPM_API_URL=http://rpm-server:8000/predict

docker-compose down && docker-compose up -d
```

### Xem logs

```bash
docker logs -f stitch-server
```

### Giới hạn bộ nhớ

Mặc định container được cấp tối đa 8GB RAM. Điều chỉnh trong `docker-compose.yml`:
```yaml
deploy:
  resources:
    limits:
      memory: 8G   # Tăng nếu stitch ảnh độ phân giải cao
```

---

## 🐛 Troubleshooting

| Lỗi                              | Nguyên nhân                 | Cách fix                                          |
| -------------------------------- | --------------------------- | ------------------------------------------------- |
| `"Stitching failed"` (HTTP 400)  | Các ảnh không có đủ overlap | Ảnh cần chụp overlap tối thiểu 30%                |
| `price_tags: []`                 | RPM không trả về tags       | Kiểm tra RPM `/health` và model `tag_yolo_path`   |
| `ValueError: too many values`    | Cũ — đã được fix            | Update code lên phiên bản mới nhất                |
| Response timeout                 | Quá nhiều ảnh / ảnh quá lớn | Resize ảnh xuống max 1920px width trước khi gửi   |
| `Connection refused` khi gọi RPM | RPM service chưa chạy       | Đảm bảo RPM đang up trước khi gọi `/upload-batch` |
