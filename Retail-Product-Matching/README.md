# 🛒 Retail Product Matching (RPM) Service

> **Mô tả:** Microservice nhận diện sản phẩm trên kệ hàng bán lẻ và đọc giá từ nhãn giá (price tag). Sử dụng mô hình YOLO để phát hiện, DINOv3 + LightGlue để matching, và PaddleOCR + EasyOCR để đọc giá.

---

## 🏗️ Kiến trúc tổng quan

```
Ảnh đầu vào
    │
    ▼
[YOLO OBB] ──────── Phát hiện vùng sản phẩm
    │
    ▼
[CLAHE Preprocessing] ── Tăng cường độ tương phản
    │
    ▼
[DINOv3] ──────── Trích xuất đặc trưng toàn cục (Global Features)
    │
    ▼
[Matrix Matching] ─── So khớp nhanh với Feature Database
    │
    ▼
[LightGlue ONNX] ─── Xác minh cục bộ (Local Verification)
    │
    ▼
[YOLO Tag Detector] ─ Phát hiện vùng nhãn giá
    │
    ▼
[PaddleOCR / EasyOCR] ── Đọc giá tiền
    │
    ▼
JSON Response (matches + price_tags)
```

---

## 📁 Cấu trúc thư mục

```text
Retail-Product-Matching/
├── configs/
│   └── settings.yaml          # Cấu hình chính (model paths, devices, thresholds)
│
├── data/
│   ├── weights/
│   │   ├── yolo/
│   │   │   └── best-obb.pt    # YOLO OBB — detect sản phẩm     ← PHẢI CÓ
│   │   ├── yolo/
│   │   │   └── best.pt        # YOLO — detect nhãn giá          ← PHẢI CÓ
│   │   └── lightglue/
│   │       ├── superpoint_batch.onnx                            ← PHẢI CÓ
│   │       └── lightglue_batch.onnx                             ← PHẢI CÓ
│   ├── support_images/        # Ảnh mẫu của từng SKU sản phẩm
│   ├── test_images/           # Ảnh test đầu vào
│   └── support_db.pt          # Feature Bank đã build            ← PHẢI CÓ
│
├── retail_matcher/            # Core package
│   ├── models/
│   │   ├── loader.py          # Load YOLO, DINOv3, ONNX
│   │   ├── extraction.py      # Trích xuất feature (DINOv3)
│   │   ├── matching.py        # Matrix matching + LightGlue verification
│   │   └── ocr.py             # PriceTagParser (YOLO detect + OCR đọc giá)
│   ├── utils/
│   │   ├── common.py          # Logger, utilities
│   │   ├── config.py          # Load settings.yaml
│   │   ├── processing.py      # CLAHE, preprocessing, map_products_to_price_tags
│   │   └── visualization.py   # Vẽ bounding box
│   └── pipeline.py            # ProductMatcher — orchestrator chính
│
├── server/
│   ├── app.py                 # FastAPI app (endpoint /predict, /health)
│   └── schemas.py             # Pydantic schemas (MappedItem, PriceTagResult...)
│
├── scripts/
│   ├── build_gallery.py       # Build Feature Bank từ support_images
│   └── test_api_client.py     # Script test API nhanh
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── run_server.sh              # Script khởi động nhanh local
```

---

## ⚙️ Yêu cầu hệ thống

| Thành phần | Tối thiểu     | Khuyến nghị  |
| ---------- | ------------- | ------------ |
| Python     | 3.10          | 3.10         |
| CUDA       | 11.8          | 12.1         |
| RAM        | 16 GB         | 32 GB        |
| VRAM       | 6 GB          | 12 GB        |
| Disk       | 10 GB         | 20 GB        |
| OS         | Ubuntu 20.04+ | Ubuntu 22.04 |

---

## 🚀 Hướng dẫn triển khai

### Cách 1: Docker (Khuyến nghị cho Production)

**Yêu cầu:** Docker + nvidia-container-toolkit đã cài trên host.

```bash
# Bước 1: Clone repo
git clone <repo-url>
cd Retail-Product-Matching

# Bước 2: Đặt model weights vào đúng vị trí (xem bảng bên dưới)

# Bước 3: Kiểm tra cấu hình
cat configs/settings.yaml

# Bước 4: Build và chạy
docker-compose up -d --build

# Bước 5: Xem log
docker logs -f rpm-server
```

**Kiểm tra hoạt động:**
```bash
curl http://localhost:8000/health
```

---

### Cách 2: Local không Docker

```bash
# Bước 1: Tạo môi trường ảo
python3 -m venv venv
source venv/bin/activate

# Bước 2: Cài dependencies
pip install -r requirements.txt

# Bước 3: Cài PaddleOCR + EasyOCR (riêng vì nặng)
pip install paddlepaddle-gpu paddleocr easyocr

# Bước 4: Build Feature Bank (chỉ làm 1 lần hoặc khi thêm SKU mới)
python3 scripts/build_gallery.py

# Bước 5: Khởi động server
bash run_server.sh
```

---

## 📋 Cấu hình `configs/settings.yaml`

```yaml
paths:
  support_db: "data/support_db.pt"           # Feature bank đã build
  product_yolo_path: "data/weights/yolo/best-obb.pt"  # Model detect sản phẩm
  tag_yolo_path: "data/weights/yolo/best.pt"           # Model detect nhãn giá

models:
  yolo_conf: 0.25          # Ngưỡng confidence YOLO (0.0 – 1.0)
  top_k: 5                 # Số ứng viên DINOv3 giữ lại để verify
  dino_thresh: 0.65        # Ngưỡng similarity tối thiểu của DINOv3
  lg_norm_thresh: 0.2      # Ngưỡng tỉ lệ inliers LightGlue
  lg_min_inliers: 30       # Số inliers tối thiểu để chấp nhận match

devices:
  yolo: "cuda"             # "cuda" hoặc "cpu"
  dino: "cuda"
  lg: "cuda"
```

> **Lưu ý DevOps:** Để chạy trên CPU-only server, đổi tất cả `"cuda"` → `"cpu"` và dùng `onnxruntime` thay vì `onnxruntime-gpu` trong `requirements.txt`.

---

## 📦 Danh sách Model Weights cần chuẩn bị

| File                    | Mục đích                               | Vị trí                                         |
| ----------------------- | -------------------------------------- | ---------------------------------------------- |
| `best-obb.pt`           | YOLO OBB — detect sản phẩm             | `data/weights/yolo/best-obb.pt`                |
| `best.pt`               | YOLO — detect nhãn giá                 | `data/weights/yolo/best.pt`                    |
| `superpoint_batch.onnx` | SuperPoint local features              | `data/weights/lightglue/superpoint_batch.onnx` |
| `lightglue_batch.onnx`  | LightGlue feature matching             | `data/weights/lightglue/lightglue_batch.onnx`  |
| `support_db.pt`         | Feature Bank (build từ support_images) | `data/support_db.pt`                           |

---

## 🌐 API Endpoints

**Base URL:** `http://localhost:8000`

### `GET /health`
Kiểm tra trạng thái server và model.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "device_info": {"yolo": "cuda", "dino": "cuda", "lg": "cuda"}
}
```

---

### `POST /predict`
Nhận diện sản phẩm và đọc giá trong một ảnh.

**Request:** `multipart/form-data`
| Field  | Type | Mô tả                       |
| ------ | ---- | --------------------------- |
| `file` | File | Ảnh chụp kệ hàng (JPEG/PNG) |

**Response:**
```json
{
  "matches": [
    {
      "class_name": "coca_cola_330ml",
      "score": 0.87,
      "box": [120, 45, 310, 280],
      "matched": true,
      "price_tag": {
        "tag_id": 0,
        "price": "15,000",
        "box": [120, 285, 310, 330]
      },
      "details": null
    }
  ],
  "price_tags": [
    {"tag_id": 0, "price": "15,000", "box": [120, 285, 310, 330]}
  ],
  "inference_time": 1.45,
  "image_size": [1920, 1080]
}
```

**Ý nghĩa các field:**
- `matches[].matched` — `true` nếu tìm được sản phẩm trong Feature Bank
- `matches[].score` — độ tin cậy tổng hợp (DINOv3 + LightGlue), 0.0–1.0
- `matches[].price_tag` — null nếu không phát hiện nhãn giá bên dưới sản phẩm
- `price_tags` — tất cả nhãn giá tìm thấy trong ảnh (kể cả chưa được assign cho sản phẩm nào)

---

## 🔧 Vận hành & Bảo trì

### Thêm SKU sản phẩm mới vào hệ thống

```bash
# 1. Đặt ảnh mẫu của SKU mới vào thư mục support_images/<tên_class>/
mkdir -p data/support_images/ten_san_pham_moi
cp /path/to/anh_mau*.jpg data/support_images/ten_san_pham_moi/

# 2. Rebuild Feature Bank
python3 scripts/build_gallery.py

# 3. Restart service (không cần rebuild Docker image)
docker restart rpm-server
```

### Điều chỉnh ngưỡng detect

Chỉnh sửa `configs/settings.yaml` và restart service. **Không cần rebuild image.**

```bash
# Sau khi sửa settings.yaml:
docker restart rpm-server
```

### Xem logs

```bash
docker logs -f rpm-server
```

---

## 🐛 Troubleshooting

| Lỗi                                   | Nguyên nhân                          | Cách fix                                               |
| ------------------------------------- | ------------------------------------ | ------------------------------------------------------ |
| `status: "error"` trên `/health`      | `support_db.pt` chưa tồn tại         | Chạy `python3 scripts/build_gallery.py`                |
| `CUDA out of memory`                  | VRAM không đủ                        | Đổi `dino: "cuda"` → `dino: "cpu"` trong settings.yaml |
| `price_tag: null` cho tất cả sản phẩm | Model `tag_yolo_path` không tìm thấy | Kiểm tra đường dẫn trong settings.yaml                 |
| `matched: false` cho sản phẩm         | SKU chưa có trong Feature Bank       | Thêm ảnh mẫu và rebuild gallery                        |
| Container exit ngay sau khi start     | Import error                         | Chạy `docker logs rpm-server` để xem chi tiết          |
