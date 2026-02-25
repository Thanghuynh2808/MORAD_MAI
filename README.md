# 🏪Retail-Insight-Pipeline

Hệ thống AI nhận diện sản phẩm và đọc giá trên kệ hàng bán lẻ từ ảnh chụp bằng điện thoại. Gồm 2 microservice độc lập, giao tiếp với nhau qua HTTP.

---

## 🗺️ Tổng quan hệ thống

```
Mobile App
    │
    │  N ảnh chụp kệ hàng
    ▼
┌─────────────────────────────────────┐
│         Stitch Service :8001        │
│                                     │
│  ┌──────────────────────────────┐   │
│  │  Gọi RPM API (song song)     │───────────────┐
│  └──────────────────────────────┘   │           │
│                                     │           ▼
│  ┌──────────────────────────────┐   │  ┌────────────────────┐
│  │  Image Stitching → Panorama  │   │  │  RPM Service :8000 │
│  └──────────────────────────────┘   │  │                    │
│                                     │  │  YOLO detect       │
│  ┌──────────────────────────────┐   │  │  DINOv3 + LightGlue│
│  │  Warp Boxes to Panorama      │   │  │  YOLO detect tag   │
│  └──────────────────────────────┘   │  │  PaddleOCR / Easy  │
│                                     │  └────────────────────┘
│  ┌──────────────────────────────┐   │
│  │  Cluster-based Voting        │   │
│  │  → assign tag to product     │   │
│  └──────────────────────────────┘   │
└─────────────────────────────────────┘
    │
    ▼
JSON: panorama + products + prices
```

---

## 📦 Cấu trúc Repository

```
MAI_MORAD/
├── Retail-Product-Matching/    # RPM Service — detect & match sản phẩm, đọc giá
│   ├── README.md               ← Chi tiết triển khai RPM
│   └── ...
│
└── stitch_service/             # Stitch Service — ghép ảnh, warp, voting
    ├── README.md               ← Chi tiết triển khai Stitch
    └── ...
```

---

## ⚡ Khởi động nhanh (Local)

Mở **2 terminal riêng biệt**:

**Terminal 1 — RPM Service:**
```bash
cd Retail-Product-Matching
bash run_server.sh
# → chạy trên http://localhost:8000
```

**Terminal 2 — Stitch Service:**
```bash
cd stitch_service
python3 server/app.py
# → chạy trên http://localhost:8001
```

**Test:**
```bash
# Health check RPM
curl http://localhost:8000/health

# Upload 2 ảnh để stitch + nhận diện
curl -X POST http://localhost:8001/upload-batch \
  -F "files=@anh1.jpg" \
  -F "files=@anh2.jpg"
```

---

## 🐳 Triển khai Docker (Production)

Chạy cả 2 service với Docker Compose từ root:

```bash
# Terminal 1
cd Retail-Product-Matching
docker-compose up -d --build

# Terminal 2
cd stitch_service
docker-compose up -d --build
```

> ℹ️ Stitch Service cần biết địa chỉ RPM:
> ```bash
> # Nếu 2 container cùng 1 host
> export RPM_API_URL=http://localhost:8000/predict
> ```

---

## 📋 Ports & Services

| Service        | Port | Docs                       |
| -------------- | ---- | -------------------------- |
| RPM API        | 8000 | http://localhost:8000/docs |
| Stitch Service | 8001 | http://localhost:8001/docs |

---

## 📖 Chi tiết từng service

| Service        | README                                                                   |
| -------------- | ------------------------------------------------------------------------ |
| RPM Service    | [Retail-Product-Matching/README.md](./Retail-Product-Matching/README.md) |
| Stitch Service | [stitch_service/README.md](./stitch_service/README.md)                   |
