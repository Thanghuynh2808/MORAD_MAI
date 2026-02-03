# Retail Product Matching (RPM) - Refactored Version

This project has been refactored from the original `Retail-Product-Matching` repository to improve maintainability, modularity, and to integrate model conversion tools (ONNX).

## 🚀 Refactoring Task Summary
- **Logic Separation**: Transitioned processing functions from a monolithic approach to a Class-based model with `ProductMatcher`.
- **Modularization**: Decoupled source code into specialized modules: `models` (loading/extraction/matching), `utils` (processing/visualization/common).
- **LightGlue-ONNX Integration**: Merged source code from the `LightGlue-ONNX` repository into the `tools/` directory to support local feature model export and optimization.
- **Standardization**: Updated function naming, variables, and directory structure to follow modern Python package standards.
- **Entry Point Optimization**: Clearly separated the main execution application (`app/main.py`) from auxiliary scripts (`scripts/`).

## 📁 Project Structure

```text
RPM_modified/
├── app/                        # Main application entry points
│   └── main.py                 # Batch image processing script
├── configs/                    # Configuration management (YAML/JSON)
├── data/                       # Models weights and data
│   ├── support_images/         # Template images for gallery building
│   ├── test_images/            # Input images for testing
│   ├── result_images/          # Output results after matching
│   ├── weights/                # YOLO weights, SuperPoint/LightGlue (ONNX)
│   └── support_db.pt           # Built Feature Bank (Feature database)
├── retail_matcher/             # CORE PACKAGE
│   ├── models/                 # Deep Learning model wrappers
│   │   ├── loader.py           # Model loading logic (YOLO, DINO, ONNX)
│   │   ├── extraction.py       # Feature extraction (Global & Local)
│   │   └── matching.py         # Matching logic (Matrix & Hybrid)
│   ├── utils/                  # Helper utilities
│   │   ├── common.py           # Logging, image loading
│   │   ├── processing.py       # Preprocessing, CLAHE, normalization
│   │   └── visualization.py    # Bounding box and label drawing
│   └── pipeline.py             # ProductMatcher class (Pipeline orchestrator)
├── scripts/                    # Auxiliary scripts
│   ├── build_gallery.py        # Build feature bank from support_images
│   └── test_api_client.py      # Client to test the FastAPI endpoint
├── server/                     # API SERVER (FastAPI)
│   ├── app.py                  # Main API entry point
│   └── schemas.py              # Pydantic data models
├── tools/                      # Development and extension tools
│   └── lightglue_export/       # Tools to convert/quantize LightGlue to ONNX
├── requirements.txt
├── Dockerfile                  # Containerization for deployment
├── docker-compose.yml          # GPU-enabled orchestration
├── run_server.sh               # Quick-start script for local API
└── README.md
```

## 🛠️ Installation & Usage

### 1. Environment Setup
Ensure you have installed the required libraries (GPU recommended):
```bash
pip install -r requirements.txt
```

### 2. Prepare Weights & Data
The project expects weights to be located in `data/weights/`:
- YOLO: `data/weights/yolo/best-obb.pt`
- ONNX: `data/weights/lightglue/superpoint_batch.onnx` & `lightglue_batch.onnx`

### 3. Build Feature Bank (Gallery)
Before running the matching process, you must extract features for the template products:
```bash
python3 scripts/build_gallery.py
```

### 4. Run Matching (Batch Mode)
Process images in the `test_images` folder and save results to disk:
```bash
python3 app/main.py
```

## 🌐 API Deployment (FastAPI)

The project includes a robust API server to serve predictions in real-time.

### Local Run
Use the convenience script to start the server:
```bash
chmod +x run_server.sh
./run_server.sh
```
The API will be available at `http://localhost:8000`. You can access the interactive documentation at `http://localhost:8000/docs`.

### Test the API
You can test the prediction endpoint using the provided client:
```bash
python3 scripts/test_api_client.py data/test_images/1.jpg
```

### Docker Deployment (Production)
To deploy with GPU support using Docker Compose:
```bash
docker-compose up -d --build
```
*Note: Requires `nvidia-container-toolkit` installed on the host.*

## 📈 Technical Improvements
- **ProductMatcher Class**: Holds the state of loaded models, making the pipeline flexible for use in different environments (e.g., APIs, Notebooks).
- **Visualization Decoupling**: Drawing logic is separated from matching logic, allowing easy UI/style changes without affecting the algorithm.
- **Path Management**: Uses `pathlib` throughout for cross-platform compatibility.
- **Hybrid Matching**: Optimized combination of DINOv3 (global) and LightGlue (local) for robust product identification.
