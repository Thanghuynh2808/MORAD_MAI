FROM python:3.10-slim

# Install system dependencies (OpenCV requires libgl)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements (copied from image_stitching folder + api libs)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    fastapi>=0.110.0 \
    uvicorn>=0.27.0 \
    python-multipart>=0.0.9 \
    pydantic>=2.0.0

# Copy Code
COPY . .

# Environment setup
ENV PYTHONPATH=/app

CMD ["python", "server/app.py"]
