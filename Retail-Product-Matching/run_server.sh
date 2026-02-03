#!/bin/bash

# Configuration
HOST="0.0.0.0"
PORT=8000
PROJECT_ROOT=$(pwd)

echo "--------------------------------------------------"
echo "🚀 Starting Retail Product Matching API Server..."
echo "📍 Root: $PROJECT_ROOT"
echo "🌐 URL: http://$HOST:$PORT"
echo "📖 Docs: http://$HOST:$PORT/docs"
echo "--------------------------------------------------"

# Optional: Add environment variables for CUDA optimization
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT

# Check if server/app.py exists
if [ ! -f "server/app.py" ]; then
    echo "❌ Error: server/app.py not found. Please run this script from the project root."
    exit 1
fi

# Run the server using uvicorn
# --reload is good for development but remove it for production
# If you want to use multiple workers, note that each worker will load models (CPU/GPU sync needed)
python3 server/app.py
