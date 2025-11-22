#!/bin/bash
set -e

echo ""
echo "vLLM Setup for Classroom Cluster"
echo "================================="
echo ""

echo "Checking prerequisites..."

# Check for nvidia-smi
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Install NVIDIA drivers."
    exit 1
fi

CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
echo "✓ CUDA Version: $CUDA_VERSION"

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [ "$(python3 -c "import sys; print(sys.version_info >= (3, 10))")" != "True" ]; then
    echo "ERROR: Python 3.10+ required (found $PYTHON_VERSION)"
    exit 1
fi
echo "✓ Python Version: $PYTHON_VERSION"

# Check GPU count
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n1)
echo "✓ GPUs Detected: $GPU_COUNT"

if [ "$GPU_COUNT" -lt 4 ]; then
    echo "WARNING: Expected 4 GPUs, found $GPU_COUNT"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "Creating Python virtual environment..."
python3 -m venv vllm-env
source vllm-env/bin/activate

echo ""
echo "Installing vLLM with CUDA support..."
pip install --upgrade pip
pip install vllm

echo ""
echo "Verifying vLLM installation..."
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"

# Model download
if ping -c 1 huggingface.co &> /dev/null; then
    echo ""
    echo "Downloading Gemma-3-4B model..."
    # Try Gemma-3 first, fallback to Gemma-2
    python -c "
from huggingface_hub import snapshot_download
import os
try:
    # Try Gemma-3
    model_path = snapshot_download(
        repo_id='google/gemma-3-4b-it',
        cache_dir='./models',
        local_dir='./models/gemma-3-4b-it'
    )
    print(f'✓ Downloaded Gemma-3-4B to {model_path}')
except Exception as e:
    print(f'Gemma-3-4B not found, using Gemma-2-4B... ({e})')
    # Fallback to Gemma-2
    model_path = snapshot_download(
        repo_id='google/gemma-2-4b-it',
        cache_dir='./models',
        local_dir='./models/gemma-2-4b-it'
    )
    print(f'✓ Downloaded Gemma-2-4B to {model_path}')
"
else
    echo "OFFLINE MODE: Skipping model download"
    echo "Transfer model manually to ./models/ directory"
fi

echo ""
echo "Creating systemd service..."
sudo tee /etc/systemd/system/vllm-classroom.service > /dev/null <<EOF
[Unit]
Description=vLLM Server for Classroom Cluster (Gemma-3-4B)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$(id -un)
WorkingDirectory=$(pwd)
Environment="PATH=$(pwd)/vllm-env/bin"
ExecStart=$(pwd)/vllm-env/bin/python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-3-4b-it \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 128 \
  --host 0.0.0.0 \
  --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload

echo ""
echo "================================="
echo "✓ Setup Complete!"
echo ""
echo "Next steps:"
echo "1. Start vLLM server:"
echo "   sudo systemctl start vllm-classroom"
echo ""
echo "2. Enable auto-start on boot:"
echo "   sudo systemctl enable vllm-classroom"
echo ""
echo "3. Check status:"
echo "   sudo systemctl status vllm-classroom"
echo ""
echo "4. Test server:"
echo "   curl http://localhost:8000/health"
echo ""
echo "5. Run concurrency test:"
echo "   python vllm-deployment/tests/test_concurrency_vllm.py --students 24 --duration 600"
echo "================================="
