#!/bin/bash
# 4× Llama-2-7B-hf Stress Test: 6 Users × 4 Models = 24 Concurrent Students

echo "🧪 LAUNCHING: 4× Llama-2-7B-hf GPU Stress Test"
echo "============================================="
echo "• GPU 0: meta-llama/Llama-2-7b-hf (port 8000)"
echo "• GPU 1: meta-llama/Llama-2-7b-hf (port 8001)"
echo "• GPU 2: meta-llama/Llama-2-7b-hf (port 8002)"
echo "• GPU 3: meta-llama/Llama-2-7b-hf (port 8003)"
echo "• Load Balancer: HAProxy on port 8042"
echo "• Test: 24 concurrent users, 10-minute stress test"
echo ""

# Create HAProxy config for 4 identical Llama-2-7B instances
cat > haproxy-4llama.cfg << EOF
global
    log /dev/log local0
    log /dev/log local1 notice
    chroot /var/lib/haproxy
    stats socket /run/haproxy/admin.sock mode 660 level admin expose-fd listeners
    stats timeout 30s
    user haproxy
    group haproxy
    daemon

defaults
    log global
    option httplog
    option dontlognull
    timeout connect 5000
    timeout client  50000
    timeout server  50000

frontend llama_cluster_frontend
    bind *:8042
    default_backend llama_cluster_backend
    # Enable request logging
    mode http
    log global

backend llama_cluster_backend
    balance roundrobin
    mode http
    server llama_gpu0 127.0.0.1:8000 check inter 10s rise 2 fall 3
    server llama_gpu1 127.0.0.1:8001 check inter 10s rise 2 fall 3
    server llama_gpu2 127.0.0.1:8002 check inter 10s rise 2 fall 3
    server llama_gpu3 127.0.0.1:8003 check inter 10s rise 2 fall 3
EOF

echo "📝 HAProxy config created: haproxy-4llama.cfg"
echo ""

# Start HAProxy
echo "🔄 Starting HAProxy load balancer..."
sudo haproxy -f haproxy-4llama.cfg &
sleep 3

# Start 4 vLLM instances with identical Llama-2-7B models
echo ""
echo "🚀 Starting 4 vLLM instances with Llama-2-7B-hf..."

cd vllm-deployment/scripts/vllm-env/bin

# GPU 0
echo "Starting GPU 0 (Llama-2-7B #1)..."
CUDA_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 ./python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.4 \
    --max-model-len 2048 \
    --max-num-seqs 8 \
    --host 127.0.0.1 \
    --port 8000 &
sleep 5

# GPU 1
echo "Starting GPU 1 (Llama-2-7B #2)..."
CUDA_VISIBLE_DEVICES=1 ./python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.4 \
    --max-model-len 2048 \
    --max-num-seqs 8 \
    --host 127.0.0.1 \
    --port 8001 &
sleep 5

# GPU 2
echo "Starting GPU 2 (Llama-2-7B #3)..."
CUDA_VISIBLE_DEVICES=2 ./python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.4 \
    --max-model-len 2048 \
    --max-num-seqs 8 \
    --host 127.0.0.1 \
    --port 8002 &
sleep 5

# GPU 3
echo "Starting GPU 3 (Llama-2-7B #4)..."
CUDA_VISIBLE_DEVICES=3 ./python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.4 \
    --max-model-len 2048 \
    --max-num-seqs 8 \
    --host 127.0.0.1 \
    --port 8003 &
sleep 5

echo ""
echo "⏳ Waiting 4 minutes for all Llama-2-7B models to load..."
sleep 240

# Check all instances are ready
echo ""
echo "🔍 Verifying all 4 models are ready..."
for port in 8000 8001 8002 8003; do
    if curl -s http://127.0.0.1:$port/health > /dev/null; then
        echo "✅ Port $port (GPU $((port-8000))): Ready"
    else
        echo "❌ Port $port (GPU $((port-8000))): Not ready"
    fi
done

if curl -s http://127.0.0.1:8042/health > /dev/null; then
    echo "✅ HAProxy load balancer (port 8042): Ready"
else
    echo "❌ HAProxy load balancer (port 8042): Not ready"
fi

echo ""
echo "🧪 READY FOR TESTING!"
echo "====================="
echo "Load balanced endpoint: http://localhost:8042"
echo ""
echo "Individual model endpoints:"
echo "http://localhost:8000 (GPU 0)"
echo "http://localhost:8001 (GPU 1)"
echo "http://localhost:8002 (GPU 2)"
echo "http://localhost:8003 (GPU 3)"
echo ""
echo "Now run the 24-user stress test to validate real multi-GPU performance!"
echo ""
echo "When finished: sudo pkill haproxy && pkill -f vllm"
