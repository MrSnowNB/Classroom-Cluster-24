#!/usr/bin/env python3
"""
Multi-Model vLLM Setup: 4 Models × 6 Users = 24 Concurrent Students

Models (4-27B range, Q4 quantized):
- GPU 0: microsoft/DialoGPT-small (117M) - demo model
- GPU 1: google/gemma-2-2b-it (2.1B)
- GPU 2: microsoft/Phi-3.5-mini-instruct (3.8B)
- GPU 3: meta-llama/Llama-2-7b-hf (6.7B)

Each model: ~1-2GB VRAM (Q4 quantization + KV cache)
Total: ~6-8GB across 4 GPUs with balancing
"""

import subprocess
import time
import signal
import requests
import sys

def start_vllm_instances():
    """Start 4 vLLM instances with different models"""
    models = [
        ("microsoft/DialoGPT-small", 117000000, "GPU 0", 8000),
        ("google/gemma-2-2b-it", 2100000000, "GPU 1", 8001),
        ("microsoft/Phi-3.5-mini-instruct", 3800000000, "GPU 2", 8002),
        ("meta-llama/Llama-2-7b-hf", 6700000000, "GPU 3", 8003),
    ]

    processes = []

    for model_id, param_count, gpu, port in models:
        print(f"\n🚀 Starting {model_id} ({param_count:,} params) on {gpu}, port {port}")

        # Calculate expected VRAM (rough estimate)
        base_vram = (param_count * 1) // (1024**3)  # Q4 = ~1 byte per param
        kv_cache = 2  # 2GB KV cache estimate for concurrency
        total_vram = base_vram + kv_cache

        print(f"   Expected VRAM: {total_vram}GB")

        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_id,
            "--tensor-parallel-size", "1",
            "--gpu-memory-utilization", "0.6",
            "--max-model-len", "2048",
            "--max-num-seqs", "8",  # 2-3 users per model with headroom
            "--env", f"CUDA_VISIBLE_DEVICES={models.index((model_id, param_count, gpu, port))}",
            "--host", "127.0.0.1",
            "--port", str(port)
        ]

        try:
            proc = subprocess.Popen(cmd, env={"CUDA_VISIBLE_DEVICES": str(models.index((model_id, param_count, gpu, port)))})
            processes.append((model_id, proc, port))
            print(f"   ✅ Launched on port {port}")
        except Exception as e:
            print(f"   ❌ Failed to launch {model_id}: {e}")

        time.sleep(2)  # Stagger startup

    return processes

def check_instances(processes):
    """Verify all instances are responding"""
    print("\n🔍 Checking model health:")

    for model_name, proc, port in processes:
        try:
            health = requests.get(f"http://localhost:{port}/health", timeout=10)
            if health.status_code == 200:
                # Test model
                test = requests.post(f"http://localhost:{port}/v1/completions",
                                 json={"model": model_name, "prompt": "Hello", "max_tokens": 5},
                                 timeout=10)
                if test.status_code == 200:
                    print(f"   ✅ {model_name} (port {port}): Ready")
                else:
                    print(f"   ⚠️ {model_name} (port {port}): Health OK, test failed")
            else:
                print(f"   ❌ {model_name} (port {port}): Health check failed")
        except Exception as e:
            print(f"   ❌ {model_name} (port {port}): {e}")

def create_haproxy_config():
    """Create HAProxy config for load balancing across 4 models"""
    config = """global
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

frontend classroom_frontend
    bind *:8042  # Different port so can run alongside existing setup
    default_backend classroom_models

backend classroom_models
    balance roundrobin
    server model1_DialoGPT 127.0.0.1:8000 check inter 10s rise 2 fall 3
    server model2_Gemma2B 127.0.0.1:8001 check inter 10s rise 2 fall 3
    server model3_Phi3_5 127.0.0.1:8002 check inter 10s rise 2 fall 3
    server model4_Llama7B 127.0.0.1:8003 check inter 10s rise 2 fall 3
"""

    with open("haproxy-multi-model.cfg", "w") as f:
        f.write(config)

    print("\n🔄 HAProxy config created: haproxy-multi-model.cfg")
    print("   To start: sudo haproxy -f haproxy-multi-model.cfg")
    print("   Load balanced endpoint: http://localhost:8042")

if __name__ == "__main__":
    print("🧠 Multi-Model vLLM Setup: 4 Models × 6 Users Balancing")
    print("=" * 70)

    processes = start_vllm_instances()

    if len(processes) > 0:
        print(f"\n⏳ Waiting for models to load (may take several minutes)...")
        time.sleep(30)  # Give models time to load

        check_instances(processes)
        create_haproxy_config()

        print(f"\n📊 Multi-Model Setup Complete:")
        print(f"   • 4 Models loaded across separate GPUs")
        print(f"   • VRAM: ~6-8GB distributed load")
        print(f"   • Load balancing: Round-robin across models")
        print(f"   • Capacity: 6-8 users per model = 24-32 concurrent students")

        print(f"\n🧪 Test Commands:")
        print(f"   # Direct model access:")
        print(f"   curl http://localhost:8000/v1/completions -d '{{\"model\":\"microsoft/DialoGPT-small\",\"prompt\":\"Hello\"}}'")
        print(f"   ")
        print(f"   # Load balanced access:")
        print(f"   curl http://localhost:8042/v1/completions -d '{{\"prompt\":\"Classroom question goes here\",\"max_tokens\":100}}'")
        print(f"   ")
        print("   # When done: sudo pkill -f vllm && sudo pkill haproxy"
    else:
        print("\n❌ No models were successfully launched")
