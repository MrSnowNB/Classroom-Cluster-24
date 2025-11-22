#!/usr/bin/env python3
"""
Simple vLLM demonstration with tensor parallelism to prove multi-GPU utilization
"""
import concurrent.futures
import requests
import subprocess
import time

def concurrent_requests(num_concurrent=8, num_requests=10):
    """Send multiple concurrent requests to demonstrate load distribution"""
    print(f"🚀 Testing with {num_concurrent} concurrent users...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
        # Submit concurrent requests
        futures = []
        for i in range(num_concurrent):
            for j in range(num_requests):
                prompt = f"Student {i+1}, question {j+1}: Generate one sentence about AI."
                future = executor.submit(make_request, prompt)
                futures.append(future)

        # Check GPU every 2 seconds
        gpu_snapshots = []
        for i in range(30):  # Monitor for 60 seconds
            time.sleep(2)
            gpu_snapshots.append(get_gpu_snapshot())

        # Wait for all requests
        successful = 0
        failed = 0
        for future in futures:
            try:
                result = future.result(timeout=30)
                if "error" not in result:
                    successful += 1
                else:
                    failed += 1
            except:
                failed += 1

        print(f"\n📊 Results: {successful} successful, {failed} failed")

        # Show GPU utilization timeline
        print("\n🔥 GPU Utilization Timeline:")
        for i, snapshot in enumerate(gpu_snapshots[::5]):  # Every 10 seconds
            print(f"T{i*10}s: {snapshot[0]}% GPU0, {snapshot[1]}% GPU1, {snapshot[2]}% GPU2, {snapshot[3]}% GPU3")

def make_request(prompt):
    try:
        response = requests.post("http://localhost:8000/v1/completions",
            json={"model": "facebook/opt-125m", "prompt": prompt, "max_tokens": 50},
            timeout=10)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def get_gpu_snapshot():
    """Get GPU utilization snapshot"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                              capture_output=True, text=True, timeout=2)
        gpu_utils = [int(x.strip()) for x in result.stdout.strip().split('\n')[:4]]
        return gpu_utils
    except:
        return [0] * 4

if __name__ == "__main__":
    print("🧪 vLLM Tensor Parallelism Test")
    print("=" * 50)

    # Check server
    try:
        health = requests.get("http://localhost:8000/health", timeout=5)
        print("✅ vLLM server responding")
    except:
        print("❌ vLLM server not responding. Start with:")
        print("  cd vllm-deployment && source scripts/vllm-env/bin/activate")
        print("  vllm serve facebook/opt-125m --tensor-parallel-size 4 --port 8000")
        exit(1)

    # Run concurrent testing with 4 models x 6 users balancing
    concurrent_requests(24, 2)  # 24 concurrent students, 2 requests each across 4 models
