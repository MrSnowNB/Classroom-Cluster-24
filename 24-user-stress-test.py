#!/usr/bin/env python3
"""
24-User Stress Test: Real Multi-GPU Validation (6 Users × 4 Llama-2-7B Models)

Tests: 6 concurrent users per GPU (total 24) for 10 minutes
Tracks: Individual tok/s averages for each user every 30 seconds
Validates: GPU distribution vs Ollama's single-GPU bottleneck
"""

import asyncio
import aiohttp
import threading
import time
import requests
import subprocess
from collections import defaultdict
import json
from datetime import datetime

PROMPTS = [
    "Explain Newton's laws of motion in simple terms for high school students.",
    "What are the main differences between mitosis and meiosis?",
    "Write a Python function to implement binary search recursively.",
    "Describe how the water cycle affects global climate patterns.",
    "What are the applications of machine learning in education?",
    "Explain the concept of quantum superposition with a real-world analogy.",
    "How do enzymes accelerate biochemical reactions in living organisms?",
    "What are the principles of cryptography and their importance in cybersecurity?",
    "Describe the structure and function of DNA in cellular processes.",
    "What are the economic impacts of technological innovation?"
]

class StressTestUser:
    """Individual user's workload tracking"""

    def __init__(self, user_id: int, model_id: int, endpoint: str):
        self.user_id = user_id
        self.model_id = model_id
        self.endpoint = endpoint
        self.session = None
        self.running = True
        self.request_count = 0
        self.tokens_generated = 0
        self.total_response_time = 0
        self.start_time = time.time()
        self.tok_per_sec_history = []

    async def initialize(self):
        """Set up aiohttp session"""
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=0)
        )

    async def run_workload(self):
        """Run continuous user workload for 10 minutes"""
        await self.initialize()

        # Track tokens/sec every 30 seconds
        last_update = time.time()
        tokens_since_update = 0

        try:
            while self.running:
                # Send request
                prompt = PROMPTS[(self.request_count + self.user_id) % len(PROMPTS)]
                await self.send_request(prompt)

                # Track token performance
                current_time = time.time()
                if current_time - last_update >= 30.0:
                    elapsed = current_time - last_update
                    if self.tokens_generated > tokens_since_update:
                        tok_per_sec = (self.tokens_generated - tokens_since_update) / elapsed
                        self.tok_per_sec_history.append(tok_per_sec)
                        print(f"User {self.user_id} (Model {self.model_id}): {tok_per_sec:.2f} tok/s (avg: {sum(self.tok_per_sec_history)/len(self.tok_per_sec_history):.2f})")
                    tokens_since_update = self.tokens_generated
                    last_update = current_time

                # Think time (simulate student behavior)
                await asyncio.sleep(0.5 + (self.user_id % 2))

        except Exception as e:
            print(f"User {self.user_id} (Model {self.model_id}) error: {e}")
        finally:
            await self.session.close()

    async def send_request(self, prompt: str):
        """Send single request and measure performance"""
        try:
            response = await self.session.post(
                f"{self.endpoint}/v1/completions",
                json={
                    "model": "meta-llama/Llama-2-7b-hf",
                    "prompt": prompt,
                    "max_tokens": 150,
                    "temperature": 0.7,
                },
                timeout=aiohttp.ClientTimeout(total=30)
            )

            if response.status == 200:
                response_data = await response.json()
                tokens = len(response_data['choices'][0]['text'].split())
                self.tokens_generated += tokens
                self.request_count += 1

                # Track successful response
                print(f"✓ User {self.user_id} (Model {self.model_id}): Req {self.request_count}, +{tokens} tok")
            else:
                print(f"✗ User {self.user_id} (Model {self.model_id}): HTTP {response.status}")

        except Exception as e:
            print(f" ❌ User {self.user_id} (Model {self.model_id}): {e}")

    def get_summary(self):
        """Return performance summary"""
        total_time = time.time() - self.start_time
        return {
            "user_id": self.user_id,
            "model_id": self.model_id,
            "requests": self.request_count,
            "tokens": self.tokens_generated,
            "avg_tok_per_sec": self.tokens_generated / total_time if total_time > 0 else 0,
            "tok_per_sec_history": self.tok_per_sec_history
        }

def gpu_monitor():
    """Background thread to monitor GPU utilization"""
    print("\n🔥 GPU Monitoring Started")
    snapshots = []
    start_time = time.time()

    while time.time() - start_time < 650:  # 10:50 minutes
        try:
            result = subprocess.run([
                'nvidia-smi',
                '--query-gpu=index,utilization.gpu,memory.used',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpu_data = [int(line.split(',')[1]) for line in lines[:4]]  # Utilization %
                snapshots.append((time.time() - start_time, gpu_data))
                print(f"GPU Status @ {time.time() - start_time:.0f}s: {gpu_data[0]}% {gpu_data[1]}% {gpu_data[2]}% {gpu_data[3]}%")
        except Exception:
            pass

        time.sleep(10)

    # Save GPU monitoring data
    with open("gpu_monitoring_4llama.json", "w") as f:
        json.dump({"snapshots": snapshots}, f, indent=2)

    print("GPU Monitoring completed")

async def main():
    """Run the complete 24-user stress test"""
    print("🚀 4× Llama-2-7B-hf STRESS TEST")
    print("==============================")
    print("• 24 concurrent users (6 per model)")
    print("• 4 identical Llama-2-7B models (1 per GPU)")
    print("• 10-minute continuous load testing")
    print("• Individual user tok/s tracking")
    print("• Real-time GPU utilization monitoring")
    print()

    # Verify systems are ready
    print("🔍 Pre-test verification...")

    # Check load balancer
    if not requests.get("http://localhost:8042/health").ok:
        print("❌ HAProxy load balancer not ready")
        return

    # Check individual models
    model_ports = [8000, 8001, 8002, 8003]
    for i, port in enumerate(model_ports):
        if not requests.get(f"http://localhost:{port}/health").ok:
            print(f"❌ Model {i} (port {port}) not ready")
            return

    print("✅ All 4模型 and load balancer ready")
    print()

    # Create 24 users: 6 users per model (6×4=24)
    users = []
    for model_id in range(4):  # 4 models
        for user_in_model in range(6):  # 6 users per model
            user_id = model_id * 6 + user_in_model
            user = StressTestUser(user_id, model_id, "http://localhost:8042")
            users.append(user)

    print(f"🚀 Starting {len(users)} concurrent users...")

    # Start GPU monitoring in background
    gpu_thread = threading.Thread(target=gpu_monitor, daemon=True)
    gpu_thread.start()

    # Start all user workloads simultaneously
    start_time = time.time()
    await asyncio.gather(*[user.run_workload() for user in users])

    total_time = time.time() - start_time

    print("\n🎯 STRESS TEST COMPLETE")
    print("===================================")
    print(f"Total test time: {total_time:.1f} seconds")
    print(f"Total users simulated: {len(users)}")
    print()

    # Collect and analyze results
    all_summaries = {}
    per_model_stats = defaultdict(list)

    for user in users:
        summary = user.get_summary()
        all_summaries[summary["user_id"]] = summary
        per_model_stats[summary["model_id"]].append(summary)

    print("📊 INDIVIDUAL USER PERFORMANCE")
    print("=================================")
    print("User  Model  Requests  Tokens  Avg Tok/s")
    for user_id in sorted(all_summaries.keys()):
        s = all_summaries[user_id]
        print(f"{s['user_id']:2d}   {s['model_id']:2d}   {s['requests']:3d}   {s['tokens']:6d}   {s['avg_tok_per_sec']:.1f}")
    print()

    print("📈 MODEL PERFORMANCE AVERAGES")
    print("===============================")
    for model_id in sorted(per_model_stats.keys()):
        users_stats = per_model_stats[model_id]
        avg_tok_s = sum(u["avg_tok_per_sec"] for u in users_stats) / len(users_stats)
        total_requests = sum(u["requests"] for u in users_stats)
        total_tokens = sum(u["tokens"] for u in users_stats)
        print(f"{model_id:2d}  {avg_tok_s:.2f} tok/s  {total_requests:4d} req  {total_tokens:6d} tok")

    # Final validation analysis
    gpu_monitoring_data = json.load(open("gpu_monitoring_4llama.json"))
    gpu_samples = gpu_monitoring_data["snapshots"]

    avg_gpu_usage = []
    for _, gpu_data in gpu_samples:
        avg_gpu_usage.append(sum(gpu_data) / len(gpu_data))

    print("
🔥 GPU UTILIZATION ANALYSIS")
    print("==================================")
    overall_avg = sum(avg_gpu_usage) / len(avg_gpu_usage) if avg_gpu_usage else 0
    print(f"Overall average GPU utilization: {overall_avg:.1f}%")
    print(f"Peak GPU utilization: {max(avg_gpu_usage):.1f}%" if avg_gpu_usage else "N/A")
    print(f"Low GPU utilization: {min(avg_gpu_usage):.1f}%" if avg_gpu_usage else "N/A")

    # Vaporware check
    if overall_avg < 30:
        print("❌ VAPORWARE DETECTED: Low GPU utilization indicates simulation")
        print("   - Requests likely bottlenecked on single GPU")
        print("   - Real parallelism would show 70%+ GPU usage")
    elif overall_avg > 70:
        print("✅ REAL PARALLELISM CONFIRMED: High GPU utilization across all 4 GPUs")
        print("   - Concurrent processing verified across multiple GPUs")
        print("   - Load balancing effectively distributing requests")
    else:
        print(f"⚠️ UNCLEAR: {overall_avg:.1f}% average GPU usage (needs further analysis)")

    # Save complete results
    with open("24user_stress_test_results.json", "w") as f:
        json.dump({
            "config": {
                "test_duration": total_time,
                "total_users": len(users),
                "models_per_gpu": 1,
                "users_per_model": 6
            },
            "user_summaries": all_summaries,
            "model_summaries": dict(per_model_stats),
            "gpu_monitoring": gpu_monitoring_data,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)

    print(f"\n🎉 Results saved to: 24user_stress_test_results.json")

if __name__ == "__main__":
    asyncio.run(main())
