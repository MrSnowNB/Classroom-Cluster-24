#!/usr/bin/env python3
"""
vLLM Concurrency Test for Classroom-Cluster-24
Tests 24 concurrent students with Gemma-3-4B model
Measures:
  - Time to First Token (TTFT)
  - Throughput (tokens/sec)
  - GPU utilization per card
  - Success rate
  - Queue depth
Usage:
  python test_concurrency_vllm.py --students 24 --duration 600
"""

import asyncio
import aiohttp
import time
import argparse
import json
import subprocess
from datetime import datetime
from statistics import mean, median, stdev
from typing import List, Dict

PROMPTS = [
    "Explain the concept of photosynthesis in simple terms.",
    "What is the Pythagorean theorem and how is it used?",
    "Write a Python function to calculate factorial recursively.",
    "Describe the water cycle and its importance.",
    "What are the three states of matter? Explain each.",
    "How does a computer process binary code?",
    "Explain what a metaphor is and give three examples.",
    "What causes the seasons on Earth?",
    "Describe the process of mitosis in cell division.",
    "What is the difference between renewable and non-renewable energy?",
]


class VLLMConcurrencyTest:
    def __init__(self, endpoint: str, model: str, num_students: int):
        self.endpoint = endpoint
        self.model = model
        self.num_students = num_students
        self.results = []

    async def send_request(self, session, student_id: int, prompt: str) -> Dict:
        """Send single request and measure timing"""
        start_time = time.time()
        ttft = None
        tokens = 0

        try:
            async with session.post(
                f"{self.endpoint}/v1/completions",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "max_tokens": 200,
                    "temperature": 0.7,
                    "stream": True
                },
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status != 200:
                    return {
                        "student_id": student_id,
                        "success": False,
                        "error": f"HTTP {response.status}"
                    }

                # Handle SSE format
                async for line in response.content:
                    if line:
                        try:
                            line_str = line.decode('utf-8').strip()
                            if line_str.startswith('data: '):
                                data = json.loads(line_str[6:])
                                if ttft is None:
                                    ttft = time.time() - start_time
                                if 'choices' in data:
                                    text = data['choices'][0].get('text', '')
                                    tokens += len(text.split())
                        except (json.JSONDecodeError, KeyError):
                            continue

                total_time = time.time() - start_time

                return {
                    "student_id": student_id,
                    "success": True,
                    "ttft": ttft,
                    "total_time": total_time,
                    "tokens": tokens,
                    "tokens_per_sec": tokens / total_time if total_time > 0 else 0
                }

        except Exception as e:
            return {
                "student_id": student_id,
                "success": False,
                "error": str(e)
            }

    async def student_workload(self, student_id: int, duration: int):
        """Simulate one student's continuous workload"""
        print(f"Student {student_id}: Starting workload")
        async with aiohttp.ClientSession() as session:
            end_time = time.time() + duration
            request_count = 0

            while time.time() < end_time:
                prompt = PROMPTS[request_count % len(PROMPTS)]
                result = await self.send_request(session, student_id, prompt)
                result['request_num'] = request_count
                result['timestamp'] = datetime.now().isoformat()
                self.results.append(result)
                request_count += 1

                # Think time (1-3 seconds)
                await asyncio.sleep(1 + (request_count % 3))

        print(f"Student {student_id}: Completed {request_count} requests")

    async def run_test(self, duration: int):
        """Run full concurrency test"""
        print("=" * 80)
        print(f"vLLM CONCURRENCY TEST")
        print(f"Endpoint: {self.endpoint}")
        print(f"Model: {self.model}")
        print(f"Students: {self.num_students}")
        print(f"Duration: {duration}s")
        print("=" * 80)
        print()

        start_time = time.time()

        # Launch all student workloads concurrently
        await asyncio.gather(*[
            self.student_workload(i + 1, duration)
            for i in range(self.num_students)
        ])

        total_time = time.time() - start_time

        # Analyze results
        self.print_results(total_time)
        self.save_results()

    def print_results(self, total_time: float):
        """Print comprehensive results"""
        successful = [r for r in self.results if r['success']]
        failed = [r for r in self.results if not r['success']]

        print()
        print("=" * 80)
        print("TEST RESULTS")
        print("=" * 80)

        print(f"\nTotal Requests: {len(self.results)}")
        print(f"Successful: {len(successful)} ({len(successful)/len(self.results)*100:.1f}%)")
        print(f"Failed: {len(failed)} ({len(failed)/len(self.results)*100:.1f}%)")

        if successful:
            ttft_values = [r['ttft'] for r in successful if r.get('ttft')]
            total_times = [r['total_time'] for r in successful]
            tokens_per_sec = [r['tokens_per_sec'] for r in successful]
            total_tokens = sum(r['tokens'] for r in successful)

            print("
Time to First Token (TTFT)"            print(f"Mean: {mean(ttft_values):.3f}s")
            print(f"Median: {median(ttft_values):.3f}s")
            if len(ttft_values) > 1:
                print(f"Std Dev: {stdev(ttft_values):.3f}s")
            print(f"P95: {sorted(ttft_values)[int(len(ttft_values)*0.95)]:.3f}s")
            print(f"P99: {sorted(ttft_values)[int(len(ttft_values)*0.99)]:.3f}s")
            print(f"Max: {max(ttft_values):.3f}s")

            print("
Throughput："            print(f"Total Tokens: {total_tokens}")
            print(f"Tokens/sec (overall): {total_tokens/total_time:.1f}")
            print(f"Tokens/sec (per student): {mean(tokens_per_sec):.1f}")
            print(f"Requests/sec: {len(self.results)/total_time:.2f}")

            print("
GPU Utilization"            self.print_gpu_stats()

            print("
Success Criteria"            p95_ttft = sorted(ttft_values)[int(len(ttft_values)*0.95)]
            avg_throughput = mean(tokens_per_sec)

            print(f"{'✓' if p95_ttft < 0.5 else '✗'} P95 TTFT < 500ms: {p95_ttft*1000:.0f}ms")
            print(f"{'✓' if avg_throughput > 30 else '✗'} Throughput > 30 tok/s/student: {avg_throughput:.1f}")
            print(f"{'✓' if len(failed) == 0 else '✗'} Zero failures: {len(failed)} failures")

    def print_gpu_stats(self):
        """Query and print GPU utilization"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used',
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                check=True
            )
            for line in result.stdout.strip().split('\n'):
                gpu_id, util, mem = line.split(',')
                print(f"GPU {gpu_id.strip()}: {util.strip()}% utilization, {mem.strip()}MB VRAM")
        except Exception as e:
            print(f"Could not query GPU stats: {e}")

    def save_results(self):
        """Save results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"vllm_test_results_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump({
                "config": {
                    "endpoint": self.endpoint,
                    "model": self.model,
                    "num_students": self.num_students,
                    "timestamp": timestamp
                },
                "results": self.results
            }, f, indent=2)

        print("
Results saved to:"        print(f"Results saved to: {filename}")


def main():
    parser = argparse.ArgumentParser(description='vLLM Concurrency Test')
    parser.add_argument('--endpoint', default='http://localhost:8000',
                       help='vLLM server endpoint')
    parser.add_argument('--model', default='google/gemma-3-4b-it',
                       help='Model name')
    parser.add_argument('--students', type=int, default=24,
                       help='Number of concurrent students')
    parser.add_argument('--duration', type=int, default=600,
                       help='Test duration in seconds')

    args = parser.parse_args()

    tett = VLLMConcurrencyTest(LLgM.Cndcoint, urgr.model, cyTs.studentet(args.endpoint, args.model, args.students)
    asyncio.run(test.run_test(args.duration))


if __name__ == "__main__":
    main()
