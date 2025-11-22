#!/usr/bin/env python3
"""
Ollama Concurrency Test for Quad RTX 6000 Ada G8
Tests 24 concurrent student workloads through load-balanced Ollama instances

Usage:
    python concurrency_test.py --endpoint http://load-balancer:11434 --students 24 --duration 600
"""

import asyncio
import aiohttp
import time
import json
import argparse
from datetime import datetime
from statistics import mean, median, stdev
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test prompts with varying complexity
TEST_PROMPTS = [
    "Explain the concept of recursion in programming with a simple example.",
    "What are the main differences between Python lists and tuples?",
    "Write a function to calculate the factorial of a number.",
    "Describe the water cycle in detail.",
    "What is machine learning and how does it work?",
    "Explain quantum computing to a high school student.",
    "What are the key principles of object-oriented programming?",
    "How does photosynthesis work in plants?",
    "Describe the structure of the solar system.",
    "What is the difference between DNA and RNA?",
]

class StudentWorkload:
    """Simulates a single student's AI workload"""

    def __init__(self, student_id, endpoint, model="gemma2:9b"):
        self.student_id = student_id
        self.endpoint = endpoint
        self.model = model
        self.metrics = {
            'requests': 0,
            'successes': 0,
            'failures': 0,
            'ttft_times': [],
            'total_times': [],
            'tokens_generated': 0,
            'errors': []
        }

    async def send_request(self, session, prompt):
        """Send a single request and measure timing"""
        start_time = time.time()
        ttft = None
        tokens = 0

        try:
            async with session.post(
                f"{self.endpoint}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": True
                },
                timeout=aiohttp.ClientTimeout(total=120)
            ) as response:

                if response.status != 200:
                    raise Exception(f"HTTP {response.status}")

                # Read streaming response
                first_token = True
                async for line in response.content:
                    if line:
                        try:
                            data = json.loads(line)

                            if first_token:
                                ttft = time.time() - start_time
                                first_token = False

                            if 'response' in data:
                                tokens += len(data['response'].split())

                            if data.get('done', False):
                                break

                        except json.JSONDecodeError:
                            continue

                total_time = time.time() - start_time

                self.metrics['requests'] += 1
                self.metrics['successes'] += 1
                self.metrics['tokens_generated'] += tokens

                if ttft:
                    self.metrics['ttft_times'].append(ttft)
                self.metrics['total_times'].append(total_time)

                logger.debug(
                    f"Student {self.student_id}: Request completed in {total_time:.2f}s "
                    f"(TTFT: {ttft:.2f}s, {tokens} tokens)"
                )

                return True

        except Exception as e:
            self.metrics['requests'] += 1
            self.metrics['failures'] += 1
            self.metrics['errors'].append(str(e))
            logger.error(f"Student {self.student_id}: Request failed - {e}")
            return False

    async def run_workload(self, duration_seconds):
        """Run continuous workload for specified duration"""
        logger.info(f"Student {self.student_id}: Starting workload")

        async with aiohttp.ClientSession() as session:
            end_time = time.time() + duration_seconds

            while time.time() < end_time:
                # Select random prompt
                prompt = TEST_PROMPTS[self.metrics['requests'] % len(TEST_PROMPTS)]

                # Send request
                await self.send_request(session, prompt)

                # Random delay between requests (1-5 seconds)
                await asyncio.sleep(1 + (self.metrics['requests'] % 4))

        logger.info(f"Student {self.student_id}: Workload completed")

    def get_summary(self):
        """Get performance summary for this student"""
        return {
            'student_id': self.student_id,
            'total_requests': self.metrics['requests'],
            'successful': self.metrics['successes'],
            'failed': self.metrics['failures'],
            'success_rate': (
                self.metrics['successes'] / self.metrics['requests'] * 100 
                if self.metrics['requests'] > 0 else 0
            ),
            'avg_ttft': mean(self.metrics['ttft_times']) if self.metrics['ttft_times'] else 0,
            'median_ttft': median(self.metrics['ttft_times']) if self.metrics['ttft_times'] else 0,
            'p95_ttft': (
                sorted(self.metrics['ttft_times'])[int(len(self.metrics['ttft_times']) * 0.95)]
                if len(self.metrics['ttft_times']) > 1 else 0
            ),
            'avg_total_time': mean(self.metrics['total_times']) if self.metrics['total_times'] else 0,
            'total_tokens': self.metrics['tokens_generated'],
            'tokens_per_second': (
                self.metrics['tokens_generated'] / sum(self.metrics['total_times'])
                if sum(self.metrics['total_times']) > 0 else 0
            )
        }


async def run_test(endpoint, num_students, duration, model):
    """Run the complete concurrency test"""
    logger.info("="*80)
    logger.info(f"CONCURRENCY TEST STARTING")
    logger.info(f"Endpoint: {endpoint}")
    logger.info(f"Students: {num_students}")
    logger.info(f"Duration: {duration}s")
    logger.info(f"Model: {model}")
    logger.info("="*80)

    # Create student workloads
    students = [
        StudentWorkload(i+1, endpoint, model) 
        for i in range(num_students)
    ]

    # Run all workloads concurrently
    start_time = time.time()
    await asyncio.gather(*[student.run_workload(duration) for student in students])
    total_time = time.time() - start_time

    # Collect results
    logger.info("\n" + "="*80)
    logger.info("TEST RESULTS")
    logger.info("="*80)

    all_summaries = [student.get_summary() for student in students]

    # Overall statistics
    total_requests = sum(s['total_requests'] for s in all_summaries)
    total_successful = sum(s['successful'] for s in all_summaries)
    total_failed = sum(s['failed'] for s in all_summaries)
    total_tokens = sum(s['total_tokens'] for s in all_summaries)

    all_ttft = [t for s in students for t in s.metrics['ttft_times']]

    logger.info(f"\nOverall Performance:")
    logger.info(f"  Total Duration: {total_time:.2f}s")
    logger.info(f"  Total Requests: {total_requests}")
    logger.info(f"  Successful: {total_successful} ({total_successful/total_requests*100:.1f}%)")
    logger.info(f"  Failed: {total_failed} ({total_failed/total_requests*100:.1f}%)")
    logger.info(f"  Requests/Second: {total_requests/total_time:.2f}")
    logger.info(f"  Total Tokens Generated: {total_tokens}")
    logger.info(f"  Overall Throughput: {total_tokens/total_time:.2f} tokens/sec")

    if all_ttft:
        logger.info(f"\nTime to First Token (TTFT):")
        logger.info(f"  Mean: {mean(all_ttft):.3f}s")
        logger.info(f"  Median: {median(all_ttft):.3f}s")
        logger.info(f"  Std Dev: {stdev(all_ttft):.3f}s" if len(all_ttft) > 1 else "  Std Dev: N/A")
        logger.info(f"  P95: {sorted(all_ttft)[int(len(all_ttft)*0.95)]:.3f}s")
        logger.info(f"  P99: {sorted(all_ttft)[int(len(all_ttft)*0.99)]:.3f}s")
        logger.info(f"  Max: {max(all_ttft):.3f}s")

    # Success criteria check
    logger.info(f"\nSuccess Criteria:")
    success_rate = total_successful / total_requests * 100 if total_requests > 0 else 0
    p95_ttft = sorted(all_ttft)[int(len(all_ttft)*0.95)] if all_ttft else float('inf')
    avg_throughput = total_tokens / total_time / num_students

    logger.info(f"  ✓ All 24 students concurrent: {'PASS' if num_students == 24 else 'FAIL'}")
    logger.info(f"  {'✓' if success_rate > 95 else '✗'} Success rate > 95%: {success_rate:.1f}%")
    logger.info(f"  {'✓' if p95_ttft < 2.0 else '✗'} P95 TTFT < 2s: {p95_ttft:.3f}s")
    logger.info(f"  {'✓' if avg_throughput > 30 else '✗'} Throughput > 30 tok/s/student: {avg_throughput:.1f}")

    # Save detailed results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"concurrency_test_results_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump({
            'test_config': {
                'endpoint': endpoint,
                'num_students': num_students,
                'duration': duration,
                'model': model,
                'timestamp': timestamp
            },
            'overall': {
                'total_time': total_time,
                'total_requests': total_requests,
                'successful': total_successful,
                'failed': total_failed,
                'success_rate': success_rate,
                'requests_per_second': total_requests/total_time,
                'total_tokens': total_tokens,
                'overall_throughput': total_tokens/total_time,
                'mean_ttft': mean(all_ttft) if all_ttft else 0,
                'median_ttft': median(all_ttft) if all_ttft else 0,
                'p95_ttft': p95_ttft if all_ttft else 0,
                'p99_ttft': sorted(all_ttft)[int(len(all_ttft)*0.99)] if all_ttft else 0,
            },
            'per_student': all_summaries
        }, f, indent=2)

    logger.info(f"\nDetailed results saved to: {results_file}")
    logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(description='Ollama Concurrency Test')
    parser.add_argument('--endpoint', type=str, required=True, 
                       help='Ollama endpoint URL (e.g., http://load-balancer:11434)')
    parser.add_argument('--students', type=int, default=24, 
                       help='Number of concurrent students (default: 24)')
    parser.add_argument('--duration', type=int, default=600, 
                       help='Test duration in seconds (default: 600)')
    parser.add_argument('--model', type=str, default='gemma2:9b', 
                       help='Model to test (default: gemma2:9b)')

    args = parser.parse_args()

    # Run the test
    asyncio.run(run_test(args.endpoint, args.students, args.duration, args.model))


if __name__ == "__main__":
    main()
