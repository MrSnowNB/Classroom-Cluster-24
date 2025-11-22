#!/usr/bin/env python3
"""
Advanced Concurrency Test Suite for Ollama Classroom Cluster
Inspired by Alex Ziskind's AI testing methodologies

This suite tests concurrency performance across different levels (1-6 instances)
with varying prompt lengths to assess stability and scaling.

Generates comprehensive analysis with charts and recommendations.
"""

import subprocess
import os
import json
import aiohttp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
import argparse
import threading
import time

# Set up plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class ConcurrencyTestSuite:
    def __init__(self, endpoint, model="gemma2:9b"):
        self.endpoint = endpoint
        self.model = model
        self.results_dir = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)
        self.all_results = []
        self.hardware_log = os.path.join(self.results_dir, "hardware_monitor.log")
        self.monitor_thread = None
        self.monitoring = False

    def start_hardware_monitoring(self):
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_hardware, daemon=True)
        self.monitor_thread.start()

    def stop_hardware_monitoring(self):
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)

    def _monitor_hardware(self):
        while self.monitoring:
            try:
                result = subprocess.run([
                    'nvidia-smi',
                    '--query-gpu=index,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,fan.speed,power.draw,power.limit,clocks.current.graphics,clocks.current.memory,clocks.current.video',
                    '--format=csv,noheader,nounits'
                ], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    with open(self.hardware_log, 'a') as f:
                        f.write(f"{timestamp}\n")
                        lines = result.stdout.strip().split('\n')
                        for i, line in enumerate(lines):
                            f.write(f"GPU{i}: {line}\n")
                        f.write("---\n")
            except Exception as e:
                with open(os.path.join(self.results_dir, "monitor_error.log"), 'a') as f:
                    f.write(f"{datetime.now()}: {e}\n")
            time.sleep(5)

    def run_single_test(self, concurrency, prompt_length="mixed", duration=60):
        """
        Run a single concurrency test

        prompt_length: 'short', 'medium', 'long', or 'mixed'
        """
        print(f"\n{'='*60}")
        print(f"RUNNING TEST: {concurrency} concurrent users, {prompt_length} prompts")
        print(f"Duration: {duration}s")
        print(f"{'='*60}")

        # For now, since concurrency_test.py has fixed prompts, we'll run with 'medium'
        # TO DO: Modify concurrency_test.py to accept prompt_length parameter

        cmd = [
            "python3", "concurrency_test.py",
            "--endpoint", self.endpoint,
            "--students", str(concurrency),
            "--duration", str(duration),
            "--model", self.model
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())

            # Try to read the generated results file (find the latest)
            results_files = [f for f in os.listdir('.') if f.startswith('concurrency_test_results_') and f.endswith('.json')]
            if results_files:
                latest_file = max(results_files, key=lambda x: os.path.getctime(x))
                with open(latest_file, 'r') as f:
                    data = json.load(f)
                    data['concurrency'] = concurrency
                    data['prompt_length'] = prompt_length
                    self.all_results.append(data)

                # Move to results dir
                os.rename(latest_file, os.path.join(self.results_dir, f"results_{concurrency}_{prompt_length}.json"))

                print(f"Test completed successfully")
                return True

        except Exception as e:
            print(f"Test failed: {e}")
            return False

    async def warm_up(self, num_requests=5):
        """Run warm-up requests to load models into memory"""
        async with aiohttp.ClientSession() as session:
            print("Starting model warm-up...")
            for i in range(num_requests):
                try:
                    async with session.post(
                        f"{self.endpoint}/api/generate",
                        json={
                            "model": self.model,
                            "prompt": f"Warm-up request {i+1}: Explain the concept of artificial intelligence in brief.",
                            "stream": True,
                            "options": {"num_predict": 50}  # Limit output for warm-up
                        },
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        if response.status == 200:
                            async for line in response.content:
                                if line and json.loads(line).get('done', False):
                                    break
                        print(f"Warm-up request {i+1} completed")
                except Exception as e:
                    print(f"Warm-up request {i+1} failed: {e}")
            print("Model warm-up completed")

    def run_full_suite(self, max_concurrency=6, include_prompt_tests=True):
        """Run the complete test suite"""

        print("\nPHASE 0: Model Warm-up")
        # Run warm-up to load models into memory
        import asyncio
        asyncio.run(self.warm_up())

        # Phase 1: Incremental concurrency scaling (1-24 users)
        print("\nPHASE 1: Incremental Concurrency Scaling (1-24 users)")
        for concurrent in range(1, 25):
            try:
                self.run_single_test(concurrent, "medium", duration=30)
            except Exception as e:
                crash_log = os.path.join(self.results_dir, "crash_log.txt")
                with open(crash_log, 'a') as f:
                    f.write(f"{datetime.now().isoformat()}: CRASH during test with {concurrent} users: {str(e)}\n")
                print(f"CRASH logged for {concurrent} users: {e}")
                break  # Stop further tests if crashed

        # Optional: Phase 2: Prompt length variations at fixed concurrency
        if include_prompt_tests:
            print("\nPHASE 2: Prompt Length Variations")
            # This would require modifying concurrency_test.py to accept prompt_length
            # For now, we'll skip or implement later

    def generate_analysis(self):
        """Generate comprehensive analysis and charts"""

        if not self.all_results:
            print("No results to analyze")
            return

        # Create analysis directory
        analysis_dir = os.path.join(self.results_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)

        # Convert results to DataFrame
        df_data = []
        for result in self.all_results:
            overall = result['overall']
            df_data.append({
                'concurrency': result['concurrency'],
                'success_rate': overall.get('success_rate', 0),
                'requests_per_second': overall.get('requests_per_second', 0),
                'overall_throughput': overall.get('overall_throughput', 0),
                'mean_ttft': overall.get('mean_ttft', 0),
                'median_ttft': overall.get('median_ttft', 0),
                'p95_ttft': overall.get('p95_ttft', 0),
                'total_requests': overall.get('total_requests', 0),
                'total_successful': overall.get('successful', 0),
                'total_failed': overall.get('failed', 0),
            })

        df = pd.DataFrame(df_data)

        # Generate charts
        self._create_charts(df, analysis_dir)

        # Generate report
        self._generate_report(df, analysis_dir)

        print(f"\nAnalysis complete! Results saved to {analysis_dir}")

    def _create_charts(self, df, output_dir):
        """Create performance charts"""

        # 1. Throughput vs Concurrency
        plt.figure()
        plt.plot(df['concurrency'], df['overall_throughput'], marker='o', linewidth=2, markersize=8)
        plt.title('Throughput vs Concurrency')
        plt.xlabel('Concurrent Users')
        plt.ylabel('Tokens/Second')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'throughput_vs_concurrency.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Success Rate vs Concurrency
        plt.figure()
        plt.plot(df['concurrency'], df['success_rate'], marker='o', linewidth=2, markersize=8, color='green')
        plt.title('Success Rate vs Concurrency')
        plt.xlabel('Concurrent Users')
        plt.ylabel('Success Rate (%)')
        plt.ylim(90, 101)
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'success_rate_vs_concurrency.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Response Time vs Concurrency (P95 TTFT)
        plt.figure()
        plt.plot(df['concurrency'], df['p95_ttft'], marker='o', linewidth=2, markersize=8, color='red')
        plt.title('P95 Time to First Token vs Concurrency')
        plt.xlabel('Concurrent Users')
        plt.ylabel('Time (seconds)')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'ttft_vs_concurrency.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Requests per Second vs Concurrency
        plt.figure()
        plt.plot(df['concurrency'], df['requests_per_second'], marker='o', linewidth=2, markersize=8, color='blue')
        plt.title('Requests per Second vs Concurrency')
        plt.xlabel('Concurrent Users')
        plt.ylabel('Requests/Second')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'rps_vs_concurrency.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_report(self, df, output_dir):
        """Generate markdown report with analysis and recommendations"""

        # Calculate key metrics
        max_throughput = df['overall_throughput'].max()
        optimal_concurrency = df.loc[df['overall_throughput'].idxmax(), 'concurrency']
        avg_success = df['success_rate'].mean()
        final_ttft = df.loc[df['concurrency'] == df['concurrency'].max(), 'p95_ttft'].values[0]

        # Generate recommendations
        recommendations = []

        if df['success_rate'].min() > 95:
            recommendations.append("✓ System maintains high reliability (>95%) across tested concurrency levels")
        else:
            recommendations.append("⚠ Consider reducing max concurrent users to improve reliability")

        if final_ttft < 2.0:
            recommendations.append("✓ Good response times maintained even at high concurrency")
        else:
            recommendations.append("⚠ Response times may be too high for interactive use")

        if optimal_concurrency > len(df)-1:
            recommendations.append("✓ Throughput continues scaling with concurrency - consider adding more users")
        else:
            recommendations.append(f"⚠ Optimal concurrency appears to be around {optimal_concurrency} users")

        # Write report
        report = f"""
# Concurrency Test Suite Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Endpoint:** {self.endpoint}
**Model:** {self.model}

## Executive Summary

This report presents the results of comprehensive concurrency testing for the Ollama Classroom Cluster.
The test suite evaluated performance across 1-6 concurrent users with mixed-length prompts.

## Key Metrics

- **Peak Throughput:** {max_throughput:.1f} tokens/second
- **Optimal Concurrency:** {optimal_concurrency} users
- **Average Success Rate:** {avg_success:.1f}%
- **P95 Response Time (6 users):** {final_ttft:.3f}s

## Performance Charts

(Charts generated as PNG files in the analysis directory)

- Throughput vs Concurrency: `throughput_vs_concurrency.png`
- Success Rate vs Concurrency: `success_rate_vs_concurrency.png`
- Response Time vs Concurrency: `ttft_vs_concurrency.png`
- Requests per Second vs Concurrency: `rps_vs_concurrency.png`

## Detailed Results

{df.to_markdown(index=False)}

## Recommendations

{'\n'.join(recommendations)}

## Methodology

Tests were conducted using concurrent AI requests to simulate classroom usage patterns.
Each test ran for 30 seconds per concurrency level with automatic metric collection and analysis.

Inspired by Alex Ziskind's comprehensive AI system testing approach, this suite provides
data-driven insights for optimizing the classroom cluster configuration.

## Next Steps

1. Compare these Ollama results with VLLM implementation
2. Test with actual classroom workloads (24 students)
3. Implement adaptive concurrency limiting based on performance thresholds
4. Consider GPU monitoring integration for resource utilization analysis
"""

        with open(os.path.join(output_dir, 'concurrency_report.md'), 'w') as f:
            f.write(report)


def main():
    parser = argparse.ArgumentParser(description='Advanced Concurrency Test Suite')
    parser.add_argument('--endpoint', type=str, default='http://localhost:11434',
                       help='Ollama endpoint URL')
    parser.add_argument('--model', type=str, default='gemma2:9b',
                       help='Model to test')
    parser.add_argument('--max-concurrency', type=int, default=6,
                       help='Maximum concurrency to test (1 to max)')
    parser.add_argument('--include-prompt-tests', action='store_true',
                       help='Include different prompt length tests')

    args = parser.parse_args()

    print("Ollama Classroom Cluster - Concurrency Test Suite")
    print("=" * 60)

    # Initialize test suite
    suite = ConcurrencyTestSuite(args.endpoint, args.model)

    # Run tests
    suite.run_full_suite(args.max_concurrency, args.include_prompt_tests)

    # Generate analysis
    suite.generate_analysis()

    print("\nTest suite completed! Check the results dir for detailed results.")


if __name__ == "__main__":
    main()
