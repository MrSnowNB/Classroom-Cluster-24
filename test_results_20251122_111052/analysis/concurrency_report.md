
# Concurrency Test Suite Analysis Report

**Generated:** 2025-11-22 11:14:39
**Endpoint:** http://localhost:11434
**Model:** gemma2:9b

## Executive Summary

This report presents the results of comprehensive concurrency testing for the Ollama Classroom Cluster.
The test suite evaluated performance across 1-6 concurrent users with mixed-length prompts.

## Key Metrics

- **Peak Throughput:** 86.0 tokens/second
- **Optimal Concurrency:** 6 users
- **Average Success Rate:** 100.0%
- **P95 Response Time (6 users):** 0.318s

## Performance Charts

(Charts generated as PNG files in the analysis directory)

- Throughput vs Concurrency: `throughput_vs_concurrency.png`
- Success Rate vs Concurrency: `success_rate_vs_concurrency.png`
- Response Time vs Concurrency: `ttft_vs_concurrency.png`
- Requests per Second vs Concurrency: `rps_vs_concurrency.png`

## Detailed Results

|   concurrency |   success_rate |   requests_per_second |   overall_throughput |   mean_ttft |   median_ttft |   p95_ttft |   total_requests |   total_successful |   total_failed |
|--------------:|---------------:|----------------------:|---------------------:|------------:|--------------:|-----------:|-----------------:|-------------------:|---------------:|
|             2 |            100 |              0.172261 |              64.6264 |    0.191019 |      0.188619 |   0.23898  |                6 |                  6 |              0 |
|             3 |            100 |              0.16338  |              66.1224 |    0.257957 |      0.235891 |   0.339807 |                7 |                  7 |              0 |
|             4 |            100 |              0.20089  |              74.3797 |    0.203503 |      0.213052 |   0.26342  |                8 |                  8 |              0 |
|             5 |            100 |              0.187488 |              74.1704 |    0.250696 |      0.252987 |   0.304056 |               10 |                 10 |              0 |
|             6 |            100 |              0.208585 |              86.0317 |    0.268664 |      0.284609 |   0.317821 |               11 |                 11 |              0 |

## Recommendations

✓ System maintains high reliability (>95%) across tested concurrency levels
✓ Good response times maintained even at high concurrency
✓ Throughput continues scaling with concurrency - consider adding more users

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
