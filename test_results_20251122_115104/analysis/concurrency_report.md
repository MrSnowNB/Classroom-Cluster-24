
# Concurrency Test Suite Analysis Report

**Generated:** 2025-11-22 11:56:43
**Endpoint:** http://localhost:11434
**Model:** gemma2:9b

## Executive Summary

This report presents the results of comprehensive concurrency testing for the Ollama Classroom Cluster.
The test suite evaluated performance across 1-6 concurrent users with mixed-length prompts.

## Key Metrics

- **Peak Throughput:** 125.9 tokens/second
- **Optimal Concurrency:** 8 users
- **Average Success Rate:** 66.7%
- **P95 Response Time (6 users):** 0.000s

## Performance Charts

(Charts generated as PNG files in the analysis directory)

- Throughput vs Concurrency: `throughput_vs_concurrency.png`
- Success Rate vs Concurrency: `success_rate_vs_concurrency.png`
- Response Time vs Concurrency: `ttft_vs_concurrency.png`
- Requests per Second vs Concurrency: `rps_vs_concurrency.png`

## Detailed Results

|   concurrency |   success_rate |   requests_per_second |   overall_throughput |   mean_ttft |   median_ttft |   p95_ttft |   total_requests |   total_successful |   total_failed |
|--------------:|---------------:|----------------------:|---------------------:|------------:|--------------:|-----------:|-----------------:|-------------------:|---------------:|
|             4 |            100 |              0.264616 |             106.949  |    0.190241 |      0.200274 |    0.21098 |               12 |                 12 |              0 |
|             8 |            100 |              0.296755 |             125.888  |    4.08284  |      0.278882 |   16.3844  |               14 |                 14 |              0 |
|            12 |            100 |              0.201794 |              89.3544 |   16.273    |     21.7188   |   33.2425  |               15 |                 15 |              0 |
|            16 |            100 |              0.185099 |              81.5668 |   27.0563   |     29.5243   |   64.0645  |               18 |                 18 |              0 |
|            20 |              0 |              6.1297   |               0      |    0        |      0        |    0       |              200 |                  0 |            200 |
|            24 |              0 |              9.56539  |               0      |    0        |      0        |    0       |              288 |                  0 |            288 |

## Recommendations

⚠ Consider reducing max concurrent users to improve reliability
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
