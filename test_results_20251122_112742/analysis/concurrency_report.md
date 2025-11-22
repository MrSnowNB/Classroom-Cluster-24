
# Concurrency Test Suite Analysis Report

**Generated:** 2025-11-22 11:35:55
**Endpoint:** http://localhost:11434
**Model:** gemma2:9b

## Executive Summary

This report presents the results of comprehensive concurrency testing for the Ollama Classroom Cluster.
The test suite evaluated performance across 1-6 concurrent users with mixed-length prompts.

## Key Metrics

- **Peak Throughput:** 121.6 tokens/second
- **Optimal Concurrency:** 8 users
- **Average Success Rate:** 96.7%
- **P95 Response Time (6 users):** 94.907s

## Performance Charts

(Charts generated as PNG files in the analysis directory)

- Throughput vs Concurrency: `throughput_vs_concurrency.png`
- Success Rate vs Concurrency: `success_rate_vs_concurrency.png`
- Response Time vs Concurrency: `ttft_vs_concurrency.png`
- Requests per Second vs Concurrency: `rps_vs_concurrency.png`

## Detailed Results

|   concurrency |   success_rate |   requests_per_second |   overall_throughput |   mean_ttft |   median_ttft |   p95_ttft |   total_requests |   total_successful |   total_failed |
|--------------:|---------------:|----------------------:|---------------------:|------------:|--------------:|-----------:|-----------------:|-------------------:|---------------:|
|             4 |            100 |              0.225615 |              91.9946 |     2.21071 |      2.23423  |    4.21518 |                8 |                  8 |              0 |
|             8 |            100 |              0.304874 |             121.623  |     3.75334 |      0.280764 |   17.8241  |               14 |                 14 |              0 |
|            12 |            100 |              0.227175 |              96.0015 |    15.2393  |     21.6626   |   28.6706  |               17 |                 17 |              0 |
|            16 |            100 |              0.249456 |             102.696  |    25.3393  |     26.4491   |   46.5807  |               22 |                 22 |              0 |
|            20 |            100 |              0.243224 |             103.857  |    34.4094  |     43.2627   |   66.5037  |               26 |                 26 |              0 |
|            24 |             80 |              0.215947 |              76.1142 |    44.4289  |     40.8141   |   94.9068  |               30 |                 24 |              6 |

## Recommendations

⚠ Consider reducing max concurrent users to improve reliability
⚠ Response times may be too high for interactive use
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
