
# Concurrency Test Suite Analysis Report

**Generated:** 2025-11-22 13:01:43
**Endpoint:** http://localhost:11434
**Model:** gemma2:9b

## Executive Summary

This report presents the results of comprehensive concurrency testing for the Ollama Classroom Cluster.
The test suite evaluated performance across 1-6 concurrent users with mixed-length prompts.

## Key Metrics

- **Peak Throughput:** 147.0 tokens/second
- **Optimal Concurrency:** 14 users
- **Average Success Rate:** 95.2%
- **P95 Response Time (6 users):** 86.685s

## Performance Charts

(Charts generated as PNG files in the analysis directory)

- Throughput vs Concurrency: `throughput_vs_concurrency.png`
- Success Rate vs Concurrency: `success_rate_vs_concurrency.png`
- Response Time vs Concurrency: `ttft_vs_concurrency.png`
- Requests per Second vs Concurrency: `rps_vs_concurrency.png`

## Detailed Results

|   concurrency |   success_rate |   requests_per_second |   overall_throughput |   mean_ttft |   median_ttft |   p95_ttft |   total_requests |   total_successful |   total_failed |
|--------------:|---------------:|----------------------:|---------------------:|------------:|--------------:|-----------:|-----------------:|-------------------:|---------------:|
|             1 |       100      |             0.0324769 |              13.2668 |     15.9574 |     15.9574   |    31.4984 |                2 |                  2 |              0 |
|             2 |       100      |             0.0466474 |              19.0904 |     20.9328 |     19.8118   |    43.6278 |                4 |                  4 |              0 |
|             3 |       100      |             0.0926013 |              41.8867 |     10.211  |     10.0179   |    11.8284 |                3 |                  3 |              0 |
|             4 |       100      |             0.0502338 |              23.434  |     51.5209 |     52.7049   |    56.9451 |                4 |                  4 |              0 |
|             5 |       100      |             0.0559546 |              23.5489 |     21.1581 |      0.495338 |    81.1549 |                7 |                  7 |              0 |
|             6 |       100      |             0.0592552 |              27.0006 |     67.9452 |     65.8694   |    79.9415 |                6 |                  6 |              0 |
|             7 |       100      |             0.0570011 |              24.8881 |     19.1322 |      5.77888  |    98.0307 |                8 |                  8 |              0 |
|             8 |       100      |             0.0680347 |              32.0954 |     80.4661 |     83.1632   |    92.7229 |                8 |                  8 |              0 |
|             9 |       100      |             0.0804489 |              34.9733 |     29.2659 |     14.8702   |    92.9119 |               11 |                 11 |              0 |
|            10 |         0      |             0.0814737 |               0      |      0      |      0        |     0      |               10 |                  0 |             10 |
|            11 |        91.6667 |             0.0790493 |              35.335  |     25.8057 |     22.7089   |    51.8129 |               12 |                 11 |              1 |
|            12 |       100      |             0.320041  |             131.484  |     12.4658 |     15.0446   |    21.049  |               18 |                 18 |              0 |
|            13 |       100      |             0.332514  |             142.858  |     13.0351 |     16.6929   |    32.6803 |               19 |                 19 |              0 |
|            14 |       100      |             0.342533  |             147.032  |     14.3502 |     16.9479   |    31.7587 |               20 |                 20 |              0 |
|            15 |       100      |             0.270187  |             116.078  |     21.1125 |     22.5989   |    41.0658 |               21 |                 21 |              0 |
|            16 |       100      |             0.250956  |             107.74   |     25.2644 |     27.4542   |    47.2001 |               22 |                 22 |              0 |
|            17 |       100      |             0.206621  |              90.992  |     29.2223 |     29.5547   |    55.1654 |               21 |                 21 |              0 |
|            18 |       100      |             0.207229  |              93.6573 |     30.6117 |     28.9347   |    59.7103 |               20 |                 20 |              0 |
|            19 |       100      |             0.218928  |              96.8613 |     34.8795 |     31.1284   |    59.5934 |               23 |                 23 |              0 |
|            20 |       100      |             0.221557  |              99.2877 |     35.0057 |     30.7786   |    75.1571 |               22 |                 22 |              0 |
|            21 |       100      |             0.19927   |              88.0476 |     44.8632 |     50.3449   |    90.523  |               27 |                 27 |              0 |
|            22 |       100      |             0.204966  |              96.8805 |     41.339  |     43.6203   |    85.5593 |               24 |                 24 |              0 |
|            23 |        96.1538 |             0.201459  |              89.8817 |     42.9113 |     51.9901   |    83.336  |               26 |                 25 |              1 |
|            24 |        96      |             0.199257  |              90.3672 |     42.5155 |     42.9428   |    86.6849 |               25 |                 24 |              1 |

## Recommendations

⚠ Consider reducing max concurrent users to improve reliability
⚠ Response times may be too high for interactive use
⚠ Optimal concurrency appears to be around 14 users

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
