
# Concurrency Test Suite Analysis Report

**Generated:** 2025-11-22 12:40:50
**Endpoint:** http://localhost:11434
**Model:** gemma2:9b

## Executive Summary

This report presents the results of comprehensive concurrency testing for the Ollama Classroom Cluster.
The test suite evaluated performance across 1-6 concurrent users with mixed-length prompts.

## Key Metrics

- **Peak Throughput:** 147.5 tokens/second
- **Optimal Concurrency:** 8 users
- **Average Success Rate:** 97.7%
- **P95 Response Time (6 users):** 84.582s

## Performance Charts

(Charts generated as PNG files in the analysis directory)

- Throughput vs Concurrency: `throughput_vs_concurrency.png`
- Success Rate vs Concurrency: `success_rate_vs_concurrency.png`
- Response Time vs Concurrency: `ttft_vs_concurrency.png`
- Requests per Second vs Concurrency: `rps_vs_concurrency.png`

## Detailed Results

|   concurrency |   success_rate |   requests_per_second |   overall_throughput |   mean_ttft |   median_ttft |   p95_ttft |   total_requests |   total_successful |   total_failed |
|--------------:|---------------:|----------------------:|---------------------:|------------:|--------------:|-----------:|-----------------:|-------------------:|---------------:|
|             1 |       100      |              0.113445 |              43.1093 |    0.380135 |      0.381826 |   0.386364 |                4 |                  4 |              0 |
|             2 |       100      |              0.16522  |              67.8227 |    0.396707 |      0.395757 |   0.443626 |                6 |                  6 |              0 |
|             3 |       100      |              0.227013 |              91.7889 |    0.414027 |      0.401471 |   0.505914 |                9 |                  9 |              0 |
|             4 |       100      |              0.257476 |             115.607  |    0.409492 |      0.405423 |   0.449955 |               11 |                 11 |              0 |
|             5 |       100      |              0.27203  |             109.377  |    0.424971 |      0.427813 |   0.462257 |               13 |                 13 |              0 |
|             6 |       100      |              0.30727  |             120.994  |    0.436071 |      0.429068 |   0.522346 |               13 |                 13 |              0 |
|             7 |       100      |              0.298546 |             115.985  |    1.59507  |      0.44327  |  14.0567   |               16 |                 16 |              0 |
|             8 |       100      |              0.346661 |             147.48   |    3.58215  |      0.437479 |  16.1539   |               14 |                 14 |              0 |
|             9 |       100      |              0.26854  |             112.107  |    7.42195  |      1.09812  |  20.4058   |               15 |                 15 |              0 |
|            10 |       100      |              0.338569 |             147.066  |    7.68179  |      7.73688  |  18.4199   |               16 |                 16 |              0 |
|            11 |       100      |              0.295996 |             129.96   |    9.53218  |     13.9794   |  18.6174   |               17 |                 17 |              0 |
|            12 |       100      |              0.24706  |             110.901  |   14.751    |     21.3823   |  29.5383   |               17 |                 17 |              0 |
|            13 |       100      |              0.256025 |             114.538  |   17.734    |     22.2574   |  42.3712   |               19 |                 19 |              0 |
|            14 |       100      |              0.262715 |             114.728  |   18.3275   |     21.794    |  43.0548   |               20 |                 20 |              0 |
|            15 |       100      |              0.306435 |             135.707  |   16.8401   |     17.653    |  31.6803   |               21 |                 21 |              0 |
|            16 |       100      |              0.243103 |             108.634  |   27.0014   |     26.9309   |  45.736    |               22 |                 22 |              0 |
|            17 |       100      |              0.202225 |              90.243  |   29.8688   |     32.026    |  55.9098   |               20 |                 20 |              0 |
|            18 |       100      |              0.160537 |              74.1934 |   48.01     |     52.0423   |  79.6914   |               19 |                 19 |              0 |
|            19 |       100      |              0.214416 |              89.8575 |   33.9857   |     38.0074   |  68.4251   |               25 |                 25 |              0 |
|            20 |        90      |              0.162647 |              70.8736 |   51.2212   |     50.8316   |  91.6144   |               20 |                 18 |              2 |
|            21 |       100      |              0.196155 |              87.9355 |   41.8014   |     43.4892   |  90.4499   |               27 |                 27 |              0 |
|            22 |       100      |              0.187802 |              88.685  |   55.1937   |     56.2506   |  94.7135   |               22 |                 22 |              0 |
|            23 |        96.2963 |              0.179829 |              77.926  |   46.6348   |     52.9908   |  93.7168   |               27 |                 26 |              1 |
|            24 |        58.3333 |              0.196463 |              53.8555 |   59.6018   |     55.6118   |  84.5816   |               24 |                 14 |             10 |

## Recommendations

⚠ Consider reducing max concurrent users to improve reliability
⚠ Response times may be too high for interactive use
⚠ Optimal concurrency appears to be around 8 users

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
