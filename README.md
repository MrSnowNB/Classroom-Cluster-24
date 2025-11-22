# Classroom Cluster 24.1

This is a project for managing classroom clusters, version 24.1.

## Getting Started

1. Ensure Node.js and npm are installed.
2. Clone the repository: `git clone https://github.com/MrSnowNB/Classroom-Cluster-24.git`
3. Install dependencies: `npm install`

## Test Suite Usage

Run the advanced concurrency test suite with hardware monitoring:

```bash
python3 test_suite.py --max-concurrency 25
```

This tests incremental concurrency from 1-24 users with hardware utilization logging and comprehensive analytics.

### ⚠️ Important: Vaporware Analysis

During testing, critical hardware-software decoupling issues were discovered that rendered the "multi-GPU cluster" implementation ineffective. See the comprehensive analysis in [OLLAMA_CLASSROOM_CLUSTER_VAPORWARE_ANALYSIS.md](OLLAMA_CLASSROOM_CLUSTER_VAPORWARE_ANALYSIS.md) for details on:

- Hardware configuration mismatches despite correct setup
- Load balancing verification failures at GPU utilization level
- Ollama architecture limitations preventing true parallelism
- Test suite design flaws that masked sequential processing
- Systemic monitoring gaps that allowed vaporware deployment

**Key Finding:** Despite 4 RTX A6000 GPUs correctly configured, only 1 GPU was utilized during peak "24-user" loads, exposing fundamental architectural misconceptions in the concurrency implementation.

## Contributing

Feel free to contribute to this project with awareness of the documented vaporware issues.

## License

ISC
