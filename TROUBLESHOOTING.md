# Quad RTX 6000 Ada Classroom Ollama Setup - Troubleshooting Guide

## Common Issues and Solutions

### Issue 1: Ollama instances not using correct GPU

**Symptoms:**
- All workloads appear on single GPU
- nvidia-smi shows uneven GPU utilization
- Some GPUs idle

**Solution:**
```bash
# Verify CUDA_VISIBLE_DEVICES is set correctly
sudo systemctl status ollama-gpu0 | grep CUDA
sudo systemctl status ollama-gpu1 | grep CUDA
sudo systemctl status ollama-gpu2 | grep CUDA
sudo systemctl status ollama-gpu3 | grep CUDA

# Each should show different GPU ID (0, 1, 2, 3)

# If not working, check service files:
sudo systemctl cat ollama-gpu0.service
```

### Issue 2: Students experiencing slow response times

**Symptoms:**
- Time to First Token > 5 seconds
- Requests timing out
- High queue wait times

**Diagnosis:**
```bash
# Check GPU memory usage
nvidia-smi

# Check if models are loaded
curl http://localhost:11434/api/tags
curl http://localhost:11435/api/tags
curl http://localhost:11436/api/tags
curl http://localhost:11437/api/tags

# Check HAProxy backend status
curl http://localhost:8404/stats

# Monitor Ollama logs
sudo journalctl -u ollama-gpu0 -n 100 --no-pager
```

**Solutions:**
1. Increase OLLAMA_NUM_PARALLEL if GPU has spare capacity
2. Ensure model is preloaded on all instances
3. Check network latency between students and server
4. Verify no CPU bottleneck (check with htop)

### Issue 3: One Ollama instance failing health checks

**Symptoms:**
- HAProxy shows backend as "DOWN"
- Only 3 GPUs being utilized
- Increased load on remaining instances

**Diagnosis:**
```bash
# Check specific instance
sudo systemctl status ollama-gpu2
sudo journalctl -u ollama-gpu2 -n 50

# Try manual request
curl http://localhost:11436/api/tags
```

**Solutions:**
```bash
# Restart the failed instance
sudo systemctl restart ollama-gpu2

# Check for port conflicts
sudo netstat -tlnp | grep 11436

# Verify GPU is accessible
CUDA_VISIBLE_DEVICES=2 nvidia-smi
```

### Issue 4: Requests stuck in queue

**Symptoms:**
- OLLAMA_MAX_QUEUE exceeded
- 503 errors from HAProxy
- Long wait times

**Solution:**
```bash
# Increase queue size in service files
sudo nano /etc/systemd/system/ollama-gpu0.service
# Change: Environment="OLLAMA_MAX_QUEUE=256"

# Reload and restart
sudo systemctl daemon-reload
sudo systemctl restart ollama-gpu0
sudo systemctl restart ollama-gpu1
sudo systemctl restart ollama-gpu2
sudo systemctl restart ollama-gpu3
```

### Issue 5: Model not found errors

**Symptoms:**
- Error: "model 'gemma2:9b' not found"
- Inconsistent behavior across backends

**Solution:**
```bash
# Pull model on each instance separately
OLLAMA_HOST=http://localhost:11434 ollama pull gemma2:9b
OLLAMA_HOST=http://localhost:11435 ollama pull gemma2:9b
OLLAMA_HOST=http://localhost:11436 ollama pull gemma2:9b
OLLAMA_HOST=http://localhost:11437 ollama pull gemma2:9b

# Or use shared model directory (recommended for air-gapped)
# All instances can share /usr/share/ollama models
```

### Issue 6: HAProxy not distributing load evenly

**Symptoms:**
- One GPU consistently more loaded than others
- Uneven request distribution

**Diagnosis:**
```bash
# Check HAProxy stats
curl http://localhost:8404/stats

# Monitor request distribution
sudo tail -f /var/log/haproxy.log | grep ollama-gpu
```

**Solution:**
```bash
# Verify roundrobin is configured
sudo grep "balance" /etc/haproxy/haproxy.cfg

# Should show: balance roundrobin

# Try least-connections algorithm instead
sudo nano /etc/haproxy/haproxy.cfg
# Change: balance leastconn

sudo systemctl restart haproxy
```

## Performance Optimization

### Tuning OLLAMA_NUM_PARALLEL

Start conservative (6 per GPU) and increase based on testing:

```bash
# Monitor GPU utilization during test
watch -n 1 nvidia-smi

# If GPU utilization < 80%, increase parallelism
# Edit service file:
Environment="OLLAMA_NUM_PARALLEL=8"

# Restart and retest
sudo systemctl restart ollama-gpu0
```

### Preloading Models

To avoid cold-start delays:

```bash
# Create preload script
cat > /usr/local/bin/ollama-preload.sh << 'EOF'
#!/bin/bash
for port in 11434 11435 11436 11437; do
    curl http://localhost:$port/api/generate -d '{
        "model": "gemma2:9b",
        "prompt": "test",
        "keep_alive": -1
    }' > /dev/null 2>&1 &
done
EOF

chmod +x /usr/local/bin/ollama-preload.sh

# Add to systemd startup
sudo /usr/local/bin/ollama-preload.sh
```

### Monitoring Dashboard

Set up continuous monitoring:

```bash
# Install monitoring tools
sudo apt-get install prometheus-node-exporter grafana

# HAProxy stats endpoint: http://server:8404/stats
# GPU metrics: nvidia-smi dmon
# System metrics: prometheus on port 9100
```

## Air-Gapped Deployment Notes

For completely offline operation:

1. **Download Ollama and models on internet-connected machine:**
```bash
# Download Ollama binary
wget https://ollama.com/download/ollama-linux-amd64

# Pull models
ollama pull gemma2:9b

# Export models directory
tar -czf ollama-models.tar.gz ~/.ollama/models
```

2. **Transfer to air-gapped server:**
```bash
# Copy files via USB or secure transfer
scp ollama-linux-amd64 user@airgapped-server:/tmp/
scp ollama-models.tar.gz user@airgapped-server:/tmp/
```

3. **Install on air-gapped server:**
```bash
# Install Ollama
sudo mv /tmp/ollama-linux-amd64 /usr/local/bin/ollama
sudo chmod +x /usr/local/bin/ollama

# Extract models
sudo mkdir -p /usr/share/ollama
sudo tar -xzf /tmp/ollama-models.tar.gz -C /usr/share/ollama
```

## Monitoring Commands

### GPU Monitoring
```bash
# Real-time GPU usage
watch -n 1 nvidia-smi

# GPU memory details
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv

# Continuous monitoring
nvidia-smi dmon -s mu
```

### Service Health
```bash
# Check all Ollama services
systemctl status ollama-gpu{0..3}.service

# Check HAProxy
systemctl status haproxy

# View recent errors
sudo journalctl -u ollama-gpu0 --since "5 minutes ago"
```

### Network Traffic
```bash
# Monitor requests to load balancer
sudo tcpdump -i any port 11434 -c 100

# HAProxy connection stats
echo "show stat" | sudo socat stdio /run/haproxy/admin.sock
```

## Performance Benchmarks

Expected performance with recommended setup:

| Metric | Target | Good | Excellent |
|--------|--------|------|-----------|
| Time to First Token (TTFT) | < 2s | < 1s | < 0.5s |
| Tokens/sec per student | > 30 | > 50 | > 80 |
| Success rate | > 95% | > 98% | 100% |
| GPU utilization | > 70% | > 85% | > 95% |
| Request throughput | > 5 req/s | > 10 req/s | > 20 req/s |

## Support and Additional Resources

- Ollama docs: https://github.com/ollama/ollama/tree/main/docs
- HAProxy docs: https://www.haproxy.org/documentation.html
- NVIDIA Multi-GPU Best Practices: https://docs.nvidia.com/cuda/
- Gemma 2 Model Card: https://ai.google.dev/gemma/docs/model_card_2

## Emergency Recovery

If system becomes unresponsive:

```bash
# Stop all services
sudo systemctl stop ollama-gpu{0..3}
sudo systemctl stop haproxy

# Kill any hung processes
sudo killall ollama

# Clear GPU memory
sudo nvidia-smi --gpu-reset

# Restart services one by one
sudo systemctl start ollama-gpu0
# Wait 10 seconds, verify it works
curl http://localhost:11434/

# Repeat for other GPUs
sudo systemctl start ollama-gpu{1..3}
sudo systemctl start haproxy
```
