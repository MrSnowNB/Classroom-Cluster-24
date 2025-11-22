
# Step 1: Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Step 2: Create ollama user (if not exists)
sudo useradd -r -s /bin/false -d /usr/share/ollama ollama

# Step 3: Pull the model on one instance first
ollama pull gemma2:9b

# Step 4: Copy systemd service files
sudo cp ollama-gpu*.service /etc/systemd/system/

# Step 5: Reload systemd
sudo systemctl daemon-reload

# Step 6: Enable all Ollama services
sudo systemctl enable ollama-gpu0.service
sudo systemctl enable ollama-gpu1.service
sudo systemctl enable ollama-gpu2.service
sudo systemctl enable ollama-gpu3.service

# Step 7: Start all Ollama services
sudo systemctl start ollama-gpu0.service
sudo systemctl start ollama-gpu1.service
sudo systemctl start ollama-gpu2.service
sudo systemctl start ollama-gpu3.service

# Step 8: Check status
sudo systemctl status ollama-gpu0.service
sudo systemctl status ollama-gpu1.service
sudo systemctl status ollama-gpu2.service
sudo systemctl status ollama-gpu3.service

# Step 9: Verify each instance
curl http://localhost:11434/api/tags  # GPU 0
curl http://localhost:11435/api/tags  # GPU 1
curl http://localhost:11436/api/tags  # GPU 2
curl http://localhost:11437/api/tags  # GPU 3

# Step 10: Install and configure HAProxy
sudo apt-get install haproxy -y
sudo cp haproxy.cfg /etc/haproxy/
sudo haproxy -c -f /etc/haproxy/haproxy.cfg
sudo systemctl restart haproxy
sudo systemctl enable haproxy

# Step 11: Test the load balancer
curl http://localhost:11434/  # Should return "Ollama is running"

# Step 12: Monitor GPU usage
watch -n 1 nvidia-smi

# Step 13: Check HAProxy stats
# Open browser to: http://your-server-ip:8404/stats

# Step 14: Run concurrency test
python3 concurrency_test.py --endpoint http://localhost:11434 --students 24 --duration 600

# Useful monitoring commands:
# - View Ollama logs: sudo journalctl -u ollama-gpu0 -f
# - Check HAProxy logs: sudo tail -f /var/log/haproxy.log
# - GPU monitoring: nvidia-smi dmon -s mu
