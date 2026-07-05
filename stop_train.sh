#!/bin/bash
# Stop the local RL training to free GPUs (internal pkill; run detached).
pkill -9 -f open_instruct/grpo_fast.py 2>/dev/null
pkill -9 -f raylet 2>/dev/null
pkill -9 -f 'ray::' 2>/dev/null
pkill -9 -f gcs_server 2>/dev/null
pkill -9 -f 'default_worker.py' 2>/dev/null
pkill -9 -f 'EngineCore' 2>/dev/null
pkill -9 -f 'run_9b.sh' 2>/dev/null
/home/yichuan/tmax/training/open-instruct/.venv/bin/ray stop --force >/dev/null 2>&1
podman rm -af >/dev/null 2>&1
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u); do
  kill -9 "$pid" 2>/dev/null
done
sleep 6
echo "stopped; GPU free MiB total: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1}END{print s}')" > /tmp/stop_train.done
