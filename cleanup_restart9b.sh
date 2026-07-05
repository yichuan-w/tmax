#!/bin/bash
pkill -9 -f eval_harbor.py 2>/dev/null
pkill -9 -f watchdog_eval.sh 2>/dev/null
pkill -9 -f "vllm serve" 2>/dev/null
pkill -9 -f "VLLM::EngineCore\|EngineCore_DP\|from_ray\|vllm.v1.engine" 2>/dev/null
# kill anything still holding a GPU (only the stale tmax-9b serve should be left)
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u); do kill -9 "$pid" 2>/dev/null; done
sleep 8
# wait for GPUs to clear
for i in $(seq 1 20); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1}END{print s}')
  [ "${u:-9999}" -lt 3000 ] && break; sleep 2
done
echo "gpu_after=$u" > /tmp/cleanup9b.done
setsid bash /home/yichuan/tmax/serve_tmax9b.sh >/dev/null 2>&1 < /dev/null &
