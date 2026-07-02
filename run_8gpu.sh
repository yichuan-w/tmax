#!/bin/bash
# Orchestrator for the 8-GPU tmax-15k DPPO run. Internal cleanup (kill/pkill here,
# never in a harness command), then run detached, logging everything to a /home log.
LOG=/home/yichuan/tmax/training/open-instruct/output/train_8gpu.log
cd /home/yichuan/tmax/training/open-instruct

pkill -9 -f open_instruct/grpo_fast.py 2>/dev/null
pkill -9 -f raylet 2>/dev/null
pkill -9 -f 'ray::' 2>/dev/null
pkill -9 -f gcs_server 2>/dev/null
pkill -9 -f 'default_worker.py' 2>/dev/null
pkill -9 -f 'EngineCore' 2>/dev/null
/home/yichuan/tmax/training/open-instruct/.venv/bin/ray stop --force >/dev/null 2>&1
# kill any process still holding a GPU (orphaned vllm/learner procs from a crash)
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u); do
  kill -9 "$pid" 2>/dev/null
done
sleep 8
# wait until all GPUs are actually free
for i in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s}')
  [ "${used:-9999}" -lt 2000 ] && break
  sleep 2
done
echo "GPU total used after cleanup: ${used} MiB"

if [ ! -S /run/user/$(id -u)/podman/podman.sock ]; then
  nohup podman system service --time=0 unix:///run/user/$(id -u)/podman/podman.sock >/tmp/podman_service.log 2>&1 &
  sleep 3
fi

echo "=== $(date -Is) starting 8-GPU tmax-15k run ===" > "$LOG"
bash scripts/tmax/RL/qwen3_8b_8gpu_local.sh >> "$LOG" 2>&1
echo "=== $(date -Is) 8-GPU run exited rc=$? ===" >> "$LOG"
