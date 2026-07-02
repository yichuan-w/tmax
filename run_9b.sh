#!/bin/bash
# Orchestrator for the 9B (Qwen3.5-9B) reward-increase RL run. Internal cleanup (kill/pkill
# here, never in a harness command), then run detached, logging everything to a /home log.
LOG=/home/yichuan/tmax/training/open-instruct/output/train_9b.log
cd /home/yichuan/tmax/training/open-instruct

pkill -9 -f open_instruct/grpo_fast.py 2>/dev/null
pkill -9 -f raylet 2>/dev/null
pkill -9 -f 'ray::' 2>/dev/null
pkill -9 -f gcs_server 2>/dev/null
pkill -9 -f 'default_worker.py' 2>/dev/null
pkill -9 -f 'EngineCore' 2>/dev/null
/home/yichuan/tmax/training/open-instruct/.venv/bin/ray stop --force >/dev/null 2>&1
podman rm -af >/dev/null 2>&1   # remove orphaned sleep-infinity sandbox containers from prior run
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u); do
  kill -9 "$pid" 2>/dev/null
done
sleep 8
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

echo "=== $(date -Is) starting 9B tmax-15k reward-increase run ===" > "$LOG"
bash scripts/tmax/RL/qwen35_9b_8gpu_local.sh >> "$LOG" 2>&1
echo "=== $(date -Is) 9B run exited rc=$? ===" >> "$LOG"
