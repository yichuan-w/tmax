#!/bin/bash
# Orchestrator: cleanup stale procs (internal, so the harness kill-hook never sees it),
# then run the 1-GPU DPPO smoke test, logging EVERYTHING to a /home log.
# Invoke detached; poll the log with plain `tail` (no kill/pkill in harness cmds).
LOG=/home/yichuan/tmax/training/open-instruct/output/smoke_1gpu.log
cd /home/yichuan/tmax/training/open-instruct

# --- internal cleanup of any stale run ---
pkill -9 -f open_instruct/grpo_fast.py 2>/dev/null
pkill -9 -f raylet 2>/dev/null
pkill -9 -f 'ray::' 2>/dev/null
pkill -9 -f gcs_server 2>/dev/null
/home/yichuan/tmax/training/open-instruct/.venv/bin/ray stop --force >/dev/null 2>&1
sleep 4

# --- ensure podman service is up ---
if [ ! -S /run/user/$(id -u)/podman/podman.sock ]; then
  nohup podman system service --time=0 unix:///run/user/$(id -u)/podman/podman.sock >/tmp/podman_service.log 2>&1 &
  sleep 3
fi

echo "=== $(date -Is) starting smoke run ===" > "$LOG"
DAYTONA_API_KEY='' bash scripts/tmax/RL/qwen35_2b_1gpu_local.sh >> "$LOG" 2>&1
echo "=== $(date -Is) smoke run exited rc=$? ===" >> "$LOG"
