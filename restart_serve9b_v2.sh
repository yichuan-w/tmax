#!/bin/bash
pkill -9 -f eval_harbor.py 2>/dev/null
pkill -9 -f watchdog_eval.sh 2>/dev/null
pkill -9 -f monitor_eval.sh 2>/dev/null
pkill -9 -f "serve_tmax9b.sh" 2>/dev/null
pkill -9 -f "vllm serve.*8009" 2>/dev/null
pkill -9 -f "port 8009" 2>/dev/null
# free the tmax-9b serve GPUs (2,3) by pid
for pid in $(nvidia-smi --query-compute-apps=gpu_bus_id,pid --format=csv,noheader 2>/dev/null | awk '{print $NF}'); do :; done
sleep 6
setsid bash /home/yichuan/tmax/serve_tmax9b.sh >/dev/null 2>&1 < /dev/null &
echo done > /tmp/restart9bv2.done
