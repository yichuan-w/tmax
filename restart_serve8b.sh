#!/bin/bash
# kill old eval + vllm serve, then relaunch the 8B serve with tool-choice flags
pkill -9 -f eval_harbor.py 2>/dev/null
pkill -9 -f "vllm serve" 2>/dev/null
pkill -9 -f "VLLM::EngineCore" 2>/dev/null
sleep 5
setsid bash /home/yichuan/tmax/serve_tmax8b.sh >/dev/null 2>&1 < /dev/null &
echo "relaunched serve" > /tmp/restart_serve8b.done
