#!/bin/bash
pkill -9 -f "uvx vllm" 2>/dev/null
pkill -9 -f "serve allenai/tmax-9b" 2>/dev/null
pkill -9 -f "port 8009" 2>/dev/null
pkill -9 -f "VLLM::EngineCore" 2>/dev/null
sleep 6
setsid bash /home/yichuan/tmax/serve_tmax9b.sh >/dev/null 2>&1 < /dev/null &
echo done > /tmp/switch_serve9b.done
