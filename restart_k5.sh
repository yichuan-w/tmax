#!/bin/bash
pkill -9 -f "run_k5.sh" 2>/dev/null
pkill -9 -f "eval_harbor.py" 2>/dev/null
sleep 4
rm -rf /home/yichuan/tmax/jobs/base-k5 /home/yichuan/tmax/jobs/tmax9b-k5 2>/dev/null
setsid bash /home/yichuan/tmax/run_k5.sh >/dev/null 2>&1 < /dev/null &
echo done > /tmp/restart_k5.done
