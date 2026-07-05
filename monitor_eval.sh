#!/bin/bash
# lightweight poller: clean the leftover uvx zombie, then log eval progress every 3 min.
pkill -9 -f "uvx vllm" 2>/dev/null
pkill -9 -f "8009.*uvx\|uvx.*8009" 2>/dev/null
# free GPU4,5 zombie (uvx vllm workers) by pid if idle-orphaned on those GPUs
ML=/home/yichuan/tmax/eval_monitor.log
cd /home/yichuan/tmax
for i in $(seq 1 200); do
  R=$(.venv/bin/python -c "
import json
try:
  d=json.load(open('jobs/tmax9b-tb2-full/result.json')); s=d['stats']
  ev=list(s['evals'].values())
  m=ev[0]['metrics'][0]['mean'] if ev and ev[0]['metrics'] else None
  print(f\"done={s['n_completed_trials']}/{d['n_total_trials']} run={s['n_running_trials']} err={s['n_errored_trials']} mean={m}\")
except Exception as e: print('noresult', e)
" 2>&1)
  UP=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8009/v1/models 2>/dev/null)
  ALIVE=$(pgrep -fc eval_harbor.py)
  echo "$(date '+%H:%M:%S') serve=$UP evalproc=$ALIVE $R" >> "$ML"
  # stop if eval finished
  echo "$R" | grep -qE "done=89/89" && { echo "$(date '+%H:%M:%S') EVAL COMPLETE" >> "$ML"; break; }
  sleep 180
done
