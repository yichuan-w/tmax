#!/bin/bash
pkill -9 -f "watchdog_eval.sh" 2>/dev/null
ML=/home/yichuan/tmax/eval_monitor_v2.log; cd /home/yichuan/tmax
for i in $(seq 1 400); do
  UP=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8009/v1/models 2>/dev/null)
  if [ "$UP" != "200" ] && ! pgrep -f "vllm serve" >/dev/null 2>&1; then
    echo "$(date '+%H:%M:%S') !! serve DOWN -> relaunch" >> "$ML"; setsid bash /home/yichuan/tmax/serve_tmax9b.sh >/dev/null 2>&1 < /dev/null &
  fi
  R=$(.venv/bin/python -c "
import json
try:
  d=json.load(open('jobs/tmax9b-tb2-v2/result.json')); s=d['stats']; ev=list(s['evals'].values())
  m=ev[0]['metrics'][0]['mean'] if ev and ev[0].get('metrics') else None
  print(f\"done={s['n_completed_trials']}/{d['n_total_trials']} run={s['n_running_trials']} err={s['n_errored_trials']} retries={s.get('n_retries')} mean={m}\")
except Exception: print('noresult')" 2>&1)
  echo "$(date '+%H:%M:%S') serve=$UP eval=$(pgrep -fc eval_harbor.py) $R" >> "$ML"
  echo "$R" | grep -qE "done=89/89" && { echo "$(date '+%H:%M:%S') === V2 COMPLETE ===" >> "$ML"; break; }
  sleep 120
done
