#!/bin/bash
# Self-healing watchdog for the tmax-9b TB-2.0 eval:
#  - logs progress every ~2min to eval_monitor.log
#  - if the vLLM serve (port 8009) is down, relaunch it (main long-run failure risk)
#  - stops when 89/89 done
pkill -9 -f "monitor_eval.sh" 2>/dev/null   # retire the old simple monitor
ML=/home/yichuan/tmax/eval_monitor.log
cd /home/yichuan/tmax
for i in $(seq 1 240); do
  UP=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8009/v1/models 2>/dev/null)
  if [ "$UP" != "200" ]; then
    # serve down mid-eval -> relaunch and give it time to load
    if ! pgrep -f "serve_tmax9b.sh\|vllm serve.*8009" >/dev/null 2>&1; then
      echo "$(date '+%H:%M:%S') !! serve DOWN -> relaunching serve_tmax9b.sh" >> "$ML"
      setsid bash /home/yichuan/tmax/serve_tmax9b.sh >/dev/null 2>&1 < /dev/null &
    fi
  fi
  R=$(.venv/bin/python -c "
import json
try:
  d=json.load(open('jobs/tmax9b-tb2-full/result.json')); s=d['stats']; ev=list(s['evals'].values())
  m=ev[0]['metrics'][0]['mean'] if ev and ev[0].get('metrics') else None
  print(f\"done={s['n_completed_trials']}/{d['n_total_trials']} run={s['n_running_trials']} err={s['n_errored_trials']} mean={m}\")
except Exception as e: print('noresult')
" 2>&1)
  echo "$(date '+%H:%M:%S') serve=$UP eval=$(pgrep -fc eval_harbor.py) $R" >> "$ML"
  echo "$R" | grep -qE "done=89/89" && { echo "$(date '+%H:%M:%S') === EVAL COMPLETE ===" >> "$ML"; break; }
  sleep 120
done
