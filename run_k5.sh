#!/bin/bash
# k=5 (avg@5) for base vs tmax-9b, both on the existing cudagraph serves.
# concurrency 8 each (16 total Daytona ~ README's 16), retry timeouts+Daytona x3.
cd /home/yichuan/tmax
export DAYTONA_API_KEY='dtn_5bfd6b7bf4ae20197feda35ccd2c5816a7aeea705de8d1aa9e93ec4ad4a72d42'
export https_proxy=http://fwdproxy:8080 HTTPS_PROXY=http://fwdproxy:8080 HF_HUB_ENABLE_HF_TRANSFER=1
CL=/home/yichuan/tmax/k5_compare.log
RETRY="--max-retries 3 --retry-include AgentTimeoutError --retry-include DaytonaNotFoundError --retry-include DaytonaAuthenticationError --retry-include DaytonaError --retry-include VerifierTimeoutError"
runeval() { # port model job log
  setsid /home/yichuan/tmax/.venv/bin/python eval_harbor.py run \
    --dataset terminal-bench@2.0 --env daytona \
    --agent-import-path Vanillux2Agent:Vanillux2Agent \
    --model openai/$2 --agent-kwarg api_base=http://localhost:$1/v1 \
    --agent-kwarg max_format_errors=64 --n-concurrent 16 -k 5 $RETRY \
    --job-name $3 > "$4" 2>&1 < /dev/null &
}
echo "$(date '+%F %T') launching k=5 (base:8009 tmax:8010)" > "$CL"
runeval 8009 qwen35-9b-base base-k5   /home/yichuan/tmax/eval_base_k5.log
runeval 8010 tmax-9b        tmax9b-k5 /home/yichuan/tmax/eval_tmax_k5.log
sleep 20
prog() { .venv/bin/python -c "
import json
try:
  d=json.load(open('jobs/$1/result.json')); s=d['stats']; ev=list(s['evals'].values())[0]
  m=ev['metrics'][0]['mean'] if ev.get('metrics') else None
  pk=ev.get('pass_at_k',{})
  print(f\"$1 done={s['n_completed_trials']}/{d['n_total_trials']} err={s['n_errored_trials']} avg@5={m} pass@k={pk}\")
except Exception: print('$1 noresult')" 2>&1; }
for i in $(seq 1 400); do
  for p in 8009 8010; do
    up=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:$p/v1/models)
    if [ "$up" != "200" ] && ! pgrep -f "port $p" >/dev/null; then
      echo "$(date '+%T') !! serve $p down->relaunch" >> "$CL"
      [ "$p" = 8009 ] && setsid bash /home/yichuan/tmax/serve_qwen35_9b_base.sh >/dev/null 2>&1 </dev/null &
      [ "$p" = 8010 ] && setsid bash /home/yichuan/tmax/serve_tmax9b.sh >/dev/null 2>&1 </dev/null &
    fi
  done
  echo "$(date '+%T') $(prog base-k5) | $(prog tmax9b-k5)" >> "$CL"
  db=$(.venv/bin/python -c "import json;print(json.load(open('jobs/base-k5/result.json'))['stats']['n_completed_trials'])" 2>/dev/null)
  dt=$(.venv/bin/python -c "import json;print(json.load(open('jobs/tmax9b-k5/result.json'))['stats']['n_completed_trials'])" 2>/dev/null)
  [ "$db" = "445" ] && [ "$dt" = "445" ] && { echo "$(date '+%T') === K5 BOTH COMPLETE ===" >> "$CL"; break; }
  sleep 150
done
