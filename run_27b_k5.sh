#!/bin/bash
# k=5 (avg@5) TB-2.0 for base Qwen3.6-27B (port 8011) vs tmax-27b RL (port 8012),
# on the cudagraph 27B serves. High concurrency (24/model). Self-heals serves; runs the
# tmaxeval-only orphan daemon so we coexist with the co-tenant titan_swe_r2e run.
cd /home/yichuan/tmax
export DAYTONA_API_KEY='dtn_5bfd6b7bf4ae20197feda35ccd2c5816a7aeea705de8d1aa9e93ec4ad4a72d42'
export https_proxy=http://fwdproxy:8080 HTTPS_PROXY=http://fwdproxy:8080 HF_HUB_ENABLE_HF_TRANSFER=1
CL=/home/yichuan/tmax/run_27b.log
RETRY="--max-retries 3 --retry-include AgentTimeoutError --retry-include DaytonaNotFoundError --retry-include DaytonaAuthenticationError --retry-include DaytonaError --retry-include VerifierTimeoutError"
pyget(){ .venv/bin/python -c "import json;print(json.load(open('jobs/$1/result.json'))['stats']['n_completed_trials'])" 2>/dev/null; }
heal(){ for p in 8011 8012; do u=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:$p/v1/models); if [ "$u" != 200 ] && ! pgrep -f "port $p">/dev/null; then
  [ $p = 8011 ] && setsid bash /home/yichuan/tmax/serve_qwen36_27b_base.sh >/dev/null 2>&1 </dev/null & [ $p = 8012 ] && setsid bash /home/yichuan/tmax/serve_tmax27b.sh >/dev/null 2>&1 </dev/null & fi; done; }
runeval(){ # port model job
  setsid /home/yichuan/tmax/.venv/bin/python eval_harbor.py run --dataset terminal-bench@2.0 --env daytona \
    --agent-import-path Vanillux2Agent:Vanillux2Agent --model openai/$2 --agent-kwarg api_base=http://localhost:$1/v1 \
    --agent-kwarg max_format_errors=64 --n-concurrent 24 -k 5 $RETRY --job-name $3 > "$4" 2>&1 < /dev/null & }

pgrep -f tua_daemon.py >/dev/null || setsid /home/yichuan/tmax/.venv/bin/python tua_daemon.py > /home/yichuan/tmax/tua_daemon.log 2>&1 </dev/null &
echo "$(date '+%F %T') 27B k=5 TB-2.0 start (base:8011 tmax:8012, n-conc 24)" > "$CL"
runeval 8011 qwen36-27b-base base27b-k5  /home/yichuan/tmax/eval_base27b_k5.log
runeval 8012 tmax-27b        tmax27b-k5  /home/yichuan/tmax/eval_tmax27b_k5.log
sleep 30
last=-1; stall=0
for i in $(seq 1 4000); do heal
  a=$(pyget base27b-k5); b=$(pyget tmax27b-k5); a=${a:-0}; b=${b:-0}; sum=$((a+b))
  if [ "$sum" = "$last" ]; then stall=$((stall+1)); else stall=0; last=$sum; fi
  echo "$(date '+%T') base27b=$a/445 tmax27b=$b/445 stall=$stall" >> "$CL"
  { [ "$a" -ge 445 ] && [ "$b" -ge 445 ]; } && { echo "$(date '+%T') === 27B k5 DONE ===" >> "$CL"; break; }
  { [ "$a" -ge 441 ] && [ "$b" -ge 441 ] && [ "$stall" -ge 8 ]; } && { echo "$(date '+%T') === 27B near+stalled DONE ===" >> "$CL"; break; }
  sleep 90; done
