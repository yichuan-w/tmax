#!/bin/bash
# TB-Lite (openthoughts-tblite@2.0) eval, base vs tmax-9b, k=5 avg@5, on the existing
# cudagraph serves. TB-Lite images build FAST (like TB-2.0/Pro) so it does NOT hit the
# TUA slow-build async-poll hang. Sandboxes tagged tmaxeval=1 (eval_harbor.py) + orphan
# daemon so we coexist with the co-tenant titan_swe_r2e run on the shared Daytona key.
cd /home/yichuan/tmax
export DAYTONA_API_KEY='dtn_5bfd6b7bf4ae20197feda35ccd2c5816a7aeea705de8d1aa9e93ec4ad4a72d42'
export https_proxy=http://fwdproxy:8080 HTTPS_PROXY=http://fwdproxy:8080 HF_HUB_ENABLE_HF_TRANSFER=1
QL=/home/yichuan/tmax/tblite.log
RETRY="--max-retries 3 --retry-include AgentTimeoutError --retry-include DaytonaNotFoundError --retry-include DaytonaAuthenticationError --retry-include DaytonaError --retry-include VerifierTimeoutError"
pyget(){ .venv/bin/python -c "import json;print(json.load(open('jobs/$1/result.json'))['stats']['n_completed_trials'])" 2>/dev/null; }
pytot(){ .venv/bin/python -c "import json;print(json.load(open('jobs/$1/result.json'))['n_total_trials'])" 2>/dev/null; }
heal(){ for p in 8009 8010; do u=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:$p/v1/models); if [ "$u" != 200 ] && ! pgrep -f "port $p">/dev/null; then
  [ $p = 8009 ] && setsid bash /home/yichuan/tmax/serve_qwen35_9b_base.sh >/dev/null 2>&1 </dev/null & [ $p = 8010 ] && setsid bash /home/yichuan/tmax/serve_tmax9b.sh >/dev/null 2>&1 </dev/null & fi; done; }
runeval(){ # port model job
  setsid /home/yichuan/tmax/.venv/bin/python eval_harbor.py run --dataset openthoughts-tblite@2.0 --env daytona --yes \
    --agent-import-path Vanillux2Agent:Vanillux2Agent --model openai/$2 --agent-kwarg api_base=http://localhost:$1/v1 \
    --agent-kwarg max_format_errors=64 --n-concurrent 8 -k 5 $RETRY --job-name $3 \
    > /home/yichuan/tmax/eval_$3.log 2>&1 < /dev/null & }

pkill -9 -f "job-name tblite-smoke" 2>/dev/null; pkill -9 -f "job-name tblite-base" 2>/dev/null; pkill -9 -f "job-name tblite-tmax" 2>/dev/null
sleep 4
rm -rf jobs/tblite-base jobs/tblite-tmax 2>/dev/null
# orphan cleaner (deletes ONLY tmaxeval=1, never titan) if not already running
pgrep -f tua_daemon.py >/dev/null || setsid /home/yichuan/tmax/.venv/bin/python tua_daemon.py > /home/yichuan/tmax/tua_daemon.log 2>&1 </dev/null &
echo "$(date '+%F %T') TB-Lite k=5 start" > "$QL"
runeval 8010 tmax-9b        tblite-tmax
echo "$(date '+%T') launched tblite-tmax" >> "$QL"
sleep 40  # stagger to dodge shared harbor-cache extraction race
runeval 8009 qwen35-9b-base tblite-base
echo "$(date '+%T') launched tblite-base" >> "$QL"

# stall-aware wait; totals read dynamically from result.json
last=-1; stall=0
for i in $(seq 1 20000); do heal
  a=$(pyget tblite-tmax); b=$(pyget tblite-base); ta=$(pytot tblite-tmax); tb=$(pytot tblite-base)
  a=${a:-0}; b=${b:-0}; ta=${ta:-999}; tb=${tb:-999}; sum=$((a+b))
  if [ "$sum" = "$last" ]; then stall=$((stall+1)); else stall=0; last=$sum; fi
  echo "$(date '+%T') wait tmax=$a/$ta base=$b/$tb stall=$stall" >> "$QL"
  { [ "$a" -ge "$ta" ] && [ "$b" -ge "$tb" ] && [ "$ta" -gt 0 ] && [ "$tb" -gt 0 ]; } && { echo "$(date '+%T') === TB-Lite DONE ===" >> "$QL"; break; }
  { [ "$stall" -ge 10 ] && [ "$a" -ge $((ta-4)) ] && [ "$b" -ge $((tb-4)) ] && [ "$ta" -gt 0 ]; } && { echo "$(date '+%T') === TB-Lite near+stalled DONE ===" >> "$QL"; break; }
  sleep 60; done
