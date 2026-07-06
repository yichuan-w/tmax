#!/bin/bash
# 27B base (Qwen3.6-27B, :8011) vs tmax-27b RL (:8012) on the OTHER Terminal-Bench
# benchmarks: TB-Lite (k5) -> TB-2.1 (k3) -> TB-Pro (k3). High concurrency, self-heal,
# tmaxeval orphan daemon. Same avg@k methodology as the 9B suite for direct comparison.
cd /home/yichuan/tmax
export DAYTONA_API_KEY='dtn_5bfd6b7bf4ae20197feda35ccd2c5816a7aeea705de8d1aa9e93ec4ad4a72d42'
export https_proxy=http://fwdproxy:8080 HTTPS_PROXY=http://fwdproxy:8080 HF_HUB_ENABLE_HF_TRANSFER=1
QL=/home/yichuan/tmax/run_27b_other.log
RETRY="--max-retries 3 --retry-include AgentTimeoutError --retry-include DaytonaNotFoundError --retry-include DaytonaAuthenticationError --retry-include DaytonaError --retry-include VerifierTimeoutError"
pyget(){ .venv/bin/python -c "import json;print(json.load(open('jobs/$1/result.json'))['stats']['n_completed_trials'])" 2>/dev/null; }
pytot(){ .venv/bin/python -c "import json;print(json.load(open('jobs/$1/result.json'))['n_total_trials'])" 2>/dev/null; }
heal(){ for p in 8011 8012; do u=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:$p/v1/models); if [ "$u" != 200 ] && ! pgrep -f "port $p">/dev/null; then
  [ $p = 8011 ] && setsid bash /home/yichuan/tmax/serve_qwen36_27b_base.sh >/dev/null 2>&1 </dev/null & [ $p = 8012 ] && setsid bash /home/yichuan/tmax/serve_tmax27b.sh >/dev/null 2>&1 </dev/null & fi; done; }
runeval(){ # port model job dataset k
  setsid /home/yichuan/tmax/.venv/bin/python eval_harbor.py run --dataset "$4" --env daytona --yes \
    --agent-import-path Vanillux2Agent:Vanillux2Agent --model openai/$2 --agent-kwarg api_base=http://localhost:$1/v1 \
    --agent-kwarg max_format_errors=64 --n-concurrent 24 -k $5 $RETRY --job-name $3 > /home/yichuan/tmax/eval_$3.log 2>&1 < /dev/null & }
wait_done(){ # basejob tmaxjob
  local last=-1 stall=0
  for i in $(seq 1 20000); do heal
    a=$(pyget $1); b=$(pyget $2); ta=$(pytot $1); tb=$(pytot $2)
    a=${a:-0}; b=${b:-0}; ta=${ta:-99999}; tb=${tb:-99999}; sum=$((a+b))
    if [ "$sum" = "$last" ]; then stall=$((stall+1)); else stall=0; last=$sum; fi
    echo "$(date '+%T') $1=$a/$ta $2=$b/$tb stall=$stall" >> "$QL"
    { [ "$a" -ge "$ta" ] && [ "$b" -ge "$tb" ] && [ "$ta" -gt 0 ] && [ "$tb" -gt 0 ]; } && return 0
    { [ "$stall" -ge 10 ] && [ "$a" -ge $((ta-4)) ] && [ "$b" -ge $((tb-4)) ] && [ "$ta" -gt 100 ]; } && return 0
    sleep 60; done; }

pgrep -f tua_daemon.py >/dev/null || setsid /home/yichuan/tmax/.venv/bin/python tua_daemon.py > /home/yichuan/tmax/tua_daemon.log 2>&1 </dev/null &
echo "$(date '+%F %T') 27B other-bench queue start" > "$QL"

# 1) TB-Lite k=5
rm -rf jobs/tblite27b-base jobs/tblite27b-tmax 2>/dev/null
runeval 8012 tmax-27b        tblite27b-tmax openthoughts-tblite@2.0 5
echo "$(date '+%T') launched tblite27b-tmax" >> "$QL"; sleep 40
runeval 8011 qwen36-27b-base tblite27b-base openthoughts-tblite@2.0 5
echo "$(date '+%T') launched tblite27b-base" >> "$QL"; sleep 20
wait_done tblite27b-base tblite27b-tmax
echo "$(date '+%T') === TB-Lite 27B DONE ===" >> "$QL"

# 2) TB-2.1 k=3
rm -rf jobs/tb21-27b-base jobs/tb21-27b-tmax 2>/dev/null
runeval 8012 tmax-27b        tb21-27b-tmax terminal-bench/terminal-bench-2-1 3
echo "$(date '+%T') launched tb21-27b-tmax" >> "$QL"; sleep 40
runeval 8011 qwen36-27b-base tb21-27b-base terminal-bench/terminal-bench-2-1 3
echo "$(date '+%T') launched tb21-27b-base" >> "$QL"; sleep 20
wait_done tb21-27b-base tb21-27b-tmax
echo "$(date '+%T') === TB-2.1 27B DONE ===" >> "$QL"

# 3) TB-Pro k=3
rm -rf jobs/tbpro27b-base jobs/tbpro27b-tmax 2>/dev/null
runeval 8012 tmax-27b        tbpro27b-tmax terminal-bench-pro/terminal-bench-pro 3
echo "$(date '+%T') launched tbpro27b-tmax" >> "$QL"; sleep 40
runeval 8011 qwen36-27b-base tbpro27b-base terminal-bench-pro/terminal-bench-pro 3
echo "$(date '+%T') launched tbpro27b-base" >> "$QL"; sleep 20
wait_done tbpro27b-base tbpro27b-tmax
echo "$(date '+%T') === TB-Pro 27B DONE === ALL 27B OTHER-BENCH COMPLETE ===" >> "$QL"
