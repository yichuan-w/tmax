#!/bin/bash
# step-100 on the OTHER benchmarks (TB-Lite k5 -> TB-2.1 k3 -> TB-Pro k3), using the FAST
# stable serve (serve_step100_cg.sh, FULL_DECODE_ONLY). Self-heals the :8013 serve. Logs -> /dev/shm
# to spare the /home quota. Same avg@k methodology as the 9B/27B suites.
cd /home/yichuan/tmax
export DAYTONA_API_KEY='dtn_5bfd6b7bf4ae20197feda35ccd2c5816a7aeea705de8d1aa9e93ec4ad4a72d42'
export https_proxy=http://fwdproxy:8080 HTTPS_PROXY=http://fwdproxy:8080 HF_HUB_ENABLE_HF_TRANSFER=1
QL=/dev/shm/run_step100_bench.log
RETRY="--max-retries 3 --retry-include AgentTimeoutError --retry-include DaytonaNotFoundError --retry-include DaytonaAuthenticationError --retry-include DaytonaError --retry-include VerifierTimeoutError"
pyget(){ .venv/bin/python -c "import json;print(json.load(open('jobs/$1/result.json'))['stats']['n_completed_trials'])" 2>/dev/null; }
pytot(){ .venv/bin/python -c "import json;print(json.load(open('jobs/$1/result.json'))['n_total_trials'])" 2>/dev/null; }
heal(){ u=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8013/v1/models 2>/dev/null)
  if [ "$u" != 200 ] && ! pgrep -f 'ckpt_step100_hf' >/dev/null; then
    echo "$(date '+%T') !! serve down -> restart" >> "$QL"
    setsid bash /home/yichuan/tmax/serve_step100_cg.sh >/dev/null 2>&1 </dev/null & fi; }
runeval(){ # job dataset k
  setsid /home/yichuan/tmax/.venv/bin/python eval_harbor.py run --dataset "$2" --env daytona --yes \
    --agent-import-path Vanillux2Agent:Vanillux2Agent --model openai/step100 --agent-kwarg api_base=http://localhost:8013/v1 \
    --agent-kwarg max_format_errors=64 --n-concurrent 24 -k $3 $RETRY --job-name "$1" > /dev/shm/eval_$1.log 2>&1 </dev/null & }
wait_done(){ local last=-1 stall=0
  for i in $(seq 1 20000); do heal
    a=$(pyget $1); a=${a:-0}; ta=$(pytot $1); ta=${ta:-99999}
    if [ "$a" = "$last" ]; then stall=$((stall+1)); else stall=0; last=$a; fi
    echo "$(date '+%T') $1=$a/$ta stall=$stall" >> "$QL"
    { [ "$a" -ge "$ta" ] && [ "$ta" -gt 0 ]; } && return 0
    { [ "$stall" -ge 15 ] && [ "$a" -ge $((ta-4)) ] && [ "$ta" -gt 100 ]; } && return 0
    sleep 60; done; }

echo "$(date '+%F %T') step100 other-bench start" > "$QL"; heal
for i in $(seq 1 20); do [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8013/v1/models)" = 200 ] && break; sleep 15; done

rm -rf jobs/step100-tblite jobs/step100-tb21 jobs/step100-tbpro 2>/dev/null
runeval step100-tblite openthoughts-tblite@2.0 5
echo "$(date '+%T') launched tblite" >> "$QL"; wait_done step100-tblite
echo "$(date '+%T') === TB-Lite DONE ===" >> "$QL"

runeval step100-tb21 terminal-bench/terminal-bench-2-1 3
echo "$(date '+%T') launched tb21" >> "$QL"; wait_done step100-tb21
echo "$(date '+%T') === TB-2.1 DONE ===" >> "$QL"

runeval step100-tbpro terminal-bench-pro/terminal-bench-pro 3
echo "$(date '+%T') launched tbpro" >> "$QL"; wait_done step100-tbpro
echo "$(date '+%T') === TB-Pro DONE === ALL STEP100 OTHER-BENCH COMPLETE ===" >> "$QL"
