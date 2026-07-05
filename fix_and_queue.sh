#!/bin/bash
# Fixer: the k5 TB-2.0 runs hang on the final 2-3 trials (Daytona sandbox teardown
# with no timeout after the trial already exhausted retries). Their avg@5 is final
# (tmax .233 / base .186). This script:
#   1. kills the stuck original queue + the hung k5 evals (results preserved on disk)
#   2. waits for base-k5 to reach 445 OR stall near-complete, then stops it
#   3. launches TB-2.1 (k=3) then TB-Pro (k=3) with a STALL-AWARE wait_done so a
#      hung tail never blocks queue advance. Self-heals serves.
cd /home/yichuan/tmax
export DAYTONA_API_KEY='dtn_5bfd6b7bf4ae20197feda35ccd2c5816a7aeea705de8d1aa9e93ec4ad4a72d42'
export https_proxy=http://fwdproxy:8080 HTTPS_PROXY=http://fwdproxy:8080 HF_HUB_ENABLE_HF_TRANSFER=1
QL=/home/yichuan/tmax/bench_queue2.log
RETRY="--max-retries 3 --retry-include AgentTimeoutError --retry-include DaytonaNotFoundError --retry-include DaytonaAuthenticationError --retry-include DaytonaError --retry-include VerifierTimeoutError"
pyget(){ .venv/bin/python -c "import json;print(json.load(open('jobs/$1/result.json'))['stats']['n_completed_trials'])" 2>/dev/null; }
heal(){ for p in 8009 8010; do u=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:$p/v1/models); if [ "$u" != 200 ] && ! pgrep -f "port $p">/dev/null; then
  [ $p = 8009 ] && setsid bash /home/yichuan/tmax/serve_qwen35_9b_base.sh >/dev/null 2>&1 </dev/null & [ $p = 8010 ] && setsid bash /home/yichuan/tmax/serve_tmax9b.sh >/dev/null 2>&1 </dev/null & fi; done; }
runeval(){ # port model job dataset k log
  setsid /home/yichuan/tmax/.venv/bin/python eval_harbor.py run --dataset "$4" --env daytona \
    --agent-import-path Vanillux2Agent:Vanillux2Agent --model openai/$2 --agent-kwarg api_base=http://localhost:$1/v1 \
    --agent-kwarg max_format_errors=64 --n-concurrent 16 -k $5 $RETRY --job-name $3 > "$6" 2>&1 < /dev/null & }
# stall-aware wait: done when both==total, OR both>=thresh(total-ceil(1.5%)) AND sum unchanged 8 checks (~12min)
wait_done(){ # job1 job2 total
  local thresh=$(( $3 - ( ($3+65)/66 ) - 2 )); local last=-1; local stall=0
  for i in $(seq 1 4000); do heal; a=$(pyget $1); b=$(pyget $2); a=${a:-0}; b=${b:-0}; sum=$((a+b))
    if [ "$sum" = "$last" ]; then stall=$((stall+1)); else stall=0; last=$sum; fi
    echo "$(date '+%T') wait $1=$a/$3 $2=$b/$3 thresh=$thresh stall=$stall" >> "$QL"
    { [ "$a" -ge "$3" ] && [ "$b" -ge "$3" ]; } && return 0
    { [ "$a" -ge "$thresh" ] && [ "$b" -ge "$thresh" ] && [ "$stall" -ge 8 ]; } && { echo "$(date '+%T') >> near-complete + stalled, advancing" >> "$QL"; return 0; }
    sleep 90; done; }

echo "$(date '+%F %T') fixer started" > "$QL"
# 1. stop stuck original queue + hung k5 evals (results already on disk)
pkill -f run_bench_queue.sh 2>/dev/null
pkill -f "job-name tmax9b-k5" 2>/dev/null
echo "$(date '+%T') killed old queue + hung tmax9b-k5 eval (was 443/445 avg@5 .233)" >> "$QL"

# 2. let base-k5 finish or stall, then stop it
last=-1; stall=0
for i in $(seq 1 14); do a=$(pyget base-k5); a=${a:-0}
  if [ "$a" = "$last" ]; then stall=$((stall+1)); else stall=0; last=$a; fi
  echo "$(date '+%T') base-k5=$a/445 stall=$stall" >> "$QL"
  [ "$a" -ge 445 ] && break
  { [ "$a" -ge 442 ] && [ "$stall" -ge 6 ]; } && { echo "$(date '+%T') base-k5 near-complete+stalled" >> "$QL"; break; }
  sleep 90; done
pkill -f "job-name base-k5" 2>/dev/null
bf=$(pyget base-k5); echo "$(date '+%T') stopped base-k5 at $bf/445 -> TB-2.0 k5 FINAL locked in" >> "$QL"
sleep 5; heal; sleep 20

# 3. TB-2.1 (k=3)
echo "$(date '+%T') === launch TB-2.1 (k=3) ===" >> "$QL"
runeval 8009 qwen35-9b-base tb21-base terminal-bench/terminal-bench-2-1 3 /home/yichuan/tmax/eval_tb21_base.log
runeval 8010 tmax-9b        tb21-tmax terminal-bench/terminal-bench-2-1 3 /home/yichuan/tmax/eval_tb21_tmax.log
sleep 30; wait_done tb21-base tb21-tmax 267
pkill -f "job-name tb21-base" 2>/dev/null; pkill -f "job-name tb21-tmax" 2>/dev/null
echo "$(date '+%T') === TB-2.1 done ===" >> "$QL"; sleep 10; heal; sleep 20

# 4. TB-Pro (k=3)
echo "$(date '+%T') === launch TB-Pro (k=3) ===" >> "$QL"
runeval 8009 qwen35-9b-base tbpro-base terminal-bench-pro/terminal-bench-pro 3 /home/yichuan/tmax/eval_tbpro_base.log
runeval 8010 tmax-9b        tbpro-tmax terminal-bench-pro/terminal-bench-pro 3 /home/yichuan/tmax/eval_tbpro_tmax.log
sleep 30; wait_done tbpro-base tbpro-tmax 600
pkill -f "job-name tbpro-base" 2>/dev/null; pkill -f "job-name tbpro-tmax" 2>/dev/null
echo "$(date '+%T') === ALL DONE (TB-2.1 + TB-Pro) ===" >> "$QL"
