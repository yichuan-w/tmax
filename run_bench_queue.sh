#!/bin/bash
# Queue: after k=5 TB-2.0 finishes -> TB-2.1 (k=3) -> TB-Pro (k=3), both models,
# on the existing cudagraph serves (base:8009, tmax:8010). Self-heals serves. k=3 = avg@3.
cd /home/yichuan/tmax
export DAYTONA_API_KEY='dtn_5bfd6b7bf4ae20197feda35ccd2c5816a7aeea705de8d1aa9e93ec4ad4a72d42'
export https_proxy=http://fwdproxy:8080 HTTPS_PROXY=http://fwdproxy:8080 HF_HUB_ENABLE_HF_TRANSFER=1
QL=/home/yichuan/tmax/bench_queue.log
RETRY="--max-retries 3 --retry-include AgentTimeoutError --retry-include DaytonaNotFoundError --retry-include DaytonaAuthenticationError --retry-include DaytonaError --retry-include VerifierTimeoutError"
pyget(){ .venv/bin/python -c "import json;print(json.load(open('jobs/$1/result.json'))['stats']['n_completed_trials'])" 2>/dev/null; }
heal(){ for p in 8009 8010; do u=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:$p/v1/models); if [ "$u" != 200 ] && ! pgrep -f "port $p">/dev/null; then
  [ $p = 8009 ] && setsid bash /home/yichuan/tmax/serve_qwen35_9b_base.sh >/dev/null 2>&1 </dev/null & [ $p = 8010 ] && setsid bash /home/yichuan/tmax/serve_tmax9b.sh >/dev/null 2>&1 </dev/null & fi; done; }
runeval(){ # port model job dataset k log
  setsid /home/yichuan/tmax/.venv/bin/python eval_harbor.py run --dataset "$4" --env daytona \
    --agent-import-path Vanillux2Agent:Vanillux2Agent --model openai/$2 --agent-kwarg api_base=http://localhost:$1/v1 \
    --agent-kwarg max_format_errors=64 --n-concurrent 16 -k $5 $RETRY --job-name $3 > "$6" 2>&1 < /dev/null & }
wait_done(){ # job1 job2 total
  for i in $(seq 1 4000); do heal; a=$(pyget $1); b=$(pyget $2)
    echo "$(date '+%T') wait $1=$a/$3 $2=$b/$3" >> "$QL"
    [ "$a" = "$3" ] && [ "$b" = "$3" ] && return 0; sleep 90; done; }

echo "$(date '+%F %T') queue started; waiting for k=5 TB-2.0..." > "$QL"
wait_done base-k5 tmax9b-k5 445
echo "$(date '+%T') === k5 TB-2.0 done -> TB-2.1 (k=3) ===" >> "$QL"
runeval 8009 qwen35-9b-base tb21-base terminal-bench/terminal-bench-2-1 3 /home/yichuan/tmax/eval_tb21_base.log
runeval 8010 tmax-9b        tb21-tmax terminal-bench/terminal-bench-2-1 3 /home/yichuan/tmax/eval_tb21_tmax.log
sleep 30; wait_done tb21-base tb21-tmax 267
echo "$(date '+%T') === TB-2.1 done -> TB-Pro (k=3) ===" >> "$QL"
runeval 8009 qwen35-9b-base tbpro-base terminal-bench-pro/terminal-bench-pro 3 /home/yichuan/tmax/eval_tbpro_base.log
runeval 8010 tmax-9b        tbpro-tmax terminal-bench-pro/terminal-bench-pro 3 /home/yichuan/tmax/eval_tbpro_tmax.log
sleep 30; wait_done tbpro-base tbpro-tmax 600
echo "$(date '+%T') === ALL QUEUE DONE (TB-2.1 + TB-Pro) ===" >> "$QL"
