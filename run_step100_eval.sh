#!/bin/bash
# Self-healing step-100 eval: keeps the :8013 serve alive (auto-restart on worker death,
# which crashed the first attempt) and runs TB-2.0 k=5. Same avg@k methodology as the
# base/step40/tmax-9b runs for a direct comparison.
cd /home/yichuan/tmax
export DAYTONA_API_KEY='dtn_5bfd6b7bf4ae20197feda35ccd2c5816a7aeea705de8d1aa9e93ec4ad4a72d42'
export https_proxy=http://fwdproxy:8080 HTTPS_PROXY=http://fwdproxy:8080 HF_HUB_ENABLE_HF_TRANSFER=1
QL=/home/yichuan/tmax/run_step100.log
RETRY="--max-retries 3 --retry-include AgentTimeoutError --retry-include DaytonaNotFoundError --retry-include DaytonaAuthenticationError --retry-include DaytonaError --retry-include VerifierTimeoutError"

pyget(){ .venv/bin/python -c "import json;print(json.load(open('jobs/step100-k5/result.json'))['stats']['n_completed_trials'])" 2>/dev/null; }
pytot(){ .venv/bin/python -c "import json;print(json.load(open('jobs/step100-k5/result.json'))['n_total_trials'])" 2>/dev/null; }
perr(){ .venv/bin/python -c "import json;print(json.load(open('jobs/step100-k5/result.json'))['stats']['n_errored_trials'])" 2>/dev/null; }

heal(){ # restart serve if :8013 is down and no vllm process is alive
  u=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8013/v1/models 2>/dev/null)
  if [ "$u" != 200 ] && ! pgrep -f 'ckpt_step100_hf' >/dev/null; then
    echo "$(date '+%T') !! serve 8013 down -> restart" >> "$QL"
    pkill -9 -f 'ckpt_step100_hf' 2>/dev/null; sleep 3
    setsid bash /home/yichuan/tmax/serve_step100.sh >/dev/null 2>&1 </dev/null &
  fi
}

echo "$(date '+%F %T') step100 self-healing eval start" > "$QL"
# ensure serve is up (may already be from serve_step100.sh)
heal
for i in $(seq 1 20); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8013/v1/models)" = 200 ] && break
  sleep 15
done
echo "$(date '+%T') serve up, launching eval" >> "$QL"

rm -rf jobs/step100-k5 2>/dev/null
setsid .venv/bin/python eval_harbor.py run --dataset terminal-bench@2.0 --env daytona --yes \
  --agent-import-path Vanillux2Agent:Vanillux2Agent --model openai/step100 \
  --agent-kwarg api_base=http://localhost:8013/v1 --agent-kwarg max_format_errors=64 \
  --n-concurrent 12 -k 5 $RETRY --job-name step100-k5 > /home/yichuan/tmax/eval_step100_k5.log 2>&1 </dev/null &
sleep 30

last=-1; stall=0
for i in $(seq 1 20000); do
  heal
  d=$(pyget); d=${d:-0}; t=$(pytot); t=${t:-445}; e=$(perr); e=${e:-0}
  if [ "$d" = "$last" ]; then stall=$((stall+1)); else stall=0; last=$d; fi
  echo "$(date '+%T') step100-k5=$d/$t err=$e stall=$stall" >> "$QL"
  { [ "$d" -ge "$t" ] && [ "$t" -gt 0 ]; } && { echo "$(date '+%T') === STEP100 EVAL DONE ===" >> "$QL"; break; }
  { [ "$stall" -ge 12 ] && [ "$d" -ge $((t-4)) ] && [ "$t" -gt 100 ]; } && { echo "$(date '+%T') === STEP100 near-done+stalled ===" >> "$QL"; break; }
  sleep 60
done
