#!/bin/bash
# Full TUA-Bench run (120 tasks, k=1) on the existing cudagraph serves, patched wrapper.
# TUA tasks are Dockerfile-build + request cpus=6/disk=30GB -> needs BOTH the S3 build
# -context proxy fix AND the resource clamp (cpu<=4, disk<=10) now in eval_harbor.py.
# Local dataset via -p tasks. Staggered start to avoid the shared harbor-cache race.
# kill patterns match ONLY eval cmdlines, never this script's filename.
cd /home/yichuan/tmax
export DAYTONA_API_KEY='dtn_5bfd6b7bf4ae20197feda35ccd2c5816a7aeea705de8d1aa9e93ec4ad4a72d42'
export https_proxy=http://fwdproxy:8080 HTTPS_PROXY=http://fwdproxy:8080 HF_HUB_ENABLE_HF_TRANSFER=1
QL=/home/yichuan/tmax/tua_queue.log
TUA=/home/yichuan/TUA-Bench/tasks
RETRY="--max-retries 3 --retry-include AgentTimeoutError --retry-include DaytonaNotFoundError --retry-include DaytonaAuthenticationError --retry-include DaytonaError --retry-include VerifierTimeoutError"
pyget(){ .venv/bin/python -c "import json;print(json.load(open('jobs/$1/result.json'))['stats']['n_completed_trials'])" 2>/dev/null; }
heal(){ for p in 8009 8010; do u=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:$p/v1/models); if [ "$u" != 200 ] && ! pgrep -f "port $p">/dev/null; then
  [ $p = 8009 ] && setsid bash /home/yichuan/tmax/serve_qwen35_9b_base.sh >/dev/null 2>&1 </dev/null & [ $p = 8010 ] && setsid bash /home/yichuan/tmax/serve_tmax9b.sh >/dev/null 2>&1 </dev/null & fi; done; }
runeval(){ # port model job
  # --agent-setup-timeout-multiplier 7: TUA's post-build in-sandbox exec (a trivial mkdir)
  #   sometimes hangs ~360s on freshly-built heavy images -> 360*7=2520s window.
  # NOTE: no -o (that + --job-name nests dirs); --job-name alone -> jobs/<name>/result.json.
  setsid /home/yichuan/tmax/.venv/bin/python eval_harbor.py run -p "$TUA" --env daytona --yes \
    --agent-import-path Vanillux2Agent:Vanillux2Agent --model openai/$2 --agent-kwarg api_base=http://localhost:$1/v1 \
    --agent-kwarg max_format_errors=64 --n-concurrent 16 -k 1 --agent-setup-timeout-multiplier 7 $RETRY --job-name $3 \
    > /home/yichuan/tmax/eval_$3.log 2>&1 < /dev/null & }

# clean any smoke/aborted procs + stale result dirs
pkill -9 -f "job-name tua-smoke" 2>/dev/null; pkill -9 -f "job-name tua-setupsmoke" 2>/dev/null; pkill -9 -f "eval_harbor.py run -p /home/yichuan/TUA-Bench" 2>/dev/null; sleep 3
rm -rf jobs/tua-tmax jobs/tua-base 2>/dev/null
echo "$(date '+%F %T') TUA start (120 tasks, k=1)" > "$QL"
runeval 8010 tmax-9b        tua-tmax
echo "$(date '+%T') launched tua-tmax" >> "$QL"
sleep 45   # let tmax finish upfront cache extraction before base joins
runeval 8009 qwen35-9b-base tua-base
echo "$(date '+%T') launched tua-base" >> "$QL"

# stall-aware wait: done when both==120 OR both>=117 & stalled 8 checks
last=-1; stall=0
for i in $(seq 1 4000); do heal; a=$(pyget tua-tmax); b=$(pyget tua-base); a=${a:-0}; b=${b:-0}; sum=$((a+b))
  if [ "$sum" = "$last" ]; then stall=$((stall+1)); else stall=0; last=$sum; fi
  echo "$(date '+%T') wait tua-tmax=$a/120 tua-base=$b/120 stall=$stall" >> "$QL"
  { [ "$a" -ge 120 ] && [ "$b" -ge 120 ]; } && { echo "$(date '+%T') === TUA DONE ===" >> "$QL"; break; }
  { [ "$a" -ge 117 ] && [ "$b" -ge 117 ] && [ "$stall" -ge 8 ]; } && { echo "$(date '+%T') === TUA near-complete+stalled, DONE ===" >> "$QL"; break; }
  sleep 90; done
