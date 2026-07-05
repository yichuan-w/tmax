#!/bin/bash
# Low-concurrency TUA run (n-concurrent 4/model = 8 sandboxes) to coexist with the
# co-tenant titan_swe_r2e RL run on the shared Daytona key without saturating quota.
# All sandboxes tagged tmaxeval=1 (via eval_harbor.py); a daemon deletes ONLY our
# orphans. Fixes in wrapper: S3 proxy, resource clamp cpu<=4/disk<=10, --yes,
# --agent-setup-timeout-multiplier 7.  kill patterns match only eval job-names.
cd /home/yichuan/tmax
export DAYTONA_API_KEY='dtn_5bfd6b7bf4ae20197feda35ccd2c5816a7aeea705de8d1aa9e93ec4ad4a72d42'
export https_proxy=http://fwdproxy:8080 HTTPS_PROXY=http://fwdproxy:8080 HF_HUB_ENABLE_HF_TRANSFER=1
QL=/home/yichuan/tmax/tua_lowconc.log
TUA=/home/yichuan/TUA-Bench/tasks
RETRY="--max-retries 3 --retry-include AgentTimeoutError --retry-include DaytonaNotFoundError --retry-include DaytonaAuthenticationError --retry-include DaytonaError --retry-include VerifierTimeoutError"
pyget(){ .venv/bin/python -c "import json;print(json.load(open('jobs/$1/result.json'))['stats']['n_completed_trials'])" 2>/dev/null; }
heal(){ for p in 8009 8010; do u=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:$p/v1/models); if [ "$u" != 200 ] && ! pgrep -f "port $p">/dev/null; then
  [ $p = 8009 ] && setsid bash /home/yichuan/tmax/serve_qwen35_9b_base.sh >/dev/null 2>&1 </dev/null & [ $p = 8010 ] && setsid bash /home/yichuan/tmax/serve_tmax9b.sh >/dev/null 2>&1 </dev/null & fi; done; }
runeval(){ # port model job
  setsid /home/yichuan/tmax/.venv/bin/python eval_harbor.py run -p "$TUA" --env daytona --yes \
    --agent-import-path Vanillux2Agent:Vanillux2Agent --model openai/$2 --agent-kwarg api_base=http://localhost:$1/v1 \
    --agent-kwarg max_format_errors=64 --n-concurrent 4 -k 1 --agent-setup-timeout-multiplier 7 $RETRY --job-name $3 \
    > /home/yichuan/tmax/eval_$3.log 2>&1 < /dev/null & }

# clean any prior TUA smokes/runs (ours only) + stale result dirs
pkill -9 -f "job-name tua-smoke" 2>/dev/null; pkill -9 -f "job-name tua-setupsmoke" 2>/dev/null
pkill -9 -f "job-name tua-lbltest" 2>/dev/null; pkill -9 -f "job-name tua-tmax" 2>/dev/null; pkill -9 -f "job-name tua-base" 2>/dev/null
pkill -9 -f "tua_daemon.py" 2>/dev/null
sleep 5
rm -rf jobs/tua-tmax jobs/tua-base 2>/dev/null

# start orphan-cleanup daemon (deletes ONLY tmaxeval=1 orphans)
setsid /home/yichuan/tmax/.venv/bin/python tua_daemon.py > /home/yichuan/tmax/tua_daemon.log 2>&1 < /dev/null &
echo "$(date '+%F %T') started tua_daemon + low-conc TUA (n-concurrent 4)" > "$QL"

runeval 8010 tmax-9b        tua-tmax
echo "$(date '+%T') launched tua-tmax" >> "$QL"
sleep 45   # stagger to avoid shared harbor-cache extraction race
runeval 8009 qwen35-9b-base tua-base
echo "$(date '+%T') launched tua-base" >> "$QL"

# stall-aware wait: done when both==120 OR both>=116 & stalled 10 checks (~15min)
last=-1; stall=0
for i in $(seq 1 20000); do heal; a=$(pyget tua-tmax); b=$(pyget tua-base); a=${a:-0}; b=${b:-0}; sum=$((a+b))
  if [ "$sum" = "$last" ]; then stall=$((stall+1)); else stall=0; last=$sum; fi
  echo "$(date '+%T') wait tua-tmax=$a/120 tua-base=$b/120 stall=$stall" >> "$QL"
  { [ "$a" -ge 120 ] && [ "$b" -ge 120 ]; } && { echo "$(date '+%T') === TUA DONE ===" >> "$QL"; break; }
  { [ "$a" -ge 116 ] && [ "$b" -ge 116 ] && [ "$stall" -ge 10 ]; } && { echo "$(date '+%T') === TUA near-complete+stalled DONE ===" >> "$QL"; break; }
  sleep 90; done
