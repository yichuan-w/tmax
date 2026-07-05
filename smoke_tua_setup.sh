#!/bin/bash
# Kill the failing TUA run (all trials hit the 360s agent-setup timeout because heavy
# TUA image builds bleed into that window), then smoke ONE task with a bumped
# --agent-setup-timeout-multiplier to confirm it reaches agent execution (Step 1/).
cd /home/yichuan/tmax
export DAYTONA_API_KEY='dtn_5bfd6b7bf4ae20197feda35ccd2c5816a7aeea705de8d1aa9e93ec4ad4a72d42'
export https_proxy=http://fwdproxy:8080 HTTPS_PROXY=http://fwdproxy:8080
pkill -9 -f "eval_harbor.py run -p /home/yichuan/TUA-Bench" 2>/dev/null
pkill -9 -f "start_tua.sh" 2>/dev/null
sleep 5
rm -rf jobs/tua-setupsmoke 2>/dev/null
# task 016-count-invoice-pivot = lightweight data task (pandas), fast build
setsid /home/yichuan/tmax/.venv/bin/python eval_harbor.py run -p /home/yichuan/TUA-Bench/tasks --env daytona --yes \
  --agent-import-path Vanillux2Agent:Vanillux2Agent --model openai/tmax-9b --agent-kwarg api_base=http://localhost:8010/v1 \
  --agent-kwarg max_format_errors=64 -i local/016-count-invoice-pivot -k 1 \
  --agent-setup-timeout-multiplier 7 \
  --job-name tua-setupsmoke > /home/yichuan/tmax/eval_tua_setupsmoke.log 2>&1 < /dev/null &
echo "$(date '+%T') tua setup smoke launched (multiplier 7)" >> /home/yichuan/tmax/tua_queue.log
