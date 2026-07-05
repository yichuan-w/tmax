#!/bin/bash
cd /home/yichuan/tmax
CL=/home/yichuan/tmax/chain_base.log
echo "$(date '+%F %T') chain started, waiting for v2 to finish..." > "$CL"
# 1) wait until v2 completes 89/89 (max ~10h)
for i in $(seq 1 600); do
  D=$(.venv/bin/python -c "import json;print(json.load(open('jobs/tmax9b-tb2-v2/result.json'))['stats']['n_completed_trials'])" 2>/dev/null)
  [ "$D" = "89" ] && { echo "$(date '+%T') v2 done (89)" >> "$CL"; break; }
  sleep 60
done
sleep 20
# 2) swap serve: kill tmax-9b serve, free GPUs, start base serve
pkill -9 -f "vllm serve" 2>/dev/null; pkill -9 -f "watchdog_v2.sh" 2>/dev/null
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null|sort -u); do kill -9 "$pid" 2>/dev/null; done
sleep 8
setsid bash /home/yichuan/tmax/serve_qwen35_9b_base.sh >/dev/null 2>&1 < /dev/null &
# 3) wait for base serve ready
for i in $(seq 1 60); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8009/v1/models)" = "200" ] && { echo "$(date '+%T') base serve ready" >> "$CL"; break; }
  sleep 10
done
sleep 10
# 4) run identical eval on the base
export DAYTONA_API_KEY='dtn_5bfd6b7bf4ae20197feda35ccd2c5816a7aeea705de8d1aa9e93ec4ad4a72d42'
export https_proxy=http://fwdproxy:8080 HTTPS_PROXY=http://fwdproxy:8080 HF_HUB_ENABLE_HF_TRANSFER=1
echo "$(date '+%T') launching base eval" >> "$CL"
.venv/bin/python eval_harbor.py run \
  --dataset terminal-bench@2.0 --env daytona \
  --agent-import-path Vanillux2Agent:Vanillux2Agent \
  --model openai/qwen35-9b-base --agent-kwarg api_base=http://localhost:8009/v1 \
  --agent-kwarg max_format_errors=64 --n-concurrent 4 -k 1 \
  --max-retries 3 --retry-include AgentTimeoutError \
  --job-name qwen35-9b-base-tb2 > /home/yichuan/tmax/eval_tb_base.log 2>&1
echo "$(date '+%T') === base eval done rc=$? ===" >> "$CL"
