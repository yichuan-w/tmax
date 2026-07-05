#!/bin/bash
cd /home/yichuan/tmax
export DAYTONA_API_KEY='dtn_5bfd6b7bf4ae20197feda35ccd2c5816a7aeea705de8d1aa9e93ec4ad4a72d42'
export https_proxy=http://fwdproxy:8080 HTTPS_PROXY=http://fwdproxy:8080
export HF_HUB_ENABLE_HF_TRANSFER=1
# v2: 65k context (serve) + concurrency 4 + retry timeouts up to 3x (paper "restart timed-out runs up to 3x")
.venv/bin/python eval_harbor.py run \
  --dataset terminal-bench@2.0 \
  --env daytona \
  --agent-import-path Vanillux2Agent:Vanillux2Agent \
  --model openai/tmax-9b \
  --agent-kwarg api_base=http://localhost:8009/v1 \
  --agent-kwarg max_format_errors=64 \
  --n-concurrent 4 \
  -k 1 \
  --max-retries 3 --retry-include AgentTimeoutError \
  --job-name tmax9b-tb2-v2 > /home/yichuan/tmax/eval_tb_9b_v2.log 2>&1
echo "=== v2 eval done rc=$? ===" >> /home/yichuan/tmax/eval_tb_9b_v2.log
