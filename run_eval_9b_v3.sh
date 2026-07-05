#!/bin/bash
cd /home/yichuan/tmax
export DAYTONA_API_KEY='dtn_5bfd6b7bf4ae20197feda35ccd2c5816a7aeea705de8d1aa9e93ec4ad4a72d42'
export https_proxy=http://fwdproxy:8080 HTTPS_PROXY=http://fwdproxy:8080 HF_HUB_ENABLE_HF_TRANSFER=1
# v3: FAST serve (cudagraph, ~5x) on :8010 + 40k + retry timeouts AND Daytona infra errors up to 3x
.venv/bin/python eval_harbor.py run \
  --dataset terminal-bench@2.0 --env daytona \
  --agent-import-path Vanillux2Agent:Vanillux2Agent \
  --model openai/tmax-9b-fast --agent-kwarg api_base=http://localhost:8010/v1 \
  --agent-kwarg max_format_errors=64 --n-concurrent 4 -k 1 \
  --max-retries 3 \
  --retry-include AgentTimeoutError --retry-include DaytonaNotFoundError \
  --retry-include DaytonaAuthenticationError --retry-include DaytonaError \
  --retry-include VerifierTimeoutError \
  --job-name tmax9b-tb2-v3-fast > /home/yichuan/tmax/eval_tb_9b_v3.log 2>&1
echo "=== v3 done rc=$? ===" >> /home/yichuan/tmax/eval_tb_9b_v3.log
