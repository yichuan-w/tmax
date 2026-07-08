#!/bin/bash
cd /home/yichuan/tmax
export DAYTONA_API_KEY='dtn_5bfd6b7bf4ae20197feda35ccd2c5816a7aeea705de8d1aa9e93ec4ad4a72d42'
export https_proxy=http://fwdproxy:8080 HTTPS_PROXY=http://fwdproxy:8080 HF_HUB_ENABLE_HF_TRANSFER=1
rm -rf jobs/tua-base 2>/dev/null
exec .venv/bin/python eval_harbor.py run -p /home/yichuan/TUA-Bench/tasks --env daytona --yes --force-build \
  --agent-import-path Vanillux2Agent:Vanillux2Agent --model openai/qwen35-9b-base --agent-kwarg api_base=http://localhost:8017/v1 \
  --agent-kwarg max_format_errors=64 --n-concurrent 8 -k 1 --agent-setup-timeout-multiplier 7 \
  --max-retries 3 --retry-include AgentTimeoutError --retry-include DaytonaError --retry-include DaytonaNotFoundError --retry-include VerifierTimeoutError \
  --job-name tua-base > /dev/shm/eval_tua-base.log 2>&1
