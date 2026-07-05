#!/bin/bash
# tbpro-base crashed on the SAME startup cache race as tb21-base (harbor unpacks ALL
# task packages upfront into shared ~/.cache/harbor; both evals raced -> base hit
# "Directory not empty" on a dir tmax was writing). tmax is now past extraction
# (running trials) -> relaunch base; it will read the populated cache, no re-race.
cd /home/yichuan/tmax
export DAYTONA_API_KEY='dtn_5bfd6b7bf4ae20197feda35ccd2c5816a7aeea705de8d1aa9e93ec4ad4a72d42'
export https_proxy=http://fwdproxy:8080 HTTPS_PROXY=http://fwdproxy:8080 HF_HUB_ENABLE_HF_TRANSFER=1
RETRY="--max-retries 3 --retry-include AgentTimeoutError --retry-include DaytonaNotFoundError --retry-include DaytonaAuthenticationError --retry-include DaytonaError --retry-include VerifierTimeoutError"
pkill -f "job-name tbpro-base" 2>/dev/null
sleep 6
setsid /home/yichuan/tmax/.venv/bin/python eval_harbor.py run --dataset terminal-bench-pro/terminal-bench-pro --env daytona \
  --agent-import-path Vanillux2Agent:Vanillux2Agent --model openai/qwen35-9b-base --agent-kwarg api_base=http://localhost:8009/v1 \
  --agent-kwarg max_format_errors=64 --n-concurrent 16 -k 3 $RETRY --job-name tbpro-base \
  > /home/yichuan/tmax/eval_tbpro_base.log 2>&1 < /dev/null &
echo "$(date '+%T') relaunched tbpro-base on 8009 (cache race fix)" >> /home/yichuan/tmax/bench_queue2.log
