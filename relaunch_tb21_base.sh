#!/bin/bash
# tb21-base crashed on a task-package extraction RACE (both evals unpacked the shared
# ~/.cache/harbor cache at once; tmax won, base hit a half-written pdb_ids.txt).
# tmax has now fully populated the cache -> relaunch base cleanly on 8009.
cd /home/yichuan/tmax
export DAYTONA_API_KEY='dtn_5bfd6b7bf4ae20197feda35ccd2c5816a7aeea705de8d1aa9e93ec4ad4a72d42'
export https_proxy=http://fwdproxy:8080 HTTPS_PROXY=http://fwdproxy:8080 HF_HUB_ENABLE_HF_TRANSFER=1
RETRY="--max-retries 3 --retry-include AgentTimeoutError --retry-include DaytonaNotFoundError --retry-include DaytonaAuthenticationError --retry-include DaytonaError --retry-include VerifierTimeoutError"
pkill -f "job-name tb21-base" 2>/dev/null
sleep 6
setsid /home/yichuan/tmax/.venv/bin/python eval_harbor.py run --dataset terminal-bench/terminal-bench-2-1 --env daytona \
  --agent-import-path Vanillux2Agent:Vanillux2Agent --model openai/qwen35-9b-base --agent-kwarg api_base=http://localhost:8009/v1 \
  --agent-kwarg max_format_errors=64 --n-concurrent 16 -k 3 $RETRY --job-name tb21-base \
  > /home/yichuan/tmax/eval_tb21_base.log 2>&1 < /dev/null &
echo "$(date '+%T') relaunched tb21-base on 8009" >> /home/yichuan/tmax/bench_queue2.log
