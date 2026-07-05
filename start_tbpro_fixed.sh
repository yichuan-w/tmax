#!/bin/bash
# Start the patched (S3-proxy) TB-Pro runs. All old tbpro evals are already dead.
# NOTE: kill patterns match ONLY eval job-names, never this script's own filename
# (the previous script self-killed via a 'relaunch_tbpro' pattern that matched itself).
cd /home/yichuan/tmax
export DAYTONA_API_KEY='dtn_5bfd6b7bf4ae20197feda35ccd2c5816a7aeea705de8d1aa9e93ec4ad4a72d42'
export https_proxy=http://fwdproxy:8080 HTTPS_PROXY=http://fwdproxy:8080 HF_HUB_ENABLE_HF_TRANSFER=1
RETRY="--max-retries 3 --retry-include AgentTimeoutError --retry-include DaytonaNotFoundError --retry-include DaytonaAuthenticationError --retry-include DaytonaError --retry-include VerifierTimeoutError"
runeval(){ # port model job log
  setsid /home/yichuan/tmax/.venv/bin/python eval_harbor.py run --dataset terminal-bench-pro/terminal-bench-pro --env daytona \
    --agent-import-path Vanillux2Agent:Vanillux2Agent --model openai/$2 --agent-kwarg api_base=http://localhost:$1/v1 \
    --agent-kwarg max_format_errors=64 --n-concurrent 16 -k 3 $RETRY --job-name $3 > "$4" 2>&1 < /dev/null & }

# insurance kill (job-name patterns only; procs should already be gone)
pkill -9 -f "eval_harbor.py run --dataset terminal-bench-pro" 2>/dev/null
sleep 4
# wipe polluted result dirs for a clean patched run
rm -rf jobs/tbpro-base jobs/tbpro-tmax jobs/tbpro-smoke 2>/dev/null

runeval 8010 tmax-9b tbpro-tmax /home/yichuan/tmax/eval_tbpro_tmax.log
echo "$(date '+%T') PATCHED start tbpro-tmax" >> /home/yichuan/tmax/bench_queue2.log
sleep 40
runeval 8009 qwen35-9b-base tbpro-base /home/yichuan/tmax/eval_tbpro_base.log
echo "$(date '+%T') PATCHED start tbpro-base" >> /home/yichuan/tmax/bench_queue2.log
