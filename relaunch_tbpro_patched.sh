#!/bin/bash
# The first TB-Pro runs failed ~85% because TB-Pro tasks build from Dockerfile and the
# daytona SDK's obstore S3Store (Rust) upload of the build context ignored https_proxy
# (+ isolated_env cleared env) -> S3 egress blocked -> "Sandbox not found".
# eval_harbor.py now injects client_options.proxy_url into S3Store (VERIFIED: smoke task
# built + ran). Relaunch both full TB-Pro runs with the patched wrapper. Stagger start
# so they don't race the shared harbor cache rmdir.
cd /home/yichuan/tmax
export DAYTONA_API_KEY='dtn_5bfd6b7bf4ae20197feda35ccd2c5816a7aeea705de8d1aa9e93ec4ad4a72d42'
export https_proxy=http://fwdproxy:8080 HTTPS_PROXY=http://fwdproxy:8080 HF_HUB_ENABLE_HF_TRANSFER=1
RETRY="--max-retries 3 --retry-include AgentTimeoutError --retry-include DaytonaNotFoundError --retry-include DaytonaAuthenticationError --retry-include DaytonaError --retry-include VerifierTimeoutError"
runeval(){ # port model job log
  setsid /home/yichuan/tmax/.venv/bin/python eval_harbor.py run --dataset terminal-bench-pro/terminal-bench-pro --env daytona \
    --agent-import-path Vanillux2Agent:Vanillux2Agent --model openai/$2 --agent-kwarg api_base=http://localhost:$1/v1 \
    --agent-kwarg max_format_errors=64 --n-concurrent 16 -k 3 $RETRY --job-name $3 > "$4" 2>&1 < /dev/null & }
pkill -f "job-name tbpro-base" 2>/dev/null
pkill -f "job-name tbpro-tmax" 2>/dev/null
pkill -f "job-name tbpro-smoke" 2>/dev/null
sleep 8
runeval 8010 tmax-9b tbpro-tmax /home/yichuan/tmax/eval_tbpro_tmax.log
echo "$(date '+%T') relaunched tbpro-tmax (patched S3 proxy)" >> /home/yichuan/tmax/bench_queue2.log
sleep 35   # let tmax pass the upfront cache extraction before base joins
runeval 8009 qwen35-9b-base tbpro-base /home/yichuan/tmax/eval_tbpro_base.log
echo "$(date '+%T') relaunched tbpro-base (patched S3 proxy)" >> /home/yichuan/tmax/bench_queue2.log
