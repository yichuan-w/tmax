#!/bin/bash
# Decisive clean restart of TB-Pro with the patched (S3-proxy) wrapper.
# The previous pkill (SIGTERM) left the old unpatched tbpro-base alive, racing the
# fresh run's result.json. Here: SIGKILL every tbpro eval until none remain, wipe the
# polluted result dirs, then relaunch both patched runs staggered.
cd /home/yichuan/tmax
export DAYTONA_API_KEY='dtn_5bfd6b7bf4ae20197feda35ccd2c5816a7aeea705de8d1aa9e93ec4ad4a72d42'
export https_proxy=http://fwdproxy:8080 HTTPS_PROXY=http://fwdproxy:8080 HF_HUB_ENABLE_HF_TRANSFER=1
RETRY="--max-retries 3 --retry-include AgentTimeoutError --retry-include DaytonaNotFoundError --retry-include DaytonaAuthenticationError --retry-include DaytonaError --retry-include VerifierTimeoutError"
runeval(){ # port model job log
  setsid /home/yichuan/tmax/.venv/bin/python eval_harbor.py run --dataset terminal-bench-pro/terminal-bench-pro --env daytona \
    --agent-import-path Vanillux2Agent:Vanillux2Agent --model openai/$2 --agent-kwarg api_base=http://localhost:$1/v1 \
    --agent-kwarg max_format_errors=64 --n-concurrent 16 -k 3 $RETRY --job-name $3 > "$4" 2>&1 < /dev/null & }

# 1. SIGKILL all tbpro evals until confirmed gone
for i in 1 2 3 4 5 6; do
  pkill -9 -f "job-name tbpro-base" 2>/dev/null
  pkill -9 -f "job-name tbpro-tmax" 2>/dev/null
  pkill -9 -f "job-name tbpro-smoke" 2>/dev/null
  pkill -9 -f "relaunch_tbpro" 2>/dev/null
  sleep 3
  n=$(pgrep -f "job-name tbpro-" | wc -l)
  [ "$n" = 0 ] && break
done
echo "$(date '+%T') killed all tbpro evals (remaining=$(pgrep -f 'job-name tbpro-'|wc -l))" >> /home/yichuan/tmax/bench_queue2.log

# 2. wipe polluted result dirs so fresh patched runs start clean
rm -rf jobs/tbpro-base jobs/tbpro-tmax jobs/tbpro-smoke 2>/dev/null

# 3. relaunch both patched, staggered (avoid shared-cache rmdir race)
runeval 8010 tmax-9b tbpro-tmax /home/yichuan/tmax/eval_tbpro_tmax.log
echo "$(date '+%T') CLEAN relaunched tbpro-tmax (patched)" >> /home/yichuan/tmax/bench_queue2.log
sleep 40
runeval 8009 qwen35-9b-base tbpro-base /home/yichuan/tmax/eval_tbpro_base.log
echo "$(date '+%T') CLEAN relaunched tbpro-base (patched)" >> /home/yichuan/tmax/bench_queue2.log
