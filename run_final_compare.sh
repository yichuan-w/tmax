#!/bin/bash
# Clean matched comparison: Qwen3.5-9B base vs tmax-9b (RL), BOTH on cudagraph serves,
# identical eval config (TB-2.0, 40960 ctx via serve max-len, k=1, concurrency 4, retry
# timeouts+Daytona x3). Self-contained + self-healing. Logs to final_compare.log.
OI=/home/yichuan/tmax/training/open-instruct
CL=/home/yichuan/tmax/final_compare.log
BASE=$(ls -d /home/yichuan/.cache/huggingface/hub/models--hamishivi--Qwen3.5-9B/snapshots/*/ | head -1)
TMAX=$(ls -d /home/yichuan/.cache/huggingface/hub/models--allenai--tmax-9b/snapshots/*/ | head -1)
export DAYTONA_API_KEY='dtn_5bfd6b7bf4ae20197feda35ccd2c5816a7aeea705de8d1aa9e93ec4ad4a72d42'
export https_proxy=http://fwdproxy:8080 HTTPS_PROXY=http://fwdproxy:8080 HF_HUB_ENABLE_HF_TRANSFER=1

echo "$(date '+%F %T') === cleanup ===" > "$CL"
pkill -9 -f eval_harbor.py 2>/dev/null; pkill -9 -f "vllm serve" 2>/dev/null
pkill -9 -f "watchdog" 2>/dev/null; pkill -9 -f "chain_base" 2>/dev/null; pkill -9 -f "run_eval_9b" 2>/dev/null
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u); do kill -9 "$pid" 2>/dev/null; done
sleep 10

# ensure preprocessor present for tmax-9b (text-only serve of multimodal config)
cp -L "$BASE/preprocessor_config.json" "$TMAX/preprocessor_config.json" 2>/dev/null
cp -L "$BASE/video_preprocessor_config.json" "$TMAX/video_preprocessor_config.json" 2>/dev/null

serve() { # $1=gpus $2=port $3=name $4=model_dir $5=log
  cd "$OI"
  CUDA_VISIBLE_DEVICES=$1 CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH \
  VLLM_USE_V1=1 TRITON_CACHE_DIR=/home/yichuan/.cache/triton VLLM_CACHE_ROOT=/home/yichuan/.cache/vllm \
  setsid "$OI/.venv/bin/vllm" serve "$4" --served-model-name "$3" \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --tensor-parallel-size 2 --gpu-memory-utilization 0.9 --max-model-len 40960 \
    --limit-mm-per-prompt '{"image":0,"video":0}' --gdn-prefill-backend triton \
    --port "$2" > "$5" 2>&1 < /dev/null &
}
echo "$(date '+%T') launching cudagraph serves (base:8009 gpu2,3 | tmax:8010 gpu4,5)" >> "$CL"
serve 2,3 8009 qwen35-9b-base "$BASE" /home/yichuan/tmax/serve_base_final.log
serve 4,5 8010 tmax-9b        "$TMAX" /home/yichuan/tmax/serve_tmax_final.log

# wait both ready (<=12min: cudagraph capture)
for i in $(seq 1 72); do
  a=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8009/v1/models)
  b=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8010/v1/models)
  [ "$a" = "200" ] && [ "$b" = "200" ] && { echo "$(date '+%T') both serves ready" >> "$CL"; break; }
  sleep 10
done
sleep 10

RETRY="--max-retries 3 --retry-include AgentTimeoutError --retry-include DaytonaNotFoundError --retry-include DaytonaAuthenticationError --retry-include DaytonaError --retry-include VerifierTimeoutError"
runeval() { # $1=port $2=model $3=job $4=log
  cd /home/yichuan/tmax
  setsid "/home/yichuan/tmax/.venv/bin/python" eval_harbor.py run \
    --dataset terminal-bench@2.0 --env daytona \
    --agent-import-path Vanillux2Agent:Vanillux2Agent \
    --model openai/$2 --agent-kwarg api_base=http://localhost:$1/v1 \
    --agent-kwarg max_format_errors=64 --n-concurrent 4 -k 1 $RETRY \
    --job-name $3 > "$4" 2>&1 < /dev/null &
}
echo "$(date '+%T') launching both evals" >> "$CL"
runeval 8009 qwen35-9b-base base-final     /home/yichuan/tmax/eval_base_final.log
runeval 8010 tmax-9b        tmax9b-final   /home/yichuan/tmax/eval_tmax_final.log

# monitor + self-heal
prog() { .venv/bin/python -c "
import json
try:
  d=json.load(open('jobs/$1/result.json')); s=d['stats']; ev=list(s['evals'].values())
  m=ev[0]['metrics'][0]['mean'] if ev and ev[0].get('metrics') else None
  print(f\"$1 done={s['n_completed_trials']}/89 err={s['n_errored_trials']} mean={m}\")
except Exception: print('$1 noresult')" 2>&1; }
cd /home/yichuan/tmax
for i in $(seq 1 400); do
  a=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8009/v1/models)
  b=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8010/v1/models)
  [ "$a" != "200" ] && ! pgrep -f "port 8009" >/dev/null && { echo "$(date '+%T') !! base serve down->relaunch" >>"$CL"; serve 2,3 8009 qwen35-9b-base "$BASE" /home/yichuan/tmax/serve_base_final.log; }
  [ "$b" != "200" ] && ! pgrep -f "port 8010" >/dev/null && { echo "$(date '+%T') !! tmax serve down->relaunch" >>"$CL"; serve 4,5 8010 tmax-9b "$TMAX" /home/yichuan/tmax/serve_tmax_final.log; }
  echo "$(date '+%T') $(prog base-final) | $(prog tmax9b-final)" >> "$CL"
  db=$(.venv/bin/python -c "import json;print(json.load(open('jobs/base-final/result.json'))['stats']['n_completed_trials'])" 2>/dev/null)
  dt=$(.venv/bin/python -c "import json;print(json.load(open('jobs/tmax9b-final/result.json'))['stats']['n_completed_trials'])" 2>/dev/null)
  [ "$db" = "89" ] && [ "$dt" = "89" ] && { echo "$(date '+%T') === BOTH COMPLETE ===" >> "$CL"; break; }
  sleep 150
done
