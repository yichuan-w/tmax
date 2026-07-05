#!/bin/bash
cd /home/yichuan/tmax/training/open-instruct
export CUDA_VISIBLE_DEVICES=4,5
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
export VLLM_USE_V1=1 TRITON_CACHE_DIR=/home/yichuan/.cache/triton VLLM_CACHE_ROOT=/home/yichuan/.cache/vllm
LOG=/home/yichuan/tmax/serve_tmax9b_fast.log
MODEL_DIR=$(ls -d /home/yichuan/.cache/huggingface/hub/models--allenai--tmax-9b/snapshots/*/ | head -1)
# NO --enforce-eager -> vLLM captures CUDA graphs (faster decode). Keep GDN triton + text-only.
.venv/bin/vllm serve "$MODEL_DIR" \
    --served-model-name tmax-9b-fast \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --tensor-parallel-size 2 --gpu-memory-utilization 0.9 --max-model-len 40960 \
    --limit-mm-per-prompt '{"image":0,"video":0}' --gdn-prefill-backend triton \
    --port 8010 > "$LOG" 2>&1
