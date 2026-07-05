#!/bin/bash
cd /home/yichuan/tmax/training/open-instruct
export CUDA_VISIBLE_DEVICES=2,3
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
export VLLM_USE_V1=1 TRITON_CACHE_DIR=/home/yichuan/.cache/triton VLLM_CACHE_ROOT=/home/yichuan/.cache/vllm
LOG=/home/yichuan/tmax/serve_qwen35_9b_base.log
MODEL_DIR=$(ls -d /home/yichuan/.cache/huggingface/hub/models--hamishivi--Qwen3.5-9B/snapshots/*/ | head -1)
.venv/bin/vllm serve "$MODEL_DIR" \
    --served-model-name qwen35-9b-base \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --tensor-parallel-size 2 --gpu-memory-utilization 0.9 --max-model-len 65536 \
    --limit-mm-per-prompt '{"image":0,"video":0}' --gdn-prefill-backend triton --enforce-eager \
    --port 8009 > "$LOG" 2>&1
