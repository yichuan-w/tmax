#!/bin/bash
# Serve the DCP->HF converted step-100 RL checkpoint (torchtitan Qwen3.5-9B GDN hybrid).
# Same recipe as tmax-9b/step40: text-only (limit-mm 0) + triton GDN prefill + qwen3_xml.
# TP=2 on GPU 2,3, port 8013.
cd /home/yichuan/tmax/training/open-instruct
export CUDA_VISIBLE_DEVICES=2,3
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
export VLLM_USE_V1=1
export TRITON_CACHE_DIR=/home/yichuan/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/home/yichuan/.cache/torchinductor
export VLLM_CACHE_ROOT=/home/yichuan/.cache/vllm
LOG=/home/yichuan/tmax/serve_step100.log

.venv/bin/vllm serve /home/yichuan/ckpt_step100_hf \
    --served-model-name step100 \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 40960 \
    --max-num-batched-tokens 8192 \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --gdn-prefill-backend triton \
    --enforce-eager \
    --port 8013 >> "$LOG" 2>&1
