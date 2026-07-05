#!/bin/bash
# Serve Qwen/Qwen3.6-27B (the tmax-27b RL base). Same Qwen3_5ForConditionalGeneration
# multimodal arch as the 9B, so same recipe: text-only (limit-mm 0) + triton GDN prefill
# + qwen3_xml tool parser + cudagraph. TP=2 on GPU 0,1, port 8011.
cd /home/yichuan/tmax/training/open-instruct
export CUDA_VISIBLE_DEVICES=0,1
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
export VLLM_USE_V1=1
export TRITON_CACHE_DIR=/home/yichuan/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/home/yichuan/.cache/torchinductor
export VLLM_CACHE_ROOT=/home/yichuan/.cache/vllm
LOG=/home/yichuan/tmax/serve_qwen36_27b_base.log
MODEL_DIR=$(ls -d /home/yichuan/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/*/ | head -1)

.venv/bin/vllm serve "$MODEL_DIR" \
    --served-model-name qwen36-27b-base \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 40960 \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --gdn-prefill-backend triton \
    --port 8011 > "$LOG" 2>&1
