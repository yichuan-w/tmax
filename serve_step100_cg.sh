#!/bin/bash
# step-100 serve, CUDA-graph FAST variant: graph only pure-decode (FULL_DECODE_ONLY),
# run prefill eager (the prefill GDN path is what crashed under FULL_AND_PIECEWISE).
# Goal: recover most of the cudagraph speed while staying stable. Log -> /dev/shm (spare /home quota).
cd /home/yichuan/tmax/training/open-instruct
export CUDA_VISIBLE_DEVICES=2,3,4,5
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
export VLLM_USE_V1=1
export TRITON_CACHE_DIR=/home/yichuan/.cache/triton
export VLLM_CACHE_ROOT=/home/yichuan/.cache/vllm
LOG=/dev/shm/serve_step100_cg.log

.venv/bin/vllm serve /dev/shm/ckpt_step100_hf \
    --served-model-name step100 \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 40960 \
    --max-num-batched-tokens 8192 \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --gdn-prefill-backend triton \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --port 8013 >> "$LOG" 2>&1
