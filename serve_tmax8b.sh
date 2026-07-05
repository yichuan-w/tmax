#!/bin/bash
# Serve hamishivi/sft_qwen3_8b_our_tmax_sft (dense Qwen3-8B, tmax SFT) with vLLM for Harbor eval.
cd /home/yichuan/tmax/training/open-instruct
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
export VLLM_USE_V1=1
export TRITON_CACHE_DIR=/home/yichuan/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/home/yichuan/.cache/torchinductor
export VLLM_CACHE_ROOT=/home/yichuan/.cache/vllm
LOG=/home/yichuan/tmax/serve_tmax8b.log
MODEL_DIR=$(ls -d /home/yichuan/.cache/huggingface/hub/models--hamishivi--sft_qwen3_8b_our_tmax_sft/snapshots/*/ | head -1)

.venv/bin/vllm serve "$MODEL_DIR" \
    --served-model-name tmax-8b \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 40960 \
    --port 8008 > "$LOG" 2>&1
