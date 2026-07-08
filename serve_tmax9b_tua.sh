#!/bin/bash
# Serve allenai/tmax-9b (Qwen3.5-9B, GatedDeltaNet hybrid) with the TRAINING venv vLLM
# (has fla + causal_conv1d built with cuda-12.9, so GDN kernels work — uvx's fresh env
# can't build them). Two fixes for the text-only RL checkpoint of a multimodal config:
#   1) copy image/video processor from the base (repo lacks preprocessor_config.json)
#   2) --limit-mm-per-prompt 0  -> vLLM runs text-only, skips the (weightless) vision tower
cd /home/yichuan/tmax/training/open-instruct
export CUDA_VISIBLE_DEVICES=4,5
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
export VLLM_USE_V1=1
export TRITON_CACHE_DIR=/home/yichuan/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/home/yichuan/.cache/torchinductor
export VLLM_CACHE_ROOT=/home/yichuan/.cache/vllm
LOG=/dev/shm/serve_tmax9b_tua.log

BASE=$(ls -d /home/yichuan/.cache/huggingface/hub/models--hamishivi--Qwen3.5-9B/snapshots/*/ | head -1)
MODEL_DIR=$(ls -d /home/yichuan/.cache/huggingface/hub/models--allenai--tmax-9b/snapshots/*/ | head -1)
cp -L "$BASE/preprocessor_config.json" "$MODEL_DIR/preprocessor_config.json" 2>/dev/null
cp -L "$BASE/video_preprocessor_config.json" "$MODEL_DIR/video_preprocessor_config.json" 2>/dev/null

.venv/bin/vllm serve "$MODEL_DIR" \
    --served-model-name tmax-9b \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 65536 \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --gdn-prefill-backend triton \
    --enforce-eager \
    --port 8016 > "$LOG" 2>&1
