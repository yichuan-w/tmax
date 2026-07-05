#!/bin/bash
# Serve allenai/tmax-9b the AUTHORS' way: fresh isolated vllm==0.19.1 env via uvx
# (its pinned transformers handles the Qwen3.5 text-only serve, unlike our training env).
export CUDA_VISIBLE_DEVICES=4,5
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_USE_V1=1
export TRITON_CACHE_DIR=/home/yichuan/.cache/triton
export VLLM_CACHE_ROOT=/home/yichuan/.cache/vllm
LOG=/home/yichuan/tmax/serve_tmax9b_uvx.log

# tmax-9b lacks preprocessor_config.json (it's a text-only RL checkpoint of a multimodal
# Qwen3.5 config). Copy the image/video processor from the base so it loads, and cap
# multimodal inputs to 0 so the (weightless) vision tower isn't exercised.
BASE=$(ls -d /home/yichuan/.cache/huggingface/hub/models--hamishivi--Qwen3.5-9B/snapshots/*/ | head -1)
TMAX=$(ls -d /home/yichuan/.cache/huggingface/hub/models--allenai--tmax-9b/snapshots/*/ | head -1)
cp -L "$BASE/preprocessor_config.json" "$TMAX/preprocessor_config.json" 2>/dev/null
cp -L "$BASE/video_preprocessor_config.json" "$TMAX/video_preprocessor_config.json" 2>/dev/null

uvx vllm==0.19.1 serve allenai/tmax-9b \
    --served-model-name tmax-9b \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 40960 \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --port 8009 > "$LOG" 2>&1
