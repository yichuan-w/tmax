#!/bin/bash
# Serve allenai/tmax-27b (RL flagship, Qwen3.6-27B DPPO). Same Qwen3_5ForConditionalGeneration
# multimodal arch as tmax-9b -> same recipe. Copy preprocessor from the base if the RL
# ckpt lacks it (tmax-9b did). TP=2 on GPU 6,7, port 8012, cudagraph.
cd /home/yichuan/tmax/training/open-instruct
export CUDA_VISIBLE_DEVICES=6,7
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
export VLLM_USE_V1=1
export TRITON_CACHE_DIR=/home/yichuan/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/home/yichuan/.cache/torchinductor
export VLLM_CACHE_ROOT=/home/yichuan/.cache/vllm
LOG=/home/yichuan/tmax/serve_tmax27b.log

BASE=$(ls -d /home/yichuan/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/*/ | head -1)
MODEL_DIR=$(ls -d /home/yichuan/.cache/huggingface/hub/models--allenai--tmax-27b/snapshots/*/ | head -1)
# RL ckpt may lack the image/video processor; copy from base so vLLM loads (text-only mode anyway)
[ -f "$MODEL_DIR/preprocessor_config.json" ] || cp -L "$BASE/preprocessor_config.json" "$MODEL_DIR/preprocessor_config.json" 2>/dev/null
[ -f "$MODEL_DIR/video_preprocessor_config.json" ] || cp -L "$BASE/video_preprocessor_config.json" "$MODEL_DIR/video_preprocessor_config.json" 2>/dev/null

.venv/bin/vllm serve "$MODEL_DIR" \
    --served-model-name tmax-27b \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 40960 \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --gdn-prefill-backend triton \
    --port 8012 > "$LOG" 2>&1
