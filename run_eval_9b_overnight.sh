#!/bin/bash
# Free the idle 8b serve GPUs (0,1); keep the tmax-9b serve (GPU 2,3, port 8009) up.
pkill -9 -f "serve_tmax8b\|sft_qwen3_8b_our_tmax_sft" 2>/dev/null
sleep 4
cd /home/yichuan/tmax
export DAYTONA_API_KEY='dtn_5bfd6b7bf4ae20197feda35ccd2c5816a7aeea705de8d1aa9e93ec4ad4a72d42'
export https_proxy=http://fwdproxy:8080 HTTPS_PROXY=http://fwdproxy:8080
export HF_HUB_ENABLE_HF_TRANSFER=1
# Full Terminal-Bench 2.0 (headline benchmark; paper tmax-9b = 27.2%), k=1, 8 concurrent.
.venv/bin/python eval_harbor.py run \
  --dataset terminal-bench@2.0 \
  --env daytona \
  --agent-import-path Vanillux2Agent:Vanillux2Agent \
  --model openai/tmax-9b \
  --agent-kwarg api_base=http://localhost:8009/v1 \
  --agent-kwarg max_format_errors=64 \
  --n-concurrent 8 \
  -k 1 \
  --job-name tmax9b-tb2-full > /home/yichuan/tmax/eval_tb_9b.log 2>&1
echo "=== eval done rc=$? ===" >> /home/yichuan/tmax/eval_tb_9b.log
