#!/bin/bash
# LOCAL single-node 8xH100 adaptation of qwen3_8b.sh (no Beaker/mason/dockerhub PAT).
# DPPO RL on Qwen3-8B (tmax SFT) with the REAL tmax-15k data + swerl sandbox tool.
# GPU split: 4 DeepSpeed ZeRO-3 learners + 4 vLLM engines. Reduced rollout/seq shape
# vs the paper config so it fits one node and produces steps in a reasonable time.
set -e
cd "$(dirname "$0")/../../.."   # -> training/open-instruct

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH

export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_V1=1
export SWERL_SANDBOX_TIMING_LOGS=1
export SWERL_RESET_FAILURE_ZERO_REWARD=1
export SWERL_DOCKER_AUTO_REMOVE=1
export SWERL_SANDBOX_TIMING_LOG_THRESHOLD_S=1.0
# rootless podman: use host net (no netavark/ip_tables); tmax task images are self-contained
export SWERL_DOCKER_NETWORK_MODE=host

export DOCKER_HOST="${DOCKER_HOST:-unix:///run/user/$(id -u)/podman/podman.sock}"
export SWERL_PODMAN_DOCKER_HOSTS="$DOCKER_HOST"

export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
# single-node rendezvous over IPv4 loopback (box's primary IP is IPv6; AF_INET bind fails)
export OPEN_INSTRUCT_MASTER_ADDR=127.0.0.1
export TMPDIR=/home/yichuan/tmp
export TRITON_CACHE_DIR=/home/yichuan/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/home/yichuan/.cache/torchinductor
export RAY_TMPDIR=/home/yichuan/ray_tmp
export VLLM_CACHE_ROOT=/home/yichuan/.cache/vllm

# launch with venv python directly so Ray workers inherit it (not `uv run`)
export VIRTUAL_ENV=/home/yichuan/tmax/training/open-instruct/.venv
export PATH=$VIRTUAL_ENV/bin:$PATH
export UV_NO_SYNC=1
PY=$VIRTUAL_ENV/bin/python

echo "DOCKER_HOST=$DOCKER_HOST ; GPUs: $(nvidia-smi -L | wc -l)"

$PY open_instruct/grpo_fast.py \
    --dataset_mixer_list allenai/tmax-15k-open-instruct 1.0 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 2048 \
    --per_turn_max_tokens 4096 \
    --response_length 8192 \
    --pack_length 10240 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 2 \
    --num_samples_per_prompt_rollout 4 \
    --async_steps 1 \
    --model_name_or_path hamishivi/sft_qwen3_8b_our_tmax_sft \
    --temperature 1.0 \
    --learning_rate 1e-6 \
    --total_episodes 24 \
    --lr_scheduler_type constant \
    --deepspeed_stage 3 \
    --sequence_parallel_size 1 \
    --num_epochs 1 \
    --num_learners_per_node 4 \
    --vllm_num_engines 4 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.0 \
    --use_vllm_logprobs true \
    --truncated_importance_sampling_ratio_cap 0.0 \
    --seed 42 \
    --gradient_checkpointing \
    --vllm_enable_prefix_caching \
    --vllm_enforce_eager \
    --vllm_gpu_memory_utilization 0.45 \
    --push_to_hub false \
    --save_traces \
    --save_trainer_logprobs false \
    --tools swerl_vanillux_sandbox \
    --tool_configs '{"task_data_hf_repo": "allenai/tmax-15k-open-instruct", "test_timeout": 120, "image": "python:3.12-slim"}' \
    --pool_size 32 \
    --max_steps 8 \
    --verification_reward 1.0 \
    --tool_parser_type vllm_hermes \
    --system_prompt_override_file scripts/train/debug/envs/swerl_vanillux_sandbox_system_prompt.txt \
    --backend_timeout 1200 \
    --checkpoint_state_freq 1000 \
    --inflight_updates true \
    --lm_head_fp32 true --use_liger_grpo_loss --liger_grpo_loss_chunk_size 8 \
    --advantage_normalization_type centered \
    --loss_fn dppo \
    --dppo_divergence_type tv \
    --dppo_divergence_threshold 0.1 \
    --rollouts_save_path /home/yichuan/tmax/training/open-instruct/output/rollouts_8b \
    --output_dir /home/yichuan/tmax/training/open-instruct/output/qwen3_8b_dppo_8gpu_local \
    --exp_name qwen3_8b_dppo_8gpu_local \
    --local_eval_every 1000 \
    --save_freq 1000
