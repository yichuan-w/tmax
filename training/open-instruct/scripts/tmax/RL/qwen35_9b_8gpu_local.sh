#!/bin/bash
# LOCAL single-node 8xH100 adaptation of qwen35_9b.sh (TMAX-9B recipe: Qwen3.5-9B + DPPO,
# NO SFT) on the REAL tmax-15k data. Goal: reproduce the reward-increase curve (paper Fig 7).
# GPU split: 4 DeepSpeed ZeRO-3 learners + 4 vLLM engines. Rollout/seq shape reduced from the
# 64-GPU paper config (65k seq, group 32, 48 engines) so it fits one node and yields steps.
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
export SWERL_DOCKER_NETWORK_MODE=host          # rootless podman: host net (no netavark/ip_tables)
export PYTORCH_ALLOC_CONF=expandable_segments:True

export DOCKER_HOST="${DOCKER_HOST:-unix:///run/user/$(id -u)/podman/podman.sock}"
export SWERL_PODMAN_DOCKER_HOSTS="$DOCKER_HOST"

export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export OPEN_INSTRUCT_MASTER_ADDR=127.0.0.1     # IPv4 rendezvous (box primary IP is IPv6)
export TMPDIR=/home/yichuan/tmp
export TRITON_CACHE_DIR=/home/yichuan/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/home/yichuan/.cache/torchinductor
export RAY_TMPDIR=/home/yichuan/ray_tmp
export VLLM_CACHE_ROOT=/home/yichuan/.cache/vllm

export VIRTUAL_ENV=/home/yichuan/tmax/training/open-instruct/.venv
export PATH=$VIRTUAL_ENV/bin:$PATH
export UV_NO_SYNC=1
PY=$VIRTUAL_ENV/bin/python

echo "DOCKER_HOST=$DOCKER_HOST ; GPUs: $(nvidia-smi -L | wc -l)"

$PY open_instruct/grpo_fast.py \
    --dataset_mixer_list allenai/tmax-15k-open-instruct 1.0 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 2048 \
    --per_turn_max_tokens 3072 \
    --response_length 10240 \
    --pack_length 12288 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 4 \
    --num_samples_per_prompt_rollout 8 \
    --async_steps 2 \
    --model_name_or_path hamishivi/Qwen3.5-9B \
    --temperature 1.0 \
    --learning_rate 1e-6 \
    --total_episodes 400000 \
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
    --pool_size 48 \
    --max_steps 16 \
    --verification_reward 1.0 \
    --tool_parser_type vllm_qwen3_xml \
    --system_prompt_override_file scripts/train/debug/envs/swerl_vanillux_sandbox_system_prompt.txt \
    --filter_zero_std_samples true \
    --backend_timeout 1200 \
    --vllm_gdn_prefill_backend triton \
    --checkpoint_state_freq 100000 \
    --inflight_updates true \
    --lm_head_fp32 true --use_liger_grpo_loss --liger_grpo_loss_chunk_size 8 \
    --advantage_normalization_type centered \
    --loss_fn dppo \
    --dppo_divergence_type tv \
    --dppo_divergence_threshold 0.1 \
    --rollouts_save_path /home/yichuan/tmax/training/open-instruct/output/rollouts_9b \
    --output_dir /home/yichuan/tmax/training/open-instruct/output/qwen35_9b_dppo_8gpu_local \
    --exp_name qwen35_9b_dppo_8gpu_local \
    --local_eval_every 100000 \
    --save_freq 25
