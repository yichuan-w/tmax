#!/bin/bash
# LOCAL adaptation of qwen35_2b_1gpu.sh (no Beaker / no mason / no dockerhub PAT).
# DPPO smoke test on Qwen3.5-2B with openthoughts terminal tasks, 1 GPU.
# Uses the already-running rootless podman service for the swerl sandbox tool.
set -e
cd "$(dirname "$0")/../../.."   # -> training/open-instruct

# ---- runtime CUDA (match torch cu128; deepspeed JIT builds ops for local sm_90) ----
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH

# ---- vLLM / swerl sandbox knobs (from the original 1gpu debug script) ----
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_V1=1
export SWERL_SANDBOX_TIMING_LOGS=1
export SWERL_RESET_FAILURE_ZERO_REWARD=1
export SWERL_DOCKER_AUTO_REMOVE=1
export SWERL_SANDBOX_TIMING_LOG_THRESHOLD_S=1.0
# rootless podman on this box can't modprobe ip_tables for netavark bridge -> use host net
export SWERL_DOCKER_NETWORK_MODE=host

# ---- point the docker SDK at our rootless podman socket (no dockerhub login) ----
export DOCKER_HOST="${DOCKER_HOST:-unix:///run/user/$(id -u)/podman/podman.sock}"
export SWERL_PODMAN_DOCKER_HOSTS="$DOCKER_HOST"

export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false

# ---- keep all scratch/caches on the big /home disk (avoid small tmpfs ENOSPC) ----
export TMPDIR=/home/yichuan/tmp
export TRITON_CACHE_DIR=/home/yichuan/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/home/yichuan/.cache/torchinductor
export RAY_TMPDIR=/home/yichuan/ray_tmp
export VLLM_CACHE_ROOT=/home/yichuan/.cache/vllm

echo "DOCKER_HOST=$DOCKER_HOST"
echo "podman socket: $(ls -la ${DOCKER_HOST#unix://} 2>&1)"

# Use the venv python DIRECTLY (not `uv run`) so Ray spawns workers with this same
# interpreter instead of re-invoking `uv run` (which creates a fresh empty .venv
# on the worker side -> "ModuleNotFoundError: No module named 'ray'").
export VIRTUAL_ENV=/home/yichuan/tmax/training/open-instruct/.venv
export PATH=$VIRTUAL_ENV/bin:$PATH
export UV_NO_SYNC=1
PY=$VIRTUAL_ENV/bin/python
$PY open_instruct/grpo_fast.py \
    --dataset_mixer_list allenai/open-instruct-openthoughts 1.0 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 2048 \
    --per_turn_max_tokens 4096 \
    --response_length 8192 \
    --pack_length 10240 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 1 \
    --num_samples_per_prompt_rollout 8 \
    --async_steps 4 \
    --model_name_or_path hamishivi/Qwen3.5-2B \
    --temperature 1.0 \
    --learning_rate 1e-6 \
    --total_episodes 32 \
    --lr_scheduler_type constant \
    --deepspeed_stage 3 \
    --num_epochs 1 \
    --num_learners_per_node 1 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --single_gpu_mode \
    --vllm_sync_backend gloo \
    --vllm_gpu_memory_utilization 0.3 \
    --vllm_enforce_eager \
    --beta 0.0 \
    --use_vllm_logprobs true \
    --truncated_importance_sampling_ratio_cap 0.0 \
    --seed 42 \
    --gradient_checkpointing \
    --vllm_enable_prefix_caching \
    --push_to_hub false \
    --save_traces \
    --save_trainer_logprobs false \
    --tools swerl_vanillux_sandbox \
    --tool_configs '{"task_data_hf_repo": "allenai/open-instruct-openthoughts", "test_timeout": 120, "image": "python:3.12-slim"}' \
    --pool_size 16 \
    --max_steps 64 \
    --verification_reward 1.0 \
    --tool_parser_type vllm_qwen3_xml \
    --system_prompt_override_file scripts/train/debug/envs/swerl_vanillux_sandbox_system_prompt.txt \
    --filter_zero_std_samples false \
    --backend_timeout 1200 \
    --vllm_gdn_prefill_backend triton \
    --checkpoint_state_freq 10 \
    --inflight_updates true \
    --lm_head_fp32 true --use_liger_grpo_loss --liger_grpo_loss_chunk_size 8 \
    --advantage_normalization_type centered \
    --loss_fn dppo \
    --dppo_divergence_type tv \
    --dppo_divergence_threshold 0.1 \
    --rollouts_save_path /home/yichuan/tmax/training/open-instruct/output/rollouts \
    --output_dir output/qwen35_2b_dppo_1gpu_local \
    --exp_name qwen35_2b_dppo_1gpu_local \
    --local_eval_every 10 \
    --save_freq 20
