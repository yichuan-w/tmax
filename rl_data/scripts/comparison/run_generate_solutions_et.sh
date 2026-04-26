#!/bin/bash
#SBATCH --job-name=rl-gen-sol-et
#SBATCH --output=logs/gen_sol_et_%j.out
#SBATCH --error=logs/gen_sol_et_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=960G

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Run our solution-generation harness on the (converted) Endless-     ║
# ║  Terminals dataset, using THE SAME model+settings as our 10k run so  ║
# ║  head-to-head comparison is apples-to-apples.                        ║
# ║                                                                       ║
# ║  Prerequisite: run ingest_endless_terminals.py to populate TASKS_DIR. ║
# ║  ET does NOT use shared base SIFs (its container.defs are self-       ║
# ║  contained), so per-task SIF builds happen lazily.                    ║
# ╚═══════════════════════════════════════════════════════════════════════╝

set -euo pipefail

# ---- Parameters (edit here) ----
TASKS_DIR="rl_data/output/tasks_endless_terminals"
# Override via env.  Examples:
#   API model:   MODEL="gemini/gemini-3-flash-preview"
#   Local vLLM:  MODEL="hosted_vllm/Qwen/Qwen2.5-Coder-7B-Instruct" \
#                HOSTED_VLLM_API_BASE="http://localhost:8000/v1"
#   Ollama:      MODEL="ollama_chat/qwen2.5-coder:7b" \
#                OLLAMA_API_BASE="http://localhost:11434"
MODEL="${MODEL:-gemini/gemini-3-flash-preview}"
NUM_SOLUTIONS=1              # match 10k run
MAX_ACTIONS=16
MAX_TOKENS=65536
NUM_TASKS=999999
START_AT=0
SOLUTION_TEMPERATURE=0.7
COMMAND_TIMEOUT=60
SHELL_INIT_TIMEOUT=240
SHELL_INIT_ATTEMPTS=3
BUILD_WORKERS=12
BUILD_RETRIES=3
# NOTE: no BASE_SIFS_DIR — ET tasks don't share the 9 prebuilt bases.
FORCE_RERUN=0
LOG_COMMANDS=0
DISABLE_TERMINAL_LOG=0

# Optional cost-bounded subsample. Set SAMPLE_SIZE=250 (or similar) in the
# environment to randomly pick N tasks rather than processing the whole set.
SAMPLE_SIZE="${SAMPLE_SIZE:-0}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"

WORKERS=12
NUM_POOL_WORKERS=16
# --------------------------------

_MODEL_TAG=$(echo "$MODEL" | tr '/' '_')
_RUN_TS=$(date -u +%Y%m%d_%H%M%S)
TERMINAL_LOG="${TASKS_DIR}/logs/${_MODEL_TAG}_${_RUN_TS}.log"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJECT_ROOT"
mkdir -p logs

export APPTAINER_DOCKER_USERNAME="${APPTAINER_DOCKER_USERNAME:?Set APPTAINER_DOCKER_USERNAME before running}"
export APPTAINER_DOCKER_PASSWORD="${APPTAINER_DOCKER_PASSWORD:?Set APPTAINER_DOCKER_PASSWORD before running}"

# Local-model support via litellm env passthrough.
# These are harmless if unset (API model is used instead).
export HOSTED_VLLM_API_BASE="${HOSTED_VLLM_API_BASE:-}"
export OLLAMA_API_BASE="${OLLAMA_API_BASE:-}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-}"
if [[ -n "${HOSTED_VLLM_API_BASE:-}${OLLAMA_API_BASE:-}${OPENAI_API_BASE:-}" && -z "${OPENAI_API_KEY:-}" ]]; then
  export OPENAI_API_KEY="EMPTY"  # some local servers don't care but litellm checks
fi

export APPTAINER_CACHEDIR="/gpfs/projects/h2lab/osey/apptainer_cache"
export APPTAINER_TMPDIR="/tmp/apptainer_tmp"
mkdir -p "$APPTAINER_TMPDIR"

# Keep apptainer instance logs off GPFS (rootless cgroups aren't supported there).
# Unconditionally ensure the /tmp target dir exists -- it can disappear between
# runs (node reboots, /tmp cleanup) while the symlink on GPFS persists, leaving
# a dangling symlink that makes `apptainer instance start` fail with
# `mkdir ...: file exists`. Creating the target first heals that case cheaply.
mkdir -p /tmp/apptainer_instances
if [ ! -L "$HOME/.apptainer/instances" ]; then
  rm -rf "$HOME/.apptainer/instances"
  ln -s /tmp/apptainer_instances "$HOME/.apptainer/instances"
fi

# ---- ET-only base SIF bootstrap -----------------------------------------
# Every ET container.def uses `Bootstrap: localimage` + `From: ./ubuntu_22.04.sif`
# resolved against the CWD (PROJECT_ROOT). Build that base SIF once here so the
# 2.5k per-task builds can layer on top of it.
#
# LEVER 2: Pre-install python3 + pip + pytest + ca-certificates into the base
# SIF so the 2,488/2,492 ET defs that install those same packages turn into
# essentially-free no-ops. Without this, running N parallel builds thundering-
# herds archive.ubuntu.com + files.pythonhosted.org and pip times out.
#
# LEVER 3: Keep /var/lib/apt/lists/* populated in the base SIF. Each ET def
# still does `apt-get update` at the top of its %post; with cached lists,
# that becomes a handful of small HEAD/InRelease requests (~1-3s) instead of
# redownloading the full ~46MB package index (~20-30s) from archive.ubuntu.com.
# Adds ~20MB to the base SIF; saves >3 hours across 500 per-task builds.
#
# A version marker file (BASE_UBUNTU_SIF.version) lets us force a rebuild when
# the recipe below changes -- bump BASE_UBUNTU_SIF_VERSION to invalidate.
BASE_UBUNTU_SIF="$PROJECT_ROOT/ubuntu_22.04.sif"
BASE_UBUNTU_SIF_VERSION="v3-py+pytest+aptlists"
BASE_UBUNTU_SIF_MARK="${BASE_UBUNTU_SIF}.version"

_need_base_build=0
if [ ! -f "$BASE_UBUNTU_SIF" ]; then
  _need_base_build=1
  _reason="missing"
elif [ ! -f "$BASE_UBUNTU_SIF_MARK" ] || \
     [ "$(cat "$BASE_UBUNTU_SIF_MARK" 2>/dev/null)" != "$BASE_UBUNTU_SIF_VERSION" ]; then
  _need_base_build=1
  _reason="stale (want ${BASE_UBUNTU_SIF_VERSION}, got $(cat "$BASE_UBUNTU_SIF_MARK" 2>/dev/null || echo none))"
fi

if [ "$_need_base_build" = "1" ]; then
  echo "=== ET base SIF ${_reason}; building enriched base (python3 + pip + pytest) ==="
  BASE_UBUNTU_DEF="$(mktemp --suffix=.def)"
  trap 'rm -f "$BASE_UBUNTU_DEF"' EXIT
  cat > "$BASE_UBUNTU_DEF" <<'EOF'
Bootstrap: docker
From: ubuntu:22.04

%post
    set -e
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -y
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-setuptools \
        python3-wheel \
        ca-certificates
    pip3 install --no-cache-dir pytest
    apt-get clean
    # NOTE: Intentionally keep /var/lib/apt/lists/* populated (Lever 3) so
    # per-task `apt-get update` is nearly a no-op instead of re-fetching 46MB.

%labels
    Author rl_data-et-bootstrap
    Description "Ubuntu 22.04 + python3 + pip + pytest + prepopulated apt lists, prebaked for ET per-task builds."
EOF
  if ! apptainer build --force "$BASE_UBUNTU_SIF" "$BASE_UBUNTU_DEF"; then
    echo "ERROR: failed to build enriched $BASE_UBUNTU_SIF (ET per-task builds require this)." >&2
    exit 1
  fi
  echo "$BASE_UBUNTU_SIF_VERSION" > "$BASE_UBUNTU_SIF_MARK"
  rm -f "$BASE_UBUNTU_DEF"
  trap - EXIT
  echo "=== ET base SIF ready: $BASE_UBUNTU_SIF (version=${BASE_UBUNTU_SIF_VERSION}) ==="
else
  echo "=== ET base SIF already present: $BASE_UBUNTU_SIF (version=${BASE_UBUNTU_SIF_VERSION}, skipping build) ==="
fi

EXTRA_ARGS=()
if [[ "${FORCE_RERUN:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--force-rerun)
fi
if [[ "${LOG_COMMANDS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--log-commands)
fi
if [[ "${SAMPLE_SIZE:-0}" != "0" ]]; then
  EXTRA_ARGS+=(--sample-size "$SAMPLE_SIZE" --sample-seed "$SAMPLE_SEED")
fi
if [[ "${DISABLE_TERMINAL_LOG:-0}" != "1" ]]; then
  TL="${TERMINAL_LOG}"
  if [[ "$TL" != /* ]]; then
    TL="$PROJECT_ROOT/$TL"
  fi
  mkdir -p "$(dirname "$TL")"
  EXTRA_ARGS+=(--terminal-log "$TL")
fi

echo "=== ET comparison run: WORKERS=${WORKERS}, NUM_SOLUTIONS=${NUM_SOLUTIONS} ==="
echo "=== Concurrent containers: $(( WORKERS * NUM_SOLUTIONS )) ==="

uv run python -m rl_data.generate_solutions \
    --tasks-dir "$TASKS_DIR" \
    --model "$MODEL" \
    --num-solutions "$NUM_SOLUTIONS" \
    --max-actions "$MAX_ACTIONS" \
    --max-tokens "$MAX_TOKENS" \
    --num-tasks "$NUM_TASKS" \
    --start-at "$START_AT" \
    --workers "$WORKERS" \
    --num-pool-workers "$NUM_POOL_WORKERS" \
    --solution-temperature "$SOLUTION_TEMPERATURE" \
    --command-timeout "$COMMAND_TIMEOUT" \
    --shell-init-timeout "$SHELL_INIT_TIMEOUT" \
    --shell-init-attempts "$SHELL_INIT_ATTEMPTS" \
    --build-workers "$BUILD_WORKERS" \
    --build-retries "$BUILD_RETRIES" \
    --verbose \
    "${EXTRA_ARGS[@]}"
