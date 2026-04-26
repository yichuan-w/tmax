#!/bin/bash
#SBATCH --job-name=rl-gen-sol-ot
#SBATCH --output=logs/gen_sol_ot_%j.out
#SBATCH --error=logs/gen_sol_ot_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Run our solution-generation harness on the (converted) OpenThoughts- ║
# ║  Agent-v1-RL dataset, using THE SAME model+settings as our 10k run.    ║
# ║                                                                        ║
# ║  Prerequisite: run rl_data/scripts/comparison/run_ingest_openthoughts.sh║
# ║  once to extract 728 tasks into TASKS_DIR. This script then prebuilds  ║
# ║  a single shared base SIF (ubuntu_24.04 + all of the dataset's common  ║
# ║  apt deps + pytest) at PROJECT_ROOT so per-task SIFs are cheap to      ║
# ║  layer (seeds + expected_output via %files).                           ║
# ╚═══════════════════════════════════════════════════════════════════════╝

set -euo pipefail

# ---- Parameters (edit here) ----
TASKS_DIR="rl_data/output/tasks_openthoughts_agent_rl"
# Override via env (see run_generate_solutions_et.sh header for examples).
MODEL="${MODEL:-gemini/gemini-3-flash-preview}"
NUM_SOLUTIONS=1              # match ET + 10k comparison runs
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
FORCE_RERUN=0
LOG_COMMANDS=0
DISABLE_TERMINAL_LOG=0

# Optional cost-bounded subsample: set SAMPLE_SIZE=250 (or similar) in env.
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
export HOSTED_VLLM_API_BASE="${HOSTED_VLLM_API_BASE:-}"
export OLLAMA_API_BASE="${OLLAMA_API_BASE:-}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-}"
if [[ -n "${HOSTED_VLLM_API_BASE:-}${OLLAMA_API_BASE:-}${OPENAI_API_BASE:-}" && -z "${OPENAI_API_KEY:-}" ]]; then
  export OPENAI_API_KEY="EMPTY"
fi

export APPTAINER_CACHEDIR="/gpfs/projects/h2lab/osey/apptainer_cache"
export APPTAINER_TMPDIR="/tmp/apptainer_tmp"
mkdir -p "$APPTAINER_TMPDIR"

# Keep apptainer instance logs off GPFS; heal dangling symlinks left behind
# when /tmp was cleaned between runs.
mkdir -p /tmp/apptainer_instances
if [ ! -L "$HOME/.apptainer/instances" ]; then
  rm -rf "$HOME/.apptainer/instances"
  ln -s /tmp/apptainer_instances "$HOME/.apptainer/instances"
fi

# ---- OT-Agent-v1-RL shared base SIF bootstrap ---------------------------
# Every OT-Agent-v1-RL container.def uses `Bootstrap: localimage` +
# `From: ./ubuntu_24.04_ot.sif` resolved against CWD (PROJECT_ROOT). We build
# that base here once, pre-installing every apt dep that the dataset's
# (identical) Dockerfile asks for, plus pytest, plus a populated
# /var/lib/apt/lists (lever 3 from the ET run) so per-task `apt-get update`
# -- if tasks ever run one -- is a no-op.
#
# Per-task SIFs layer payload (seeds/, test.sh, expected_output.txt) on top,
# so their build time is dominated by squashfs repack (~2-5s).
#
# Bump BASE_OT_SIF_VERSION to invalidate when the recipe below changes.
BASE_OT_SIF="$PROJECT_ROOT/ubuntu_24.04_ot.sif"
BASE_OT_SIF_VERSION="v2-ubuntu24.04-dev+pytest+home-user"
BASE_OT_SIF_MARK="${BASE_OT_SIF}.version"

_need_base_build=0
if [ ! -f "$BASE_OT_SIF" ]; then
  _need_base_build=1
  _reason="missing"
elif [ ! -f "$BASE_OT_SIF_MARK" ] || \
     [ "$(cat "$BASE_OT_SIF_MARK" 2>/dev/null)" != "$BASE_OT_SIF_VERSION" ]; then
  _need_base_build=1
  _reason="stale (want ${BASE_OT_SIF_VERSION}, got $(cat "$BASE_OT_SIF_MARK" 2>/dev/null || echo none))"
fi

if [ "$_need_base_build" = "1" ]; then
  echo "=== OT base SIF ${_reason}; building $BASE_OT_SIF (ubuntu:24.04 + dataset deps + pytest) ==="
  BASE_OT_DEF="$(mktemp --suffix=.def)"
  trap 'rm -f "$BASE_OT_DEF"' EXIT
  # Mirrors the OT-Agent-v1-RL Dockerfile (identical across all 728 tasks),
  # plus pytest for our verifier wrapper. We intentionally do NOT
  # `rm -rf /var/lib/apt/lists/*` so subsequent apt invocations are cheap.
  cat > "$BASE_OT_DEF" <<'EOF'
Bootstrap: docker
From: ubuntu:24.04

%post
    set -e
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        curl \
        git \
        jq \
        less \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        unzip \
        vim \
        wget
    # pytest outside the system dirs; --break-system-packages is required on
    # Ubuntu 24.04's PEP 668-flagged `python3-pip`.
    python3 -m pip install --no-cache-dir --break-system-packages pytest
    apt-get clean
    # tmax convention: every base SIF has /home/user present so the
    # harness's generate_solutions._patch_def_chmod can inject its
    # `chmod 755 /home/user` unconditionally across adapters.
    mkdir -p /home/user
    chmod 755 /home/user
    # (Intentionally keep /var/lib/apt/lists/* populated.)

%environment
    export DEBIAN_FRONTEND=noninteractive

%labels
    Author rl_data-ot-agent-rl-bootstrap
    Description "Ubuntu 24.04 + OT-Agent-v1-RL dataset deps + pytest, prebaked for per-task builds."
EOF
  if ! apptainer build --force "$BASE_OT_SIF" "$BASE_OT_DEF"; then
    echo "ERROR: failed to build $BASE_OT_SIF (OT-Agent-v1-RL per-task builds require this)." >&2
    exit 1
  fi
  echo "$BASE_OT_SIF_VERSION" > "$BASE_OT_SIF_MARK"
  rm -f "$BASE_OT_DEF"
  trap - EXIT
  echo "=== OT base SIF ready: $BASE_OT_SIF (version=${BASE_OT_SIF_VERSION}) ==="
else
  echo "=== OT base SIF already present: $BASE_OT_SIF (version=${BASE_OT_SIF_VERSION}, skipping build) ==="
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

echo "=== OpenThoughts-Agent-v1-RL comparison run: WORKERS=${WORKERS}, NUM_SOLUTIONS=${NUM_SOLUTIONS} ==="
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
