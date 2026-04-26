#!/bin/bash
set -euo pipefail

# ---- Parameters (edit here) ----
TASKS_DIR="/gpfs/scrubbed/osey/tmax/rl_data/output/tasks_openthoughts_agent_rl"
PLOTS_DIR=""   # leave empty to default to <TASKS_DIR>/analysis
MODEL=""       # e.g. "gemini/gemini-3-flash-preview"; leave empty to auto-discover all models
# --------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJECT_ROOT"

ARGS=(--tasks-dir "$TASKS_DIR")
if [[ -n "$PLOTS_DIR" ]]; then
    ARGS+=(--plots-dir "$PLOTS_DIR")
fi
if [[ -n "$MODEL" ]]; then
    ARGS+=(--model "$MODEL")
fi

uv run python -m rl_data.analyze "${ARGS[@]}"
