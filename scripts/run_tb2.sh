#!/usr/bin/env bash
set -euo pipefail

# Run TassieAgent on Terminal-Bench 2.0 (Daytona backend)
# Resumable: re-running this script resumes the previous job if it exists.
#
# Required env vars:
#   DAYTONA_API_KEY    - Daytona API key
#   OPENAI_API_BASE    - vLLM server URL (e.g. http://localhost:8000/v1)
#
# Optional env vars:
#   MODEL              - Model name (default: openai/default)
#   N_CONCURRENT       - Number of concurrent trials (default: 25)
#   MAX_STEPS          - Max agent steps per trial (default: 50)
#   JOB_NAME           - Job name for resumability (default: tb2)

export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
MODEL="${MODEL:-openai/default}"
N_CONCURRENT="${N_CONCURRENT:-25}"
MAX_STEPS="${MAX_STEPS:-50}"
JOB_NAME="${JOB_NAME:-tb2}"
JOB_DIR="jobs/${JOB_NAME}"

if [ -d "$JOB_DIR" ]; then
    echo "Resuming job from $JOB_DIR"
    uv run harbor jobs resume \
        --job-path "$JOB_DIR" \
        --filter-error-type DaytonaError
else
    uv run harbor run \
        --dataset terminal-bench@2.0 \
        --agent-import-path TassieAgent:TassieAgent \
        --model "$MODEL" \
        --env daytona \
        --n-concurrent "$N_CONCURRENT" \
        --agent-kwarg "max_steps=$MAX_STEPS" \
        --job-name "$JOB_NAME"
fi
