#!/bin/bash

usage() {
    echo "Usage: $0 --hf_token <token> --flock_api_key <key> --task_id <id> [other options]"
    exit 1
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --hf_token) HF_TOKEN="$2"; shift ;;
        --flock_api_key) FLOCK_API_KEY="$2"; shift ;;
        --task_id) TASK_ID="$2"; shift ;;
        *) OTHER_ARGS+="$1 "; shift ;;  # Collect other arguments
    esac
    shift
done

if [ -z "$HF_TOKEN" ] || [ -z "$FLOCK_API_KEY" ] || [ -z "$TASK_ID" ]; then
    usage
fi

export HF_TOKEN
export FLOCK_API_KEY

while true; do
    python validate.py loop --task_id ${TASK_ID} ${OTHER_ARGS}
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 100 ]; then
        echo "CUDA error detected, restarting the process..."
        continue
    elif [ $EXIT_CODE -ne 0 ]; then
        echo "Validation failed with exit code $EXIT_CODE"
        exit $EXIT_CODE
    fi
    break
done