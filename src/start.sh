#!/bin/bash

usage() {
    echo "Usage: $0 --hf_token <token> --flock_api_key <key> --task_id <id> [other options]"
    exit 1
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --hf_token) HF_TOKEN="$2"; shift 2 ;;
        --flock_api_key) FLOCK_API_KEY="$2"; shift 2 ;;
        --task_id) TASK_ID="$2"; shift 2 ;;
        --validation_args_file) VALIDATION_ARGS_FILE="$2"; shift 2 ;;
        *)
            if [[ "$2" != --* ]]; then
                OTHER_ARGS+="$1 $2 "; shift 2
            else
                OTHER_ARGS+="$1 "; shift
            fi
            ;;
    esac
done

if [ -z "$HF_TOKEN" ] || [ -z "$FLOCK_API_KEY" ] || [ -z "$TASK_ID" ]; then
    usage
fi

export HF_TOKEN
export FLOCK_API_KEY

while true; do
    python validate.py loop --task_id ${TASK_ID} --validation_args_file ${VALIDATION_ARGS_FILE} ${OTHER_ARGS}
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
