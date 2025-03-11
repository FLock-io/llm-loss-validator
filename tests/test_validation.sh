#!/bin/bash

python3 -m venv venv

source venv/bin/activate

# Install requirements
echo "Downloading requirements..."
pip install -r requirements.txt > /dev/null 2>&1

declare -A MODEL_MAP=(
    ["Qwen/Qwen1.5-1.8B-Chat"]="qwen1.5"
    ["google/gemma-2b"]="gemma"
    ["microsoft/Phi-3-mini-4k-instruct"]="phi3"
    ["mistralai/Ministral-8B-Instruct-2410"]="mistral"
    ["01-ai/Yi-1.5-6B-Chat"]="yi"
    ["meta-llama/Llama-3.1-8B"]="llama3"
    ["random-sequence/task-1-microsoft-Phi-3-mini-4k-instruct"]="phi3"
    ["random-sequence/task-1-microsoft-Phi-3.5-mini-instruct"]="phi3"
    ["random-sequence/task-1-Qwen-Qwen2.5-7B-Instruct"]="qwen1.5"
    ["random-sequence/task-2-google-gemma-2-2b-it"]="gemma"
)

# Define models to validate - mix of full models and LORA
MODELS=(
    "Qwen/Qwen1.5-1.8B-Chat"
    "google/gemma-2b"
    "microsoft/Phi-3-mini-4k-instruct"
    "mistralai/Ministral-8B-Instruct-2410"
    "01-ai/Yi-1.5-6B-Chat"
    "meta-llama/Llama-3.1-8B"
    "random-sequence/task-1-microsoft-Phi-3-mini-4k-instruct"
    "random-sequence/task-1-microsoft-Phi-3.5-mini-instruct"
    "random-sequence/task-1-Qwen-Qwen2.5-7B-Instruct"
    "random-sequence/task-2-google-gemma-2-2b-it"
)

# Run validation for each model
for MODEL in "${MODELS[@]}"; do
    echo "Validating model: $MODEL"

    cd src

    # Determine the base model based on the model name using the dictionary
    BASE_MODEL=${MODEL_MAP[$MODEL]}

    if [ -z "$BASE_MODEL" ]; then
        echo "Unknown model: $MODEL"
        continue
    fi

    OUTPUT=$(FLOCK_API_KEY="$FLOCK_API_KEY" HF_TOKEN="$HF_TOKEN" python validate.py validate \
    --model_name_or_path "$MODEL" \
    --base_model "$BASE_MODEL" \
    --eval_file ./data/dummy_data.jsonl \
    --context_length 4096 \
    --max_params 9000000000 \
    --local_test \
    --lora_only False \
    --validation_args_file validation_config.json.example 2>&1)

    # Display relevant logs for the current model. Add INFO to see info logs
    echo "$OUTPUT" | grep -E " ERROR|WARNING"  # Show only INFO, ERROR, and WARNING logs

    # Check if model is validated
    if echo "$OUTPUT" | grep -q "The model can be correctly validated by validators."; then
        echo -e "\e[32m$MODEL \xE2\x9C\x94\e[0m"  # Green text and tick symbol
    else
        # Check for CUDA error
        if echo "$OUTPUT" | grep -q "CUDA error"; then
            echo -e "$MODEL \xE2\x9D\x8C"
        else
            echo -e "$MODEL \xE2\x9D\x8C"
            echo "Error details: $OUTPUT"
        fi
    fi

    echo "---------------------------------------"

    cd ..
done

deactivate
