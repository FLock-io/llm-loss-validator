# llm-loss-validator

Validator that computes the validation loss for a huggingface-compatible LLM

## Environment Setup

We [recommend] you to use `conda` to manage the python env for this repo.

```bash
conda create -n llm-loss-validator python==3.10.12
conda activate llm-loss-validator
pip install -r requirements.txt
```

## How to run validation script

### Automation with GPU

If you wish to continuously receive task assignments, you should use the following command:

```bash
cd /src
CUDA_VISIBLE_DEVICES=0 \
bash start.sh \
--hf_token your_hf_token \
--flock_api_key your_flock_api_key \
--task_id your_task_id \
--validation_args_file validation_config.json.example \
--auto_clean_cache False \
--lora_only True
```
### Explanation of Parameters
- `CUDA_VISIBLE_DEVICES=0`: Specifies which GPU to use. `0` indicates the first GPU. Adjust this based on your available GPUs.
- `--hf_token`: Your Hugging Face token, required for accessing certain models. This should token should have write access.
- `--flock_api_key`: Your FLock API key.
- `--task_id`: The ID of the task you want to validate. If you are validating multiple tasks, you can pass a list [e.g.] if you are validating tasks 8 and 9, you can pass `--task_id 8,9`
- `--validation_args_file`: The path to the validation arguments file.
- `--auto_clean_cache`: A flag to determine whether to automatically clean the model cache.
- `--lora_only`: A flag to indicate whether to validate only repositories with LoRA (Low-Rank Adaptation) weights. `True` means only LoRA weights will be validated. This is useful for validators with limited network bandwidth, as LoRA weights are significantly smaller (10-500 MiB) compared to full model files (>10 GiB).

### Validate only one assignment

#### With CPU

```bash
cd /src
FLOCK_API_KEY="<your-api-key>" python validate.py validate \
--model_name_or_path Qwen/Qwen1.5-1.8B-Chat \
--base_model qwen1.5 \
--eval_file ./data/dummy_data.jsonl \
--context_length 128 \
--max_params 7000000000 \
--local_test \
--validation_args_file validation_config_cpu.json.example
```

#### With GPU

```bash
cd /src
CUDA_VISIBLE_DEVICES=0 FLOCK_API_KEY="<your-api-key>" python validate.py validate \
--model_name_or_path Qwen/Qwen1.5-1.8B-Chat \
--base_model qwen1.5 \
--eval_file ./data/dummy_data.jsonl \
--context_length 128 \
--max_params 7000000000 \
--local_test \
--validation_args_file validation_config.json.example
```

The `--local_test` flag is for both validator and training node to test that whether they can successfully run validation for a given model submission and dataset. It won't interact with the Fed Ledger service.

To actually calculate and submit the score for a given task assignment. You should use the following command

```bash
CUDA_VISIBLE_DEVICES=0 FLOCK_API_KEY="<your-api-key>" python validate.py validate \
--model_name_or_path Qwen/Qwen1.5-1.8B-Chat \
--base_model qwen1.5 \
--eval_file ./data/dummy_data.jsonl \
--context_length 128 \
--max_params 7000000000 \
--assignment_id <assignment-id> \
--validation_args_file validation_config.json.example
```
