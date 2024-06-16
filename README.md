# llm-loss-validator
Validator that computes the validation loss for a huggingface-compatible LLM

## Environment Setup

We recommand you to use `conda` to manage the python env for this repo.

```bash
conda create -n llm-loss-validator python==3.10.12
conda activate llm-loss-validator
pip install -r requirements.txt
```

## How to run validation script

### With CPU

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

### With GPU

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

If you wish to continuously receive task assignments, you should use the following command:

```bash
CUDA_VISIBLE_DEVICES=0 FLOCK_API_KEY="<your-api-key>" HF_TOKEN="your-hf-token-to-access-gated-based-models" python validate.py loop \
--task_id <list-of-task-ids> \ # For example: "2,3" OR "2"
--validation_args_file validation_config.json.example \
--auto_clean_cache False # whether to automatically clean up non base models cache
```
