import os
import click
import torch
from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)

from core.collator import SFTDataCollator
from core.dataset import UnifiedSFTDataset
from core.template import template_dict
from client.fed_ledger import FedLedger

FLOCK_API_KEY = os.getenv("FLOCK_API_KEY")
if FLOCK_API_KEY is None:
    raise ValueError("FLOCK_API_KEY is not set")
LOSS_FOR_MODEL_PARAMS_EXCEED = 999.0


def load_tokenizer(model_name_or_path: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
    )
    if "gemma" in model_name_or_path.lower():
        tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<start_of_turn>", "<end_of_turn>"]}
        )

    if tokenizer.__class__.__name__ == "QWenTokenizer":
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.pad_token_id is not None, "pad_token_id should not be None"
    assert tokenizer.eos_token_id is not None, "eos_token_id should not be None"
    logger.info(f"vocab_size of tokenizer: {tokenizer.vocab_size}")
    return tokenizer


def load_model(model_name_or_path: str, val_args: TrainingArguments) -> Trainer:
    # logger.info(f'Loading model from base model: {args.model_name_or_path}')

    if val_args.use_cpu:
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.float16 if val_args.fp16 else torch.bfloat16
    model_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        use_cache=False,
        device_map=None,
    )
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)

    if "output_router_logits" in model.config.to_dict():
        logger.info("set output_router_logits as True")
        model.config.output_router_logits = True
    logger.info(
        f"memory footprint of model: {model.get_memory_footprint() / (1024 * 1024 * 1024)} GB"
    )

    total = sum(p.numel() for p in model.parameters())
    logger.info("Total model params: %.2fM" % (total / 1e6))

    return model


def load_sft_dataset(
    eval_file: str, max_seq_length: int, template_name: str, tokenizer: AutoTokenizer
) -> UnifiedSFTDataset:
    if template_name not in template_dict.keys():
        raise ValueError(
            f"template_name doesn't exist, all template_name: {template_dict.keys()}"
        )
    template = template_dict[template_name]
    logger.info("Loading data with UnifiedSFTDataset")
    return UnifiedSFTDataset(eval_file, tokenizer, max_seq_length, template)


@click.command()
@click.option("--model_name_or_path", required=True, type=str, help="")
@click.option("--base_model", required=True, type=str, help="")
@click.option("--eval_file", default="./data/dummy_data.jsonl", type=str, help="")
@click.option("--context_length", required=True, type=int)
@click.option("--max_params", required=True, type=int)
@click.option(
    "--validation_args_file",
    type=str,
    default="validation_config.json.example",
    help="",
)
@click.option(
    "--assignment_id",
    type=str,
    help="The id of the validation assignment",
)
@click.option("--local_test", is_flag=True, help="Run the script in local test mode to avoid submitting to the server")
def main(
    model_name_or_path: str,
    base_model: str,
    eval_file: str,
    context_length: int,
    max_params: int,
    validation_args_file: str,
    assignment_id: str = None,
    local_test: bool = False,
):  
    
    if not local_test and assignment_id is None:
        raise ValueError("assignment_id is required for submitting validation result to the server")
    
    fed_ledger = FedLedger(FLOCK_API_KEY)
    parser = HfArgumentParser(TrainingArguments)
    val_args = parser.parse_json_file(json_file=validation_args_file)[0]

    tokenizer = load_tokenizer(model_name_or_path)
    eval_dataset = load_sft_dataset(
        eval_file, context_length, template_name=base_model, tokenizer=tokenizer
    )
    model = load_model(model_name_or_path, val_args)
    # if the number of parameters exceeds the limit, submit a validation result with a large loss
    total = sum(p.numel() for p in model.parameters())
    if total > max_params:
        logger.error(
            f"Total model params: {total} exceeds the limit {max_params}, submitting validation result with a large loss"
        )
        if local_test:
            return
        resp = fed_ledger.submit_validation_result(
            assignment_id=assignment_id,
            loss=LOSS_FOR_MODEL_PARAMS_EXCEED,
        )
        # check response is 200
        if resp.status_code != 200:
            logger.error(f"Failed to submit validation result: {resp.content}")
        return
    data_collator = SFTDataCollator(tokenizer, max_seq_length=context_length)

    trainer = Trainer(
        model=model,
        args=val_args,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    eval_result = trainer.evaluate()
    eval_loss = eval_result["eval_loss"]
    logger.info("evaluate result is %s" % str(eval_result))
    if local_test:
        logger.info("The model can be correctly validated by validators.")
        return
    resp = fed_ledger.submit_validation_result(
        assignment_id=assignment_id,
        loss=eval_loss,
    )
    # check response is 200
    if resp.status_code != 200:
        logger.error(f"Failed to submit validation result: {resp.content}")
        return


if __name__ == "__main__":
    main()
