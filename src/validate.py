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
    assert val_args.bf16 or val_args.fp16, "bf16 or fp16 should be True"
    # logger.info(f'Loading model from base model: {args.model_name_or_path}')

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
@click.option("--template_name", required=True, type=str, help="")
@click.option("--eval_file", default="./data/dummy_data.jsonl", type=str, help="")
@click.option("--max_seq_length", required=True, type=int)
@click.option(
    "--validation_args_file",
    type=str,
    default="validation_config.json.example",
    help="",
)
def main(
    model_name_or_path, template_name, eval_file, max_seq_length, validation_args_file
):
    parser = HfArgumentParser(TrainingArguments)
    val_args = parser.parse_json_file(json_file=validation_args_file)[0]

    tokenizer = load_tokenizer(model_name_or_path)
    eval_dataset = load_sft_dataset(
        eval_file, max_seq_length, template_name=template_name, tokenizer=tokenizer
    )
    model = load_model(model_name_or_path, val_args)
    data_collator = SFTDataCollator(tokenizer, max_seq_length=max_seq_length)

    trainer = Trainer(
        model=model,
        args=val_args,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    eval_result = trainer.evaluate()
    logger.info("evaluate result is %s" % str(eval_result))


if __name__ == "__main__":
    main()
