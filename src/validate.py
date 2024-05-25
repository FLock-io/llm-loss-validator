import os
import time

import click
import torch
import requests
import tempfile
from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)

from dotenv import load_dotenv
from core.collator import SFTDataCollator
from core.dataset import UnifiedSFTDataset
from core.template import template_dict
from tenacity import retry, stop_after_attempt, wait_exponential
from client.fed_ledger import FedLedger

load_dotenv()
TIME_SLEEP = int(os.getenv("TIME_SLEEP", 10))
FLOCK_API_KEY = os.getenv("FLOCK_API_KEY")
if FLOCK_API_KEY is None:
    raise ValueError("FLOCK_API_KEY is not set")
LOSS_FOR_MODEL_PARAMS_EXCEED = 999.0


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True,
)
def download_file(url):
    try:
        # Send a GET request to the signed URL
        response = requests.get(url, stream=True)
        # Raise an HTTPError if the HTTP request returned an unsuccessful status code
        response.raise_for_status()

        # Create a temporary file to save the content
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            # Write the content to the temp file in binary mode
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    temp_file.write(chunk)

            # move the file pointer to the beginning of the file
            temp_file.flush()
            temp_file.seek(0)

            # get the file path
            file_path = temp_file.name
            logger.info(f"Downloaded the file to {file_path}")

            return file_path

    except requests.exceptions.RequestException as e:
        # Handle any exception that can be raised by the requests library
        logger.error(f"An error occurred while downloading the file: {e}")
        raise e


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


@click.group()
def cli():
    pass


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
@click.option(
    "--local_test",
    is_flag=True,
    help="Run the script in local test mode to avoid submitting to the server",
)
def validate(
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
        raise ValueError(
            "assignment_id is required for submitting validation result to the server"
        )

    try:
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
    except (OSError, RuntimeError) as e:
        # log the type of the exception
        logger.error(f"An error occurred while validating the model: {e}")
        # fail this assignment
        fed_ledger.mark_assignment_as_failed(assignment_id)


@click.command()
@click.option(
    "--validation_args_file",
    type=str,
    default="validation_config.json.example",
    help="",
)
@click.option(
    "--task_id",
    type=str,
    help="The id of the task",
)
def loop(validation_args_file: str, task_id: str = None):
    fed_ledger = FedLedger(FLOCK_API_KEY)

    if task_id is None:
        raise ValueError("task_id is required for asking assignment_id")

    while True:
        resp = fed_ledger.request_validation_assignment(task_id)
        if resp.status_code != 200:
            logger.error(f"Failed to ask assignment_id: {resp.content}")
            time.sleep(TIME_SLEEP)
            continue
        resp = resp.json()
        eval_file = download_file(resp["data"]["validation_set_url"])
        ctx = click.Context(validate)
        ctx.invoke(
            validate,
            model_name_or_path=resp["task_submission"]["data"]["hg_repo_id"],
            base_model=resp["data"]["base_model"],
            eval_file=eval_file,
            context_length=resp["data"]["context_length"],
            max_params=resp["data"]["max_params"],
            validation_args_file=validation_args_file,
            assignment_id=resp["id"],
            local_test=False,
        )
        os.remove(eval_file)
        time.sleep(TIME_SLEEP)


cli.add_command(validate)
cli.add_command(loop)

if __name__ == "__main__":
    cli()
