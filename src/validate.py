import json
import os
import time

import click
import shutil
import torch
import requests
import tempfile
from dotenv import load_dotenv
from loguru import logger

load_dotenv()
os.environ["HF_HOME"] = os.getenv("HF_HOME")
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
from core.hf_utils import download_lora_config, download_lora_repo
from tenacity import retry, stop_after_attempt, wait_exponential
from client.fed_ledger import FedLedger
from peft import PeftModel

TIME_SLEEP = int(os.getenv("TIME_SLEEP", 60 * 10))
ASSIGNMENT_LOOKUP_INTERVAL = 60 * 3  # 3 minutes
FLOCK_API_KEY = os.getenv("FLOCK_API_KEY")
if FLOCK_API_KEY is None:
    raise ValueError("FLOCK_API_KEY is not set")
LOSS_FOR_MODEL_PARAMS_EXCEED = 999.0
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError(
        "You need to set HF_TOKEN to download some gated model from HuggingFace"
    )


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
    # check whether it is a lora weight
    if download_lora_config(model_name_or_path):
        logger.info("Repo is a lora weight, loading model with adapter weights")
        with open("lora/adapter_config.json", "r") as f:
            adapter_config = json.load(f)
        base_model = adapter_config["base_model_name_or_path"]
        model = AutoModelForCausalLM.from_pretrained(
            base_model, token=HF_TOKEN, **model_kwargs
        )
        # download the adapter weights
        download_lora_repo(model_name_or_path)
        model = PeftModel.from_pretrained(
            model,
            "lora",
            device_map=None,
        )
        model = model.merge_and_unload()
        logger.info("Loaded model with adapter weights")
    # assuming full fine-tuned model
    else:
        logger.info("Repo is a full fine-tuned model, loading model directly")
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, token=HF_TOKEN, **model_kwargs
        )

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


def check_cache_size(folder, max_cache_size):
    cnt = 0
    size = 0
    for root, dirs, files in os.walk(folder):
        size += sum([os.path.getsize(os.path.join(root, name)) for name in files])
    size = size / (1024 ** 3)
    while int(max_cache_size) - size < 25:
        # if still in dead loop
        if cnt > 100:
            raise ValueError("max_cache_size number error")
        delete_file = []
        # del tmp file
        for root, dirs, files in os.walk(folder):
            for file in files:
                if "tmp" in file:
                    os.remove(root + os.sep + file)
                    delete_file.append(root + os.sep + file)

        # del smallest model
        min_model_size = float("inf")
        min_model_path = ""
        for model_path in next(os.walk(folder + os.sep + "hub"))[1]:
            if "model" in model_path:
                temp_size = 0
                model_path = folder + os.sep + "hub" + os.sep + model_path
                for root, dirs, files in os.walk(model_path):
                    temp_size += sum([os.path.getsize(os.path.join(root, name)) for name in files])
                # logger.info("model: ", model_path, "size: ", str(temp_size))
                if min_model_size > temp_size:
                    min_model_path = model_path
                    min_model_size = temp_size
        delete_file.append(min_model_path)
        shutil.rmtree(min_model_path)
        logger.info("delete file and folder as below: " + ",".join(delete_file))

        size = 0
        for root, dirs, files in os.walk(folder):
            size += sum([os.path.getsize(os.path.join(root, name)) for name in files])
        size = size / (1024 ** 3)
        cnt += 1


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
            if resp.json() == {
                "detail": "Validation assignment is not in validating status"
            }:
                logger.info(
                    "Validation assignment is not in validating status anymore, marking it as failed"
                )
                fed_ledger.mark_assignment_as_failed(assignment_id)
            return
        logger.info(
            f"Successfully submitted validation result for assignment {assignment_id}"
        )
    except (OSError, RuntimeError) as e:
        # log the type of the exception
        logger.error(f"An error occurred while validating the model: {e}")
        # fail this assignment
        fed_ledger.mark_assignment_as_failed(assignment_id)
    # raise for other exceptions
    except Exception as e:
        raise e
    finally:
        # offload the model to save memory
        del model
        torch.cuda.empty_cache()
        # remove lora folder
        if os.path.exists("lora"):
            os.system("rm -rf lora")


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
@click.option("--max_cache_size", type=str, default="100", help="use GB as the unit")
def loop(
        validation_args_file: str,
        max_cache_size: str,
        task_id: str = None,
):
    fed_ledger = FedLedger(FLOCK_API_KEY)

    if task_id is None:
        raise ValueError("task_id is required for asking assignment_id")
    if int(max_cache_size) < 25:
        raise ValueError(" max_cache_size should be greater than 25")
    task_id_list = task_id.split(",")
    last_successful_request_time = [time.time()]*len(task_id_list)
    resp = None
    while True:
        for index, task_id_num in enumerate(task_id_list):
            resp = fed_ledger.request_validation_assignment(task_id_num)
            if resp.status_code != 200:
                logger.error(f"Failed to ask assignment_id: {resp.content}")
                # handle lookup rate limit
                if resp.json() == {
                    "detail": "Rate limit reached for validation assignment lookup: 1 per 5 minutes"
                }:
                    # if not passed, sleep until the next assignment lookup interval
                    if (
                            time.time() - last_successful_request_time[index]
                            < ASSIGNMENT_LOOKUP_INTERVAL
                    ):
                        time_to_sleep = ASSIGNMENT_LOOKUP_INTERVAL - (
                                time.time() - last_successful_request_time[index]
                        )
                        logger.info(f"Sleeping for {int(time_to_sleep / len(task_id_list))} seconds")
                        time.sleep(time_to_sleep / len(task_id_list))
                    continue
                else:
                    logger.info(f"Sleeping for {TIME_SLEEP} seconds")
                    time.sleep(TIME_SLEEP / len(task_id_list))
                    continue
            else:
                last_successful_request_time[index] = time.time()
                break
        if resp is None:
            raise ValueError("task_id format is incorrect")
        if resp.status_code != 200:
            continue
        resp = resp.json()
        eval_file = download_file(resp["data"]["validation_set_url"])
        assignment_id = resp["id"]
        for attempt in range(3):
            try:
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
                # if no exception, break the loop
                break
            # if keyboard interrupt, break the loop
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                # If it's the last attempt, mark the assignment as failed
                if attempt == 2:
                    logger.error(
                        f"Marking assignment {assignment_id} as failed after 3 attempts"
                    )
                    fed_ledger.mark_assignment_as_failed(assignment_id)
        os.remove(eval_file)


cli.add_command(validate)
cli.add_command(loop)

if __name__ == "__main__":
    cli()
