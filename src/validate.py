import json
import os
import time
import shutil

import gc
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
    file_utils,
)

from dotenv import load_dotenv
from pathlib import Path
from core.collator import SFTDataCollator
from core.dataset import UnifiedSFTDataset
from core.template import template_dict
from core.hf_utils import download_lora_config, download_lora_repo
from core.gpu_utils import get_gpu_type
from core.constant import SUPPORTED_BASE_MODELS
from core.exception import (
    handle_os_error,
    handle_runtime_error,
    handle_value_error,
)
from tenacity import retry, stop_after_attempt, wait_exponential
from client.fed_ledger import FedLedger
from peft import PeftModel
import sys

load_dotenv()
TIME_SLEEP = int(os.getenv("TIME_SLEEP", 60 * 10))
ASSIGNMENT_LOOKUP_INTERVAL = 60 * 3  # 3 minutes
FLOCK_API_KEY = os.getenv("FLOCK_API_KEY")
if FLOCK_API_KEY is None:
    raise ValueError("FLOCK_API_KEY is not set")
LOSS_FOR_MODEL_PARAMS_EXCEED = 999.0
HF_TOKEN = os.getenv("HF_TOKEN")
IS_DOCKER_CONTAINER = os.getenv("IS_DOCKER_CONTAINER", False)

if not IS_DOCKER_CONTAINER:
    import git  # only import git in non-docker container environment because it is not installed in docker image

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


def load_model(
    model_name_or_path: str, lora_only: bool, revision: str, val_args: TrainingArguments
) -> Trainer:
    logger.info(f"Loading model from base model: {model_name_or_path}")

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
    if download_lora_config(model_name_or_path, revision):
        logger.info("Repo is a lora weight, loading model with adapter weights")
        with open("lora/adapter_config.json", "r") as f:
            adapter_config = json.load(f)
        base_model = adapter_config["base_model_name_or_path"]
        model = AutoModelForCausalLM.from_pretrained(
            base_model, token=HF_TOKEN, **model_kwargs
        )
        # download the adapter weights
        download_lora_repo(model_name_or_path, revision)
        model = PeftModel.from_pretrained(
            model,
            "lora",
            device_map=None,
        )
        model = model.merge_and_unload()
        logger.info("Loaded model with adapter weights")
    # assuming full fine-tuned model
    else:
        if lora_only:
            logger.error(
                "Repo is not a lora weight, but lora_only flag is set to True. Will mark the assignment as failed"
            )
            return None
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


def is_latest_version(repo_path: str):
    """
    Check if the current branch is up-to-date with the remote main branch.
    Parameters:
    - repo_path (str or Path): The path to the git repository.
    """
    try:
        repo = git.Repo(repo_path)
        origin = repo.remotes.origin
        origin.fetch()

        local_commit = repo.commit("main")
        remote_commit = repo.commit("origin/main")

        if local_commit.hexsha != remote_commit.hexsha:
            logger.error(
                "The local code is not up to date with the main branch.Pls update your version"
            )
            raise
    except git.exc.InvalidGitRepositoryError:
        logger.error("This is not a git repository.")
        raise
    except Exception as e:
        logger.error("An error occurred: %s", str(e))
        raise


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


def clean_model_cache(
    auto_clean_cache: bool, cache_path: str = file_utils.default_cache_path
):
    """
    Cleans up the local model cache directory by removing directories that are not
    listed in SUPPORTED_BASE_MODELS.

    Parameters:
    - auto_clean_cache (bool): A flag to determine whether to clean the cache.
    - cache_path (str): The path to the cache directory. Defaults to file_utils.default_cache_path.
    """
    if not auto_clean_cache:
        return

    try:
        cache_path = Path(cache_path)
        for item in cache_path.iterdir():
            if item.is_dir() and item.name.startswith("models"):
                if item.name not in {
                    f"models--{BASE_MODEL.replace('/', '--')}"
                    for BASE_MODEL in SUPPORTED_BASE_MODELS
                }:
                    shutil.rmtree(item)
                    logger.info(f"Removed directory: {item}")
        logger.info("Successfully cleaned up the local model cache")
    except (OSError, shutil.Error) as e:
        logger.error(f"Failed to clean up the local model cache: {e}")


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
    lora_only: bool = True,
    revision: str = "main",
):
    if not local_test and assignment_id is None:
        raise ValueError(
            "assignment_id is required for submitting validation result to the server"
        )

    model = None
    eval_dataset = None

    try:
        fed_ledger = FedLedger(FLOCK_API_KEY)
        parser = HfArgumentParser(TrainingArguments)
        val_args = parser.parse_json_file(json_file=validation_args_file)[0]
        gpu_type = get_gpu_type()

        tokenizer = load_tokenizer(model_name_or_path)
        eval_dataset = load_sft_dataset(
            eval_file, context_length, template_name=base_model, tokenizer=tokenizer
        )
        model = load_model(model_name_or_path, lora_only, revision, val_args)
        # if model is not loaded, mark the assignment as failed and return
        if model is None:
            fed_ledger.mark_assignment_as_failed(assignment_id)
            return
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
                gpu_type=gpu_type,
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
            assignment_id=assignment_id, loss=eval_loss, gpu_type=gpu_type
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

    # raise for exceptions, will handle at `loop` level
    except Exception as e:
        raise e
    finally:
        # offload the model to save memory
        gc.collect()
        if model is not None:
            logger.debug("Offloading model to save memory")
            model.cpu()
            del model
        if eval_dataset is not None:
            logger.debug("Offloading eval_dataset to save memory")
            del eval_dataset
        torch.cuda.empty_cache()
        # remove lora folder
        if os.path.exists("lora"):
            logger.debug("Removing lora folder")
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
@click.option(
    "--auto_clean_cache",
    type=bool,
    default=True,
    help="Auto clean the model cache except for the base model",
)
@click.option(
    "--lora_only", type=bool, default=True, help="Only validate repo with lora weight"
)
def loop(
    validation_args_file: str,
    task_id: str = None,
    auto_clean_cache: bool = True,
    lora_only: bool = True,
):
    if task_id is None:
        raise ValueError("task_id is required for asking assignment_id")
    if auto_clean_cache:
        logger.info("Auto clean the model cache except for the base model")
    else:
        logger.info("Skip auto clean the model cache")

    repo_path = Path(__file__).resolve().parent.parent

    if not IS_DOCKER_CONTAINER:
        is_latest_version(repo_path)
    else:
        logger.info("Skip checking the latest version in docker container")
        logger.info(
            "Please make sure you are using the latest version of the docker image."
        )

    fed_ledger = FedLedger(FLOCK_API_KEY)
    task_id_list = task_id.split(",")
    logger.info(f"Validating task_id: {task_id_list}")
    last_successful_request_time = [time.time()] * len(task_id_list)
    while True:
        clean_model_cache(auto_clean_cache)

        for index, task_id_num in enumerate(task_id_list):
            resp = fed_ledger.request_validation_assignment(task_id_num)
            if resp.status_code == 200:
                last_successful_request_time[index] = time.time()
                break
            else:
                if resp.json() == {
                    "detail": "No task submissions available to validate"
                }:
                    logger.info(
                        "Failed to ask assignment_id: No task submissions available to validate"
                    )
                else:
                    logger.error(f"Failed to ask assignment_id: {resp.content}")
                if resp.json() == {
                    "detail": "Rate limit reached for validation assignment lookup: 1 per 3 minutes"
                }:
                    time_since_last_success = (
                        time.time() - last_successful_request_time[index]
                    )
                    if time_since_last_success < ASSIGNMENT_LOOKUP_INTERVAL:
                        time_to_sleep = (
                            ASSIGNMENT_LOOKUP_INTERVAL - time_since_last_success
                        )
                        logger.info(f"Sleeping for {int(time_to_sleep)} seconds")
                        time.sleep(time_to_sleep)
                    continue
                else:
                    logger.info(f"Sleeping for {int(TIME_SLEEP)} seconds")
                    time.sleep(TIME_SLEEP)
                    continue

        if resp is None or resp.status_code != 200:
            continue
        resp = resp.json()
        eval_file = download_file(resp["data"]["validation_set_url"])
        revision = resp["task_submission"]["data"].get("revision", "main")
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
                    lora_only=lora_only,
                    revision=revision,
                )
                break  # Break the loop if no exception
            except KeyboardInterrupt:
                # directly terminate the process if keyboard interrupt
                sys.exit(1)
            except OSError as e:
                handle_os_error(e, assignment_id, fed_ledger)
            except RuntimeError as e:
                handle_runtime_error(e, assignment_id, fed_ledger)
            except ValueError as e:
                handle_value_error(e, assignment_id, fed_ledger)
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
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
