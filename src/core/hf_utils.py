from huggingface_hub import HfApi
from huggingface_hub.utils._errors import EntryNotFoundError
from loguru import logger

api = HfApi()


def download_lora_config(repo_id: str) -> bool:
    try:
        api.hf_hub_download(
            repo_id=repo_id, filename="adapter_config.json", local_dir="lora"
        )
    except EntryNotFoundError:
        logger.info("No adapter_config.json found in the repo, assuming full model")
        return False
    return True


def download_lora_repo(repo_id: str) -> None:
    api.snapshot_download(repo_id=repo_id, local_dir="lora")
