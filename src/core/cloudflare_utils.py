import os
import boto3
from loguru import logger


class CloudStorage:
    def __init__(
        self,
        access_key=None,
        secret_key=None,
        endpoint_url=None,
        bucket=None,
        session_token=None,
    ):
        self.access_key = access_key
        self.secret_key = secret_key
        self.endpoint_url = endpoint_url
        self.bucket = bucket
        self.client = None
        self.session_token = session_token

    def initialize(self):
        if (
            self.access_key is None
            or self.secret_key is None
            or self.endpoint_url is None
        ):
            logger.error(
                "Please provide access_key, secret_key, session_token and endpoint_url"
            )
            raise
        self.client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            aws_session_token=self.session_token,
        )
        return self

    def download_files(self, prefix: str, local_dir: str) -> bool:
        response = self.client.list_objects_v2(Prefix=prefix, Bucket=self.bucket)

        if "Contents" in response:
            for obj in response["Contents"]:
                file_key = obj["Key"]
                local_file_path = os.path.join(local_dir, file_key)

                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

                self.client.download_file(self.bucket, file_key, local_file_path)
                logger.info(f"Downloaded {file_key} to {local_file_path}")
            return True
        else:
            logger.info("No files found in the bucket.")
            return False
