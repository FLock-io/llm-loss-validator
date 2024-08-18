import requests
from tenacity import retry, stop_after_attempt, wait_exponential


class FedLedger:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://fed-ledger-prod.flock.io/api"
        self.api_version = "v1"
        self.url = f"{self.base_url}/{self.api_version}"
        self.headers = {
            "flock-api-key": self.api_key,
            "Content-Type": "application/json",
        }

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=3, max=10)
    )
    def request_validation_assignment(self, task_id: str):
        url = f"{self.url}/tasks/request-validation-assignment/{task_id}"
        response = requests.post(url, headers=self.headers)
        return response

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=3, max=10)
    )
    def submit_validation_result(self, assignment_id: str, loss: float):
        url = f"{self.url}/tasks/update-validation-assignment/{assignment_id}"
        response = requests.post(
            url,
            headers=self.headers,
            json={
                "status": "completed",
                "data": {
                    "loss": loss,
                },
            },
        )
        if response.status_code != 200:
            raise Exception(f"Failed to submit validation result: {response.text}")
        return response

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=3, max=10)
    )
    def mark_assignment_as_failed(self, assignment_id: str):
        url = f"{self.url}/tasks/update-validation-assignment/{assignment_id}"
        response = requests.post(
            url,
            headers=self.headers,
            json={
                "status": "failed",
            },
        )
        if response.status_code != 200:
            raise Exception(f"Failed to mark assignment as failed: {response.text}")
        return response
