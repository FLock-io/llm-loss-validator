import requests


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

    def request_validation_assignment(self, task_id: str):
        url = f"{self.url}/tasks/request-validation-assignment/{task_id}"
        response = requests.post(url, headers=self.headers)
        return response.json()

    def submit_validation_result(self, task_id: str, loss: float):
        url = f"{self.url}/tasks/update-validation-assignment/{task_id}"
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
        return response.json()

    def mark_assignment_as_failed(self, task_id: str):
        url = f"{self.url}/tasks/update-validation-assignment/{task_id}"
        response = requests.post(
            url,
            headers=self.headers,
            json={
                "status": "failed",
            },
        )
        return response.json()
    

