# Tests

This directory contains test scripts to ensure that the loos validator can successfully validate models. It checks if variou supported models and LORAs can be validated.
## Running Tests

To run the tests, use the following command:

```bash
FLOCK_API_KEY=<your_flock_api_key> HF_TOKEN=<your_hf_token> bash tests/test_validation.sh
