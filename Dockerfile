FROM python:3.11.9-slim-bullseye

WORKDIR /app

COPY . .

RUN pip3 install -r requirements.txt

WORKDIR /app/src


CMD ["sh", "-c", "./monitor.sh --hf_token ${HF_TOKEN} --flock_api_key ${FLOCK_API_KEY} --task_id ${TASK_ID}"]