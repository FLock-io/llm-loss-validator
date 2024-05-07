FROM python:3.11.9-slim-bullseye

WORKDIR /app

COPY . .

RUN pip3 install -r requirements.txt

WORKDIR /app/src


CMD ["python", "validate.py","loop","--task_id","$task_id","--validation_args_file", "validation_config_cpu.json.example"]

