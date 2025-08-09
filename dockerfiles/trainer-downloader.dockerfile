FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir huggingface_hub aiohttp pydantic transformers
COPY image/configs/ configs/
COPY dockerfiles/trainer_downloader.py trainer_downloader.py

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "trainer_downloader.py"]