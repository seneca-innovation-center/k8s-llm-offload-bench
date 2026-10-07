FROM python:3.12-slim
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir "transformers>=4.51,<5" safetensors numpy huggingface_hub accelerate
ENV OMP_WAIT_POLICY=PASSIVE PYTHONUNBUFFERED=1
WORKDIR /app
COPY *.py /app/
