FROM python:3.10-slim

WORKDIR /app

COPY inference_server.py .

# Nếu bạn có requirements.txt riêng thì COPY và cài
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir \
    torch \
    transformers \
    litserve

# Nếu có Hugging Face cache/model volume
VOLUME ["/app/models"]

CMD ["python", "inference_server.py"]
