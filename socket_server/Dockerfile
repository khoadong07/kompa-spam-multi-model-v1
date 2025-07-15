FROM python:3.10-slim

WORKDIR /app

COPY socket_server.py .

RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    aiohttp \
    tenacity \
    python-socketio

CMD ["uvicorn", "socket_server:asgi_app", "--host", "0.0.0.0", "--port", "5001", "--workers", "4"]
