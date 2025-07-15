#!/bin/bash
source venv/bin/activate

cd socket_server
# Start the server in background
nohup uvicorn socket_server:asgi_app --host 0.0.0.0 --port 5001 --workers 4 > ../logs/socket_server.log 2>&1 &

# Save the PID
echo $! > ../socket_server.pid
echo "Inference server started with PID $(cat socket_server.pid)"
