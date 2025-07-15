#!/bin/bash

if [ -f inference_server.pid ]; then
    PID=$(cat inference_server.pid)
    echo "Killing inference server with PID $PID"
    kill $PID
    rm inference_server.pid
else
    echo "No PID file found. Server may not be running."
fi
