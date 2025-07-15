#!/bin/bash

if [ -f socket_server.pid ]; then
    PID=$(cat socket_server.pid)
    echo "Killing socket server with PID $PID"
    kill $PID
    rm socket_server.pid
else
    echo "No PID file found. Server may not be running."
fi
