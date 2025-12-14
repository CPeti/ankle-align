#!/usr/bin/env bash
# docker-run.sh - Convenience script to run the Docker container with proper mounts
#
# Usage:
#   ./docker-run.sh              - Run full pipeline
#   ./docker-run.sh train        - Run training only
#   ./docker-run.sh app          - Start web app only
#   ./docker-run.sh bash         - Open shell in container

set -euo pipefail

COMMAND=${1:-pipeline}
IMAGE_NAME="ankle-align"

# Build the image if it doesn't exist
if ! docker images | grep -q "$IMAGE_NAME"; then
    echo "Building Docker image..."
    docker build -t "$IMAGE_NAME" .
fi

echo "Starting container with command: $COMMAND"
echo "Mounting local directories: data, models, log, anklealign"
echo ""

if [ "$COMMAND" == "bash" ]; then
    # Interactive shell
    docker run -it --rm \
        -p 7860:7860 \
        -v "$(pwd)/data:/app/data" \
        -v "$(pwd)/models:/app/models" \
        -v "$(pwd)/log:/app/log" \
        -v "$(pwd)/anklealign:/app/anklealign" \
        -e PYTHONUNBUFFERED=1 \
        "$IMAGE_NAME" \
        /bin/bash
else
    # Run pipeline command
    docker run -it --rm \
        -p 7860:7860 \
        -v "$(pwd)/data:/app/data" \
        -v "$(pwd)/models:/app/models" \
        -v "$(pwd)/log:/app/log" \
        -v "$(pwd)/anklealign:/app/anklealign" \
        -e PYTHONUNBUFFERED=1 \
        "$IMAGE_NAME" \
        ./run.sh "$COMMAND"
fi

