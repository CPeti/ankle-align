# Dockerfile for Ankle Alignment Classification
# 
# This container runs the complete ML pipeline:
# 1. Data preprocessing
# 2. Model training
# 3. Model evaluation
# 4. Demo inference
# 5. Gradio web application
#
# Build:
#   docker build -t ankle-align .
#
# Run full pipeline with local directory mounted:
#   docker run -it -p 7860:7860 \
#     -v $(pwd)/data:/app/data \
#     -v $(pwd)/models:/app/models \
#     -v $(pwd)/log:/app/log \
#     -v $(pwd)/anklealign:/app/anklealign \
#     ankle-align
#
# Run specific command:
#   docker run -it -v $(pwd):/app ankle-align ./run.sh train
#
# Access the web app at: http://localhost:7860

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    findutils \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY run.sh run.sh

# Create directories for data, models, and logs
# These should be mounted from host for persistence
RUN mkdir -p /app/data /app/models /app/log /app/anklealign

# Make scripts executable
# Normalize Windows CRLF line endings to Unix LF for shell scripts
RUN dos2unix /app/run.sh || true \
    && chmod +x /app/run.sh

# Expose Gradio port
EXPOSE 7860

# Environment variables
ENV GRADIO_SERVER_NAME="127.0.0.1"
ENV GRADIO_SERVER_PORT="7860"
ENV PORT="7860"

# Set Python to run unbuffered for real-time logs
ENV PYTHONUNBUFFERED=1

# Default command: run the full pipeline
# This will:
# 1. Preprocess data
# 2. Train the model
# 3. Evaluate on test set
# 4. Run demo inference
# 5. Start the Gradio web app
CMD ["./run.sh", "pipeline"]
