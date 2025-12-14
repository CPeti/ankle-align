#!/usr/bin/env bash
# run.sh - Run the complete ML pipeline for Ankle Alignment Classification
# 
# This script runs all pipeline stages and logs output to log/run.log
#
# Usage:
#   ./run.sh              - Run full pipeline (preprocess -> train -> evaluate -> inference -> app)
#   ./run.sh pipeline     - Same as above
#   ./run.sh train        - Run training only
#   ./run.sh evaluate     - Run evaluation only
#   ./run.sh app          - Start web application only
#   ./run.sh demo         - Run inference on demo data

set -euo pipefail

COMMAND=${1:-pipeline}

# Ensure log directory exists
mkdir -p log

# Log file path
LOG_FILE="log/run.log"

# Function to log with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Clear log file for fresh pipeline run
if [[ "$COMMAND" == "pipeline" || "$COMMAND" == "all" ]]; then
    > "$LOG_FILE"
fi

log_message "========================================"
log_message "ANKLE ALIGNMENT ML PIPELINE"
log_message "========================================"
log_message "Command: $COMMAND"
log_message "Working directory: $(pwd)"
log_message ""

case $COMMAND in
    preprocess)
        log_message "Running data preprocessing..."
        python src/01-data-preprocessing.py
        ;;
    train)
        log_message "Running training..."
        python src/02-training.py
        ;;
    evaluate)
        log_message "Running evaluation..."
        python src/03-evaluation.py
        ;;
    inference|demo)
        log_message "Running inference on demo data..."
        # Find some test images to use as demo
        if [ -d "data/processed/test" ]; then
            DEMO_IMAGES=$(find data/processed/test -name "*.png" | head -5)
            if [ -n "$DEMO_IMAGES" ]; then
                python src/04-inference.py --input $DEMO_IMAGES
            else
                log_message "No demo images found in data/processed/test"
            fi
        else
            log_message "Test data directory not found. Run preprocessing first."
        fi
        ;;
    app|serve|web)
        log_message "Starting web application..."
        python src/app.py --port ${PORT:-7860}
        ;;
    pipeline|all)
        log_message "========================================"
        log_message "RUNNING FULL ML PIPELINE"
        log_message "========================================"
        log_message ""
        
        # Step 1: Data Preprocessing
        log_message "========================================"
        log_message "STEP 1/5: DATA PREPROCESSING"
        log_message "========================================"
        python src/01-data-preprocessing.py
        log_message ""
        
        # Step 2: Model Training
        log_message "========================================"
        log_message "STEP 2/5: MODEL TRAINING"
        log_message "========================================"
        python src/02-training.py
        log_message ""
        
        # Step 3: Model Evaluation
        log_message "========================================"
        log_message "STEP 3/5: MODEL EVALUATION"
        log_message "========================================"
        python src/03-evaluation.py
        log_message ""
        
        # Step 4: Demo Inference
        log_message "========================================"
        log_message "STEP 4/5: DEMO INFERENCE"
        log_message "========================================"
        if [ -d "data/processed/test" ]; then
            python src/04-inference.py --input-dir data/processed/test --recursive
        else
            log_message "Test data not found, skipping inference demo"
        fi
        log_message ""
        
        # Step 5: Start Web Application
        log_message "========================================"
        log_message "STEP 5/5: STARTING WEB APPLICATION"
        log_message "========================================"
        log_message "Gradio app starting on port ${PORT:-7860}..."
        log_message "Access the app at: http://localhost:${PORT:-7860}"
        log_message ""
        python src/app.py --port ${PORT:-7860}
        ;;
    *)
        echo "Unknown command: $COMMAND"
        echo ""
        echo "Available commands:"
        echo "  pipeline   - Run full pipeline (preprocess -> train -> evaluate -> inference -> app)"
        echo "  preprocess - Run data preprocessing only"
        echo "  train      - Run model training only"
        echo "  evaluate   - Run model evaluation only"
        echo "  inference  - Run inference on demo data"
        echo "  app        - Start web application only"
        exit 1
        ;;
esac

log_message ""
log_message "========================================"
log_message "PIPELINE FINISHED"
log_message "========================================"
log_message "Log file: $LOG_FILE"
