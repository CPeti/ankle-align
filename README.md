# Deep Learning Class (VITMMA19) Project Work template

## Project Details

### Project Information

- **Selected Topic**: AnkleAlign
- **Student Name**: Peter Czumbel
- **Aiming for +1 Mark**: Yes

### Solution Description

This project addresses ankle alignment classification from foot images, mapping each input to one of three clinically relevant classes: Neutral, Pronation, and Supination. The goal is to build a reliable end-to-end pipeline that prepares heterogeneous labeled data, trains a robust image classifier, and serves predictions via a simple GUI.

Model architecture: We use a compact, convolutional backbone suitable for 224×224 images, initialized with ImageNet-pretrained weights to accelerate convergence and improve generalization. The final layers are adapted for three-way classification with a softmax output. The choice emphasizes a strong baseline that is efficient to train and deploy, while leaving room for incremental improvements (e.g., tuning interpolation, aspect ratio padding, and augmentation strategies).

Training methodology: The pipeline produces stratified train/val/test splits and resizes images to a consistent resolution while preserving aspect ratio with padding. Training uses cross-entropy loss with an optimizer configured in `src/config.py` (e.g., learning rate, batch size, epochs). We apply standard data augmentations to improve robustness (random flips/rotations where appropriate), track accuracy and loss over epochs, and validate against the held-out set. Checkpoints and the best-performing model are saved to `models/`, and logs are captured to `log/run.log` for reproducibility.

Results: The final model demonstrates stable training curves and balanced performance across the three classes on the test split. Evaluation includes per-split metrics, confusion matrix, and summary reports stored under `log/` and `models/`. While this baseline focuses on reliability and clarity, the codebase supports further enhancement (e.g., stronger backbones, fine-grained augmentation, class weighting) to push accuracy and robustness.

### Extra Credit Justification

I am aiming for +1 grade based on these aspects:

- Innovative pipeline hardening: Built a robust, reproducible data pipeline that auto-downloads, normalizes, deduplicates, and stratifies the dataset with comprehensive manifests, enabling reliable experimentation at scale.
- Model improvements with ablations: Ran targeted ablations on interpolation methods, aspect-ratio padding, and augmentation strength to quantify their impact on generalization. The final configuration improved test accuracy and class balance compared to the baseline.
- Deployable ML service: Delivered a containerized Gradio app with inference tooling and clean logs, turning the model into a usable service.

### Docker Instructions

This project is containerized using Docker. Follow the instructions below to build and run the solution.

#### Build

Run the following command in the root directory of the repository to build the Docker image:

```bash
docker compose build
```

#### Run

To run the solution, use the following command. You must mount your local data directory to `/app/data` inside the container.

**To capture the logs for submission (required), redirect the output to a file:**

```bash
docker compose up
```
Logs are captured automatically where the `LOG_FILE` variable points in the run.sh script. Default value is `log/run.log`.

The gradio server will be available at `http://localhost:7860`.

### File Structure and Functions

The repository is structured as follows:

- **`src/`**: Contains the source code for the machine learning pipeline.
    - `01-data-preprocessing.py`: Complete 4-phase preprocessing pipeline (download, fetch, cleanse, split/resize).
    - `02-training.py`: Model definition and training loop with checkpointing.
    - `03-evaluation.py`: Model evaluation on test data, generates metrics and confusion matrix.
    - `04-inference.py`: Inference script for running predictions on new images.
    - `app.py`: Gradio web application for interactive model inference.
    - `config.py`: Central configuration file with hyperparameters, paths, and training settings.
    - `utils.py`: Shared utilities and logger configuration.

- **`notebook/`**: Jupyter notebooks for exploratory analysis and experimentation.
    - `01-consensus-labels-analysis.ipynb`: Analysis of consensus labels across labelers.
    - `02-data-preparation-1.ipynb`: Initial data preparation experiments.
    - `03-data-exploration.ipynb`: Exploratory data analysis and visualization.
    - `04-data-cleansing.ipynb`: Data quality analysis and cleansing experiments.
    - `05-incremental-modeling.ipynb`: Incremental model development and ablation studies.

- **`log/`**: Contains execution logs.
    - `run.log`: Complete pipeline execution log (preprocessing, training, evaluation, inference).

- **Root Directory**:
    - `Dockerfile`: Docker image configuration with dependency installation and CRLF normalization.
    - `docker-compose.yml`: Docker Compose configuration for container orchestration.
    - `docker-run.sh`: Docker run helper script.
    - `run.sh`: Main execution script for pipeline stages (preprocess/train/evaluate/app).
    - `requirements.txt`: Python package dependencies with versions.
    - `README.md`: Project documentation and instructions.
