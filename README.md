# MNIST Classification — PyTorch & scikit-learn

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Neural%20Network-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Reproducible MNIST digit classification using a PyTorch neural network integrated with a scikit-learn pipeline.

This project implements an end-to-end machine learning pipeline for classifying handwritten digits from the MNIST dataset.

In addition to model training, the project provides **stratified cross-validation, reproducible experiments, automated evaluation, model persistence, experiment logging, and result visualization**.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Technologies](#technologies)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Training Process](#training-process)
- [Model](#model)
- [Evaluation](#evaluation)
- [Experiment Logging](#experiment-logging)
- [Saved Results](#saved-results)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [CPU and GPU](#cpu-and-gpu)
- [Reproducibility](#reproducibility)
- [Limitations](#limitations)
- [Possible Extensions](#possible-extensions)
- [License](#license)

## Features

- PyTorch neural network for classifying digits `0–9`
- scikit-learn pipeline with integrated `MinMaxScaler`
- 10-fold stratified cross-validation
- 80/20 stratified train/test split
- Adam optimizer
- Accuracy, precision, recall, and F1-score
- Confusion matrix generation
- Training-loss visualization
- Random test-image visualization
- Model-weight persistence
- Scaler persistence
- Detailed experiment logging
- Configuration and metric export
- Automatic CUDA usage when available
- Loading of a previously trained model without retraining

## Architecture

The overall data flow is:

```text
                         MNIST CSV
                             │
                             ▼
                    ┌─────────────────┐
                    │  Data Loading   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Train / Test    │
                    │     Split       │
                    │      80/20      │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
      10-Fold Cross-Validation          Test Set
              │                             │
              ▼                             │
        MinMaxScaler                       │
              │                             │
              ▼                             │
       PyTorch Classifier                  │
              │                             │
              ▼                             │
      Validation Metrics                   │
                                            │
                             ┌──────────────┘
                             ▼
                      Final Model
                             │
                             ▼
                         Evaluation
                             │
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
         Metrics       Confusion Matrix   Test Image
            │
            ▼
      Experiment Logging
```

The train/test split is performed before cross-validation. Cross-validation is applied exclusively to the training data.

## Technologies

| Technology | Purpose |
|---|---|
| Python | Programming language |
| PyTorch | Neural network and training |
| scikit-learn | Pipeline, cross-validation, preprocessing, and metrics |
| NumPy | Numerical processing |
| pandas | Data processing and CSV handling |
| Matplotlib | Visualization |
| joblib | Scaler persistence |

## Requirements

- Python 3.x
- pip
- Optional: NVIDIA GPU with a compatible CUDA/PyTorch installation

Required Python packages:

```text
torch
numpy
pandas
scikit-learn
matplotlib
joblib
```

## Installation

### Clone the repository

```bash
git clone https://github.com/<USERNAME>/<REPOSITORY>.git
cd <REPOSITORY>
```

### Install dependencies

```bash
pip install torch numpy pandas scikit-learn matplotlib joblib
```

For reproducible environments, a `requirements.txt` file is recommended:

```bash
pip install -r requirements.txt
```

> The appropriate PyTorch package should be selected according to the target CPU/CUDA environment.

## Dataset

The program expects the following file by default:

```text
mnist_data_60k.csv
```

The CSV file is read without a header. The first column contains the label, followed by exactly 784 pixel values.

Expected format:

```text
label,pixel_1,pixel_2,...,pixel_784
```

The program validates that the input contains exactly 784 pixel features.

> The dataset itself is not generated or downloaded by the script. It must be provided separately.

## Usage

After installing the dependencies and placing the dataset in the expected location:

```bash
python mnist.py
```

The program checks whether a previously trained model and scaler are available in:

```text
mnist_experiments/
├── mnist_model_weights.pth
└── mnist_scaler.joblib
```

If both files exist, the program asks:

```text
Should a new training be performed? [j/n]:
```

### Start a new training

Enter:

```text
j
```

A new experiment will:

1. Split the dataset into training and test data.
2. Run 10-fold stratified cross-validation.
3. Store the results of each fold.
4. Train a final model on the complete training set.
5. Save the model and scaler.
6. Evaluate the final model on the test set.
7. Generate visualizations.
8. Save the experiment logs and summary.

### Load an existing model

Enter:

```text
n
```

The saved model and scaler are loaded and used for inference and evaluation without retraining.

## Training Process

### 1. Train/Test Split

The dataset is split into:

```text
80% Training
20% Test
```

A stratified split is used to preserve the class distribution.

### 2. Cross-Validation

The training data is evaluated using:

```python
StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)
```

Each fold uses the following pipeline:

```text
MinMaxScaler
      │
      ▼
PyTorchClassifier
```

### 3. Final Training

After cross-validation, the final model is trained on the complete training set.

The model weights and fitted scaler are then saved as:

```text
mnist_model_weights.pth
mnist_scaler.joblib
```

## Model

The neural network has the following architecture:

```text
Input: 784
   │
   ▼
Linear: 784 → 300
   │
 ReLU
   │
   ▼
Linear: 300 → 300
   │
 ReLU
   │
   ▼
Linear: 300 → 300
   │
 ReLU
   │
   ▼
Linear: 300 → 10
   │
   ▼
Output
```

### Training Parameters

| Parameter | Value |
|---|---:|
| Input Size | `784` |
| Hidden Size | `300` |
| Output Size | `10` |
| Optimizer | `Adam` |
| Learning Rate | `0.001` |
| Weight Decay | `0.0` |
| Epochs | `10` |
| Batch Size | `64` |
| Cross-Validation Folds | `10` |
| Test Size | `0.20` |
| Random Seed | `42` |

## Evaluation

The model is evaluated on both the training and test data.

The following metrics are calculated:

- Accuracy
- Precision
- Recall
- F1-score

Both macro and weighted averages are reported.

### Confusion Matrix

A confusion matrix is generated for the test data and saved in both CSV and PNG format.

### Training Loss

For a new training run, the loss and training accuracy are stored for every epoch.

The resulting data is saved as:

```text
best_model_loss_history.csv
```

and visualized as:

```text
best_model_loss.png
```

### Random Test Image

After evaluation, a random test image is selected.

The output includes:

- Test-image index
- Actual label
- Predicted label
- Whether the prediction was correct

The image is saved as:

```text
random_test_image.png
```

## Experiment Logging

The project contains a dedicated experiment logger.

Each training run receives a unique run ID:

```text
run_0001
run_0002
run_0003
...
```

Experiment information includes:

- Experiment and run IDs
- Start and end timestamps
- Model parameters
- Training parameters
- System information
- Python/PyTorch/NumPy/pandas versions
- CPU/GPU information
- Training metrics
- Validation metrics
- Test metrics
- Confusion matrices
- Loss history
- Runtime information

The experiment log is stored as:

```text
experiment_runs.jsonl
```

## Saved Results

All generated output is stored in:

```text
mnist_experiments/
```

| File | Description |
|---|---|
| `mnist_model_weights.pth` | PyTorch model checkpoint |
| `mnist_scaler.joblib` | Fitted `MinMaxScaler` |
| `experiment_runs.jsonl` | Detailed experiment log |
| `cross_validation_results.csv` | Results for all CV folds |
| `experiment_summary.json` | Experiment summary |
| `model_config.json` | Model and training configuration |
| `confusion_matrix_test.csv` | Test confusion matrix |
| `classification_report_test.csv` | Classification report |
| `best_model_loss_history.csv` | Loss and training-accuracy history |
| `confusion_matrix_test.png` | Confusion-matrix visualization |
| `best_model_loss.png` | Training-loss visualization |
| `random_test_image.png` | Random test image with prediction |

## Project Structure

A recommended repository structure is:

```text
mnist-pytorch-classification/
│
├── README.md
├── LICENSE
├── mnist.py
├── mnist_data_60k.csv
├── requirements.txt
├── .gitignore
│
└── mnist_experiments/
    ├── mnist_model_weights.pth
    ├── mnist_scaler.joblib
    ├── experiment_runs.jsonl
    ├── cross_validation_results.csv
    ├── experiment_summary.json
    ├── model_config.json
    ├── confusion_matrix_test.csv
    ├── classification_report_test.csv
    ├── best_model_loss_history.csv
    ├── confusion_matrix_test.png
    ├── best_model_loss.png
    └── random_test_image.png
```

For Git repositories, generated experiment artifacts can optionally be excluded from version control.

Example `.gitignore` entries:

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/

# Virtual environments
.venv/
venv/
env/

# IDEs
.vscode/
.idea/

# Experiment outputs
mnist_experiments/

# Test and cache directories
.pytest_cache/

# OS
.DS_Store
Thumbs.db
```

## Configuration

The main configuration is defined near the beginning of the Python script:

```python
SEED = 42

DATA_FILE = "mnist_data_60k.csv"
OUTPUT_DIR = "mnist_experiments"

INPUT_SIZE = 784
HIDDEN_SIZE = 300
OUTPUT_SIZE = 10

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0
EPOCHS = 10
BATCH_SIZE = 64

CV_FOLDS = 10
TEST_SIZE = 0.20
```

The configuration used for an experiment is also exported to:

```text
model_config.json
```

## CPU and GPU

The program automatically detects CUDA availability.

If CUDA is available, PyTorch uses the GPU; otherwise, computation falls back to the CPU.

The selected device and relevant CUDA/GPU information are also stored in the experiment metadata.

## Reproducibility

The project uses the fixed seed:

```python
SEED = 42
```

The seed is applied to:

- Python `random`
- NumPy
- PyTorch
- CUDA

Individual cross-validation training runs use derived random states based on the fold number.

> Exact reproducibility may still depend on hardware, CUDA version, PyTorch version, and backend behavior, especially when running on GPUs.

## Limitations

- The input dataset must contain exactly 784 pixel features.
- The dataset filename is currently configured directly in the Python script.
- Main hyperparameters are configured directly in the Python script.
- The loaded model is intended for inference and is not designed for further training.
- 10-fold cross-validation can require significant computation time, particularly on CPU-only systems.
- The current project does not provide a dedicated command-line interface for configuring hyperparameters.
- Actual evaluation metrics depend on the supplied dataset and execution environment.

## Possible Extensions

Potential future improvements include:

- [ ] Version-pinned `requirements.txt`
- [ ] External configuration file (`YAML`/`JSON`)
- [ ] Command-line interface using `argparse`
- [ ] Early stopping
- [ ] Learning-rate scheduling
- [ ] Hyperparameter optimization
- [ ] TensorBoard integration
- [ ] Automated tests
- [ ] Dedicated inference script
- [ ] REST API
- [ ] Docker support
- [ ] GitHub Actions / CI
- [ ] Automated experiment reports

## License

This project is licensed under the **MIT License**.

See the [LICENSE](LICENSE) file for the complete license text.

## Author

**<YOUR NAME>**

GitHub: `https://github.com/<USERNAME>`

## Project Status

**Status:** Experimental / Educational

This project is intended for practical exploration of MNIST classification with PyTorch and scikit-learn, as well as structured documentation and evaluation of machine-learning experiments.
