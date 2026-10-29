"""Central configuration: paths, constants and dataset metadata."""
from __future__ import annotations

import string
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
SAMPLES_DIR = PROJECT_ROOT / "sample_images"

IMG_SIZE = 28
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 1)
SEED = 42

DATASETS = {
    "mnist": {
        "title": "Digit Recognition (MNIST)",
        "short": "MNIST",
        "class_names": [str(d) for d in range(10)],
        "description": "MNIST handwritten digits 0-9 (60,000 train / 10,000 test).",
    },
    "emnist": {
        "title": "Alphabet Recognition (EMNIST)",
        "short": "EMNIST Letters",
        "class_names": list(string.ascii_uppercase),
        "description": "EMNIST Letters A-Z. Upper and lower case share a class.",
    },
}


def get_dataset_info(name: str) -> dict:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Choose one of: {list(DATASETS)}")
    return DATASETS[name]


def model_path(name: str) -> Path:
    return MODELS_DIR / f"{name}_cnn.keras"


def labels_path(name: str) -> Path:
    return MODELS_DIR / f"{name}_labels.json"


def metrics_path(name: str) -> Path:
    return METRICS_DIR / f"{name}_metrics.json"


def report_path(name: str) -> Path:
    return METRICS_DIR / f"{name}_classification_report.txt"


def history_path(name: str) -> Path:
    return METRICS_DIR / f"{name}_history.csv"


def ensure_directories() -> None:
    for folder in (DATA_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR, SAMPLES_DIR):
        folder.mkdir(parents=True, exist_ok=True)
