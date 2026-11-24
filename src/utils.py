"""Shared paths and encoding persistence helpers."""

from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
ENCODINGS_PATH = PROCESSED_DIR / "face_encodings.npy"
LABELS_PATH = PROCESSED_DIR / "labels.npy"
CLASSIFIER_PATH = MODELS_DIR / "face_recognition_model.pkl"


def save_encodings(encodings, labels, enc_file=ENCODINGS_PATH, label_file=LABELS_PATH):
    """Save aligned encodings and labels, creating parent directories as needed."""
    encodings = np.asarray(encodings, dtype=np.float64)
    labels = np.asarray(labels)
    if len(encodings) != len(labels):
        raise ValueError("The number of face encodings must match the number of labels.")
    Path(enc_file).parent.mkdir(parents=True, exist_ok=True)
    Path(label_file).parent.mkdir(parents=True, exist_ok=True)
    np.save(enc_file, encodings)
    np.save(label_file, labels)


def load_encodings(enc_file=ENCODINGS_PATH, label_file=LABELS_PATH):
    """Load and validate aligned face encodings and labels."""
    if not Path(enc_file).is_file() or not Path(label_file).is_file():
        raise FileNotFoundError(
            "Face encodings are missing. Run `python -m src.prepare_dataset` first."
        )
    encodings = np.load(enc_file, allow_pickle=False)
    labels = np.load(label_file, allow_pickle=False)
    if encodings.ndim != 2 or encodings.shape[1] != 128:
        raise ValueError("Expected face encodings with shape (n, 128).")
    if len(encodings) != len(labels):
        raise ValueError("Encoding and label files contain different numbers of entries.")
    if len(encodings) == 0:
        raise ValueError("No face encodings are available.")
    return encodings, labels
