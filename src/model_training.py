"""Train and evaluate the separate linear SVC classification pipeline."""

from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from .utils import CLASSIFIER_PATH, load_encodings


def train_model(face_encodings, labels, model_path=CLASSIFIER_PATH, test_size=0.2):
    """Split, fit, evaluate, and save a linear SVC; return model and metrics."""
    X = np.asarray(face_encodings, dtype=np.float64)
    y = np.asarray(labels)
    if X.ndim != 2 or X.shape[1] != 128:
        raise ValueError("face_encodings must have shape (n_samples, 128).")
    if len(X) != len(y):
        raise ValueError("The number of face encodings must match the number of labels.")
    classes, counts = np.unique(y, return_counts=True)
    if len(X) < 4 or len(classes) < 2:
        raise ValueError("Training requires at least four samples across at least two identities.")
    if counts.min() < 2:
        raise ValueError("Each identity needs at least two samples for a stratified split.")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")
    test_count = int(np.ceil(len(X) * test_size)) if isinstance(test_size, float) else test_size
    if test_count < len(classes) or len(X) - test_count < len(classes):
        raise ValueError("The selected split needs enough samples for every identity in both sets.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    model = SVC(kernel="linear", probability=True, random_state=42)
    model.fit(X_train, y_train)
    accuracy = float(model.score(X_test, y_test))
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model, {"accuracy": accuracy, "train_samples": len(X_train), "test_samples": len(X_test)}


def main():
    try:
        encodings, labels = load_encodings()
        _, results = train_model(encodings, labels)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(f"Held-out accuracy: {results['accuracy']:.2%}")
    print(f"Training samples: {results['train_samples']}; test samples: {results['test_samples']}")
    print(f"Saved SVC classifier to {CLASSIFIER_PATH}")


if __name__ == "__main__":
    main()
