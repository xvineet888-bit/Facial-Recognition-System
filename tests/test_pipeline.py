import sys

import cv2
import joblib
import numpy as np
import pytest

import src.prepare_dataset as prepare_dataset
from src.data_preprocessing import load_images_from_folder, preprocess_image
from src.face_detection import detect_faces
from src.face_recognition import recognize_faces
from src.feature_extraction import extract_features, load_predictor
from src.model_training import train_model
from src.prepare_dataset import extract_dataset_encodings
from src.utils import load_encodings, save_encodings


def test_preprocess_image_converts_bgr_to_rgb():
    bgr = np.zeros((3, 4, 3), dtype=np.uint8)
    bgr[0, 0] = [10, 20, 30]
    rgb = preprocess_image(bgr)
    assert rgb.shape == (3, 4, 3)
    assert rgb[0, 0].tolist() == [30, 20, 10]


def test_preprocess_rejects_invalid_image():
    with pytest.raises(ValueError):
        preprocess_image(None)


def test_load_images_skips_unreadable_files(tmp_path):
    cv2.imwrite(str(tmp_path / "valid.png"), np.zeros((4, 4, 3), dtype=np.uint8))
    (tmp_path / "broken.jpg").write_text("not an image")
    assert len(load_images_from_folder(tmp_path)) == 1


def test_detect_faces_on_blank_image_returns_empty():
    assert len(detect_faces(np.zeros((100, 100, 3), dtype=np.uint8))) == 0


def test_detect_faces_rejects_invalid_image():
    with pytest.raises(ValueError):
        detect_faces(None)


def test_recognition_returns_best_match_and_handles_unknown():
    known = np.zeros((2, 128), dtype=np.float64)
    known[1, :] = 0.5
    query = np.zeros(128)
    assert recognize_faces(known, query) == 0
    assert recognize_faces(known, np.ones(128), tolerance=0.3) is None
    assert recognize_faces([], query) is None


def test_recognition_rejects_malformed_encoding():
    with pytest.raises(ValueError):
        recognize_faces([[1, 2]], np.zeros(128))


def test_feature_extraction_requires_real_predictor(tmp_path):
    with pytest.raises(FileNotFoundError, match="68-point landmark model"):
        load_predictor(tmp_path / "missing.dat")


def test_feature_extraction_validates_image_before_predictor():
    with pytest.raises(ValueError, match="empty or invalid"):
        extract_features(None, object(), predictor=object())


def test_save_and_load_encodings(tmp_path):
    enc_path, label_path = tmp_path / "enc.npy", tmp_path / "labels.npy"
    vectors = np.zeros((2, 128))
    save_encodings(vectors, ["alex", "sam"], enc_path, label_path)
    loaded, labels = load_encodings(enc_path, label_path)
    assert loaded.shape == (2, 128)
    assert labels.tolist() == ["alex", "sam"]


def test_save_encodings_requires_aligned_labels(tmp_path):
    with pytest.raises(ValueError, match="match"):
        save_encodings(np.zeros((2, 128)), ["alex"], tmp_path / "e.npy", tmp_path / "l.npy")


def test_model_training_saves_svc_and_reports_holdout(tmp_path):
    rng = np.random.default_rng(4)
    class_a = rng.normal(0, 0.02, (10, 128))
    class_b = rng.normal(1, 0.02, (10, 128))
    X = np.vstack([class_a, class_b])
    y = np.array(["a"] * 10 + ["b"] * 10)
    model_path = tmp_path / "nested" / "model.pkl"
    model, metrics = train_model(X, y, model_path)
    assert model_path.is_file()
    assert metrics["test_samples"] > 0
    assert 0 <= metrics["accuracy"] <= 1
    loaded_model = joblib.load(model_path)
    assert loaded_model.predict(X[:1]).tolist() == ["a"]


def test_model_training_rejects_bad_shapes():
    with pytest.raises(ValueError, match="128"):
        train_model(np.zeros((4, 4)), ["a", "a", "b", "b"])


def test_dataset_without_labeled_face_images_is_actionable(tmp_path):
    with pytest.raises(ValueError, match="No face encodings"):
        extract_dataset_encodings(tmp_path)


def test_dataset_skips_ambiguous_multi_face_images(tmp_path, monkeypatch):
    identity_dir = tmp_path / "person_a"
    identity_dir.mkdir()
    cv2.imwrite(str(identity_dir / "group.png"), np.zeros((8, 8, 3), dtype=np.uint8))
    monkeypatch.setattr(
        "src.prepare_dataset.face_recognition.face_encodings",
        lambda image: [np.zeros(128), np.ones(128)],
    )
    with pytest.raises(ValueError, match="No face encodings"):
        extract_dataset_encodings(tmp_path)


def test_dataset_command_refuses_to_overwrite_existing_encodings(tmp_path, monkeypatch):
    encodings = tmp_path / "face_encodings.npy"
    labels = tmp_path / "labels.npy"
    encodings.write_bytes(b"existing encodings")
    labels.write_bytes(b"existing labels")
    monkeypatch.setattr(prepare_dataset, "ENCODINGS_PATH", encodings)
    monkeypatch.setattr(prepare_dataset, "LABELS_PATH", labels)
    monkeypatch.setattr(sys, "argv", ["prepare_dataset"])
    with pytest.raises(SystemExit, match="--overwrite"):
        prepare_dataset.main()
    assert encodings.read_bytes() == b"existing encodings"
    assert labels.read_bytes() == b"existing labels"
