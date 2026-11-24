"""Optional dlib 68-point landmark extraction (separate from face encodings)."""

from functools import lru_cache
from pathlib import Path

import cv2
import dlib

from .utils import MODELS_DIR


DEFAULT_PREDICTOR_PATH = MODELS_DIR / "shape_predictor_68_face_landmarks.dat"


@lru_cache(maxsize=4)
def load_predictor(model_path=None):
    """Load a local or face_recognition_models 68-point predictor."""
    if model_path is None:
        path = DEFAULT_PREDICTOR_PATH
        if not path.is_file():
            try:
                import face_recognition_models

                path = Path(face_recognition_models.pose_predictor_model_location())
            except ImportError:
                path = DEFAULT_PREDICTOR_PATH
    else:
        path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"The dlib 68-point landmark model is not present at {path}. Install the "
            "face_recognition_models dependency or download "
            "shape_predictor_68_face_landmarks.dat.bz2 from https://dlib.net/files/, "
            "extract the .dat file into data/models/, and review its terms."
        )
    return dlib.shape_predictor(str(path))


def extract_features(image, face, predictor=None):
    """Return the 68 landmark points for a dlib rectangle and BGR image."""
    if image is None or not hasattr(image, "shape") or image.size == 0:
        raise ValueError("Image is empty or invalid.")
    predictor = predictor if predictor is not None else load_predictor()
    if image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError("Expected a grayscale or three-channel BGR image.")
    shape = predictor(gray, face)
    return [(shape.part(i).x, shape.part(i).y) for i in range(68)]
