"""Distance based matching helpers built on face_recognition encodings."""

import numpy as np
import face_recognition as face_recognition_lib


DEFAULT_TOLERANCE = 0.6


def recognize_faces(known_face_encodings, face_encoding, tolerance=DEFAULT_TOLERANCE):
    """Return the closest matching index, or ``None`` when no match qualifies."""
    known = np.asarray(known_face_encodings, dtype=np.float64)
    query = np.asarray(face_encoding, dtype=np.float64)
    if known.size == 0:
        return None
    if known.ndim != 2 or known.shape[1] != 128 or query.shape != (128,):
        raise ValueError("Known and query encodings must have 128 values per face.")
    if not 0.0 < tolerance <= 2.0:
        raise ValueError("tolerance must be greater than 0 and at most 2.")
    distances = face_recognition_lib.face_distance(known, query)
    best = int(np.argmin(distances))
    return best if distances[best] <= tolerance else None


def main():
    raise SystemExit("Use `python -m src.prepare_dataset` to create encodings for a labeled dataset.")


if __name__ == "__main__":
    main()
