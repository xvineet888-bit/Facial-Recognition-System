"""Webcam identification using distance matching against saved encodings."""

import os
import sys
from pathlib import Path

import cv2

if __package__ in (None, ""):
    # Support both `python src/real_time_integration.py` and `python -m ...`.
    project_root = Path(__file__).resolve().parents[1]
    script_dir = Path(__file__).resolve().parent
    sys.path = [
        entry for entry in sys.path
        if not entry or Path(entry).resolve() != script_dir
    ]
    sys.path.insert(0, str(project_root))
    from src.face_recognition import recognize_faces
    from src.utils import load_encodings
else:
    from .face_recognition import recognize_faces
    from .utils import load_encodings

import face_recognition


def run_webcam(camera_index=0, tolerance=0.6):
    encodings, labels = load_encodings()
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    camera = cv2.VideoCapture(camera_index, backend)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Could not open webcam at index {camera_index}.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                raise RuntimeError("The webcam did not return a valid frame.")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(rgb)
            face_encodings = face_recognition.face_encodings(rgb, locations)
            for (top, right, bottom, left), encoding in zip(locations, face_encodings):
                match_index = recognize_faces(encodings, encoding, tolerance=tolerance)
                name = str(labels[match_index]) if match_index is not None else "Unknown"
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, name, (left + 6, max(top - 8, 20)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow("Facial Recognition (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main():
    try:
        run_webcam()
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
