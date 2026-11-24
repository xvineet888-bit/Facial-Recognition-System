"""Face detection using dlib's frontal face detector."""

import cv2
import dlib


_detector = dlib.get_frontal_face_detector()


def detect_faces(image):
    """Return dlib rectangles for a valid BGR or grayscale uint8 image."""
    if image is None or not hasattr(image, "shape") or image.size == 0:
        raise ValueError("Image is empty or invalid.")
    if image.dtype != "uint8":
        image = image.clip(0, 255).astype("uint8")
    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim != 2:
        raise ValueError("Expected a grayscale or three-channel BGR image.")
    return _detector(image, 1)


def main():
    from .data_preprocessing import load_images_from_folder
    from .utils import PROCESSED_DIR

    images = load_images_from_folder(PROCESSED_DIR)
    if not images:
        raise SystemExit(f"No processed images found in {PROCESSED_DIR}.")
    path, image = images[0]
    faces = detect_faces(image)
    print(f"Detected {len(faces)} face(s) in {path.name}.")


if __name__ == "__main__":
    main()
