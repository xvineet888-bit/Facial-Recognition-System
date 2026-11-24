"""Dataset image loading and lightweight preprocessing."""

from pathlib import Path

import cv2

from .utils import RAW_DIR


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_images_from_folder(folder):
    """Load readable images recursively, returning ``(path, BGR image)`` pairs."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {folder}")
    images = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is not None:
                images.append((path, image))
    return images


def preprocess_image(image, size=None):
    """Validate a BGR image and return RGB uint8 pixels for face_recognition."""
    if image is None or not hasattr(image, "shape") or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a valid three-channel BGR image.")
    if image.dtype != "uint8":
        image = image.clip(0, 255).astype("uint8")
    if size is not None:
        image = cv2.resize(image, tuple(size))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main():
    images = load_images_from_folder(RAW_DIR)
    if not images:
        raise SystemExit(f"No readable images found in {RAW_DIR}.")
    output = RAW_DIR.parent / "processed"
    for path, image in images:
        destination = output / path.relative_to(RAW_DIR)
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Persist in the original BGR convention; recognition converts on load.
        if not cv2.imwrite(str(destination), image):
            print(f"Warning: could not write {destination}")
    print(f"Copied {len(images)} readable images to {output}.")


if __name__ == "__main__":
    main()
