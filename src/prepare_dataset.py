"""Extract face_recognition encodings from data/raw/<identity> image folders."""

import argparse
from pathlib import Path

import face_recognition

from .data_preprocessing import load_images_from_folder, preprocess_image
from .utils import ENCODINGS_PATH, LABELS_PATH, RAW_DIR, save_encodings


def extract_dataset_encodings(dataset_dir=RAW_DIR):
    """Return one 128-value encoding per detectable face and identity label."""
    images = load_images_from_folder(dataset_dir)
    encodings, labels = [], []
    for path, image in images:
        try:
            rgb = preprocess_image(image)
            found = face_recognition.face_encodings(rgb)
        except (RuntimeError, ValueError) as error:
            print(f"Skipping {path}: {error}")
            continue
        if len(found) != 1:
            print(f"Skipping {path}: expected one face, found {len(found)}.")
            continue
        label = path.parent.name
        if path.parent.resolve() == Path(dataset_dir).resolve():
            print(f"Skipping {path}: put images inside a subfolder named for the identity.")
            continue
        encodings.append(found[0])
        labels.append(label)
    if not encodings:
        raise ValueError(
            f"No face encodings were extracted from {dataset_dir}. Add readable face images "
            "under one subfolder per identity."
        )
    return encodings, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing encoding and label files after reprocessing the dataset",
    )
    args = parser.parse_args()
    if (ENCODINGS_PATH.exists() or LABELS_PATH.exists()) and not args.overwrite:
        raise SystemExit(
            f"Existing encoding files were left unchanged. Pass --overwrite to replace "
            f"{ENCODINGS_PATH} and {LABELS_PATH}."
        )
    try:
        encodings, labels = extract_dataset_encodings()
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
    save_encodings(encodings, labels)
    print(f"Saved {len(encodings)} face encoding(s) for {len(set(labels))} identity/identities.")


if __name__ == "__main__":
    main()
