
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from insightface.app import FaceAnalysis


# --------------------------------------------------
# PATHS AND SETTINGS
# --------------------------------------------------

DESKTOP = Path.home() / "OneDrive" / "Desktop"
OUTPUT = DESKTOP / "lfw_modern_comparison"

OUTPUT.mkdir(parents=True, exist_ok=True)

PATHS = np.load(
    DESKTOP / "lfw_image_paths.npy",
    allow_pickle=True
)

LABELS = np.load(
    DESKTOP / "lfw_labels.npy",
    allow_pickle=True
)

GALLERY_LABELS = np.load(
    DESKTOP / "lfw_gallery_labels.npy",
    allow_pickle=True
)

TEST_LABELS = np.load(
    DESKTOP / "lfw_test_labels.npy",
    allow_pickle=True
)

torch.set_num_threads(4)
DEVICE = torch.device("cpu")

print("Images:", len(PATHS))
print("Labels:", len(LABELS))


# --------------------------------------------------
# RECREATE AND VERIFY EXACT ORIGINAL SPLIT
# --------------------------------------------------

from collections import defaultdict

groups = defaultdict(list)

for i, label in enumerate(LABELS):
    groups[str(label)].append(i)

rng = np.random.default_rng(42)

gallery_indices = []
test_indices = []

for person, indices in groups.items():
    indices = np.array(indices)
    rng.shuffle(indices)

    if len(indices) >= 2:
        gallery_indices.append(indices[0])
        test_indices.extend(indices[1:])
    else:
        test_indices.extend(indices)

gallery_indices = np.array(gallery_indices)
test_indices = np.array(test_indices)

assert np.array_equal(
    LABELS[gallery_indices], GALLERY_LABELS
), "Gallery split mismatch"

assert np.array_equal(
    LABELS[test_indices], TEST_LABELS
), "Test split mismatch"

print("Original split verified.")
print("Gallery:", len(gallery_indices))
print("Test:", len(test_indices))


# --------------------------------------------------
# FACENET MODEL
# --------------------------------------------------

print("\nLoading FaceNet...")

mtcnn = MTCNN(
    image_size=160,
    margin=0,
    keep_all=False,
    post_process=True,
    device=DEVICE
)

facenet = InceptionResnetV1(
    pretrained="vggface2"
).eval().to(DEVICE)

facenet_embeddings = np.zeros(
    (len(PATHS), 512),
    dtype=np.float32
)

facenet_valid = np.zeros(
    len(PATHS),
    dtype=bool
)

batch_tensors = []
batch_indices = []


def flush_facenet_batch():
    global batch_tensors, batch_indices

    if not batch_tensors:
        return

    with torch.no_grad():
        batch = torch.stack(batch_tensors).to(DEVICE)
        embeddings = facenet(batch).cpu().numpy()

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    embeddings = embeddings / np.maximum(norms, 1e-12)

    for idx, embedding in zip(batch_indices, embeddings):
        facenet_embeddings[idx] = embedding
        facenet_valid[idx] = True

    batch_tensors.clear()
    batch_indices.clear()


for i, image_path in enumerate(PATHS):

    try:
        with Image.open(str(image_path)) as image:
            image = image.convert("RGB")
            face_tensor = mtcnn(image)

        if face_tensor is not None:
            batch_tensors.append(face_tensor)
            batch_indices.append(i)

        if len(batch_tensors) >= 32:
            flush_facenet_batch()

    except Exception as error:
        print("FaceNet image error:", i, error)

    if (i + 1) % 100 == 0:
        print(
            f"FaceNet processed: {i + 1}/{len(PATHS)}"
        )

flush_facenet_batch()

np.savez_compressed(
    OUTPUT / "facenet_encodings.npz",
    embeddings=facenet_embeddings,
    valid=facenet_valid
)

print(
    "FaceNet successful images:",
    int(facenet_valid.sum())
)


# --------------------------------------------------
# ARCFACE MODEL
# --------------------------------------------------

print("\nLoading ArcFace...")

arcface = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

arcface.prepare(
    ctx_id=-1,
    det_size=(640, 640)
)

arcface_embeddings = np.zeros(
    (len(PATHS), 512),
    dtype=np.float32
)

arcface_valid = np.zeros(
    len(PATHS),
    dtype=bool
)


for i, image_path in enumerate(PATHS):

    try:
        image = cv2.imread(str(image_path))

        if image is None:
            continue

        faces = arcface.get(image)

        if not faces:
            continue

        # Select the largest detected face.
        face = max(
            faces,
            key=lambda f: (
                (f.bbox[2] - f.bbox[0]) *
                (f.bbox[3] - f.bbox[1])
            )
        )

        embedding = np.asarray(
            face.normed_embedding,
            dtype=np.float32
        )

        norm = np.linalg.norm(embedding)

        if norm > 0:
            arcface_embeddings[i] = embedding / norm
            arcface_valid[i] = True

    except Exception as error:
        print("ArcFace image error:", i, error)

    if (i + 1) % 100 == 0:
        print(
            f"ArcFace processed: {i + 1}/{len(PATHS)}"
        )


np.savez_compressed(
    OUTPUT / "arcface_encodings.npz",
    embeddings=arcface_embeddings,
    valid=arcface_valid
)

print(
    "ArcFace successful images:",
    int(arcface_valid.sum())
)


# --------------------------------------------------
# SAVE SPLIT INDICES AND SUMMARY
# --------------------------------------------------

np.savez(
    OUTPUT / "lfw_split_indices.npz",
    gallery_indices=gallery_indices,
    test_indices=test_indices
)

print("\n========== GENERATION COMPLETE ==========")
print("Output folder:", OUTPUT)
print("Total images:", len(PATHS))
print("FaceNet valid:", int(facenet_valid.sum()))
print("ArcFace valid:", int(arcface_valid.sum()))
print("Gallery:", len(gallery_indices))
print("Test:", len(test_indices))
print("=========================================")
