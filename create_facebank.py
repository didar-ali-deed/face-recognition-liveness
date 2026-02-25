"""
create_facebank.py
------------------
Build (or update) a facebank CSV from a folder of reference images.

Facebank CSV format:
    name, emb_0, emb_1, ..., emb_511

One row per image. Multiple images of the same person are stored as
separate rows — IdentityVerification will pick the closest match across
all of them automatically.

Usage examples
--------------
# Build from scratch — images in data/images/, one identity called "reynolds"
python create_facebank.py --images_dir data/images --name reynolds --output data/reynolds.csv

# Add a new identity to an existing facebank
python create_facebank.py --images_dir data/images/didar --name didar --output data/facebank.csv --append

# Build a multi-identity facebank where each sub-folder is an identity name
python create_facebank.py --images_dir data/identities --output data/facebank.csv --subdirs
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime
import pandas as pd

from facetools import FaceDetection

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_embedding(resnet: onnxruntime.InferenceSession, face_arr: np.ndarray) -> np.ndarray:
    face = np.moveaxis(face_arr, -1, 0)
    inp = np.expand_dims((face - 127.5) / 128.0, 0).astype(np.float32)
    return resnet.run(["output"], {"input": inp})[0]  # shape (1, 512)


def process_image(
    img_path: Path,
    name: str,
    detector: FaceDetection,
    resnet: onnxruntime.InferenceSession,
) -> list:
    """Return a list of rows (one per detected face) ready for the CSV."""
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"  [WARN] Could not read {img_path.name}, skipping.")
        return []

    faces, _ = detector(image)
    if not faces:
        print(f"  [WARN] No face detected in {img_path.name}, skipping.")
        return []

    rows = []
    for face_arr in faces:
        embedding = get_embedding(resnet, face_arr).flatten()
        row = [name] + embedding.tolist()
        rows.append(row)
        print(f"  [OK]   {img_path.name}  → identity: '{name}'")

    return rows


def build_header(embedding_dim: int = 512) -> list:
    return ["name"] + [f"emb_{i}" for i in range(embedding_dim)]


def main():
    parser = argparse.ArgumentParser(description="Build a labelled facebank CSV.")
    parser.add_argument(
        "--images_dir", required=True,
        help="Folder containing reference images (or sub-folders if --subdirs)."
    )
    parser.add_argument(
        "--name", default=None,
        help="Identity label for all images in images_dir. Required unless --subdirs is set."
    )
    parser.add_argument(
        "--output", required=True,
        help="Output CSV path (e.g. data/facebank.csv)."
    )
    parser.add_argument(
        "--checkpoint", default="data/checkpoints/InceptionResnetV1_vggface2.onnx",
        help="Path to InceptionResNetV1 ONNX checkpoint."
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Append rows to an existing facebank instead of overwriting."
    )
    parser.add_argument(
        "--subdirs", action="store_true",
        help="Treat each sub-folder of images_dir as a separate identity (folder name = label)."
    )
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    output_path = Path(args.output)

    if not images_dir.is_dir():
        raise NotADirectoryError(f"images_dir not found: {images_dir}")

    if not args.subdirs and args.name is None:
        parser.error("--name is required unless --subdirs is set.")

    # Load models
    print("Loading face detector and recognition model...")
    detector = FaceDetection(max_num_faces=1)
    resnet = onnxruntime.InferenceSession(
        args.checkpoint, providers=["CPUExecutionProvider"]
    )

    # Collect (image_path, identity_name) pairs
    pairs = []
    if args.subdirs:
        for subdir in sorted(images_dir.iterdir()):
            if subdir.is_dir():
                for img_path in sorted(subdir.iterdir()):
                    if img_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                        pairs.append((img_path, subdir.name))
    else:
        for img_path in sorted(images_dir.iterdir()):
            if img_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                pairs.append((img_path, args.name))

    if not pairs:
        print("No supported images found. Exiting.")
        return

    # Build new rows
    print(f"\nProcessing {len(pairs)} image(s)...\n")
    new_rows = []
    for img_path, name in pairs:
        new_rows.extend(process_image(img_path, name, detector, resnet))

    if not new_rows:
        print("\nNo faces were detected in any image. Facebank not written.")
        return

    header = build_header()
    new_df = pd.DataFrame(new_rows, columns=header)

    # Append to existing or create fresh
    if args.append and output_path.is_file():
        existing_df = pd.read_csv(output_path)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.to_csv(output_path, index=False)
        print(f"\nAppended {len(new_rows)} row(s) to {output_path}  "
              f"(total: {len(combined_df)} rows)")
    else:
        new_df.to_csv(output_path, index=False)
        print(f"\nFacebank written → {output_path}  ({len(new_rows)} rows)")


if __name__ == "__main__":
    main()