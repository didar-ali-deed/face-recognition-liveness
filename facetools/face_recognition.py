import urllib
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import onnxruntime
import pandas as pd
import progressbar

pbar = None

# Empirically tuned default: InceptionResNetV1 (VGGFace2) L2 distances
# below this value are considered the same identity.
# Lower = stricter. Typical range: 0.6 (strict) – 1.1 (lenient).
DEFAULT_THRESHOLD = 0.9


class IdentityVerification:
    """
    Verifies whether a detected face matches a known identity in the facebank.

    Facebank CSV format (with header row):
        name, emb_0, emb_1, ..., emb_511

    Each row is one stored embedding for a known person.
    Multiple rows can share the same name (multiple reference photos per person).
    """

    def __init__(
        self,
        checkpoint_path: str,
        facebank_path: str,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        """
        Args:
            checkpoint_path: Path to InceptionResNetV1 ONNX model.
            facebank_path:   Path to facebank CSV (name + 512-dim embeddings).
            threshold:       L2 distance below which a face is considered a match.
                             Default is 0.9. Lower values = stricter matching.
        """
        if not Path(checkpoint_path).is_file():
            print("Downloading the Inception ResNet V1 ONNX checkpoint...")
            urllib.request.urlretrieve(
                "https://github.com/ffletcherr/face-recognition-liveness/releases/download/v0.1/InceptionResnetV1_vggface2.onnx",
                Path(checkpoint_path).absolute().as_posix(),
                show_progress,
            )

        if not Path(facebank_path).is_file():
            raise FileNotFoundError(
                f"Facebank not found: {facebank_path}\n"
                "Run create_facebank.py to generate it."
            )

        self.threshold = threshold
        self.resnet = onnxruntime.InferenceSession(
            checkpoint_path, providers=["CPUExecutionProvider"]
        )

        df = pd.read_csv(facebank_path)
        self._validate_facebank(df, facebank_path)

        self.names: np.ndarray = df.iloc[:, 0].values          # shape (N,)
        self.embeddings: np.ndarray = df.iloc[:, 1:].values    # shape (N, 512)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(
        self, face_arr: np.ndarray
    ) -> Tuple[str, float, float, bool]:
        """
        Compare a face against every entry in the facebank.

        Args:
            face_arr: Cropped face image as BGR numpy array.

        Returns:
            matched_name:    Name of the closest identity, or "unknown" if no
                             entry passes the threshold.
            min_dist:        L2 distance to the closest stored embedding.
            mean_dist:       Mean L2 distance across all stored embeddings.
            is_match:        True if min_dist < threshold.
        """
        embedding = self._embed(face_arr)
        distances = np.linalg.norm(self.embeddings - embedding, axis=1)

        min_idx = int(np.argmin(distances))
        min_dist = float(np.round(distances[min_idx], 3))
        mean_dist = float(np.round(distances.mean(), 3))
        is_match = min_dist < self.threshold
        matched_name = self.names[min_idx] if is_match else "unknown"

        return matched_name, min_dist, mean_dist, is_match

    def compare(
        self, face_arr_a: np.ndarray, face_arr_b: np.ndarray
    ) -> Tuple[float, bool]:
        """
        Directly compare two face crops — no facebank lookup.

        Useful for ID-vs-selfie workflows where you have two photos and
        just want to know if they're the same person.

        Args:
            face_arr_a: First face crop (BGR numpy array).
            face_arr_b: Second face crop (BGR numpy array).

        Returns:
            distance: L2 distance between the two embeddings.
            is_match: True if distance < threshold.
        """
        emb_a = self._embed(face_arr_a)
        emb_b = self._embed(face_arr_b)
        distance = float(np.round(np.linalg.norm(emb_a - emb_b), 3))
        is_match = distance < self.threshold
        return distance, is_match

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed(self, face_arr: np.ndarray) -> np.ndarray:
        """Run inference and return a 512-dim L2-normalised embedding."""
        face_arr = np.moveaxis(face_arr, -1, 0)                          # HWC → CHW
        input_arr = np.expand_dims((face_arr - 127.5) / 128.0, 0)       # normalise
        embedding = self.resnet.run(
            ["output"], {"input": input_arr.astype(np.float32)}
        )[0]
        return embedding  # shape (1, 512)

    @staticmethod
    def _validate_facebank(df: pd.DataFrame, path: str) -> None:
        if df.shape[1] < 2:
            raise ValueError(
                f"Facebank at {path} must have at least 2 columns: "
                "name + embedding dimensions. "
                "Re-run create_facebank.py to regenerate."
            )
        if df.shape[1] != 513:  # 1 name col + 512 embedding dims
            raise ValueError(
                f"Expected 513 columns (name + 512 embedding dims), "
                f"got {df.shape[1]}. Re-run create_facebank.py."
            )


def show_progress(block_num, block_size, total_size):
    global pbar
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)
        pbar.start()
    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None