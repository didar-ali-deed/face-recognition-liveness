"""
liveness_detection.py
---------------------
Liveness detection using an ensemble of two complementary models:

1. DeepPixBiS  — pixel-wise binary supervision, strong on texture-based attacks
                 (printed photos, low-quality replays).
2. MiniFASNet  — Silent-Face Anti-Spoofing (CVPR 2021 workshop), trained on a
                 large-scale dataset covering printed photos, screen replay, and
                 3D mask attacks. Much better generalisation across cameras and
                 lighting conditions.

Final liveness score = weighted average of both model outputs.
A score > threshold is considered LIVE.
"""

import urllib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime
import progressbar
from PIL import Image
from torchvision import transforms as T

pbar = None

# ---------------------------------------------------------------------------
# Download URLs
# ---------------------------------------------------------------------------
_DEEPPIX_URL = (
    "https://github.com/ffletcherr/face-recognition-liveness/releases/"
    "download/v0.1/OULU_Protocol_2_model_0_0.onnx"
)
_MINIFAS_URL = (
    "https://github.com/ffletcherr/face-recognition-liveness/releases/"
    "download/v0.1/MiniFASNetV2.onnx"
)

# ---------------------------------------------------------------------------
# Default thresholds
# ---------------------------------------------------------------------------
DEFAULT_LIVENESS_THRESHOLD = 0.5   # score > 0.5 → LIVE
DEEPPIX_WEIGHT = 0.4               # contribution of DeepPixBiS to ensemble
MINIFAS_WEIGHT = 0.6               # MiniFASNet generalises better → higher weight


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class LivenessResult:
    """Structured result from liveness detection."""
    is_live: bool               # Final pass/fail decision
    score: float                # Ensemble score in [0, 1]  (higher = more live)
    deeppix_score: float        # Raw DeepPixBiS score
    minifas_score: float        # Raw MiniFASNet score
    threshold: float            # Threshold used for the decision

    def to_dict(self) -> dict:
        return {
            "is_live": self.is_live,
            "score": self.score,
            "deeppix_score": self.deeppix_score,
            "minifas_score": self.minifas_score,
            "threshold": self.threshold,
        }


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
_deeppix_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_minifas_transform = T.Compose([
    T.Resize((80, 80)),       # MiniFASNetV2 input size
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class LivenessDetection:
    """
    Ensemble liveness detector.

    Combines DeepPixBiS and MiniFASNetV2 to provide robust anti-spoofing
    against printed photo, screen replay, and 3D mask attacks.

    Args:
        deeppix_checkpoint_path: Path to OULU_Protocol_2_model_0_0.onnx
        minifas_checkpoint_path: Path to MiniFASNetV2.onnx
                                 Pass None to disable (uses DeepPixBiS only).
        threshold:               Score above which a face is considered LIVE.
                                 Default 0.5. Lower = more permissive.
        deeppix_weight:          Weight for DeepPixBiS in the ensemble (0–1).
        minifas_weight:          Weight for MiniFASNet in the ensemble (0–1).
    """

    def __init__(
        self,
        checkpoint_path: str,                    # kept for backward compatibility
        minifas_checkpoint_path: Optional[str] = None,
        threshold: float = DEFAULT_LIVENESS_THRESHOLD,
        deeppix_weight: float = DEEPPIX_WEIGHT,
        minifas_weight: float = MINIFAS_WEIGHT,
    ):
        self.threshold = threshold
        self.deeppix_weight = deeppix_weight
        self.minifas_weight = minifas_weight

        # --- DeepPixBiS ---
        self._deeppix = self._load_or_download(
            checkpoint_path, _DEEPPIX_URL, "DeepPixBiS"
        )

        # --- MiniFASNetV2 ---
        self._minifas = None
        if minifas_checkpoint_path is not None:
            self._minifas = self._load_or_download(
                minifas_checkpoint_path, _MINIFAS_URL, "MiniFASNetV2"
            )

        if self._minifas is None:
            print(
                "[LivenessDetection] MiniFASNet not loaded — running DeepPixBiS only.\n"
                "  Pass minifas_checkpoint_path to enable the full ensemble."
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(self, face_arr: np.ndarray) -> LivenessResult:
        """
        Run liveness detection on a cropped face.

        Args:
            face_arr: BGR numpy array (cropped face from FaceDetection).

        Returns:
            LivenessResult with score, per-model scores, and is_live flag.
        """
        deeppix_score = self._run_deeppix(face_arr)

        if self._minifas is not None:
            minifas_score = self._run_minifas(face_arr)
            # Normalise weights so they always sum to 1
            total_w = self.deeppix_weight + self.minifas_weight
            ensemble_score = (
                self.deeppix_weight * deeppix_score
                + self.minifas_weight * minifas_score
            ) / total_w
        else:
            minifas_score = float("nan")
            ensemble_score = deeppix_score

        ensemble_score = float(np.round(ensemble_score, 4))
        is_live = ensemble_score > self.threshold

        return LivenessResult(
            is_live=is_live,
            score=ensemble_score,
            deeppix_score=float(np.round(deeppix_score, 4)),
            minifas_score=float(np.round(minifas_score, 4)) if self._minifas else float("nan"),
            threshold=self.threshold,
        )

    # ------------------------------------------------------------------
    # Private — model runners
    # ------------------------------------------------------------------

    def _run_deeppix(self, face_arr: np.ndarray) -> float:
        face_rgb = cv2.cvtColor(face_arr, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        tensor = _deeppix_transform(face_pil).unsqueeze(0).detach().cpu().numpy()
        output_pixel, output_binary = self._deeppix.run(
            ["output_pixel", "output_binary"],
            {"input": tensor.astype(np.float32)},
        )
        return float(
            (np.mean(output_pixel.flatten()) + np.mean(output_binary.flatten())) / 2.0
        )

    def _run_minifas(self, face_arr: np.ndarray) -> float:
        face_rgb = cv2.cvtColor(face_arr, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        tensor = _minifas_transform(face_pil).unsqueeze(0).detach().cpu().numpy()
        output = self._minifas.run(
            ["output"], {"input": tensor.astype(np.float32)}
        )[0]
        # MiniFASNet outputs [spoof_prob, live_prob] — take live probability
        probs = self._softmax(output.flatten())
        return float(probs[1])  # index 1 = live

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / e.sum()

    # ------------------------------------------------------------------
    # Private — model loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_or_download(
        checkpoint_path: str, url: str, name: str
    ) -> onnxruntime.InferenceSession:
        path = Path(checkpoint_path)
        if not path.is_file():
            print(f"Downloading {name} ONNX checkpoint...")
            urllib.request.urlretrieve(
                url, path.absolute().as_posix(), show_progress
            )
        return onnxruntime.InferenceSession(
            checkpoint_path, providers=["CPUExecutionProvider"]
        )


# ---------------------------------------------------------------------------
# Progress bar helper
# ---------------------------------------------------------------------------
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