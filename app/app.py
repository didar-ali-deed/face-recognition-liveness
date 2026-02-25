"""
app.py
------
KYC-ready Flask API for NFC e-Passport verification.

Designed to integrate directly with the React Native KYC app (LivenessService.ts).

Endpoints
---------
GET  /health          — Public health check
POST /liveness        — Liveness detection on a selfie
POST /face-match      — Compare passport DG2 face vs live selfie  ← KYC primary
POST /identity        — Facebank identity lookup (internal use)
POST /main            — All-in-one: identity + liveness
POST /compare         — Legacy multipart photo comparison

Input formats accepted
----------------------
All POST endpoints accept two input formats:

  1. JSON body (used by React Native app):
     Content-Type: application/json
     {
       "image":          "<base64 JPEG>",   (for /liveness, /identity, /main)
       "reference_image": "<base64 JPEG>",  (for /face-match — passport DG2)
       "selfie_image":    "<base64 JPEG>",  (for /face-match — live selfie)
       "dg2_hex":         "<hex string>"    (for /face-match — raw DG2 bytes, alternative)
     }

  2. Raw bytes body (for direct testing via curl / client.py):
     Content-Type: image/jpeg
     <raw image bytes>

Response format for /liveness (matches LivenessService.ts expectations):
  { "passed": true, "score": 0.91 }

Response format for /face-match (matches LivenessService.ts expectations):
  { "matched": true, "score": 0.88 }

Security
--------
  - API key via X-API-Key header
  - Per-IP rate limiting
  - 10MB request size limit (DG2 can be up to 50KB, selfie larger)
  - Image format validation
  - Structured logging
"""

import base64
import logging
import os
import time
from functools import wraps
from os import environ
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from facetools import FaceDetection, IdentityVerification, LivenessDetection
from flask import Flask, Response, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from dg2_decoder import decode_dg2_to_cv2

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
root = Path(os.path.abspath(__file__)).parent.absolute()
load_dotenv((root / ".env").as_posix())

data_folder        = root.parent / environ.get("DATA_FOLDER", "data")
resnet_name        = environ.get("RESNET", "InceptionResnetV1_vggface2.onnx")
deeppix_name       = environ.get("DEEPPIX", "OULU_Protocol_2_model_0_0.onnx")
minifas_name       = environ.get("MINIFAS", "MiniFASNetV2.onnx")
facebank_name      = environ.get("FACEBANK", "facebank.csv")
max_content_mb     = float(environ.get("MAX_CONTENT_MB", "10"))   # 10MB — DG2 can be large
rate_limit         = environ.get("RATE_LIMIT", "30 per minute")

# KYC thresholds — aligned with React Native app decision logic
LIVENESS_THRESHOLD    = float(environ.get("LIVENESS_THRESHOLD", "0.80"))   # app requires >= 0.80
FACE_MATCH_THRESHOLD  = float(environ.get("FACE_MATCH_THRESHOLD", "0.85"))  # app requires >= 0.85
IDENTITY_THRESHOLD    = float(environ.get("IDENTITY_THRESHOLD", "0.9"))

_raw_keys = environ.get("API_KEYS", "")
VALID_API_KEYS = set(k.strip() for k in _raw_keys.split(",") if k.strip())
if not VALID_API_KEYS:
    log.warning("No API_KEYS set — all requests will be rejected. Set API_KEYS in app/.env")

# ---------------------------------------------------------------------------
# Image format detection (no imghdr — removed in Python 3.13)
# ---------------------------------------------------------------------------
_IMAGE_SIGNATURES = [
    (b"\xff\xd8\xff", "jpeg"),
    (b"\x89PNG",      "png"),
    (b"BM",           "bmp"),
]

def detect_image_type(data: bytes) -> str:
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    for sig, fmt in _IMAGE_SIGNATURES:
        if data[:len(sig)] == sig:
            return fmt
    return "unknown"

ALLOWED_IMAGE_TYPES = {"jpeg", "png", "bmp", "webp"}

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
resNet_path  = data_folder / "checkpoints" / resnet_name
deeppix_path = data_folder / "checkpoints" / deeppix_name
minifas_path = data_folder / "checkpoints" / minifas_name
facebank_path = data_folder / facebank_name

log.info("Loading models...")
faceDetector = FaceDetection()
identityChecker = IdentityVerification(
    checkpoint_path=resNet_path.as_posix(),
    facebank_path=facebank_path.as_posix(),
    threshold=IDENTITY_THRESHOLD,
)
livenessDetector = LivenessDetection(
    checkpoint_path=deeppix_path.as_posix(),
    minifas_checkpoint_path=(
        minifas_path.as_posix() if minifas_path.is_file() else None
    ),
    threshold=LIVENESS_THRESHOLD,
)
log.info("Models loaded successfully.")

# ---------------------------------------------------------------------------
# Flask + rate limiter
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(max_content_mb * 1024 * 1024)

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=[rate_limit],
    storage_uri="memory://",
)


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------
def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get("X-API-Key", "")
        if key not in VALID_API_KEYS:
            log.warning("Unauthorized from %s", request.remote_addr)
            return _err("Invalid or missing API key.", 401)
        return f(*args, **kwargs)
    return decorated


def log_request(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        t = time.perf_counter()
        result = f(*args, **kwargs)
        log.info("%s %s %s — %.0fms",
                 request.method, request.path,
                 request.remote_addr,
                 (time.perf_counter() - t) * 1000)
        return result
    return decorated


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------
def _err(message: str, status: int) -> Response:
    import json
    return Response(json.dumps({"error": message}), status=status,
                    mimetype="application/json")


def _ok(data: dict) -> Response:
    import json
    return Response(json.dumps(data), status=200, mimetype="application/json")


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------
def _raw_to_image(raw: bytes):
    """
    Validate and decode raw image bytes → BGR numpy array.
    Returns (image, error_response). One will always be None.
    """
    img_type = detect_image_type(raw)
    if img_type not in ALLOWED_IMAGE_TYPES:
        return None, _err(
            f"Unsupported image format '{img_type}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_IMAGE_TYPES))}.", 400)
    nparr = np.frombuffer(raw, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        return None, _err("Could not decode image.", 400)
    return image, None


def _resolve_image(field_name: str):
    """
    Resolve a single image from either:
      - JSON body: request.json[field_name] as base64 string
      - Raw bytes body: request.data (when field_name == 'image')
      - Multipart: request.files[field_name]

    Returns (image_ndarray, error_response). One will always be None.
    """
    # JSON base64
    if request.is_json:
        b64 = (request.json or {}).get(field_name)
        if not b64:
            return None, _err(f"Missing field '{field_name}' in JSON body.", 400)
        try:
            raw = base64.b64decode(b64)
        except Exception:
            return None, _err(f"'{field_name}' is not valid base64.", 400)
        return _raw_to_image(raw)

    # Raw bytes (curl / client.py style) — only for single-image endpoints
    if request.data and field_name == "image":
        return _raw_to_image(request.data)

    # Multipart
    if field_name in request.files:
        return _raw_to_image(request.files[field_name].read())

    return None, _err(f"Could not find image in request (field: '{field_name}').", 400)


def _resolve_dg2():
    """
    Resolve passport DG2 bytes from request. Accepts:
      - JSON: { "dg2_hex": "<hex>" }         — raw DG2 bytes as hex string
      - JSON: { "reference_image": "<b64>" } — already-extracted face JPEG as base64
      - Multipart: files["reference_image"]  — already-extracted face JPEG

    Returns (image_ndarray, error_response). One will always be None.
    """
    if request.is_json:
        body = request.json or {}

        # Raw DG2 TLV structure (hex) — needs full decoding
        if "dg2_hex" in body:
            image = decode_dg2_to_cv2(body["dg2_hex"])
            if image is None:
                return None, _err(
                    "Could not extract face from DG2 data. "
                    "Ensure dg2_hex is the raw TLV bytes from the NFC chip.", 400)
            return image, None

        # Pre-extracted face JPEG as base64 — just decode it
        if "reference_image" in body:
            try:
                raw = base64.b64decode(body["reference_image"])
            except Exception:
                return None, _err("'reference_image' is not valid base64.", 400)
            return _raw_to_image(raw)

    # Multipart
    if "reference_image" in request.files:
        return _raw_to_image(request.files["reference_image"].read())

    return None, _err(
        "Provide either 'dg2_hex' (raw DG2 TLV hex) or "
        "'reference_image' (base64 face JPEG) in the request.", 400)


def _distance_to_similarity(distance: float) -> float:
    """
    Convert L2 embedding distance to a 0-1 similarity score.

    InceptionResNetV1 (VGGFace2) produces L2 distances where:
      ~0.0  = same person (identical images)
      ~0.6  = likely same person
      ~1.0  = borderline
      ~1.5+ = different people

    We map this to a [0, 1] similarity score where 1 = identical.
    Using exponential decay: similarity = exp(-distance * k)
    Tuned so distance=0.6 → similarity≈0.87, distance=1.0 → similarity≈0.72
    """
    import math
    return round(math.exp(-distance * 0.22), 4)


def _detect_face(image) -> tuple:
    """Detect face and return (face_arr, error_response)."""
    faces, _ = faceDetector(image)
    if not faces:
        return None, _err("No face detected in the image.", 400)
    return faces[0], None


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------
@app.errorhandler(413)
def too_large(e):
    return _err(f"Request exceeds {max_content_mb}MB limit.", 413)


@app.errorhandler(429)
def too_many(e):
    return _err("Rate limit exceeded. Slow down.", 429)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    """Public — no auth required. Used by React Native to check connectivity."""
    return jsonify({
        "status": "ok",
        "liveness_threshold": LIVENESS_THRESHOLD,
        "face_match_threshold": FACE_MATCH_THRESHOLD,
    })


# ------ KYC PRIMARY ENDPOINTS (used by React Native LivenessService.ts) ------

@app.route("/liveness", methods=["POST"])
@require_api_key
@log_request
def liveness():
    """
    Liveness detection on a selfie.

    Request (JSON — React Native):
      { "image": "<base64 JPEG selfie>" }

    Request (raw bytes — testing):
      Content-Type: image/jpeg
      <raw bytes>

    Response (matches LivenessService.ts):
      { "passed": true, "score": 0.91 }
    """
    image, err = _resolve_image("image")
    if err:
        return err

    face, err = _detect_face(image)
    if err:
        return err

    result = livenessDetector(face)

    return _ok({
        "passed": result.is_live,
        "score": round(result.score, 4),
        # Extended info (ignored by app but useful for debugging)
        "deeppix_score": result.deeppix_score,
        "minifas_score": result.minifas_score,
        "threshold": result.threshold,
    })


@app.route("/face-match", methods=["POST"])
@require_api_key
@log_request
def face_match():
    """
    Compare passport chip face (DG2) with a live selfie.

    This is the primary KYC endpoint called by LivenessService.ts.

    Request option A — raw DG2 TLV bytes (recommended):
      {
        "dg2_hex": "<hex string of raw DG2 bytes from NFC chip>",
        "selfie_image": "<base64 JPEG selfie>"
      }

    Request option B — pre-extracted face JPEG:
      {
        "reference_image": "<base64 JPEG of face extracted from passport>",
        "selfie_image": "<base64 JPEG selfie>"
      }

    Response (matches LivenessService.ts):
      { "matched": true, "score": 0.88 }

    Score is a similarity value in [0, 1]:
      >= 0.85 = PASS (same person, app threshold)
      < 0.85  = FAIL
    """
    # Resolve passport reference face
    ref_image, err = _resolve_dg2()
    if err:
        return err

    # Resolve selfie
    selfie_image, err = _resolve_image("selfie_image")
    if err:
        return err

    # Detect faces
    ref_face, err = _detect_face(ref_image)
    if err:
        return _err("No face detected in passport reference image (DG2).", 400)

    selfie_face, err = _detect_face(selfie_image)
    if err:
        return _err("No face detected in selfie image.", 400)

    # Compare — convert L2 distance to similarity score
    distance, _ = identityChecker.compare(ref_face, selfie_face)
    similarity = _distance_to_similarity(distance)
    matched = similarity >= FACE_MATCH_THRESHOLD

    return _ok({
        "matched": matched,
        "score": similarity,
        # Extended info for debugging
        "distance": round(distance, 4),
        "threshold": FACE_MATCH_THRESHOLD,
    })


# ------ LEGACY / INTERNAL ENDPOINTS ------

@app.route("/identity", methods=["POST"])
@require_api_key
@log_request
def identity():
    """Facebank identity lookup. Internal use."""
    image, err = _resolve_image("image")
    if err:
        return err
    face, err = _detect_face(image)
    if err:
        return err

    matched_name, min_dist, mean_dist, is_match = identityChecker(face)
    return _ok({
        "matched_name": matched_name,
        "is_match": is_match,
        "min_dist": min_dist,
        "mean_dist": mean_dist,
        "threshold": identityChecker.threshold,
    })


@app.route("/main", methods=["POST"])
@require_api_key
@log_request
def main():
    """All-in-one: facebank identity + liveness. Internal use."""
    image, err = _resolve_image("image")
    if err:
        return err
    face, err = _detect_face(image)
    if err:
        return err

    matched_name, min_dist, mean_dist, is_match = identityChecker(face)
    liveness_result = livenessDetector(face)

    return _ok({
        "identity": {
            "matched_name": matched_name,
            "is_match": is_match,
            "min_dist": min_dist,
            "mean_dist": mean_dist,
        },
        "liveness": liveness_result.to_dict(),
    })


@app.route("/compare", methods=["POST"])
@require_api_key
@log_request
def compare():
    """
    Legacy direct photo comparison. Accepts multipart or JSON.
    Prefer /face-match for KYC workflows.
    """
    if request.is_json:
        image_a, err = _resolve_image("photo_a")
        if err:
            return err
        image_b, err = _resolve_image("photo_b")
        if err:
            return err
    else:
        if "photo_a" not in request.files or "photo_b" not in request.files:
            return _err("Provide photo_a and photo_b as multipart fields or JSON.", 400)
        image_a, err = _raw_to_image(request.files["photo_a"].read())
        if err:
            return err
        image_b, err = _raw_to_image(request.files["photo_b"].read())
        if err:
            return err

    face_a, err = _detect_face(image_a)
    if err:
        return _err("No face in photo_a.", 400)
    face_b, err = _detect_face(image_b)
    if err:
        return _err("No face in photo_b.", 400)

    distance, _ = identityChecker.compare(face_a, face_b)
    similarity = _distance_to_similarity(distance)
    liveness_result = livenessDetector(face_b)

    return _ok({
        "identity": {
            "similarity": similarity,
            "distance": round(distance, 4),
            "matched": similarity >= FACE_MATCH_THRESHOLD,
            "threshold": FACE_MATCH_THRESHOLD,
        },
        "liveness": liveness_result.to_dict(),
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    debug = environ.get("FLASK_DEBUG", "false").lower() == "true"
    port = int(environ.get("PORT", 5000))
    if debug:
        log.info("Starting Flask dev server on port %d", port)
        app.run(host="0.0.0.0", port=port, debug=True)
    else:
        from waitress import serve
        log.info("Starting Waitress production server on port %d", port)
        serve(app, host="0.0.0.0", port=port, threads=4)