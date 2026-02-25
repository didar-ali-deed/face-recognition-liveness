"""
app.py
------
Production-hardened Flask API for face recognition and liveness detection.

Security features:
  - API key authentication via X-API-Key header
  - Per-IP rate limiting (flask-limiter)
  - Request size limit (default 5MB)
  - Image format validation (JPEG, PNG, BMP, WEBP only)
  - Structured request/response logging

Run in production via:
  python app.py           (uses Waitress WSGI server automatically)
  FLASK_DEBUG=true python app.py   (uses Flask dev server for local debug)
"""


import logging
import os
import time
from functools import wraps
from os import environ
from pathlib import Path

import cv2
import jsonpickle
import numpy as np
from dotenv import load_dotenv
from facetools import FaceDetection, IdentityVerification, LivenessDetection
from flask import Flask, Response, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config from .env
# ---------------------------------------------------------------------------
root = Path(os.path.abspath(__file__)).parent.absolute()
load_dotenv((root / ".env").as_posix())

data_folder        = root.parent / environ.get("DATA_FOLDER", "data")
resnet_name        = environ.get("RESNET", "InceptionResnetV1_vggface2.onnx")
deeppix_name       = environ.get("DEEPPIX", "OULU_Protocol_2_model_0_0.onnx")
minifas_name       = environ.get("MINIFAS", "MiniFASNetV2.onnx")
facebank_name      = environ.get("FACEBANK", "facebank.csv")
liveness_threshold = float(environ.get("LIVENESS_THRESHOLD", "0.5"))
identity_threshold = float(environ.get("IDENTITY_THRESHOLD", "0.9"))
max_content_mb     = float(environ.get("MAX_CONTENT_MB", "5"))
rate_limit         = environ.get("RATE_LIMIT", "30 per minute")

# API keys — comma-separated list in .env, e.g. API_KEYS=key1,key2
_raw_keys = environ.get("API_KEYS", "")
VALID_API_KEYS = set(k.strip() for k in _raw_keys.split(",") if k.strip())

if not VALID_API_KEYS:
    log.warning(
        "No API_KEYS set in .env — all requests will be rejected. "
        "Add API_KEYS=your-secret-key to app/.env"
    )

ALLOWED_IMAGE_TYPES = {"jpeg", "png", "bmp", "webp"}

# Byte-header signatures — replaces deprecated imghdr (removed in Python 3.13)
_IMAGE_SIGNATURES = [
    (b"\xff\xd8\xff", "jpeg"),
    (b"\x89PNG",       "png"),
    (b"BM",             "bmp"),
]

def detect_image_type(data: bytes) -> str:
    """Detect image format from raw bytes. Returns format string or 'unknown'."""
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    for sig, fmt in _IMAGE_SIGNATURES:
        if data[:len(sig)] == sig:
            return fmt
    return "unknown"

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
resNet_checkpoint_path  = data_folder / "checkpoints" / resnet_name
deepPix_checkpoint_path = data_folder / "checkpoints" / deeppix_name
miniFas_checkpoint_path = data_folder / "checkpoints" / minifas_name
facebank_path           = data_folder / facebank_name

log.info("Loading models...")
faceDetector = FaceDetection()
identityChecker = IdentityVerification(
    checkpoint_path=resNet_checkpoint_path.as_posix(),
    facebank_path=facebank_path.as_posix(),
    threshold=identity_threshold,
)
livenessDetector = LivenessDetection(
    checkpoint_path=deepPix_checkpoint_path.as_posix(),
    minifas_checkpoint_path=(
        miniFas_checkpoint_path.as_posix()
        if miniFas_checkpoint_path.is_file() else None
    ),
    threshold=liveness_threshold,
)
log.info("Models loaded successfully.")

# ---------------------------------------------------------------------------
# Flask app + rate limiter
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
    """Reject requests that don't supply a valid X-API-Key header."""
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get("X-API-Key", "")
        if key not in VALID_API_KEYS:
            log.warning("Unauthorized request from %s", request.remote_addr)
            return error_response("Invalid or missing API key.", 401)
        return f(*args, **kwargs)
    return decorated


def log_request(f):
    """Log endpoint, caller IP, and response time for every request."""
    @wraps(f)
    def decorated(*args, **kwargs):
        start = time.perf_counter()
        result = f(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        log.info(
            "%s %s from %s — %.1fms",
            request.method, request.path,
            request.remote_addr, elapsed,
        )
        return result
    return decorated


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def error_response(message: str, status: int) -> Response:
    return Response(
        response=jsonpickle.encode({"message": message}),
        status=status,
        mimetype="application/json",
    )


def ok_response(data: dict) -> Response:
    return Response(
        response=jsonpickle.encode({"message": "OK", **data}),
        status=200,
        mimetype="application/json",
    )


def decode_and_validate_image(raw: bytes) -> tuple:
    """
    Validate and decode raw bytes to a BGR numpy array.
    Returns (image, None) on success or (None, error_response) on failure.
    """
    img_type = detect_image_type(raw)
    if img_type not in ALLOWED_IMAGE_TYPES:
        return None, error_response(
            f"Unsupported image format '{img_type}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_IMAGE_TYPES))}.",
            400,
        )
    nparr = np.frombuffer(raw, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        return None, error_response("Could not decode image.", 400)
    return image, None


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------
@app.errorhandler(413)
def request_too_large(e):
    return error_response(f"Image exceeds the {max_content_mb}MB size limit.", 413)


@app.errorhandler(429)
def rate_limit_exceeded(e):
    return error_response("Rate limit exceeded. Please slow down.", 429)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    """Public health check — no auth required."""
    return jsonify({"status": "ok"})


@app.route("/main", methods=["POST"])
@require_api_key
@log_request
def main():
    """Detection + identity verification + liveness in one call."""
    image, err = decode_and_validate_image(request.data)
    if err:
        return err

    faces, _ = faceDetector(image)
    if not faces:
        return error_response("No face detected in the image.", 400)

    face_arr = faces[0]
    matched_name, min_dist, mean_dist, is_match = identityChecker(face_arr)
    liveness = livenessDetector(face_arr)

    return ok_response({
        "identity": {
            "matched_name": matched_name,
            "is_match": is_match,
            "min_dist": min_dist,
            "mean_dist": mean_dist,
            "threshold": identityChecker.threshold,
        },
        "liveness": liveness.to_dict(),
    })


@app.route("/identity", methods=["POST"])
@require_api_key
@log_request
def identity():
    """Face detection + identity verification only."""
    image, err = decode_and_validate_image(request.data)
    if err:
        return err

    faces, _ = faceDetector(image)
    if not faces:
        return error_response("No face detected in the image.", 400)

    face_arr = faces[0]
    matched_name, min_dist, mean_dist, is_match = identityChecker(face_arr)

    return ok_response({
        "matched_name": matched_name,
        "is_match": is_match,
        "min_dist": min_dist,
        "mean_dist": mean_dist,
        "threshold": identityChecker.threshold,
    })


@app.route("/liveness", methods=["POST"])
@require_api_key
@log_request
def liveness():
    """Face detection + liveness detection only."""
    image, err = decode_and_validate_image(request.data)
    if err:
        return err

    faces, _ = faceDetector(image)
    if not faces:
        return error_response("No face detected in the image.", 400)

    liveness_result = livenessDetector(faces[0])
    return ok_response(liveness_result.to_dict())


@app.route("/compare", methods=["POST"])
@require_api_key
@log_request
def compare():
    """
    Direct photo-to-photo comparison (ID vs selfie).

    Multipart form fields:
        photo_a — reference image (e.g. ID card)
        photo_b — probe image    (e.g. selfie)
    """
    if "photo_a" not in request.files or "photo_b" not in request.files:
        return error_response("Both photo_a and photo_b fields are required.", 400)

    image_a, err = decode_and_validate_image(request.files["photo_a"].read())
    if err:
        return err
    image_b, err = decode_and_validate_image(request.files["photo_b"].read())
    if err:
        return err

    faces_a, _ = faceDetector(image_a)
    faces_b, _ = faceDetector(image_b)

    if not faces_a:
        return error_response("No face detected in photo_a.", 400)
    if not faces_b:
        return error_response("No face detected in photo_b.", 400)

    distance, is_match = identityChecker.compare(faces_a[0], faces_b[0])
    liveness_result = livenessDetector(faces_b[0])

    return ok_response({
        "identity": {
            "distance": distance,
            "is_match": is_match,
            "threshold": identityChecker.threshold,
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
        log.info("Starting Flask development server on port %d", port)
        app.run(host="0.0.0.0", port=port, debug=True)
    else:
        from waitress import serve
        log.info("Starting Waitress production server on port %d", port)
        serve(app, host="0.0.0.0", port=port, threads=4)