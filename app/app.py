import os
from os import environ
from pathlib import Path

import cv2
import jsonpickle
import numpy as np
from dotenv import load_dotenv
from facetools import FaceDetection, IdentityVerification, LivenessDetection
from flask import Flask, Response, request

root = Path(os.path.abspath(__file__)).parent.absolute()
load_dotenv((root / ".env").as_posix())

data_folder = root.parent / environ.get("DATA_FOLDER")
resnet_name = environ.get("RESNET")
deeppix_name = environ.get("DEEPPIX")
minifas_name = environ.get("MINIFAS", "MiniFASNetV2.onnx")   # optional second model
facebank_name = environ.get("FACEBANK")
liveness_threshold = float(environ.get("LIVENESS_THRESHOLD", "0.5"))
identity_threshold = float(environ.get("IDENTITY_THRESHOLD", "0.9"))

resNet_checkpoint_path = data_folder / "checkpoints" / resnet_name
facebank_path = data_folder / facebank_name
deepPix_checkpoint_path = data_folder / "checkpoints" / deeppix_name
miniFas_checkpoint_path = data_folder / "checkpoints" / minifas_name

faceDetector = FaceDetection()

identityChecker = IdentityVerification(
    checkpoint_path=resNet_checkpoint_path.as_posix(),
    facebank_path=facebank_path.as_posix(),
    threshold=identity_threshold,
)

livenessDetector = LivenessDetection(
    checkpoint_path=deepPix_checkpoint_path.as_posix(),
    minifas_checkpoint_path=miniFas_checkpoint_path.as_posix() if miniFas_checkpoint_path.is_file() else None,
    threshold=liveness_threshold,
)

app = Flask(__name__)


def decode_image(raw_data: bytes) -> np.ndarray:
    nparr = np.frombuffer(raw_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def no_face_response():
    return jsonpickle.encode({
        "message": "No face detected in the image.",
    }), 400


@app.route("/main", methods=["POST"])
def main():
    """Run face detection, identity verification, and liveness detection."""
    frame = decode_image(request.data)
    faces, boxes = faceDetector(frame)

    if not faces:
        return Response(*no_face_response(), mimetype="application/json")

    face_arr = faces[0]
    matched_name, min_dist, mean_dist, is_match = identityChecker(face_arr)
    liveness = livenessDetector(face_arr)

    response = jsonpickle.encode({
        "message": "OK",
        "identity": {
            "matched_name": matched_name,
            "is_match": is_match,
            "min_dist": min_dist,
            "mean_dist": mean_dist,
            "threshold": identityChecker.threshold,
        },
        "liveness": liveness.to_dict(),
    })
    return Response(response=response, status=200, mimetype="application/json")


@app.route("/identity", methods=["POST"])
def identity():
    """Run face detection and identity verification only."""
    frame = decode_image(request.data)
    faces, boxes = faceDetector(frame)

    if not faces:
        return Response(*no_face_response(), mimetype="application/json")

    face_arr = faces[0]
    matched_name, min_dist, mean_dist, is_match = identityChecker(face_arr)

    response = jsonpickle.encode({
        "message": "OK",
        "matched_name": matched_name,
        "is_match": is_match,
        "min_dist": min_dist,
        "mean_dist": mean_dist,
        "threshold": identityChecker.threshold,
    })
    return Response(response=response, status=200, mimetype="application/json")


@app.route("/liveness", methods=["POST"])
def liveness():
    """Run face detection and liveness detection only."""
    frame = decode_image(request.data)
    faces, boxes = faceDetector(frame)

    if not faces:
        return Response(*no_face_response(), mimetype="application/json")

    face_arr = faces[0]
    liveness_result = livenessDetector(face_arr)   # fixed: no identity call here

    response = jsonpickle.encode({
        "message": "OK",
        **liveness_result.to_dict(),
    })
    return Response(response=response, status=200, mimetype="application/json")


@app.route("/compare", methods=["POST"])
def compare():
    """
    Compare two face photos directly (ID vs selfie).

    Expects a multipart/form-data POST with two fields:
        photo_a: reference image (e.g. ID card photo)
        photo_b: probe image   (e.g. live selfie)

    Returns similarity distance, match decision, and liveness score for photo_b.
    """
    if "photo_a" not in request.files or "photo_b" not in request.files:
        return Response(
            jsonpickle.encode({"message": "Both photo_a and photo_b are required."}),
            status=400, mimetype="application/json",
        )

    frame_a = decode_image(request.files["photo_a"].read())
    frame_b = decode_image(request.files["photo_b"].read())

    faces_a, _ = faceDetector(frame_a)
    faces_b, _ = faceDetector(frame_b)

    if not faces_a:
        return Response(
            jsonpickle.encode({"message": "No face detected in photo_a."}),
            status=400, mimetype="application/json",
        )
    if not faces_b:
        return Response(
            jsonpickle.encode({"message": "No face detected in photo_b."}),
            status=400, mimetype="application/json",
        )

    distance, is_match = identityChecker.compare(faces_a[0], faces_b[0])
    liveness_result = livenessDetector(faces_b[0])   # check liveness on the probe

    response = jsonpickle.encode({
        "message": "OK",
        "identity": {
            "distance": distance,
            "is_match": is_match,
            "threshold": identityChecker.threshold,
        },
        "liveness": liveness_result.to_dict(),
    })
    return Response(response=response, status=200, mimetype="application/json")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)