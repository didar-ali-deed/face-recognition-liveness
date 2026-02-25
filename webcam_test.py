import os
from pathlib import Path

import cv2

from facetools import FaceDetection, IdentityVerification, LivenessDetection
from facetools.utils import visualize_results

root = Path(os.path.abspath(__file__)).parent.absolute()
data_folder = root / "data"

resNet_checkpoint_path = data_folder / "checkpoints" / "InceptionResnetV1_vggface2.onnx"
facebank_path = data_folder / "facebank.csv"  # updated: now includes name column

deepPix_checkpoint_path = data_folder / "checkpoints" / "OULU_Protocol_2_model_0_0.onnx"

faceDetector = FaceDetection(max_num_faces=1)
identityChecker = IdentityVerification(
    checkpoint_path=resNet_checkpoint_path.as_posix(),
    facebank_path=facebank_path.as_posix(),
    threshold=0.9,  # tune this — lower = stricter
)
livenessDetector = LivenessDetection(checkpoint_path=deepPix_checkpoint_path.as_posix())

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    canvas = frame.copy()
    faces, boxes = faceDetector(frame)
    for face_arr, box in zip(faces, boxes):
        matched_name, min_dist, mean_dist, is_match = identityChecker(face_arr)
        liveness_score = livenessDetector(face_arr)
        label = f"{matched_name} ({min_dist:.3f})" if is_match else f"unknown ({min_dist:.3f})"
        canvas = visualize_results(canvas, box, liveness_score, min_dist, label=label)
    cv2.imshow("face", canvas)
    k = cv2.waitKey(1)
    if k == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()