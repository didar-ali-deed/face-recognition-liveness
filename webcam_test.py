import os
from pathlib import Path

import cv2

from facetools import FaceDetection, IdentityVerification, LivenessDetection
from facetools.utils import visualize_results

root = Path(os.path.abspath(__file__)).parent.absolute()
data_folder = root / "data"

resNet_checkpoint_path  = data_folder / "checkpoints" / "InceptionResnetV1_vggface2.onnx"
facebank_path           = data_folder / "facebank.csv"
deepPix_checkpoint_path = data_folder / "checkpoints" / "OULU_Protocol_2_model_0_0.onnx"

faceDetector    = FaceDetection(max_num_faces=1)
identityChecker = IdentityVerification(
    checkpoint_path=resNet_checkpoint_path.as_posix(),
    facebank_path=facebank_path.as_posix(),
    threshold=0.9,
)
livenessDetector = LivenessDetection(
    checkpoint_path=deepPix_checkpoint_path.as_posix(),
    threshold=0.01,   # temporary — remove once MiniFASNet is added
)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    canvas = frame.copy()
    faces, boxes = faceDetector(frame)

    for face_arr, box in zip(faces, boxes):
        matched_name, min_dist, mean_dist, is_match = identityChecker(face_arr)
        liveness_result = livenessDetector(face_arr)

        # Build overlay label
        identity_label = f"{matched_name} ({min_dist:.3f})" if is_match else f"unknown ({min_dist:.3f})"
        liveness_label = "LIVE" if liveness_result.is_live else "SPOOF"
        label = f"{identity_label} | {liveness_label} {liveness_result.score:.2f}"

        # Draw bounding box
        x1, y1 = box[0]
        x2, y2 = box[1]
        color = (0, 255, 0) if is_match and liveness_result.is_live else (0, 0, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("face", canvas)
    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()