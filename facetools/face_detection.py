from typing import List, Tuple

import numpy as np
from mtcnn import MTCNN

from .utils import extract_face


class FaceDetection:
    """
    Face detector using MTCNN (Multi-task Cascaded Convolutional Networks).

    Replaces MediaPipe FaceMesh which was designed for landmark detection,
    not bounding-box detection. MTCNN is purpose-built for face detection
    and handles varied angles, lighting, and partial occlusion much better.
    """

    def __init__(self, max_num_faces: int = 1, min_confidence: float = 0.95):
        """
        Args:
            max_num_faces: Maximum number of faces to return (sorted by confidence).
            min_confidence: Minimum MTCNN detection confidence to accept a face.
        """
        self.detector = MTCNN()
        self.max_num_faces = max_num_faces
        self.min_confidence = min_confidence

    def __call__(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[List[int]]]:
        """
        Detect faces in an image.

        Args:
            image: BGR image as numpy array (OpenCV format).

        Returns:
            faces: List of cropped face arrays (BGR).
            boxes: List of bounding boxes as [[x1, y1], [x2, y2]].
        """
        # MTCNN expects RGB; OpenCV provides BGR
        image_rgb = image[:, :, ::-1]

        detections = self.detector.detect_faces(image_rgb)

        # Filter by confidence and sort best-first
        detections = [
            d for d in detections if d["confidence"] >= self.min_confidence
        ]
        detections = sorted(detections, key=lambda d: d["confidence"], reverse=True)
        detections = detections[: self.max_num_faces]

        faces = []
        boxes = []

        for det in detections:
            x, y, w, h = det["box"]
            # MTCNN can return slightly negative coords on edge faces
            x, y = max(0, x), max(0, y)
            x2, y2 = x + w, y + h

            bbox_flat = [x, y, x2, y2]
            face_arr = extract_face(image, bbox_flat)

            # Store as [[x1,y1],[x2,y2]] to match original interface
            bbox = np.array([[x, y], [x2, y2]], dtype=np.int32)
            faces.append(face_arr)
            boxes.append(bbox)

        return faces, boxes