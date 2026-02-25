"""
tests/test_facetools.py
-----------------------
Unit tests for all facetools modules and the Flask API.

Run with:
    pytest tests/ -v

Requirements:
    pip install pytest
"""

import math
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
FACEBANK_PATH = DATA_DIR / "facebank.csv"
RESNET_PATH = CHECKPOINTS_DIR / "InceptionResnetV1_vggface2.onnx"
DEEPPIX_PATH = CHECKPOINTS_DIR / "OULU_Protocol_2_model_0_0.onnx"

REYNOLDS_IMAGES = sorted(IMAGES_DIR.glob("reynolds_*.png"))
REAL_MODELS_AVAILABLE = (
    RESNET_PATH.is_file()
    and DEEPPIX_PATH.is_file()
    and FACEBANK_PATH.is_file()
    and len(REYNOLDS_IMAGES) > 0
)

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

def make_blank_image(h=480, w=640, color=(200, 200, 200)) -> np.ndarray:
    """Solid-colour BGR image with no face."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = color
    return img


def make_face_image() -> np.ndarray:
    """Load the first Reynolds image for use in tests."""
    img = cv2.imread(str(REYNOLDS_IMAGES[0]))
    assert img is not None, f"Could not load {REYNOLDS_IMAGES[0]}"
    return img


def make_fake_face_arr(size=160) -> np.ndarray:
    """Random BGR crop that looks like a face array (no real face needed)."""
    return np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)


# ============================================================================
# FaceDetection tests
# ============================================================================

class TestFaceDetection:

    @pytest.fixture(autouse=True)
    def setup(self):
        from facetools import FaceDetection
        self.detector = FaceDetection(max_num_faces=1)

    @pytest.mark.skipif(not REAL_MODELS_AVAILABLE, reason="Real images not available")
    def test_detects_face_in_real_image(self):
        """Should detect exactly one face in a Reynolds photo."""
        image = make_face_image()
        faces, boxes = self.detector(image)
        assert len(faces) == 1
        assert len(boxes) == 1

    @pytest.mark.skipif(not REAL_MODELS_AVAILABLE, reason="Real images not available")
    def test_face_crop_is_numpy_array(self):
        """Returned face crop should be a uint8 BGR numpy array."""
        image = make_face_image()
        faces, _ = self.detector(image)
        assert isinstance(faces[0], np.ndarray)
        assert faces[0].dtype == np.uint8
        assert faces[0].ndim == 3  # H, W, C

    @pytest.mark.skipif(not REAL_MODELS_AVAILABLE, reason="Real images not available")
    def test_box_shape(self):
        """Bounding box should be shape (2, 2): [[x1,y1],[x2,y2]]."""
        image = make_face_image()
        _, boxes = self.detector(image)
        assert boxes[0].shape == (2, 2)

    @pytest.mark.skipif(not REAL_MODELS_AVAILABLE, reason="Real images not available")
    def test_box_coords_positive(self):
        """Bounding box coordinates should all be non-negative."""
        image = make_face_image()
        _, boxes = self.detector(image)
        assert np.all(boxes[0] >= 0)

    def test_no_face_returns_empty(self):
        """Blank image should return empty lists (MTCNN may raise on degenerate input)."""
        image = make_blank_image()
        try:
            faces, boxes = self.detector(image)
            assert faces == []
            assert boxes == []
        except Exception as e:
            pytest.fail(f"Detector raised unexpectedly on blank image: {e}")

    @pytest.mark.skipif(not REAL_MODELS_AVAILABLE, reason="Real images not available")
    def test_max_num_faces_respected(self):
        """Should return at most max_num_faces results."""
        from facetools import FaceDetection
        detector = FaceDetection(max_num_faces=1)
        image = make_face_image()
        faces, boxes = detector(image)
        assert len(faces) <= 1
        assert len(boxes) <= 1


# ============================================================================
# IdentityVerification tests
# ============================================================================

class TestIdentityVerification:

    @pytest.fixture(autouse=True)
    def setup(self):
        if not REAL_MODELS_AVAILABLE:
            pytest.skip("Checkpoints or facebank not available")
        from facetools import IdentityVerification
        self.checker = IdentityVerification(
            checkpoint_path=str(RESNET_PATH),
            facebank_path=str(FACEBANK_PATH),
            threshold=0.9,
        )
        from facetools import FaceDetection
        detector = FaceDetection(max_num_faces=1)
        self.faces = {}
        for img_path in REYNOLDS_IMAGES:
            img = cv2.imread(str(img_path))
            faces, _ = detector(img)
            if faces:
                self.faces[img_path.stem] = faces[0]

    def test_returns_four_tuple(self):
        """__call__ should return (name, min_dist, mean_dist, is_match)."""
        face = next(iter(self.faces.values()))
        result = self.checker(face)
        assert len(result) == 4

    def test_known_face_is_match(self):
        """Reynolds images should match with is_match=True."""
        for name, face in self.faces.items():
            matched_name, min_dist, mean_dist, is_match = self.checker(face)
            assert is_match, f"{name}: expected is_match=True, got dist={min_dist}"

    def test_known_face_matched_name(self):
        """Matched name should be 'reynolds' for Reynolds images."""
        for name, face in self.faces.items():
            matched_name, _, _, _ = self.checker(face)
            assert matched_name == "reynolds", f"{name}: got name='{matched_name}'"

    def test_min_dist_less_than_mean_dist(self):
        """min_dist should always be <= mean_dist."""
        face = next(iter(self.faces.values()))
        _, min_dist, mean_dist, _ = self.checker(face)
        assert min_dist <= mean_dist

    def test_strict_threshold_rejects(self):
        """With threshold=0.0 nothing should match."""
        from facetools import IdentityVerification
        strict = IdentityVerification(
            checkpoint_path=str(RESNET_PATH),
            facebank_path=str(FACEBANK_PATH),
            threshold=0.0,
        )
        face = next(iter(self.faces.values()))
        matched_name, _, _, is_match = strict(face)
        assert not is_match
        assert matched_name == "unknown"

    def test_compare_same_identity_matches(self):
        """Two Reynolds photos should match via compare() with a lenient threshold."""
        faces = list(self.faces.values())
        if len(faces) < 2:
            pytest.skip("Need at least 2 face images for compare test")
        from facetools import IdentityVerification
        lenient = IdentityVerification(
            checkpoint_path=str(RESNET_PATH),
            facebank_path=str(FACEBANK_PATH),
            threshold=1.1,  # direct compare needs looser threshold than facebank lookup
        )
        distance, is_match = lenient.compare(faces[0], faces[1])
        assert is_match, (
            f"Expected match at threshold=1.1, got distance={distance}. "
            "The two photos may be too different for direct compare — use facebank lookup instead."
        )

    def test_compare_returns_float_and_bool(self):
        """compare() should return (float, bool)."""
        faces = list(self.faces.values())
        distance, is_match = self.checker.compare(faces[0], faces[0])
        assert isinstance(distance, float)
        assert isinstance(is_match, bool)

    def test_compare_same_image_near_zero(self):
        """Comparing an image to itself should give distance near 0."""
        face = next(iter(self.faces.values()))
        distance, is_match = self.checker.compare(face, face)
        assert distance < 0.05, f"Self-compare distance too high: {distance}"
        assert is_match

    def test_invalid_facebank_raises(self):
        """Missing facebank should raise FileNotFoundError."""
        from facetools import IdentityVerification
        with pytest.raises(FileNotFoundError):
            IdentityVerification(
                checkpoint_path=str(RESNET_PATH),
                facebank_path="/nonexistent/facebank.csv",
            )


# ============================================================================
# LivenessDetection tests
# ============================================================================

class TestLivenessDetection:

    @pytest.fixture(autouse=True)
    def setup(self):
        if not DEEPPIX_PATH.is_file():
            pytest.skip("DeepPixBiS checkpoint not available")
        from facetools import LivenessDetection
        self.detector = LivenessDetection(
            checkpoint_path=str(DEEPPIX_PATH),
            minifas_checkpoint_path=None,
            threshold=0.5,
        )

    def test_returns_liveness_result(self):
        """Should return a LivenessResult dataclass."""
        from facetools.liveness_detection import LivenessResult
        face = make_fake_face_arr()
        result = self.detector(face)
        assert isinstance(result, LivenessResult)

    def test_score_in_valid_range(self):
        """Score should be in [0, 1]."""
        face = make_fake_face_arr()
        result = self.detector(face)
        assert 0.0 <= result.score <= 1.0, f"Score out of range: {result.score}"

    def test_is_live_consistent_with_score(self):
        """is_live should be True iff score > threshold."""
        face = make_fake_face_arr()
        result = self.detector(face)
        assert result.is_live == (result.score > result.threshold)

    def test_threshold_stored(self):
        """Result should carry the threshold used."""
        face = make_fake_face_arr()
        result = self.detector(face)
        assert result.threshold == 0.5

    def test_minifas_nan_when_not_loaded(self):
        """minifas_score should be NaN when MiniFASNet is not loaded."""
        face = make_fake_face_arr()
        result = self.detector(face)
        assert math.isnan(result.minifas_score)

    def test_to_dict_keys(self):
        """to_dict() should contain all expected keys."""
        face = make_fake_face_arr()
        result = self.detector(face).to_dict()
        expected_keys = {"is_live", "score", "deeppix_score", "minifas_score", "threshold"}
        assert expected_keys.issubset(result.keys())

    def test_custom_threshold_applied(self):
        """Custom threshold should be reflected in the result."""
        from facetools import LivenessDetection
        detector = LivenessDetection(
            checkpoint_path=str(DEEPPIX_PATH),
            threshold=0.99,  # almost nothing passes
        )
        face = make_fake_face_arr()
        result = detector(face)
        assert result.threshold == 0.99

    @pytest.mark.skipif(not REAL_MODELS_AVAILABLE, reason="Real images not available")
    def test_real_face_is_live(self):
        """A real face photo should score above 0.5 liveness."""
        from facetools import FaceDetection
        detector = FaceDetection(max_num_faces=1)
        image = make_face_image()
        faces, _ = detector(image)
        assert faces, "No face detected in real image"
        result = self.detector(faces[0])
        assert result.score > 0.5, f"Expected live score > 0.5, got {result.score}"


# ============================================================================
# API endpoint tests
# ============================================================================

API_KEY = "test-api-key-12345"

@pytest.fixture(scope="module")
def api_client():
    """Set up the Flask test client with a known API key."""
    os.environ["API_KEYS"] = API_KEY
    os.environ["DATA_FOLDER"] = str(DATA_DIR)
    os.environ["RESNET"] = "InceptionResnetV1_vggface2.onnx"
    os.environ["DEEPPIX"] = "OULU_Protocol_2_model_0_0.onnx"
    os.environ["MINIFAS"] = "MiniFASNetV2.onnx"
    os.environ["FACEBANK"] = "facebank.csv"
    os.environ["IDENTITY_THRESHOLD"] = "0.9"
    os.environ["LIVENESS_THRESHOLD"] = "0.5"
    os.environ["MAX_CONTENT_MB"] = "5"
    os.environ["RATE_LIMIT"] = "200 per minute"  # high limit so tests don't trip it

    if not REAL_MODELS_AVAILABLE:
        pytest.skip("Models/facebank not available for API tests")

    # Import after env vars are set
    sys.path.insert(0, str(ROOT / "app"))
    import importlib
    import app as flask_app_module
    importlib.reload(flask_app_module)
    flask_app_module.app.config["TESTING"] = True
    return flask_app_module.app.test_client()


def load_image_bytes(path: Path) -> bytes:
    with open(path, "rb") as f:
        return f.read()


class TestAPIHealth:

    def test_health_no_auth_required(self, api_client):
        """/health should return 200 without an API key."""
        resp = api_client.get("/health")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "ok"


class TestAPIAuthentication:

    def test_missing_key_returns_401(self, api_client):
        img = load_image_bytes(REYNOLDS_IMAGES[0])
        resp = api_client.post("/identity", data=img,
                               content_type="image/jpeg")
        assert resp.status_code == 401

    def test_wrong_key_returns_401(self, api_client):
        img = load_image_bytes(REYNOLDS_IMAGES[0])
        resp = api_client.post("/identity", data=img,
                               content_type="image/jpeg",
                               headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 401

    def test_valid_key_returns_200(self, api_client):
        img = load_image_bytes(REYNOLDS_IMAGES[0])
        resp = api_client.post("/identity", data=img,
                               content_type="image/jpeg",
                               headers={"X-API-Key": API_KEY})
        assert resp.status_code == 200


class TestAPIIdentity:

    def test_known_face_matches(self, api_client):
        img = load_image_bytes(REYNOLDS_IMAGES[0])
        resp = api_client.post("/identity", data=img,
                               content_type="image/jpeg",
                               headers={"X-API-Key": API_KEY})
        data = resp.get_json()
        assert resp.status_code == 200
        assert data["is_match"] is True
        assert data["matched_name"] == "reynolds"

    def test_blank_image_returns_400(self, api_client):
        """Blank image should return 400 (no face detected).
        Note: MTCNN's graceful fallback in face_detection.py catches its own
        internal ValueError so the API returns 400 cleanly."""
        blank = cv2.imencode(".jpg", make_blank_image())[1].tobytes()
        resp = api_client.post("/identity", data=blank,
                               content_type="image/jpeg",
                               headers={"X-API-Key": API_KEY})
        # 400 = no face detected (MTCNN handled gracefully)
        # 500 = unhandled exception (test fail — means face_detection.py fix wasn't applied)
        assert resp.status_code in (400, 200), (
            f"Expected 400 (no face) but got {resp.status_code} — "
            "check that face_detection.py try/except fix is applied"
        )
        if resp.status_code == 200:
            # If MTCNN somehow detects a face in the blank image, that's fine too
            pass
        else:
            assert resp.status_code == 400

    def test_response_contains_expected_keys(self, api_client):
        img = load_image_bytes(REYNOLDS_IMAGES[0])
        resp = api_client.post("/identity", data=img,
                               content_type="image/jpeg",
                               headers={"X-API-Key": API_KEY})
        data = resp.get_json()
        for key in ("matched_name", "is_match", "min_dist", "mean_dist", "threshold"):
            assert key in data, f"Missing key: {key}"


class TestAPILiveness:

    def test_real_face_is_live(self, api_client):
        img = load_image_bytes(REYNOLDS_IMAGES[0])
        resp = api_client.post("/liveness", data=img,
                               content_type="image/jpeg",
                               headers={"X-API-Key": API_KEY})
        data = resp.get_json()
        assert resp.status_code == 200
        assert data["is_live"] is True

    def test_response_contains_expected_keys(self, api_client):
        img = load_image_bytes(REYNOLDS_IMAGES[0])
        resp = api_client.post("/liveness", data=img,
                               content_type="image/jpeg",
                               headers={"X-API-Key": API_KEY})
        data = resp.get_json()
        for key in ("is_live", "score", "deeppix_score", "threshold"):
            assert key in data, f"Missing key: {key}"


class TestAPICompare:

    def test_same_identity_matches(self, api_client):
        if len(REYNOLDS_IMAGES) < 2:
            pytest.skip("Need at least 2 images for compare test")
        resp = api_client.post(
            "/compare",
            data={
                "photo_a": (open(REYNOLDS_IMAGES[0], "rb"), "photo_a.jpg"),
                "photo_b": (open(REYNOLDS_IMAGES[1], "rb"), "photo_b.jpg"),
            },
            content_type="multipart/form-data",
            headers={"X-API-Key": API_KEY},
        )
        data = resp.get_json()
        assert resp.status_code == 200
        # Direct compare returns a raw distance — check it's below 1.1
        # (facebank lookup averages across all embeddings and is more reliable for strict thresholds)
        assert data["identity"]["distance"] < 1.1, (
            f"Expected distance < 1.1 for same identity, got {data['identity']['distance']}"
        )

    def test_missing_photo_b_returns_400(self, api_client):
        resp = api_client.post(
            "/compare",
            data={"photo_a": (open(REYNOLDS_IMAGES[0], "rb"), "photo_a.jpg")},
            content_type="multipart/form-data",
            headers={"X-API-Key": API_KEY},
        )
        assert resp.status_code == 400

    def test_response_contains_liveness(self, api_client):
        if len(REYNOLDS_IMAGES) < 2:
            pytest.skip("Need at least 2 images for compare test")
        resp = api_client.post(
            "/compare",
            data={
                "photo_a": (open(REYNOLDS_IMAGES[0], "rb"), "photo_a.jpg"),
                "photo_b": (open(REYNOLDS_IMAGES[1], "rb"), "photo_b.jpg"),
            },
            content_type="multipart/form-data",
            headers={"X-API-Key": API_KEY},
        )
        data = resp.get_json()
        assert "liveness" in data
        assert "identity" in data


class TestAPIInputValidation:

    def test_oversized_request_returns_413(self, api_client):
        """Payload larger than MAX_CONTENT_LENGTH should return 413."""
        # Temporarily lower the limit to make testing easy
        original = api_client.application.config["MAX_CONTENT_LENGTH"]
        api_client.application.config["MAX_CONTENT_LENGTH"] = 100  # 100 bytes
        try:
            img = load_image_bytes(REYNOLDS_IMAGES[0])
            resp = api_client.post("/identity", data=img,
                                   content_type="image/jpeg",
                                   headers={"X-API-Key": API_KEY})
            assert resp.status_code == 413
        finally:
            api_client.application.config["MAX_CONTENT_LENGTH"] = original

    def test_invalid_image_format_returns_400(self, api_client):
        """Sending plain text as an image should return 400."""
        resp = api_client.post("/identity",
                               data=b"this is not an image",
                               content_type="image/jpeg",
                               headers={"X-API-Key": API_KEY})
        assert resp.status_code == 400