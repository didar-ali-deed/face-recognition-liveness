# face-recognition-liveness

Face liveness detection and identity recognition using fast and accurate convolutional neural networks implemented in PyTorch. A production-ready Flask API, KYC integration support, and Dockerfile are included.

> **Built and improved with Claude AI (Anthropic)**

![face recognition and liveness](https://user-images.githubusercontent.com/43831412/181917410-a7df598b-8e89-419c-9505-6111676dc3a4.jpg)

---

## What's Inside

| Module | Description |
|--------|-------------|
| `facetools/face_detection.py` | MTCNN-based face detector (replaced MediaPipe) |
| `facetools/face_recognition.py` | InceptionResNetV1 (VGGFace2) identity verification with labelled facebank |
| `facetools/liveness_detection.py` | Ensemble liveness: DeepPixBiS + MiniFASNetV2 |
| `app/app.py` | Production Flask API (Waitress WSGI, API key auth, rate limiting) |
| `app/dg2_decoder.py` | NFC e-Passport DG2 face image extractor (ICAO 9303) |
| `create_facebank.py` | Build a labelled facebank CSV from reference images |
| `webcam_test.py` | Real-time webcam demo |

---

## Getting Started

### 1. Install dependencies

```bash
pip install -e .
pip install -r requirements.txt
```

### 2. Download ONNX checkpoints

Place them in `data/checkpoints/`:

- [InceptionResnetV1_vggface2.onnx](https://github.com/ffletcherr/face-recognition-liveness/releases/download/v0.1/InceptionResnetV1_vggface2.onnx)
- [OULU_Protocol_2_model_0_0.onnx](https://github.com/ffletcherr/face-recognition-liveness/releases/download/v0.1/OULU_Protocol_2_model_0_0.onnx)

> Models will be downloaded automatically if you have an internet connection.

### 3. Create a facebank

Put reference images in a folder, then run:

```bash
# Single identity
python create_facebank.py \
  --images_dir data/images \
  --name reynolds \
  --output data/facebank.csv

# Multiple identities (one subfolder per person)
python create_facebank.py \
  --images_dir data/identities \
  --subdirs \
  --output data/facebank.csv

# Add a new person to an existing facebank
python create_facebank.py \
  --images_dir data/images/didar \
  --name didar \
  --output data/facebank.csv \
  --append
```

### 4. Webcam demo

```bash
python webcam_test.py
```

---

## Flask API

### Setup

Configure `app/.env`:

```env
DATA_FOLDER=data
RESNET=InceptionResnetV1_vggface2.onnx
DEEPPIX=OULU_Protocol_2_model_0_0.onnx
MINIFAS=MiniFASNetV2.onnx
FACEBANK=facebank.csv

# Generate a key: python -c "import secrets; print(secrets.token_hex(32))"
API_KEYS=your-secret-key

# KYC thresholds
LIVENESS_THRESHOLD=0.80
FACE_MATCH_THRESHOLD=0.85
IDENTITY_THRESHOLD=0.9

MAX_CONTENT_MB=10
RATE_LIMIT=30 per minute
PORT=5000
FLASK_DEBUG=false
```

### Start the server

```bash
# Production (Waitress WSGI — Windows compatible)
python app/app.py

# Development
FLASK_DEBUG=true python app/app.py
```

### Endpoints

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | `/health` | Server health check | No |
| POST | `/liveness` | Liveness detection on a selfie | Yes |
| POST | `/face-match` | Compare passport face vs selfie (KYC) | Yes |
| POST | `/identity` | Facebank identity lookup | Yes |
| POST | `/main` | Identity + liveness in one call | Yes |
| POST | `/compare` | Direct photo-to-photo comparison | Yes |

All protected endpoints require the `X-API-Key` header.

### Input formats

**Raw bytes** (for testing):
```bash
curl -X POST http://localhost:5000/liveness \
  -H "X-API-Key: your-key" \
  -H "Content-Type: image/jpeg" \
  --data-binary @selfie.jpg
```

**JSON + base64** (for mobile apps):
```bash
curl -X POST http://localhost:5000/liveness \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64 JPEG>"}'
```

### Response examples

`POST /liveness`
```json
{ "passed": true, "score": 0.87, "threshold": 0.80 }
```

`POST /face-match`
```json
{ "matched": true, "score": 0.88, "distance": 0.384, "threshold": 0.85 }
```

`POST /identity`
```json
{ "matched_name": "reynolds", "is_match": true, "min_dist": 0.384, "mean_dist": 0.83, "threshold": 0.9 }
```

---

## Test Client

```bash
# Health check
python app/client.py --service health

# Liveness
python app/client.py --service liveness \
  --image data/images/reynolds_001.png \
  --api_key your-key

# Face match (KYC — passport vs selfie)
python app/client.py --service face-match \
  --reference data/images/reynolds_001.png \
  --selfie    data/images/reynolds_002.png \
  --api_key   your-key

# Identity lookup
python app/client.py --service identity \
  --image data/images/reynolds_001.png \
  --api_key your-key

# JSON/base64 mode (mimics mobile app behaviour)
python app/client.py --service liveness \
  --image data/images/reynolds_001.png \
  --api_key your-key \
  --json_mode
```

---

## KYC Integration (NFC e-Passport)

The `/face-match` endpoint supports direct integration with NFC e-Passport readers following ICAO 9303.

`app/dg2_decoder.py` extracts the face JPEG from raw DG2 TLV bytes read off the passport chip. Send either:

- `dg2_hex` — raw DG2 bytes as a hex string (full TLV decoding done server-side)
- `reference_image` — pre-extracted face JPEG as base64

```bash
# Using raw DG2 hex
curl -X POST http://localhost:5000/face-match \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "dg2_hex": "<hex string from NFC chip>",
    "selfie_image": "<base64 selfie JPEG>"
  }'
```

---

## Docker

Build and run:

```bash
docker build --tag face-demo .

docker run -p 5000:5000 face-demo
```

---

## Run Tests

```bash
pip install pytest
pytest
```

37 tests covering face detection, identity verification, liveness detection, and all API endpoints.

---

## Models & Credits

| Model | Source |
|-------|--------|
| **MTCNN** face detector | [facenet-pytorch](https://github.com/timesler/facenet-pytorch) via `mtcnn` package |
| **InceptionResNetV1** (VGGFace2) | [facenet-pytorch](https://github.com/timesler/facenet-pytorch) |
| **DeepPixBiS** liveness | [Deep Pixel-wise Binary Supervision](https://arxiv.org/abs/1907.04047) — [pretrained models](https://www.idiap.ch/software/bob/docs/bob/bob.paper.deep_pix_bis_pad.icb2019/master/pix_bis.html) |
| **MiniFASNetV2** liveness | [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) |dir