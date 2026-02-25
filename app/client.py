"""
client.py
---------
Test client for the face tools API.

Usage examples:

  # Identity check
  python client.py --service identity --image data/images/reynolds_001.png --api_key your-key

  # Liveness check
  python client.py --service liveness --image data/images/reynolds_001.png --api_key your-key

  # Full pipeline
  python client.py --service main --image data/images/reynolds_001.png --api_key your-key

  # Photo comparison (ID vs selfie)
  python client.py --service compare \
      --photo_a data/images/reynolds_001.png \
      --photo_b data/images/reynolds_002.png \
      --api_key your-key

  # Health check (no key needed)
  python client.py --service health
"""

import argparse
import os

import requests

parser = argparse.ArgumentParser(description="Face tools API client.")
parser.add_argument("--image", metavar="path", type=str,
                    help="Image path (for main, liveness, identity).")
parser.add_argument("--photo_a", metavar="path", type=str,
                    help="Reference image path (for /compare).")
parser.add_argument("--photo_b", metavar="path", type=str,
                    help="Probe image path (for /compare).")
parser.add_argument("--host", type=str, default="localhost")
parser.add_argument("--port", type=str, default="5000")
parser.add_argument("--service", type=str,
                    choices=["main", "liveness", "identity", "compare", "health"],
                    default="main")
parser.add_argument("--api_key", type=str,
                    default=os.environ.get("FACETOOLS_API_KEY", ""),
                    help="API key for authentication. Can also be set via "
                         "FACETOOLS_API_KEY environment variable.")

args = parser.parse_args()
BASE_URL = f"http://{args.host}:{args.port}"
HEADERS = {"X-API-Key": args.api_key, "content-type": "image/jpeg"}


def post_image(img_path: str, endpoint: str) -> dict:
    with open(img_path, "rb") as f:
        data = f.read()
    response = requests.post(f"{BASE_URL}/{endpoint}", data=data, headers=HEADERS)
    return response.status_code, response.json()


def post_compare(photo_a: str, photo_b: str) -> dict:
    with open(photo_a, "rb") as fa, open(photo_b, "rb") as fb:
        response = requests.post(
            f"{BASE_URL}/compare",
            files={
                "photo_a": ("photo_a.jpg", fa, "image/jpeg"),
                "photo_b": ("photo_b.jpg", fb, "image/jpeg"),
            },
            headers={"X-API-Key": args.api_key},
        )
    return response.status_code, response.json()


def get_health() -> dict:
    response = requests.get(f"{BASE_URL}/health")
    return response.status_code, response.json()


# Dispatch
if args.service == "health":
    status, result = get_health()
elif args.service == "compare":
    if not args.photo_a or not args.photo_b:
        raise SystemExit("--photo_a and --photo_b are required for /compare.")
    status, result = post_compare(args.photo_a, args.photo_b)
else:
    if not args.image:
        raise SystemExit(f"--image is required for /{args.service}.")
    status, result = post_image(args.image, args.service)

print(f"Status: {status}")
print(f"Response:\n", result)