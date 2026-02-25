import argparse

import requests

parser = argparse.ArgumentParser(description="Client for the face tools API.")

parser.add_argument("--image", metavar="path", type=str,
                    help="Path to image file (for main, liveness, identity endpoints).")
parser.add_argument("--photo_a", metavar="path", type=str,
                    help="Reference image path (for /compare endpoint).")
parser.add_argument("--photo_b", metavar="path", type=str,
                    help="Probe image path (for /compare endpoint).")
parser.add_argument("--host", type=str, default="localhost")
parser.add_argument("--port", type=str, default="5000")
parser.add_argument("--service", type=str,
                    choices=["main", "liveness", "identity", "compare"],
                    default="main",
                    help="API endpoint to call.")

args = parser.parse_args()
BASE_URL = f"http://{args.host}:{args.port}"


def post_image(img_path: str, endpoint: str) -> dict:
    with open(img_path, "rb") as f:
        img_data = f.read()
    response = requests.post(
        f"{BASE_URL}/{endpoint}",
        data=img_data,
        headers={"content-type": "image/jpeg"},
    )
    return response.json()


def post_compare(photo_a: str, photo_b: str) -> dict:
    with open(photo_a, "rb") as fa, open(photo_b, "rb") as fb:
        response = requests.post(
            f"{BASE_URL}/compare",
            files={
                "photo_a": ("photo_a.jpg", fa, "image/jpeg"),
                "photo_b": ("photo_b.jpg", fb, "image/jpeg"),
            },
        )
    return response.json()


if args.service == "compare":
    if not args.photo_a or not args.photo_b:
        raise SystemExit("--photo_a and --photo_b are required for the /compare endpoint.")
    result = post_compare(args.photo_a, args.photo_b)
else:
    if not args.image:
        raise SystemExit("--image is required for this endpoint.")
    result = post_image(args.image, args.service)

print("Response:\n", result)