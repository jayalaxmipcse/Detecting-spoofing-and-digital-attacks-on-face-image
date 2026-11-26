import argparse
import os
import re
import time
from datetime import datetime

import cv2


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "webcam_samples")
MIN_CAPTURE_INTERVAL = 1.0


def main():
    parser = argparse.ArgumentParser(
        description="Capture labeled webcam samples locally."
    )
    parser.add_argument(
        "--participant",
        required=True,
        help="Anonymous participant ID, for example p01.",
    )
    parser.add_argument(
        "--session",
        required=True,
        help="Session ID; use the same ID for real and spoof captures.",
    )
    parser.add_argument(
        "--label",
        required=True,
        choices=("real", "spoof"),
        help="Ground-truth label for what is presented to the camera.",
    )
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    for value, name in (
        (args.participant, "participant"),
        (args.session, "session"),
    ):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
            parser.error(f"{name} must contain only letters, numbers, _ or -")

    output_folder = os.path.join(
        OUTPUT_DIR, args.participant, args.session, args.label
    )
    os.makedirs(output_folder, exist_ok=True)

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}")

    last_capture = 0.0
    print(f"Saving {args.label} samples to: {output_folder}")
    print("Press SPACE to save a frame (at least 1 second apart); Q to finish.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Could not read a frame from the webcam")

            preview = frame.copy()
            cv2.putText(
                preview,
                f"Label: {args.label} | SPACE: save | Q: quit",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Local webcam dataset capture", preview)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == 32:
                now = time.monotonic()
                if now - last_capture < MIN_CAPTURE_INTERVAL:
                    continue
                filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f.jpg")
                path = os.path.join(output_folder, filename)
                if not cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise OSError(f"Could not save captured image: {path}")
                last_capture = now
                print(f"Saved: {path}")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
