import sys
from collections import deque

import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "face_spoof_model.keras"
IMG_SIZE = (128, 128)
THRESHOLD = 0.5
MARGIN = 0.25
SMOOTH_FRAMES = 7

model = tf.keras.models.load_model(MODEL_PATH)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def crop_face(frame_bgr):
    """Return (face_crop, box) for the biggest face, or (None, None)."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))
    if len(faces) == 0:
        return None, None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    mx, my = int(w * MARGIN), int(h * MARGIN)
    x1, y1 = max(0, x - mx), max(0, y - my)
    x2 = min(frame_bgr.shape[1], x + w + mx)
    y2 = min(frame_bgr.shape[0], y + h + my)
    return frame_bgr[y1:y2, x1:x2], (x1, y1, x2, y2)


def score_face(face_bgr):
    """Raw model score: near 0 = REAL, near 1 = SPOOF."""
    img = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)      # BGR -> RGB
    img = cv2.resize(img, IMG_SIZE)                      # same as training
    img = np.expand_dims(img.astype("float32"), 0)
    return float(model.predict(img, verbose=0)[0][0])


def predict_image(image_bgr):
    """Predict one BGR image (from cv2.imread). Returns (label, score)."""
    face, _ = crop_face(image_bgr)
    if face is None:
        return "NO FACE", None
    score = score_face(face)
    return ("SPOOF" if score > THRESHOLD else "REAL"), score


def run_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam could not be opened. Try VideoCapture(1).")
    scores = deque(maxlen=SMOOTH_FRAMES)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        face, box = crop_face(frame)
        if face is None:
            scores.clear()
            cv2.putText(frame, "No face", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            scores.append(score_face(face))
            avg = sum(scores) / len(scores)
            spoof = avg > THRESHOLD
            color = (0, 0, 255) if spoof else (0, 200, 0)
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{'SPOOF' if spoof else 'REAL'}  score={avg:.2f}",
                        (x1, max(30, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Face Spoof Detection (q = quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        if img is None:
            raise SystemExit("Could not read image: " + sys.argv[1])
        label, score = predict_image(img)
        print("Result:", label, "| score:", score)
    else:
        run_webcam()
