import os
import cv2
import mediapipe as mp
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LANDMARKER_PATH = os.path.join(BASE_DIR, "face_landmarker.task")

options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=LANDMARKER_PATH
    ),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_faces=5,
    min_face_detection_confidence=0.3,
    min_face_presence_confidence=0.3
)

face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
    options
)


def crop_face_bgr(image_bgr):

    if image_bgr is None:
        raise ValueError("Could not read the image")

    height, width = image_bgr.shape[:2]

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb
    )

    result = face_landmarker.detect(mp_image)

    if not result.face_landmarks:
        raise ValueError("No face detected. Move closer to the camera.")

    landmarks = max(
        result.face_landmarks,
        key=lambda face: (
            max(p.x for p in face) - min(p.x for p in face)
        ) * (
            max(p.y for p in face) - min(p.y for p in face)
        )
    )

    x_min = min(p.x for p in landmarks) * width
    x_max = max(p.x for p in landmarks) * width
    y_min = min(p.y for p in landmarks) * height
    y_max = max(p.y for p in landmarks) * height

    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2

    crop_size = max(x_max - x_min, y_max - y_min) * 1.8

    left = max(0, int(center_x - crop_size / 2))
    right = min(width, int(center_x + crop_size / 2))
    top = max(0, int(center_y - crop_size / 2))
    bottom = min(height, int(center_y + crop_size / 2))

    if left >= right or top >= bottom:
        raise ValueError("Invalid face crop")

    return image_bgr[top:bottom, left:right]


def preprocess_image(image_bgr):

    face_bgr = crop_face_bgr(image_bgr)

    face_rgb = cv2.cvtColor(
        face_bgr,
        cv2.COLOR_BGR2RGB
    )

    image_resized = cv2.resize(
        face_rgb,
        (128, 128)
    )

    return np.expand_dims(
        image_resized.astype("float32"),
        axis=0
    )