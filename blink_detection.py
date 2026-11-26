
import cv2
import mediapipe as mp
import numpy as np

# Load MediaPipe Face Landmarker
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="face_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)

landmarker = FaceLandmarker.create_from_options(options)

# Open webcam
cap = cv2.VideoCapture(0)

blink_count = 0
eye_closed = False

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))

    result = landmarker.detect_for_video(mp_image, timestamp)

    if result.face_landmarks:

        landmarks = result.face_landmarks[0]

        # Eye landmarks
        left_eye = [33, 160, 158, 133, 153, 144]
        right_eye = [362, 385, 387, 263, 373, 380]

        h, w, _ = frame.shape

        def get_points(indices):
            return np.array([
                [landmarks[i].x * w, landmarks[i].y * h]
                for i in indices
            ])

        def calculate_ear(points):
            a = np.linalg.norm(points[1] - points[5])
            b = np.linalg.norm(points[2] - points[4])
            c = np.linalg.norm(points[0] - points[3])

            return (a + b) / (2.0 * c)

        left_ear = calculate_ear(get_points(left_eye))
        right_ear = calculate_ear(get_points(right_eye))

        ear = (left_ear + right_ear) / 2

        if ear < 0.20:
            eye_closed = True

        elif eye_closed:
            blink_count += 1
            eye_closed = False

        cv2.putText(
            frame,
            f"Blink Count: {blink_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"EAR: {ear:.2f}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

    cv2.imshow("AI Blink Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
landmarker.close()
cv2.destroyAllWindows()
