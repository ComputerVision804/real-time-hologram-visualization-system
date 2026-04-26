import cv2
import numpy as np
import time
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =====================================================
# FACE MODEL
# =====================================================
face_model = "C:\\Users\\Qc\\Desktop\\Antigravity\\face_landmarker.task"

BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=face_model),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=3
)

face_detector = FaceLandmarker.create_from_options(face_options)

# =====================================================
# HAND MODEL
# =====================================================
hand_model = "C:\\Users\\Qc\\Desktop\\Antigravity\\hand_landmarker.task"

HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=hand_model),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

hand_detector = HandLandmarker.create_from_options(hand_options)

# =====================================================
# CAMERA
# =====================================================
cap = cv2.VideoCapture(0)

cv2.namedWindow("one Hologram System", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("one Hologram System", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

zoom = 1.0

# =====================================================
# MAIN LOOP
# =====================================================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    timestamp = int(time.time() * 1000)

    # =============================
    # FACE DETECTION
    # =============================
    face_result = face_detector.detect_for_video(mp_image, timestamp)

    # =============================
    # HAND DETECTION
    # =============================
    hand_result = hand_detector.detect_for_video(mp_image, timestamp)

    # =============================
    # GESTURE CONTROL (PINCH ZOOM)
    # =============================
    if hand_result.hand_landmarks:
        for hand_landmarks in hand_result.hand_landmarks:

            thumb = hand_landmarks[4]
            index = hand_landmarks[8]

            x1, y1 = thumb.x * w, thumb.y * h
            x2, y2 = index.x * w, index.y * h

            dist = ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5

            zoom = np.interp(dist, [20, 200], [0.6, 2.5])

    # =============================
    # HOLOGRAM LAYERS
    # =============================
    holo_left = np.zeros_like(frame)
    holo_right = np.zeros_like(frame)

    # LEFT + RIGHT OFFSETS
    left_offset = w // 4
    right_offset = int(w * 0.75)

    # =============================
    # FACE DRAWING
    # =============================
    if face_result.face_landmarks:
        for face_landmarks in face_result.face_landmarks:

            for lm in face_landmarks:

                x = lm.x * w
                y = lm.y * h

                # =============================
                # LEFT HOLOGRAM
                # =============================
                lx = int(left_offset + (x - w/2) * zoom)
                ly = int((y - h/2) * zoom + h/2)

                cv2.circle(holo_left, (lx, ly), 2, (0, 255, 0), -1)

                # =============================
                # RIGHT HOLOGRAM
                # =============================
                rx = int(right_offset + (x - w/2) * zoom)
                ry = int((y - h/2) * zoom + h/2)

                cv2.circle(holo_right, (rx, ry), 2, (0, 255, 0), -1)

    # =============================
    # COMBINE EVERYTHING
    # =============================
    output = cv2.add(holo_left, holo_right)
    output = cv2.addWeighted(frame, 0.4, output, 1.0, 0)

    # split guide lines
    cv2.line(output, (w//2, 0), (w//2, h), (80, 80, 80), 2)
    cv2.line(output, (w//4, 0), (w//4, h), (40, 40, 40), 1)
    cv2.line(output, (int(w*0.75), 0), (int(w*0.75), h), (40, 40, 40), 1)

    cv2.imshow("one Hologram System", output)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()