import cv2
import dlib
import numpy as np
import requests
import json
from datetime import datetime
from scipy.spatial import distance as dist
from imutils import face_utils

# ── Constants ──────────────────────────────────────────────────────────
EAR_THRESHOLD     = 0.25
EAR_CONSEC_FRAMES = 20
PREDICTOR_PATH    = "shape_predictor_68_face_landmarks.dat"
WINDOW_NAME       = "Drowsiness Detector"

# ── Oracle APEX REST URLs ───────────────────────────────────────────────
BASE_URL     = "https://oracleapex.com/ords/sadikshya_c7466829/drowsiness"
SESSION_URL  = f"{BASE_URL}/sessions/"
ALERT_URL    = f"{BASE_URL}/alerts/"

def log_session_start():
    try:
        response = requests.post(SESSION_URL,
            headers={"Content-Type": "application/json"},
            json={"notes": "Session started"},
            timeout=5)
        print("Session logged to Oracle:", response.status_code)
        return response.json().get("session_id", 1)
    except Exception as e:
        print("Oracle connection failed, running offline:", e)
        return None

def log_alert(session_id, ear_value, duration_frames):
    try:
        requests.post(ALERT_URL,
            headers={"Content-Type": "application/json"},
            json={
                "session_id": session_id or 1,
                "ear_value": round(float(ear_value), 4),
                "duration_frames": int(duration_frames)
            },
            timeout=5)
    except Exception as e:
        print("Alert log failed:", e)

# ── Load models ─────────────────────────────────────────────────────────
print("Loading models...")
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor(PREDICTOR_PATH)
print("Models loaded successfully!")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# ── Start session ────────────────────────────────────────────────────────
print("Logging session to Oracle...")
session_id = log_session_start()

print("Starting webcam... Press Q or ESC to quit.")
cap = cv2.VideoCapture(0)
frame_counter  = 0
alert_logged   = False

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Failed to grab frame.")
        break

    frame = cv2.resize(frame, (640, 480))
    gray  = np.ascontiguousarray(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), dtype=np.uint8)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        cv2.putText(frame, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    for (x, y, w, h) in faces:
        dlib_rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))

        try:
            shape    = predictor(gray, dlib_rect)
            shape    = face_utils.shape_to_np(shape)
        except Exception as e:
            print("Predictor failed:", e)
            continue

        leftEye  = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        ear      = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0

        cv2.drawContours(frame, [cv2.convexHull(leftEye)],  -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        if ear < EAR_THRESHOLD:
            frame_counter += 1
            if frame_counter >= EAR_CONSEC_FRAMES:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if not alert_logged:
                    log_alert(session_id, ear, frame_counter)
                    alert_logged = True
        else:
            if alert_logged:
                alert_logged = False
            frame_counter = 0

        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Closed frames: {frame_counter}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        if session_id:
            cv2.putText(frame, f"Session: {session_id}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow(WINDOW_NAME, frame)

    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break
    if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
        break

cap.release()
cv2.destroyAllWindows()
print("Session ended.")