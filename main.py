import cv2
import dlib
import numpy as np
import sqlite3
from datetime import datetime
from scipy.spatial import distance as dist
from imutils import face_utils

# ── Constants ───────────────────────────────────────────────────────────
EAR_THRESHOLD     = 0.25   # below this = eyes closing
MAR_THRESHOLD     = 0.6    # above this = yawning
EAR_CONSEC_FRAMES = 20     # frames eye must be closed to alert
MAR_CONSEC_FRAMES = 15     # frames mouth must be open to alert
PREDICTOR_PATH    = "shape_predictor_68_face_landmarks.dat"
WINDOW_NAME       = "Drowsiness Detector"
DB_PATH           = "drowsiness.db"

# ── SQLite Setup ─────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            session_start TEXT,
            session_end   TEXT,
            total_alerts  INTEGER DEFAULT 0,
            avg_ear       REAL,
            synced        INTEGER DEFAULT 0
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      INTEGER,
            alert_time      TEXT,
            alert_type      TEXT,
            ear_value       REAL,
            mar_value       REAL,
            duration_frames INTEGER,
            synced          INTEGER DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()

def start_session():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO sessions (session_start) VALUES (?)",
              (datetime.now().isoformat(),))
    session_id = c.lastrowid
    conn.commit()
    conn.close()
    print(f"Session {session_id} started in SQLite")
    return session_id

def log_alert(session_id, alert_type, ear_value, mar_value, duration_frames):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO alerts
                 (session_id, alert_time, alert_type, ear_value, mar_value, duration_frames)
                 VALUES (?, ?, ?, ?, ?, ?)''',
              (session_id, datetime.now().isoformat(), alert_type,
               round(float(ear_value), 4), round(float(mar_value), 4),
               int(duration_frames)))
    c.execute("UPDATE sessions SET total_alerts = total_alerts + 1 WHERE id = ?",
              (session_id,))
    conn.commit()
    conn.close()
    print(f"{alert_type} alert logged to SQLite!")

def end_session(session_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE sessions SET session_end = ? WHERE id = ?",
              (datetime.now().isoformat(), session_id))
    conn.commit()
    conn.close()
    print(f"Session {session_id} ended in SQLite")

# ── EAR Formula ──────────────────────────────────────────────────────────
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# ── MAR Formula ──────────────────────────────────────────────────────────
def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[1], mouth[7])
    B = dist.euclidean(mouth[2], mouth[6])
    C = dist.euclidean(mouth[0], mouth[4])
    return (A + B) / (2.0 * C)

# ── Load Models ──────────────────────────────────────────────────────────
print("Loading models...")
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor(PREDICTOR_PATH)
print("Models loaded successfully!")

# ── Landmark indices ─────────────────────────────────────────────────────
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]

# ── Init DB and start session ────────────────────────────────────────────
init_db()
session_id = start_session()

print("Starting webcam... Press Q or ESC to quit.")
cap               = cv2.VideoCapture(0)
ear_counter       = 0
mar_counter       = 0
ear_alert_logged  = False
mar_alert_logged  = False
ear_values        = []

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
            shape = predictor(gray, dlib_rect)
            shape = face_utils.shape_to_np(shape)
        except Exception:
            continue

        # ── Eyes ─────────────────────────────────────────────────────────
        leftEye  = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        ear      = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
        ear_values.append(ear)

        # ── Mouth ─────────────────────────────────────────────────────────
        mouth = shape[mStart:mEnd]
        mar   = mouth_aspect_ratio(mouth)

        # ── Draw contours ─────────────────────────────────────────────────
        cv2.drawContours(frame, [cv2.convexHull(leftEye)],  -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(mouth)],    -1, (0, 255, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # ── EAR check (eye closure) ───────────────────────────────────────
        if ear < EAR_THRESHOLD:
            ear_counter += 1
            if ear_counter >= EAR_CONSEC_FRAMES:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if not ear_alert_logged:
                    log_alert(session_id, "EAR", ear, mar, ear_counter)
                    ear_alert_logged = True
        else:
            ear_counter      = 0
            ear_alert_logged = False

        # ── MAR check (yawn detection) ────────────────────────────────────
        if mar > MAR_THRESHOLD:
            mar_counter += 1
            if mar_counter >= MAR_CONSEC_FRAMES:
                cv2.putText(frame, "YAWN DETECTED!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                if not mar_alert_logged:
                    log_alert(session_id, "MAR", ear, mar, mar_counter)
                    mar_alert_logged = True
        else:
            mar_counter      = 0
            mar_alert_logged = False

        # ── Display stats ─────────────────────────────────────────────────
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Session: {session_id}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow(WINDOW_NAME, frame)
    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break
    if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
        break

# ── Cleanup ───────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
end_session(session_id)
print("Session ended.")