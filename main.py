import cv2
import dlib
import numpy as np
import sqlite3
import requests
from datetime import datetime
from scipy.spatial import distance as dist
from imutils import face_utils

# ── Constants ───────────────────────────────────────────────────────────
EAR_THRESHOLD     = 0.25
EAR_CONSEC_FRAMES = 20
PREDICTOR_PATH    = "shape_predictor_68_face_landmarks.dat"
WINDOW_NAME       = "Drowsiness Detector"
DB_PATH           = "drowsiness.db"

# ── Oracle APEX REST URLs ────────────────────────────────────────────────
APEX_SESSION_URL = "https://oracleapex.com/ords/sadikshya_c7466829/drowsiness/sessions/"
APEX_ALERT_URL   = "https://oracleapex.com/ords/sadikshya_c7466829/drowsiness/alerts/"

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
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id     INTEGER,
            alert_time     TEXT,
            ear_value      REAL,
            duration_frames INTEGER,
            synced         INTEGER DEFAULT 0
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

def log_alert(session_id, ear_value, duration_frames):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO alerts (session_id, alert_time, ear_value, duration_frames)
                 VALUES (?, ?, ?, ?)''',
              (session_id, datetime.now().isoformat(),
               round(float(ear_value), 4), int(duration_frames)))
    c.execute("UPDATE sessions SET total_alerts = total_alerts + 1 WHERE id = ?",
              (session_id,))
    conn.commit()
    conn.close()

def end_session(session_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE sessions SET session_end = ? WHERE id = ?",
              (datetime.now().isoformat(), session_id))
    conn.commit()
    conn.close()
    print(f"Session {session_id} ended in SQLite")

# ── Sync to Oracle APEX ──────────────────────────────────────────────────
# def sync_to_oracle():
#     print("Syncing to Oracle APEX...")
#     conn = sqlite3.connect(DB_PATH)
#     c = conn.cursor()

    # Sync unsynced sessions
    c.execute("SELECT * FROM sessions WHERE synced = 0")
    sessions = c.fetchall()
    for s in sessions:
        try:
            requests.post(APEX_SESSION_URL,
                headers={"Content-Type": "application/json"},
                json={"notes": f"Session {s[0]} - {s[1]}"},
                timeout=3)
            c.execute("UPDATE sessions SET synced = 1 WHERE id = ?", (s[0],))
            print(f"Session {s[0]} synced to Oracle")
        except Exception as e:
            print(f"Session sync failed: {e}")

    # Sync unsynced alerts
    c.execute("SELECT * FROM alerts WHERE synced = 0")
    alerts = c.fetchall()
    for a in alerts:
        try:
            requests.post(APEX_ALERT_URL,
                headers={"Content-Type": "application/json"},
                json={
                    "session_id": a[1],
                    "ear_value": a[3],
                    "duration_frames": a[4]
                },
                timeout=3)
            c.execute("UPDATE alerts SET synced = 1 WHERE id = ?", (a[0],))
            print(f"Alert {a[0]} synced to Oracle")
        except Exception as e:
            print(f"Alert sync failed: {e}")

    conn.commit()
    conn.close()
    print("Sync complete!")

# ── Load Models ──────────────────────────────────────────────────────────
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

# ── Initialise DB and start session ──────────────────────────────────────
init_db()
session_id  = start_session()

print("Starting webcam... Press Q or ESC to quit.")
cap           = cv2.VideoCapture(0)
frame_counter = 0
alert_logged  = False
ear_values    = []

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
        except Exception as e:
            continue

        leftEye  = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        ear      = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
        ear_values.append(ear)

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
        cv2.putText(frame, f"Alerts: {frame_counter}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Session: {session_id}", (10, 120),
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
sync_to_oracle()
print("Session ended.")