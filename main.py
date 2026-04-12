import cv2
import dlib
import numpy as np
import sqlite3
from datetime import datetime
from scipy.spatial import distance as dist
from imutils import face_utils

# ── Constants ───────────────────────────────────────────────────────────
EAR_THRESHOLD     = 0.25
MAR_THRESHOLD     = 0.6
EAR_CONSEC_FRAMES = 20
MAR_CONSEC_FRAMES = 15
PREDICTOR_PATH    = "shape_predictor_68_face_landmarks.dat"
WINDOW_NAME       = "DrowsGuard - Driver Safety System"
DB_PATH           = "drowsiness.db"

# ── Greeting based on time ───────────────────────────────────────────────
def get_greeting():
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "Good Morning"
    elif 12 <= hour < 17:
        return "Good Afternoon"
    elif 17 <= hour < 21:
        return "Good Evening"
    else:
        return "Good Night"

# ── SQLite Setup ─────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_start TEXT, session_end TEXT,
        total_alerts INTEGER DEFAULT 0,
        avg_ear REAL, synced INTEGER DEFAULT 0)''')
    c.execute('''CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER, alert_time TEXT,
        alert_type TEXT, ear_value REAL,
        mar_value REAL, duration_frames INTEGER,
        synced INTEGER DEFAULT 0)''')
    conn.commit()
    conn.close()

def start_session():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO sessions (session_start) VALUES (?)",
              (datetime.now().isoformat(),))
    sid = c.lastrowid
    conn.commit()
    conn.close()
    return sid

def log_alert(session_id, alert_type, ear, mar, frames):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO alerts
        (session_id, alert_time, alert_type, ear_value, mar_value, duration_frames)
        VALUES (?,?,?,?,?,?)''',
        (session_id, datetime.now().isoformat(), alert_type,
         round(float(ear), 4), round(float(mar), 4), int(frames)))
    c.execute("UPDATE sessions SET total_alerts = total_alerts + 1 WHERE id=?",
              (session_id,))
    conn.commit()
    conn.close()

def end_session(session_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE sessions SET session_end=? WHERE id=?",
              (datetime.now().isoformat(), session_id))
    conn.commit()
    conn.close()

# ── EAR & MAR ────────────────────────────────────────────────────────────
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[1], mouth[7])
    B = dist.euclidean(mouth[2], mouth[6])
    C = dist.euclidean(mouth[0], mouth[4])
    return (A + B) / (2.0 * C)

# ── Draw modern UI overlay ────────────────────────────────────────────────
def draw_ui(frame, ear, mar, ear_counter, mar_counter,
            session_id, total_alerts, greeting, status):

    h, w = frame.shape[:2]

    # Dark top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Dark bottom bar
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h-70), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay2, 0.85, frame, 0.15, 0, frame)

    # App title
    cv2.putText(frame, "DrowsGuard", (12, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)

    # Greeting + time
    now_str = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, f"{greeting}  |  {now_str}", (200, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Session info top right
    cv2.putText(frame, f"Session #{session_id}", (w-160, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)

    # Bottom stats bar
    cv2.putText(frame, f"EAR: {ear:.2f}", (15, h-42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 100) if ear >= EAR_THRESHOLD else (0, 60, 255), 2)

    cv2.putText(frame, f"MAR: {mar:.2f}", (140, h-42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 100) if mar <= MAR_THRESHOLD else (0, 165, 255), 2)

    cv2.putText(frame, f"Alerts: {total_alerts}", (265, h-42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)

    cv2.putText(frame, f"Eye frames: {ear_counter}/{EAR_CONSEC_FRAMES}",
                (370, h-42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    cv2.putText(frame, "Press Q / ESC to quit", (w-220, h-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)

    # Status alert banner
    if status == "EAR":
        cv2.rectangle(frame, (0, 60), (w, 115), (0, 0, 180), -1)
        cv2.putText(frame, "⚠  DROWSINESS ALERT  — Eyes closing detected!",
                    (15, 97), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    elif status == "MAR":
        cv2.rectangle(frame, (0, 60), (w, 115), (0, 100, 180), -1)
        cv2.putText(frame, "⚠  YAWN DETECTED  — Signs of fatigue!", (15, 97),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame

# ── Load Models ──────────────────────────────────────────────────────────
print("Loading models...")
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor(PREDICTOR_PATH)
print("Models loaded!")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]

# ── Init ─────────────────────────────────────────────────────────────────
init_db()
session_id    = start_session()
greeting      = get_greeting()
total_alerts  = 0
ear_counter   = 0
mar_counter   = 0
ear_alert_logged = False
mar_alert_logged = False
status        = "OK"

print(f"{greeting}! Session {session_id} started.")
print("Starting webcam... Press Q or ESC to quit.")

cap = cv2.VideoCapture(0)
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 800, 600)

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    frame = cv2.resize(frame, (800, 600))
    gray  = np.ascontiguousarray(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), dtype=np.uint8)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    ear = 0.0
    mar = 0.0
    status = "OK"

    if len(faces) == 0:
        cv2.putText(frame, "No face detected — please face the camera",
                    (80, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    for (x, y, w, h) in faces:
        dlib_rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
        try:
            shape = predictor(gray, dlib_rect)
            shape = face_utils.shape_to_np(shape)
        except Exception:
            continue

        leftEye  = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth    = shape[mStart:mEnd]

        ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
        mar = mouth_aspect_ratio(mouth)

        # Draw contours
        cv2.drawContours(frame, [cv2.convexHull(leftEye)],  -1, (0, 255, 100), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 100), 1)
        cv2.drawContours(frame, [cv2.convexHull(mouth)],    -1, (0, 220, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 180, 255), 2)

        # EAR check
        if ear < EAR_THRESHOLD:
            ear_counter += 1
            if ear_counter >= EAR_CONSEC_FRAMES:
                status = "EAR"
                if not ear_alert_logged:
                    log_alert(session_id, "EAR", ear, mar, ear_counter)
                    total_alerts += 1
                    ear_alert_logged = True
        else:
            ear_counter      = 0
            ear_alert_logged = False

        # MAR check
        if mar > MAR_THRESHOLD:
            mar_counter += 1
            if mar_counter >= MAR_CONSEC_FRAMES:
                if status == "OK":
                    status = "MAR"
                if not mar_alert_logged:
                    log_alert(session_id, "MAR", ear, mar, mar_counter)
                    total_alerts += 1
                    mar_alert_logged = True
        else:
            mar_counter      = 0
            mar_alert_logged = False

    # Draw UI
    frame = draw_ui(frame, ear, mar, ear_counter, mar_counter,
                    session_id, total_alerts, greeting, status)

    cv2.imshow(WINDOW_NAME, frame)

    # Exit on X button or Q/ESC
    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break
    if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
        break

cap.release()
cv2.destroyAllWindows()
end_session(session_id)
print(f"Session {session_id} ended. Total alerts: {total_alerts}")