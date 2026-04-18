import cv2
import dlib
import numpy as np
import sqlite3
import threading
import pyttsx3
from datetime import datetime
from scipy.spatial import distance as dist
from imutils import face_utils

# ── Constants ───────────────────────────────────────────────────────────
EAR_THRESHOLD     = 0.25
MAR_THRESHOLD     = 0.6
EAR_CONSEC_FRAMES = 20
MAR_CONSEC_FRAMES = 15
PREDICTOR_PATH    = "shape_predictor_68_face_landmarks.dat"
WINDOW_NAME       = "DrowsGuard"
DB_PATH           = "drowsiness.db"

# ── Colors (BGR) ─────────────────────────────────────────────────────────
C_CYAN    = (255, 210, 0)
C_WHITE   = (240, 240, 240)
C_GRAY    = (120, 120, 120)
C_GREEN   = (80, 220, 100)
C_RED     = (60, 60, 220)
C_ORANGE  = (30, 150, 255)
C_DARK    = (20, 20, 20)
C_PANEL   = (30, 35, 45)

def get_greeting():
    hour = datetime.now().hour
    if 5 <= hour < 12:   return "Good Morning"
    elif 12 <= hour < 17: return "Good Afternoon"
    elif 17 <= hour < 21: return "Good Evening"
    else:                 return "Good Night"

engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

def play_alert_sound(alert_type="EAR"):
    def speak():
        if alert_type == "EAR":
            engine.say("Drowsiness detected! Please take a break.")
        else:
            engine.say("Yawning detected. You seem tired. Please rest.")
        engine.runAndWait()
    threading.Thread(target=speak, daemon=True).start()

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

def draw_rounded_rect(img, pt1, pt2, color, alpha, radius=10):
    overlay = img.copy()
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(overlay, (x1+radius, y1), (x2-radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1+radius), (x2, y2-radius), color, -1)
    cv2.circle(overlay, (x1+radius, y1+radius), radius, color, -1)
    cv2.circle(overlay, (x2-radius, y1+radius), radius, color, -1)
    cv2.circle(overlay, (x1+radius, y2-radius), radius, color, -1)
    cv2.circle(overlay, (x2-radius, y2-radius), radius, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

def draw_stat_pill(frame, x, y, label, value, val_color, bar_val=None):
    draw_rounded_rect(frame, (x, y), (x+170, y+52), C_PANEL, 0.88)
    cv2.putText(frame, label, (x+10, y+18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_GRAY, 1, cv2.LINE_AA)
    cv2.putText(frame, value, (x+10, y+42),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, val_color, 1, cv2.LINE_AA)
    if bar_val is not None:
        bar_w = int(150 * min(bar_val, 1.0))
        cv2.rectangle(frame, (x+10, y+48), (x+160, y+51), (50,50,50), -1)
        cv2.rectangle(frame, (x+10, y+48), (x+10+bar_w, y+51), val_color, -1)

def draw_ui(frame, ear, mar, ear_counter, mar_counter,
            session_id, total_alerts, greeting, status, fps):
    h, w = frame.shape[:2]

    # ── Top navbar ────────────────────────────────────────────────────────
    draw_rounded_rect(frame, (0, 0), (w, 58), C_DARK, 0.92, radius=0)
    cv2.line(frame, (0, 58), (w, 58), (50, 55, 65), 1)

    # Logo dot
    cv2.circle(frame, (22, 29), 7, C_CYAN, -1)
    cv2.putText(frame, "DrowsGuard", (38, 37),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, C_CYAN, 1, cv2.LINE_AA)

    # Center greeting + time
    now_str = datetime.now().strftime("%H:%M:%S")
    greet_text = f"{greeting}  |  {now_str}"
    tw = cv2.getTextSize(greet_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0]
    cv2.putText(frame, greet_text, ((w - tw)//2, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, C_WHITE, 1, cv2.LINE_AA)

    # Right — session + FPS
    cv2.putText(frame, f"Session #{session_id}   {fps} FPS", (w-200, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_GRAY, 1, cv2.LINE_AA)

    # ── Alert banner ──────────────────────────────────────────────────────
    if status == "EAR":
        draw_rounded_rect(frame, (0, 62), (w, 108), (0, 0, 180), 0.92, radius=0)
        cv2.line(frame, (0, 108), (w, 108), (0, 0, 220), 1)
        msg = "DROWSINESS ALERT  —  Eyes closing detected!"
        tw  = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 0.62, 1)[0][0]
        cv2.putText(frame, msg, ((w-tw)//2, 92),
                    cv2.FONT_HERSHEY_DUPLEX, 0.62, C_WHITE, 1, cv2.LINE_AA)
    elif status == "MAR":
        draw_rounded_rect(frame, (0, 62), (w, 108), (30, 100, 200), 0.92, radius=0)
        cv2.line(frame, (0, 108), (w, 108), (30, 130, 220), 1)
        msg = "YAWN DETECTED  —  Signs of fatigue!"
        tw  = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 0.62, 1)[0][0]
        cv2.putText(frame, msg, ((w-tw)//2, 92),
                    cv2.FONT_HERSHEY_DUPLEX, 0.62, C_WHITE, 1, cv2.LINE_AA)

    # ── Bottom stat pills ─────────────────────────────────────────────────
    ear_color  = C_GREEN if ear >= EAR_THRESHOLD else C_RED
    mar_color  = C_GREEN if mar <= MAR_THRESHOLD else C_ORANGE
    ear_bar    = ear / 0.4
    mar_bar    = mar / 1.0

    draw_stat_pill(frame, 10,   h-70, "EYE RATIO (EAR)",
                   f"{ear:.3f}", ear_color, ear_bar)
    draw_stat_pill(frame, 190,  h-70, "MOUTH RATIO (MAR)",
                   f"{mar:.3f}", mar_color, mar_bar)
    draw_stat_pill(frame, 370,  h-70, "TOTAL ALERTS",
                   str(total_alerts), C_ORANGE)
    draw_stat_pill(frame, 550,  h-70, "EYE FRAMES",
                   f"{ear_counter}/{EAR_CONSEC_FRAMES}", C_GRAY,
                   ear_counter/EAR_CONSEC_FRAMES)

    # ── No face message ───────────────────────────────────────────────────
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

init_db()
session_id       = start_session()
greeting         = get_greeting()
total_alerts     = 0
ear_counter      = 0
mar_counter      = 0
ear_alert_logged = False
mar_alert_logged = False
status           = "OK"
prev_time        = datetime.now()

print(f"{greeting}! Session {session_id} started.")
print("Starting webcam... Press Q or ESC to quit.")

cap = cv2.VideoCapture(0)
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 900, 660)

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    # FPS
    now       = datetime.now()
    fps       = int(1 / max((now - prev_time).total_seconds(), 0.001))
    prev_time = now

    frame = cv2.resize(frame, (900, 660))
    gray  = np.ascontiguousarray(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), dtype=np.uint8)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    ear    = 0.0
    mar    = 0.0
    status = "OK"

    if len(faces) == 0:
        tw = cv2.getTextSize("No face detected — please face the camera",
                             cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)[0][0]
        cv2.putText(frame, "No face detected — please face the camera",
                    ((900-tw)//2, 350),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_ORANGE, 1, cv2.LINE_AA)

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

        # Draw clean contours
        cv2.drawContours(frame, [cv2.convexHull(leftEye)],  -1, C_GREEN, 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, C_GREEN, 1)
        cv2.drawContours(frame, [cv2.convexHull(mouth)],    -1, C_CYAN,  1)

        # Clean face box — just corners not full rectangle
        bx, by, bw, bh = x, y, w, h
        ln = 18
        clr = C_RED if status != "OK" else (0, 180, 255)
        for px, py, dx, dy in [(bx,by,1,1),(bx+bw,by,-1,1),
                                (bx,by+bh,1,-1),(bx+bw,by+bh,-1,-1)]:
            cv2.line(frame, (px, py), (px+dx*ln, py), clr, 2)
            cv2.line(frame, (px, py), (px, py+dy*ln), clr, 2)

        # EAR check
        if ear < EAR_THRESHOLD:
            ear_counter += 1
            if ear_counter >= EAR_CONSEC_FRAMES:
                status = "EAR"
                if not ear_alert_logged:
                    log_alert(session_id, "EAR", ear, mar, ear_counter)
                    total_alerts += 1
                    ear_alert_logged = True
                    play_alert_sound("EAR")
        else:
            ear_counter      = 0
            ear_alert_logged = False

        # MAR check
        if mar > MAR_THRESHOLD:
            mar_counter += 1
            if mar_counter >= MAR_CONSEC_FRAMES:
                if status == "OK": status = "MAR"
                if not mar_alert_logged:
                    log_alert(session_id, "MAR", ear, mar, mar_counter)
                    total_alerts += 1
                    mar_alert_logged = True
                    play_alert_sound("MAR")
        else:
            mar_counter      = 0
            mar_alert_logged = False

    frame = draw_ui(frame, ear, mar, ear_counter, mar_counter,
                    session_id, total_alerts, greeting, status, fps)

    cv2.imshow(WINDOW_NAME, frame)

    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break
    if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
        break

cap.release()
cv2.destroyAllWindows()
end_session(session_id)
print(f"Session {session_id} ended. Total alerts: {total_alerts}")