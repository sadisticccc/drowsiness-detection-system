import cv2
import dlib
import numpy as np
import sqlite3
import threading
import pyttsx3
import queue as _queue
import time
import math
from datetime import datetime
from scipy.spatial import distance as dist
from imutils import face_utils

db_lock = threading.Lock()

# ── Detection Thresholds ─────────────────────────────────────────────────
EAR_THRESHOLD      = 0.25
MAR_THRESHOLD      = 0.5
EAR_CONSEC_FRAMES  = 20
MAR_CONSEC_FRAMES  = 8
YAWNS_BEFORE_ALERT = 3
PREDICTOR_PATH     = "shape_predictor_68_face_landmarks.dat"
WINDOW_NAME        = "DrowsGuard"
DB_PATH            = "drowsiness.db"

# ── Colour Palette (BGR) ─────────────────────────────────────────────────
C_BG     = (8,   10,  14)
C_PANEL  = (18,  22,  30)
C_BORDER = (45,  55,  75)
C_ACCENT = (0,   210, 255)
C_ACCENT2= (0,   160, 255)
C_GREEN  = (80,  220, 130)
C_AMBER  = (30,  160, 255)
C_RED    = (55,  65,  230)
C_WHITE  = (245, 248, 252)
C_MUTED  = (100, 115, 140)
C_DIM    = (55,  65,  85)
C_VIOLET = (210, 100, 180)

# ════════════════════════════════════════════════════════════════════════
#  AUDIO — single persistent engine on dedicated thread
# ════════════════════════════════════════════════════════════════════════
_audio_q       = _queue.Queue()
_audio_playing = False

def _audio_worker():
    global _audio_playing
    while True:
        msg = _audio_q.get()
        if msg is None:
            break
        try:
            _audio_playing = True
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            if len(voices) > 1:
                engine.setProperty('voice', voices[1].id)
            engine.setProperty('rate',   130)
            engine.setProperty('volume', 0.9)
            engine.say(msg)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print(f"[Audio] Speech failed: {e}")
        finally:
            _audio_playing = False
            _audio_q.task_done()

threading.Thread(target=_audio_worker, daemon=True).start()

_audio_lock = threading.Lock()

def play_alert_sound(alert_type="EAR"):
    msg = ("Please stay alert. Drowsiness has been detected."
           if alert_type == "EAR"
           else "You appear to be fatigued. Please consider taking a rest.")
    def _speak():
        with _audio_lock:
            try:
                e = pyttsx3.init()
                voices = e.getProperty('voices')
                if len(voices) > 1:
                    e.setProperty('voice', voices[1].id)
                e.setProperty('rate', 130)
                e.setProperty('volume', 0.9)
                e.say(msg)
                e.runAndWait()
                e.stop()
            except Exception as ex:
                print(f"[Audio] {ex}")
    threading.Thread(target=_speak, daemon=True).start()

# ════════════════════════════════════════════════════════════════════════
#  DRAWING UTILITIES
# ════════════════════════════════════════════════════════════════════════
def blend(img, overlay, alpha):
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def filled_rect(img, pt1, pt2, color, alpha=1.0):
    if alpha >= 1.0:
        cv2.rectangle(img, pt1, pt2, color, -1)
    else:
        ov = img.copy()
        cv2.rectangle(ov, pt1, pt2, color, -1)
        blend(img, ov, alpha)

def put_text(img, text, pos, scale, color, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    cv2.putText(img, text, pos, font, scale, color, thickness, cv2.LINE_AA)

def put_duplex(img, text, pos, scale, color, thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_DUPLEX, scale, color, thickness, cv2.LINE_AA)

def text_size(text, scale, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    return cv2.getTextSize(text, font, scale, thickness)[0]

def draw_arc(img, center, radius, start_angle, end_angle, color, thickness=2, alpha=1.0):
    ov = img.copy() if alpha < 1.0 else img
    cv2.ellipse(ov, center, (radius, radius), -90, start_angle, end_angle,
                color, thickness, cv2.LINE_AA)
    if alpha < 1.0:
        blend(img, ov, alpha)

def draw_ring_gauge(img, cx, cy, radius, value, vmin, vmax,
                    color_low, color_high, label, val_str, warn=False):
    pct  = max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
    span = 240.0
    fill = span * pct
    draw_arc(img, (cx, cy), radius, 0, span, C_DIM, 3, 0.6)
    for i in range(9):
        angle = math.radians(-90 + (i / 8) * span)
        ix = int(cx + (radius + 7)  * math.cos(angle))
        iy = int(cy + (radius + 7)  * math.sin(angle))
        ox = int(cx + (radius + 11) * math.cos(angle))
        oy = int(cy + (radius + 11) * math.sin(angle))
        cv2.line(img, (ix, iy), (ox, oy), C_DIM, 1, cv2.LINE_AA)
    col = color_high if warn else color_low
    if fill > 0:
        draw_arc(img, (cx, cy), radius, 0, fill, col, 3)
        tip_angle = math.radians(-90 + fill)
        tx = int(cx + radius * math.cos(tip_angle))
        ty = int(cy + radius * math.sin(tip_angle))
        cv2.circle(img, (tx, ty), 5, col, -1, cv2.LINE_AA)
        cv2.circle(img, (tx, ty), 8, col,  1, cv2.LINE_AA)
    tw, th = text_size(val_str, 0.7, 2, cv2.FONT_HERSHEY_DUPLEX)
    put_duplex(img, val_str, (cx - tw//2, cy + th//2 - 2), 0.7, col, 2)
    lw = text_size(label, 0.28)[0]
    put_text(img, label, (cx - lw//2, cy + radius + 18), 0.28, C_MUTED)

def draw_face_bracket(img, x, y, w, h, color, ln=22, thick=2, pulse=False):
    if pulse:
        ov = img.copy()
        cv2.rectangle(ov, (x, y), (x+w, y+h), color, 1)
        blend(img, ov, 0.15)
    for px, py, dx, dy in [(x,y,1,1),(x+w,y,-1,1),(x,y+h,1,-1),(x+w,y+h,-1,-1)]:
        cv2.line(img, (px, py), (px+dx*ln, py), color, thick, cv2.LINE_AA)
        cv2.line(img, (px, py), (px, py+dy*ln), color, thick, cv2.LINE_AA)
        cv2.circle(img, (px, py), 3, color, -1, cv2.LINE_AA)

def draw_stat_card(img, x, y, w, h, label, value, accent=None, bar_pct=None, alert=False):
    if accent is None:
        accent = C_ACCENT
    filled_rect(img, (x, y), (x+w, y+h), C_PANEL, 0.92)
    filled_rect(img, (x, y+4), (x+2, y+h-4), accent)
    cv2.line(img, (x, y), (x+w, y), accent if alert else C_BORDER, 1, cv2.LINE_AA)
    put_text(img, label, (x+10, y+16), 0.28, C_MUTED)
    value_y = y + h - (20 if bar_pct is not None else 14)
    put_duplex(img, value, (x+10, value_y), 0.58, accent, 1)
    if bar_pct is not None:
        bx, by_b, bw_b = x+8, y+h-8, w-16
        filled_rect(img, (bx, by_b), (bx+bw_b, by_b+3), C_DIM)
        fw = int(bw_b * min(max(bar_pct, 0), 1))
        if fw > 0:
            filled_rect(img, (bx, by_b), (bx+fw, by_b+3), accent)

def draw_pulse_line(img, x, y, w, h, value, color, history):
    history.append(value)
    if len(history) > w:
        history.pop(0)
    if len(history) < 2:
        return
    pts = [(x + int(i * w / max(len(history)-1, 1)),
            y + h - int(v * h))
           for i, v in enumerate(history)]
    for i in range(1, len(pts)):
        alpha = 0.4 + 0.6 * (i / len(pts))
        col   = tuple(int(c * alpha) for c in color)
        cv2.line(img, pts[i-1], pts[i], col, 1, cv2.LINE_AA)

_alert_anim = 0.0

def draw_alert_banner(img, w, status, frame_count):
    global _alert_anim
    _alert_anim = max(0.0, _alert_anim - 0.08) if status == "OK" \
                  else min(1.0, _alert_anim + 0.12)
    if _alert_anim < 0.05:
        return
    pulse = 0.7 + 0.3 * abs(math.sin(frame_count * 0.15))
    alpha = _alert_anim * pulse
    if status == "EAR":
        bg_col, border = (40, 40, 180), (80, 80, 240)
        icon = "!! DROWSINESS ALERT"
        sub  = "EYES CLOSING DETECTED — PLEASE STAY ALERT"
    elif status == "MAR":
        bg_col, border = (120, 50, 180), (160, 80, 220)
        icon = "!! FATIGUE ALERT"
        sub  = "YAWNING DETECTED — CONSIDER TAKING A BREAK"
    else:
        return
    filled_rect(img, (0, 62), (w, 106), bg_col, alpha * 0.88)
    cv2.line(img, (0,  62), (w,  62), border, 1, cv2.LINE_AA)
    cv2.line(img, (0, 106), (w, 106), border, 1, cv2.LINE_AA)
    iw = text_size(icon, 0.65, 2, cv2.FONT_HERSHEY_DUPLEX)[0]
    put_duplex(img, icon, ((w-iw)//2, 84), 0.65, C_WHITE, 2)
    sw = text_size(sub, 0.33)[0]
    put_text(img, sub, ((w-sw)//2, 100), 0.33, C_WHITE)

def draw_top_bar(img, w, session_id, fps, greeting, cam_name, frame_count):
    filled_rect(img, (0, 0), (w, 56), C_BG, 0.97)
    cv2.line(img, (0, 56), (w, 56), C_BORDER, 1, cv2.LINE_AA)
    pulse_r = 6 + int(2 * abs(math.sin(frame_count * 0.05)))
    cv2.circle(img, (20, 28), pulse_r,     C_ACCENT, -1, cv2.LINE_AA)
    cv2.circle(img, (20, 28), pulse_r + 3, C_ACCENT,  1, cv2.LINE_AA)
    put_duplex(img, "DROWSGUARD",     (36, 33), 0.60, C_ACCENT, 1)
    put_text(img,   "AI MONITOR v2.0",(38, 48), 0.25, C_MUTED)
    now_str    = datetime.now().strftime("%H:%M:%S")
    date_str   = datetime.now().strftime("%a %d %b %Y").upper()
    center_txt = f"{greeting.upper()}  |  {now_str}"
    cw = text_size(center_txt, 0.46)[0]
    put_text(img, center_txt, ((w-cw)//2, 30), 0.46, C_WHITE)
    dw = text_size(date_str, 0.27)[0]
    put_text(img, date_str, ((w-dw)//2, 46), 0.27, C_MUTED)
    fps_color = C_GREEN if fps >= 15 else (C_AMBER if fps >= 10 else C_RED)
    put_text(img, f"SESSION #{session_id:02d}", (w-180, 24), 0.34, C_MUTED)
    put_text(img, cam_name[:18],               (w-180, 38), 0.28, C_DIM)
    fps_str = f"{fps} FPS"
    sw = text_size(fps_str, 0.42)[0]
    put_duplex(img, fps_str, (w-sw-14, 52), 0.42, fps_color)
    for i in range(5):
        col = fps_color if (fps/30.0) > (i/5.0) else C_DIM
        bx  = w - 188 + i * 10
        filled_rect(img, (bx, 40), (bx+8, 44), col)

def draw_bottom_hud(img, h, w, ear, mar, ear_counter, mar_counter,
                    total_alerts, status, frame_count,
                    ear_history, mar_history, yawn_count=0):
    bh = 140
    by = h - bh
    filled_rect(img, (0, by), (w, h), C_BG, 0.96)
    cv2.line(img, (0, by), (w, by), C_BORDER, 1, cv2.LINE_AA)

    # EAR ring gauge
    ear_warn = ear < EAR_THRESHOLD
    draw_ring_gauge(img, 72, by+65, 46, ear, 0.0, 0.5,
                    C_GREEN, C_RED, "EYE ASPECT RATIO", f"{ear:.3f}", warn=ear_warn)
    ear_pct = ear_counter / EAR_CONSEC_FRAMES
    if ear_pct > 0:
        draw_arc(img, (72, by+65), 54, 0, ear_pct*240, C_RED, 2, 0.7)

    # MAR ring gauge
    mar_warn = mar > MAR_THRESHOLD
    draw_ring_gauge(img, 200, by+65, 46, mar, 0.0, 1.5,
                    C_ACCENT, C_VIOLET, "MOUTH ASPECT RATIO", f"{mar:.3f}", warn=mar_warn)

    cv2.line(img, (275, by+15), (275, h-10), C_BORDER, 1, cv2.LINE_AA)

    # EAR waveform
    put_text(img, "EAR SIGNAL", (288, by+18), 0.27, C_MUTED)
    draw_pulse_line(img, 288, by+22, 175, 58,
                    min(max(ear/0.5, 0), 1.0), list(C_GREEN), ear_history)
    th_y = by + 22 + 58 - int((EAR_THRESHOLD/0.5)*58)
    cv2.line(img, (288, th_y), (463, th_y), C_RED, 1, cv2.LINE_AA)
    put_text(img, "THRESHOLD", (466, th_y+4), 0.22, C_RED)

    # MAR waveform
    put_text(img, "MAR SIGNAL", (288, by+90), 0.27, C_MUTED)
    draw_pulse_line(img, 288, by+94, 175, 38,
                    min(max(mar/1.5, 0), 1.0), list(C_ACCENT2), mar_history)

    cv2.line(img, (475, by+15), (475, h-10), C_BORDER, 1, cv2.LINE_AA)

    # Alert count card
    alert_col = C_GREEN if total_alerts == 0 else (C_AMBER if total_alerts < 5 else C_RED)
    draw_stat_card(img, 485, by+8,  120, 58, "TOTAL ALERTS",
                   str(total_alerts), accent=alert_col, alert=(total_alerts > 0))

    # Yawn progress card
    yawn_col = C_GREEN if yawn_count == 0 else \
               (C_AMBER if yawn_count < YAWNS_BEFORE_ALERT else C_VIOLET)
    draw_stat_card(img, 485, by+72, 120, 58, f"YAWNS  {yawn_count}/{YAWNS_BEFORE_ALERT}",
                   f"{yawn_count} / {YAWNS_BEFORE_ALERT}",
                   accent=yawn_col,
                   bar_pct=yawn_count / YAWNS_BEFORE_ALERT)

    cv2.line(img, (618, by+15), (618, h-10), C_BORDER, 1, cv2.LINE_AA)

    # Driver status
    status_map = {
        "OK":  ("DRIVER SAFE",  "MONITORING ACTIVE",  C_GREEN),
        "EAR": ("DROWSY",       "EYES CLOSING",        C_RED),
        "MAR": ("FATIGUED",     "YAWNING DETECTED",    C_VIOLET),
    }
    st_label, st_sub, st_color = status_map.get(status, status_map["OK"])
    pulse  = 0.6 + 0.4 * abs(math.sin(frame_count * 0.1))
    stx, sty = 670, by + 50
    cv2.circle(img, (stx, sty), 15, st_color, 1,  cv2.LINE_AA)
    cv2.circle(img, (stx, sty), 10, st_color, -1, cv2.LINE_AA)
    if status != "OK":
        ov = img.copy()
        cv2.circle(ov, (stx, sty), 24, st_color, 2)
        blend(img, ov, pulse * 0.5)
    put_duplex(img, st_label, (stx+25, sty+6),  0.52, st_color, 1)
    put_text(img,  st_sub,   (stx+25, sty+22), 0.28, C_MUTED)
    put_text(img, f"LIVE  {datetime.now().strftime('%H:%M:%S')}",
             (stx+25, by+90), 0.28, C_DIM)

    tag = "DROWSGUARD — REAL-TIME BIOMETRIC MONITORING"
    tw  = text_size(tag, 0.24)[0]
    put_text(img, tag, (w-tw-10, h-5), 0.24, C_DIM)

def draw_landmarks(frame, shape, status):
    lS, lE = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    rS, rE = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    mS, mE = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    left_eye  = shape[lS:lE]
    right_eye = shape[rS:rE]
    mouth_pts = shape[mS:mE]
    eye_col   = C_RED    if status == "EAR" else C_GREEN
    mouth_col = C_VIOLET if status == "MAR" else C_ACCENT
    ov = frame.copy()
    cv2.fillConvexPoly(ov, cv2.convexHull(left_eye),  eye_col)
    cv2.fillConvexPoly(ov, cv2.convexHull(right_eye), eye_col)
    cv2.fillConvexPoly(ov, cv2.convexHull(mouth_pts), mouth_col)
    blend(frame, ov, 0.18)
    cv2.drawContours(frame, [cv2.convexHull(left_eye)],  -1, eye_col,   1, cv2.LINE_AA)
    cv2.drawContours(frame, [cv2.convexHull(right_eye)], -1, eye_col,   1, cv2.LINE_AA)
    cv2.drawContours(frame, [cv2.convexHull(mouth_pts)], -1, mouth_col, 1, cv2.LINE_AA)
    for pt in np.concatenate([left_eye, right_eye]):
        cv2.circle(frame, tuple(pt), 2, eye_col, -1, cv2.LINE_AA)
    for pt in mouth_pts:
        cv2.circle(frame, tuple(pt), 2, mouth_col, -1, cv2.LINE_AA)

def draw_corner_reticles(frame, w, h, bottom_hud_h=140, top_bar_h=56):
    vx1, vy1, vx2, vy2 = 0, top_bar_h, w, h - bottom_hud_h
    ln, col = 14, C_BORDER
    for px, py, dx, dy in [(vx1,vy1,1,1),(vx2,vy1,-1,1),(vx1,vy2,1,-1),(vx2,vy2,-1,-1)]:
        cv2.line(frame, (px, py), (px+dx*ln, py), col, 1, cv2.LINE_AA)
        cv2.line(frame, (px, py), (px, py+dy*ln), col, 1, cv2.LINE_AA)

def draw_scanlines(frame, h, w, strength=0.03):
    ov = np.zeros_like(frame)
    for y in range(0, h, 4):
        cv2.line(ov, (0, y), (w, y), (0, 0, 0), 1)
    cv2.addWeighted(frame, 1.0, ov, strength, 0, frame)

# ════════════════════════════════════════════════════════════════════════
#  DATABASE
# ════════════════════════════════════════════════════════════════════════
def get_greeting():
    hour = datetime.now().hour
    if   5  <= hour < 12: return "Good Morning"
    elif 12 <= hour < 17: return "Good Afternoon"
    elif 17 <= hour < 21: return "Good Evening"
    else:                 return "Good Night"

def init_db():
    with db_lock:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
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
        c.execute("UPDATE sessions SET session_end=datetime('now') WHERE session_end IS NULL")
        conn.commit()
        conn.close()

def start_session():
    with db_lock:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        c = conn.cursor()
        c.execute("INSERT INTO sessions (session_start) VALUES (?)",
                  (datetime.now().isoformat(),))
        sid = c.lastrowid
        conn.commit()
        conn.close()
    return sid

def log_alert(session_id, alert_type, ear, mar, frames):
    with db_lock:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
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

def end_session(session_id, ear_samples):
    with db_lock:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        c = conn.cursor()
        avg_ear = round(sum(ear_samples)/len(ear_samples), 4) if ear_samples else 0.0
        c.execute("UPDATE sessions SET session_end=?, avg_ear=? WHERE id=?",
                  (datetime.now().isoformat(), avg_ear, session_id))
        conn.commit()
        conn.close()

# ════════════════════════════════════════════════════════════════════════
#  BIOMETRICS
# ════════════════════════════════════════════════════════════════════════
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

# ════════════════════════════════════════════════════════════════════════
#  CAMERA DETECTION
# ════════════════════════════════════════════════════════════════════════
def detect_cameras():
    print("Scanning for cameras...")
    available = []
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                name = f"Camera {i} (Built-in)" if i == 0 else f"Camera {i}"
                available.append((i, name))
                print(f"  [{i}] {name}")
            cap.release()
    return available

def select_camera(available):
    if not available:
        return 0
    if len(available) == 1:
        return available[0][0]
    print("\nSelect camera (auto in 5s):")
    for idx, (ci, name) in enumerate(available):
        print(f"  {idx+1}. {name}")
    import msvcrt, time as _t
    deadline = _t.time() + 5
    s = ""
    while _t.time() < deadline:
        if msvcrt.kbhit():
            ch = msvcrt.getwche()
            if ch == '\r': break
            s += ch
    try:
        c = int(s.strip()) - 1
        if 0 <= c < len(available):
            return available[c][0]
    except Exception:
        pass
    return available[0][0]

# ════════════════════════════════════════════════════════════════════════
#  INIT
# ════════════════════════════════════════════════════════════════════════
print("Loading models...")
face_cascade    = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
profile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml")
predictor       = dlib.shape_predictor(PREDICTOR_PATH)
print("Models loaded!")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]

available_cameras = detect_cameras()
cam_index         = select_camera(available_cameras)
cam_name          = next((n for i, n in available_cameras if i == cam_index), f"Camera {cam_index}")

init_db()
session_id       = start_session()
greeting         = get_greeting()
total_alerts     = 0
ear_counter      = 0
mar_counter      = 0
ear_alert_logged = False
mar_alert_logged = False
status           = "OK"
prev_time        = time.perf_counter()
frame_count      = 0
yawn_count       = 0
yawn_in_progress = False
ear_samples      = []
ear_history      = []
mar_history      = []
fps_samples      = []

print(f"{greeting}! Session {session_id} started. Camera: {cam_name}")
print("Press Q/ESC to quit. Press C to switch camera.")

cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 900, 660)

# ════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ════════════════════════════════════════════════════════════════════════
try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.05)
            continue

        frame_count += 1
        now          = time.perf_counter()
        fps          = int(1 / max(now - prev_time, 1e-6))
        prev_time    = now
        if fps < 500:
            fps_samples.append(fps)

        frame = cv2.resize(frame, (900, 660))
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Face detection: frontal first, profile fallback
        haar_faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE)
        
        if len(haar_faces) == 0:
            profile_faces = profile_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=8, minSize=(80, 80))
            if len(profile_faces) > 0:
                haar_faces = profile_faces
            else:
                flipped = cv2.flip(gray, 1)
                flipped_faces = profile_cascade.detectMultiScale(
                    flipped, scaleFactor=1.1, minNeighbors=8, minSize=(80, 80))
                if len(flipped_faces) > 0:
                    fw = gray.shape[1]
                    haar_faces = np.array([[fw - x - w, y, w, h]
                                           for (x, y, w, h) in flipped_faces])

        faces = [dlib.rectangle(int(x), int(y), int(x+fw), int(y+fh))
                 for (x, y, fw, fh) in haar_faces] if len(haar_faces) > 0 else []

        ear    = 0.0
        mar    = 0.0
        status = "OK"

        if len(faces) == 0:
            h_f, w_f = gray.shape
            cr = gray[h_f//4:3*h_f//4, w_f//4:3*w_f//4]
            msg = ("Adjust position — face the camera directly"
                   if np.mean(cr) > 40
                   else "No face detected — move in front of camera")
            tw = text_size(msg, 0.55)[0]
            put_text(frame, msg, ((900-tw)//2, 350), 0.55, C_AMBER)

        for dlib_rect in faces:
            try:
                shape = predictor(gray, dlib_rect)
                shape = face_utils.shape_to_np(shape)
            except Exception as e:
                print(f"[Landmark] {e}")
                continue

            leftEye  = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            mouth    = shape[mStart:mEnd]

            ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
            mar = mouth_aspect_ratio(mouth)
            ear_samples.append(round(float(ear), 4))

            draw_landmarks(frame, shape, status)
            draw_face_bracket(frame, dlib_rect.left(), dlib_rect.top(),
                              dlib_rect.width(), dlib_rect.height(),
                              C_RED if status != "OK" else C_ACCENT,
                              pulse=(status != "OK"))

            # ── EAR drowsiness check ──────────────────────────────────
            if ear < EAR_THRESHOLD:
                ear_counter += 1
                if ear_counter >= EAR_CONSEC_FRAMES:
                    status = "EAR"
                    if not ear_alert_logged:
                        log_alert(session_id, "EAR", ear, mar, ear_counter)
                        total_alerts    += 1
                        ear_alert_logged = True
                        play_alert_sound("EAR")
            else:
                ear_counter      = 0
                ear_alert_logged = False

            # ── MAR yawn cycle counting ───────────────────────────────
            if mar > MAR_THRESHOLD:
                mar_counter += 1
                if mar_counter >= MAR_CONSEC_FRAMES and not yawn_in_progress:
                    yawn_in_progress = True
            else:
                if yawn_in_progress:
                    yawn_in_progress = False
                    yawn_count += 1
                    print(f"[Yawn] #{yawn_count} detected")
                    if yawn_count >= YAWNS_BEFORE_ALERT:
                        status = "MAR"
                        log_alert(session_id, "MAR", ear, mar, mar_counter)
                        total_alerts += 1
                        play_alert_sound("MAR")
                        yawn_count = 0
                        print("[Alert] MAR alert fired!")
                mar_counter = 0

        # ── Render ───────────────────────────────────────────────────
        draw_top_bar(frame, 900, session_id, fps, greeting, cam_name, frame_count)
        draw_alert_banner(frame, 900, status, frame_count)
        draw_bottom_hud(frame, 660, 900, ear, mar,
                        ear_counter, mar_counter,
                        total_alerts, status, frame_count,
                        ear_history, mar_history, yawn_count)
        draw_corner_reticles(frame, 900, 660)
        draw_scanlines(frame, 660, 900)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c') or key == ord('C'):
            cap.release()
            current = [i for i, n in available_cameras]
            if cam_index in current:
                pos = current.index(cam_index)
                cam_index = current[(pos + 1) % len(current)]
            else:
                cam_index = current[0]
            cam_name = next((n for i, n in available_cameras if i == cam_index), f"Camera {cam_index}")
            cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
            ear_counter = mar_counter = 0
            ear_alert_logged = mar_alert_logged = False
            print(f"[Camera] Switched to: {cam_name}")

        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key in [ord('q'), 27]:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    end_session(session_id, ear_samples)
    print(f"Session {session_id} ended. Total alerts: {total_alerts}")
    if fps_samples:
        avg = round(sum(fps_samples)/len(fps_samples), 2)
        print(f"Avg FPS: {avg} ({'PASS' if avg >= 15 else 'FAIL'})")