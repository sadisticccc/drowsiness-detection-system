import cv2
import dlib
import numpy as np
import sqlite3
import threading
import pyttsx3
import time
import math
from datetime import datetime
from scipy.spatial import distance as dist
from imutils import face_utils

db_lock = threading.Lock()

# ── Constants ───────────────────────────────────────────────────────────
EAR_THRESHOLD     = 0.25
MAR_THRESHOLD     = 0.6
EAR_CONSEC_FRAMES = 20
MAR_CONSEC_FRAMES = 15
PREDICTOR_PATH    = "shape_predictor_68_face_landmarks.dat"
WINDOW_NAME       = "DrowsGuard"
DB_PATH           = "drowsiness.db"

# ── Weighted Drowsiness Score ────────────────────────────────────────────
# Instead of counting plain frames, we accumulate a score based on HOW
# closed the eyes are. Deep closure = score rises fast = faster alert.
# A normal blink barely moves the score; real drowsiness triggers quickly.
DROWSY_SCORE_THRESHOLD = 3.0   # total score needed to fire EAR alert
DROWSY_SCORE_DECAY     = 0.3   # score drops this much every frame eyes are open
MAR_SCORE_THRESHOLD    = 2.0   # same idea for yawning
MAR_SCORE_DECAY        = 0.2

# ── Luxury Dark Palette (BGR) ────────────────────────────────────────────
C_BG          = (8, 10, 14)          # near-black base
C_PANEL       = (18, 22, 30)         # card background
C_PANEL2      = (24, 30, 42)         # slightly lighter card
C_BORDER      = (45, 55, 75)         # subtle border
C_ACCENT      = (0, 210, 255)        # electric cyan — primary accent
C_ACCENT2     = (0, 160, 255)        # deeper cyan
C_GREEN       = (80, 220, 130)       # safe/good
C_AMBER       = (30, 160, 255)       # warning amber
C_RED         = (55, 65, 230)        # alert red
C_WHITE       = (245, 248, 252)      # pure text
C_MUTED       = (100, 115, 140)      # secondary text
C_DIM         = (55, 65, 85)         # very muted
C_GOLD        = (40, 185, 255)       # accent gold-amber
C_VIOLET      = (210, 100, 180)      # MAR/yawn violet

# Overlay alpha blending helpers
def blend(img, overlay, alpha):
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def filled_rect(img, pt1, pt2, color, alpha=1.0):
    if alpha >= 1.0:
        cv2.rectangle(img, pt1, pt2, color, -1)
    else:
        ov = img.copy()
        cv2.rectangle(ov, pt1, pt2, color, -1)
        blend(img, ov, alpha)

def draw_line(img, pt1, pt2, color, thickness=1, alpha=1.0):
    if alpha >= 1.0:
        cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)
    else:
        ov = img.copy()
        cv2.line(ov, pt1, pt2, color, thickness, cv2.LINE_AA)
        blend(img, ov, alpha)

def put_text(img, text, pos, scale, color, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    cv2.putText(img, text, pos, font, scale, color, thickness, cv2.LINE_AA)

def put_duplex(img, text, pos, scale, color, thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_DUPLEX, scale, color, thickness, cv2.LINE_AA)

def text_size(text, scale, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    return cv2.getTextSize(text, font, scale, thickness)[0]

# ── Animated arc / ring drawing ──────────────────────────────────────────
def draw_arc(img, center, radius, start_angle, end_angle, color, thickness=2, alpha=1.0):
    ov = img.copy() if alpha < 1.0 else img
    cv2.ellipse(ov, center, (radius, radius), -90, start_angle, end_angle, color, thickness, cv2.LINE_AA)
    if alpha < 1.0:
        blend(img, ov, alpha)

def draw_ring_gauge(img, cx, cy, radius, value, vmin, vmax, color_low, color_high,
                    label, val_str, warn=False, tick_color=None):
    """Draw a circular gauge ring — used for EAR and MAR."""
    pct  = max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
    span = 240.0  # degrees of arc
    fill = span * pct

    # Track ring (background arc)
    draw_arc(img, (cx, cy), radius,     0,    span, C_DIM,    3, 0.6)
    # Tick marks
    for i in range(9):
        angle = math.radians(-90 + (i / 8) * span)
        ix = int(cx + (radius + 7) * math.cos(angle))
        iy = int(cy + (radius + 7) * math.sin(angle))
        ox = int(cx + (radius + 11) * math.cos(angle))
        oy = int(cy + (radius + 11) * math.sin(angle))
        tc = tick_color if tick_color else C_DIM
        cv2.line(img, (ix, iy), (ox, oy), tc, 1, cv2.LINE_AA)

    # Filled arc
    col = color_high if warn else color_low
    if fill > 0:
        draw_arc(img, (cx, cy), radius, 0, fill, col, 3)

    # Glow dot at tip
    if fill > 0:
        tip_angle = math.radians(-90 + fill)
        tx = int(cx + radius * math.cos(tip_angle))
        ty = int(cy + radius * math.sin(tip_angle))
        cv2.circle(img, (tx, ty), 5, col, -1, cv2.LINE_AA)
        cv2.circle(img, (tx, ty), 8, col, 1, cv2.LINE_AA)

    # Center value
    tw, th = text_size(val_str, 0.7, 2, cv2.FONT_HERSHEY_DUPLEX)
    put_duplex(img, val_str, (cx - tw//2, cy + th//2 - 2), 0.7, col, 2)

    # Label below
    lw = text_size(label, 0.32)[0]
    put_text(img, label, (cx - lw//2, cy + radius + 18), 0.32, C_MUTED)


# ── Corner bracket face box ──────────────────────────────────────────────
def draw_face_bracket(img, x, y, w, h, color, ln=22, thick=2, pulse=False):
    if pulse:
        ov = img.copy()
        cv2.rectangle(ov, (x, y), (x+w, y+h), color, 1)
        blend(img, ov, 0.15)
    for px, py, dx, dy in [(x, y, 1, 1), (x+w, y, -1, 1), (x, y+h, 1, -1), (x+w, y+h, -1, -1)]:
        cv2.line(img, (px, py), (px + dx*ln, py), color, thick, cv2.LINE_AA)
        cv2.line(img, (px, py), (px, py + dy*ln), color, thick, cv2.LINE_AA)
        # Corner dot
        cv2.circle(img, (px, py), 3, color, -1, cv2.LINE_AA)


# ── Micro stat card ──────────────────────────────────────────────────────
def draw_stat_card(img, x, y, w, h, label, value, sub="", accent=C_ACCENT,
                   bar_pct=None, alert=False):
    # Card background
    filled_rect(img, (x, y), (x+w, y+h), C_PANEL, 0.92)
    # Left accent bar
    filled_rect(img, (x, y+4), (x+2, y+h-4), accent)
    # Top separator line
    cv2.line(img, (x, y), (x+w, y), accent if alert else C_BORDER, 1, cv2.LINE_AA)

    label_y = y + 16
    value_y = y + h - (20 if bar_pct is not None else 14)

    put_text(img, label, (x+10, label_y), 0.30, C_MUTED)
    put_duplex(img, value, (x+10, value_y), 0.60, accent, 1)
    if sub:
        sw = text_size(sub, 0.28)[0]
        put_text(img, sub, (x+w-sw-6, value_y), 0.28, C_DIM)

    if bar_pct is not None:
        bar_x = x + 8
        bar_y = y + h - 8
        bar_w = w - 16
        filled_rect(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + 3), C_DIM)
        fill_w = int(bar_w * min(max(bar_pct, 0), 1))
        if fill_w > 0:
            filled_rect(img, (bar_x, bar_y), (bar_x + fill_w, bar_y + 3), accent)


# ── Heartbeat / pulse line ────────────────────────────────────────────────
_pulse_offset = 0

def draw_pulse_line(img, x, y, w, h, value, color, history):
    """Draw a mini EKG-style history line."""
    history.append(value)
    if len(history) > w:
        history.pop(0)
    if len(history) < 2:
        return
    pts = []
    for i, v in enumerate(history):
        px = x + int(i * w / max(len(history)-1, 1))
        py = y + h - int(v * h)
        pts.append((px, py))
    for i in range(1, len(pts)):
        alpha = 0.4 + 0.6 * (i / len(pts))
        col = tuple(int(c * alpha) for c in color)
        cv2.line(img, pts[i-1], pts[i], col, 1, cv2.LINE_AA)


# ── Alert banner ──────────────────────────────────────────────────────────
_alert_anim = 0.0

def draw_alert_banner(img, w, status, frame_count):
    global _alert_anim
    if status == "OK":
        _alert_anim = max(0.0, _alert_anim - 0.08)
    else:
        _alert_anim = min(1.0, _alert_anim + 0.12)

    if _alert_anim < 0.05:
        return

    pulse = 0.7 + 0.3 * abs(math.sin(frame_count * 0.15))
    alpha = _alert_anim * pulse

    if status == "EAR":
        bg_col  = (40, 40, 180)
        border  = (80, 80, 240)
        icon    = "◉ DROWSINESS ALERT"
        sub     = "EYES CLOSING DETECTED — PLEASE STAY ALERT"
    elif status == "MAR":
        bg_col  = (120, 50, 180)
        border  = (160, 80, 220)
        icon    = "◎ FATIGUE ALERT"
        sub     = "YAWNING DETECTED — CONSIDER TAKING A BREAK"
    else:
        return

    filled_rect(img, (0, 62), (w, 106), bg_col, alpha * 0.88)
    cv2.line(img, (0, 62), (w, 62), border, 1, cv2.LINE_AA)
    cv2.line(img, (0, 106), (w, 106), border, 1, cv2.LINE_AA)

    iw = text_size(icon, 0.65, 2, cv2.FONT_HERSHEY_DUPLEX)[0]
    put_duplex(img, icon, ((w - iw)//2, 84), 0.65, C_WHITE, 2)
    sw = text_size(sub, 0.33)[0]
    put_text(img, sub, ((w - sw)//2, 100), 0.33, C_WHITE)


# ── Top HUD bar ────────────────────────────────────────────────────────────
def draw_top_bar(img, w, session_id, fps, greeting, frame_count):
    filled_rect(img, (0, 0), (w, 56), C_BG, 0.97)
    cv2.line(img, (0, 56), (w, 56), C_BORDER, 1, cv2.LINE_AA)

    # ── Left: Logo ──
    # Animated logo dot — subtle pulse
    pulse_r = 6 + int(2 * abs(math.sin(frame_count * 0.05)))
    cv2.circle(img, (20, 28), pulse_r, C_ACCENT, -1, cv2.LINE_AA)
    cv2.circle(img, (20, 28), pulse_r + 3, C_ACCENT, 1, cv2.LINE_AA)
    put_duplex(img, "DROWSGUARD", (36, 34), 0.62, C_ACCENT, 1)
    put_text(img, "AI MONITOR v2.0", (38, 48), 0.25, C_MUTED)

    # ── Center: Time + greeting ──
    now_str    = datetime.now().strftime("%H:%M:%S")
    date_str   = datetime.now().strftime("%a %d %b %Y").upper()
    center_txt = f"{greeting.upper()}  ·  {now_str}"
    cw = text_size(center_txt, 0.46)[0]
    put_text(img, center_txt, ((w - cw)//2, 30), 0.46, C_WHITE)
    dw = text_size(date_str, 0.27)[0]
    put_text(img, date_str, ((w - dw)//2, 46), 0.27, C_MUTED)

    # ── Right: Session + FPS indicator ──
    fps_color = C_GREEN if fps >= 15 else (C_AMBER if fps >= 10 else C_RED)
    sess_str = f"SESSION #{session_id:02d}"
    fps_str  = f"{fps} FPS"
    put_text(img, sess_str, (w - 160, 28), 0.36, C_MUTED)
    sw = text_size(fps_str, 0.42)[0]
    put_duplex(img, fps_str, (w - sw - 14, 44), 0.42, fps_color)

    # FPS mini-bar (5 segments)
    seg_w = 8
    for i in range(5):
        seg_on = (fps / 30.0) > (i / 5.0)
        col    = fps_color if seg_on else C_DIM
        bx     = w - 168 + i * 10
        filled_rect(img, (bx, 36), (bx + seg_w, 40), col)


# ── Bottom HUD ─────────────────────────────────────────────────────────────
def draw_bottom_hud(img, h, w, ear, mar, ear_counter, mar_counter,
                    total_alerts, status, frame_count,
                    ear_history, mar_history,
                    drowsy_score=0.0, mar_score=0.0):
    bh  = 130    # bottom panel height
    by  = h - bh
    filled_rect(img, (0, by), (w, h), C_BG, 0.96)
    cv2.line(img, (0, by), (w, by), C_BORDER, 1, cv2.LINE_AA)

    # ── EAR ring gauge ─────────────────────────────────────────────────
    ear_warn   = ear < EAR_THRESHOLD
    ear_color  = C_RED if ear_warn else C_GREEN
    draw_ring_gauge(img, 70, by + 60, 44,
                    ear, 0.0, 0.5,
                    C_GREEN, C_RED,
                    "EYE ASPECT RATIO", f"{ear:.3f}",
                    warn=ear_warn)

    # EAR frame progress arc
    ear_pct = ear_counter / EAR_CONSEC_FRAMES
    if ear_pct > 0:
        draw_arc(img, (70, by + 60), 52, 0, ear_pct * 240, C_RED, 2, 0.7)

    # ── MAR ring gauge ──────────────────────────────────────────────────
    mar_warn  = mar > MAR_THRESHOLD
    mar_color = C_VIOLET if mar_warn else C_ACCENT
    draw_ring_gauge(img, 195, by + 60, 44,
                    mar, 0.0, 1.0,
                    C_ACCENT, C_VIOLET,
                    "MOUTH ASPECT RATIO", f"{mar:.3f}",
                    warn=mar_warn)

    mar_pct = mar_counter / MAR_CONSEC_FRAMES
    if mar_pct > 0:
        draw_arc(img, (195, by + 60), 52, 0, mar_pct * 240, C_VIOLET, 2, 0.7)

    # ── Divider ─────────────────────────────────────────────────────────
    cv2.line(img, (270, by + 15), (270, h - 15), C_BORDER, 1, cv2.LINE_AA)

    # ── EAR pulse waveform ───────────────────────────────────────────────
    put_text(img, "EAR SIGNAL", (285, by + 18), 0.27, C_MUTED)
    ear_norm = min(max(ear / 0.5, 0), 1.0)
    draw_pulse_line(img, 285, by + 22, 180, 60, ear_norm, list(C_GREEN), ear_history)
    # Threshold line
    th_y = by + 22 + 60 - int((EAR_THRESHOLD / 0.5) * 60)
    cv2.line(img, (285, th_y), (465, th_y), C_RED, 1, cv2.LINE_AA)
    put_text(img, "THRESHOLD", (468, th_y + 4), 0.22, C_RED)

    # ── MAR pulse waveform ───────────────────────────────────────────────
    put_text(img, "MAR SIGNAL", (285, by + 92), 0.27, C_MUTED)
    mar_norm = min(max(mar / 1.0, 0), 1.0)
    draw_pulse_line(img, 285, by + 96, 180, 30, mar_norm, list(C_ACCENT2), mar_history)

    # ── Divider ──────────────────────────────────────────────────────────
    cv2.line(img, (510, by + 15), (510, h - 15), C_BORDER, 1, cv2.LINE_AA)

    # ── Alert counter + status ──────────────────────────────────────────
    alert_color = C_GREEN if total_alerts == 0 else (C_AMBER if total_alerts < 5 else C_RED)
    draw_stat_card(img, 520, by + 8, 115, 55, "TOTAL ALERTS",
                   str(total_alerts), accent=alert_color, alert=(total_alerts > 0))

    # Show drowsy score as a 0-100% bar so it's human readable
    score_pct = min(drowsy_score / DROWSY_SCORE_THRESHOLD, 1.0)
    score_col = C_RED if score_pct > 0.7 else (C_AMBER if score_pct > 0.3 else C_MUTED)
    draw_stat_card(img, 520, by + 68, 115, 55, "DROWSY SCORE",
                   f"{min(int(score_pct * 100), 100)}%",
                   bar_pct=score_pct,
                   accent=score_col)

    # ── Driver status ────────────────────────────────────────────────────
    cv2.line(img, (648, by + 15), (648, h - 15), C_BORDER, 1, cv2.LINE_AA)

    status_map = {
        "OK":  ("DRIVER SAFE",   "MONITORING ACTIVE",  C_GREEN),
        "EAR": ("DROWSY",        "EYES CLOSING",        C_RED),
        "MAR": ("FATIGUED",      "YAWNING DETECTED",    C_VIOLET),
    }
    st_label, st_sub, st_color = status_map.get(status, status_map["OK"])

    # Animated status indicator
    pulse = 0.6 + 0.4 * abs(math.sin(frame_count * 0.1))
    radius = 10
    sty = by + 45
    stx = 700
    cv2.circle(img, (stx, sty), radius + 5, st_color, 1, cv2.LINE_AA)
    cv2.circle(img, (stx, sty), radius,     st_color, -1, cv2.LINE_AA)
    if status != "OK":
        ov = img.copy()
        cv2.circle(ov, (stx, sty), radius + 14, st_color, 2)
        blend(img, ov, pulse * 0.5)

    put_duplex(img, st_label, (stx + 22, sty + 5), 0.52, st_color, 1)
    put_text(img,  st_sub,   (stx + 22, sty + 20), 0.28, C_MUTED)

    # Session info
    now_sec = datetime.now().strftime("%H:%M:%S")
    put_text(img, f"LIVE  {now_sec}", (stx + 22, by + 85), 0.28, C_DIM)

    # ── Bottom right: branding micro tag ─────────────────────────────────
    tag = "DROWSGUARD · REAL-TIME BIOMETRIC MONITORING"
    tw  = text_size(tag, 0.24)[0]
    put_text(img, tag, (w - tw - 10, h - 6), 0.24, C_DIM)


# ── Face landmarks overlay ─────────────────────────────────────────────────
def draw_landmarks(frame, shape, status):
    lStart, lEnd = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    rStart, rEnd = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    mStart, mEnd = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]

    left_eye  = shape[lStart:lEnd]
    right_eye = shape[rStart:rEnd]
    mouth     = shape[mStart:mEnd]

    eye_col   = C_RED if status == "EAR" else C_GREEN
    mouth_col = C_VIOLET if status == "MAR" else C_ACCENT

    # Convex hull overlays
    ov = frame.copy()
    cv2.fillConvexPoly(ov, cv2.convexHull(left_eye),  eye_col)
    cv2.fillConvexPoly(ov, cv2.convexHull(right_eye), eye_col)
    cv2.fillConvexPoly(ov, cv2.convexHull(mouth),     mouth_col)
    blend(frame, ov, 0.18)

    # Outline
    cv2.drawContours(frame, [cv2.convexHull(left_eye)],  -1, eye_col,   1, cv2.LINE_AA)
    cv2.drawContours(frame, [cv2.convexHull(right_eye)], -1, eye_col,   1, cv2.LINE_AA)
    cv2.drawContours(frame, [cv2.convexHull(mouth)],     -1, mouth_col, 1, cv2.LINE_AA)

    # Individual landmark dots
    for pt in np.concatenate([left_eye, right_eye]):
        cv2.circle(frame, tuple(pt), 2, eye_col, -1, cv2.LINE_AA)
    for pt in mouth:
        cv2.circle(frame, tuple(pt), 2, mouth_col, -1, cv2.LINE_AA)


# ── No-face overlay ──────────────────────────────────────────────────────
def draw_no_face(frame, w, h):
    msg = "NO FACE DETECTED"
    sub = "PLEASE FACE THE CAMERA"
    mw = text_size(msg, 0.7, 2, cv2.FONT_HERSHEY_DUPLEX)[0]
    sw = text_size(sub, 0.32)[0]
    cy = h // 2
    filled_rect(frame, ((w - mw)//2 - 20, cy - 30), ((w + mw)//2 + 20, cy + 30), C_PANEL, 0.8)
    put_duplex(frame, msg, ((w - mw)//2, cy + 5), 0.7, C_AMBER, 2)
    put_text(frame, sub, ((w - sw)//2, cy + 24), 0.32, C_MUTED)


# ── Scanline overlay (luxury CRT effect) ───────────────────────────────────
def draw_scanlines(frame, h, w, strength=0.04):
    ov = np.zeros_like(frame)
    for y in range(0, h, 4):
        cv2.line(ov, (0, y), (w, y), (0, 0, 0), 1)
    cv2.addWeighted(frame, 1.0, ov, strength, 0, frame)


# ── Corner reticle overlays ───────────────────────────────────────────────
def draw_corner_reticles(frame, w, h, bottom_hud_h=130, top_bar_h=56):
    """Thin reticle marks at video area corners."""
    vx1, vy1 = 0, top_bar_h
    vx2, vy2 = w, h - bottom_hud_h
    ln, col = 16, C_BORDER
    for px, py, dx, dy in [(vx1, vy1, 1, 1), (vx2, vy1, -1, 1),
                            (vx1, vy2, 1, -1), (vx2, vy2, -1, -1)]:
        cv2.line(frame, (px, py), (px + dx*ln, py), col, 1, cv2.LINE_AA)
        cv2.line(frame, (px, py), (px, py + dy*ln), col, 1, cv2.LINE_AA)


# ────────────────────────────────────────────────────────────────────────────
#  DATABASE
# ────────────────────────────────────────────────────────────────────────────
def get_greeting():
    hour = datetime.now().hour
    if   5  <= hour < 12: return "Good Morning"
    elif 12 <= hour < 17: return "Good Afternoon"
    elif 17 <= hour < 21: return "Good Evening"
    else:                 return "Good Night"

def play_alert_sound(alert_type="EAR"):
    def speak():
        try:
            _engine = pyttsx3.init()
            voices  = _engine.getProperty('voices')
            _engine.setProperty('voice', voices[1].id)
            _engine.setProperty('rate', 130)
            _engine.setProperty('volume', 0.9)
            if alert_type == "EAR":
                _engine.say("Please stay alert. Drowsiness has been detected.")
            else:
                _engine.say("You appear to be fatigued. Please consider taking a rest.")
            _engine.runAndWait()
            _engine.stop()
        except Exception as e:
            print(f"[Audio] Alert failed: {e}")
    threading.Thread(target=speak, daemon=True).start()

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

def end_session(session_id):
    with db_lock:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        c = conn.cursor()
        c.execute("SELECT COALESCE(AVG(ear_value), 0.0) FROM alerts WHERE session_id=?",
                  (session_id,))
        avg_ear = round(c.fetchone()[0], 4)
        c.execute("UPDATE sessions SET session_end=?, avg_ear=? WHERE id=?",
                  (datetime.now().isoformat(), avg_ear, session_id))
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


# ── Load Models ───────────────────────────────────────────────────────────
print("Loading models...")
face_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
face_detector = dlib.get_frontal_face_detector()
predictor     = dlib.shape_predictor(PREDICTOR_PATH)
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
prev_time        = time.perf_counter()
frame_count      = 0

# Weighted drowsiness scores
drowsy_score     = 0.0   # accumulates based on how far EAR drops below threshold
mar_score        = 0.0   # accumulates based on how far MAR rises above threshold

# Signal histories for waveform
ear_history = []
mar_history = []

print(f"{greeting}! Session {session_id} started.")
print("Starting webcam... Press Q or ESC to quit.")

cap = cv2.VideoCapture(0)
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 900, 660)

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    frame_count += 1

    # FPS
    now       = time.perf_counter()
    fps       = int(1 / max(now - prev_time, 1e-6))
    prev_time = now

    frame    = cv2.resize(frame, (900, 660))
    gray_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray     = np.array(gray_raw, dtype=np.uint8)

    # ── Face Detection ──────────────────────────────────────────────────
    haar_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(haar_faces) == 0:
        faces = list(face_detector(gray, 0))
    else:
        faces = [dlib.rectangle(int(x), int(y), int(x+w), int(y+h)) for (x, y, w, h) in haar_faces]

    ear    = 0.0
    mar    = 0.0
    status = "OK"

    if len(faces) == 0:
        draw_no_face(frame, 900, 660)

    for dlib_rect in faces:
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

        # Draw new-style landmarks
        draw_landmarks(frame, shape, status)

        # Face bracket box
        bx  = dlib_rect.left()
        by_ = dlib_rect.top()
        bw  = dlib_rect.width()
        bh_ = dlib_rect.height()
        face_col = C_RED if status != "OK" else C_ACCENT
        draw_face_bracket(frame, bx, by_, bw, bh_, face_col, pulse=(status != "OK"))

        # ── EAR check (weighted score) ──────────────────────────────────
        # Score rises faster the more closed the eyes are.
        # Normal blink (EAR ~0.22, 3-4 frames): score barely reaches ~0.4 — no alert
        # Tired eyes (EAR ~0.18, 8 frames):     score reaches ~2.8   — alert in ~0.5s
        # Fully closed (EAR ~0.05, 4 frames):   score reaches ~3.2   — alert in ~0.3s
        if ear < EAR_THRESHOLD:
            ear_counter  += 1
            deficit       = EAR_THRESHOLD - ear          # how far below threshold
            drowsy_score += deficit * 10                 # weight by severity
            if drowsy_score >= DROWSY_SCORE_THRESHOLD:
                status = "EAR"
                if not ear_alert_logged:
                    log_alert(session_id, "EAR", ear, mar, ear_counter)
                    total_alerts     += 1
                    ear_alert_logged  = True
                    play_alert_sound("EAR")
        else:
            ear_counter   = 0
            ear_alert_logged = False
            # Decay score gradually — eyes briefly open shouldn't reset everything
            drowsy_score  = max(0.0, drowsy_score - DROWSY_SCORE_DECAY)

        # ── MAR check (weighted score) ──────────────────────────────────
        if mar > MAR_THRESHOLD:
            mar_counter += 1
            surplus      = mar - MAR_THRESHOLD           # how far above threshold
            mar_score   += surplus * 8
            if mar_score >= MAR_SCORE_THRESHOLD:
                if status == "OK": status = "MAR"
                if not mar_alert_logged:
                    log_alert(session_id, "MAR", ear, mar, mar_counter)
                    total_alerts    += 1
                    mar_alert_logged = True
                    play_alert_sound("MAR")
        else:
            mar_counter      = 0
            mar_alert_logged = False
            mar_score        = max(0.0, mar_score - MAR_SCORE_DECAY)

    # ── Draw all HUD layers ─────────────────────────────────────────────
    draw_top_bar(frame, 900, session_id, fps, greeting, frame_count)
    draw_alert_banner(frame, 900, status, frame_count)
    draw_bottom_hud(frame, 660, 900, ear, mar,
                    ear_counter, mar_counter,
                    total_alerts, status, frame_count,
                    ear_history, mar_history,
                    drowsy_score, mar_score)
    draw_corner_reticles(frame, 900, 660)
    draw_scanlines(frame, 660, 900)

    cv2.imshow(WINDOW_NAME, frame)

    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break
    if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
        break

cap.release()
cv2.destroyAllWindows()
end_session(session_id)
print(f"Session {session_id} ended. Total alerts: {total_alerts}")