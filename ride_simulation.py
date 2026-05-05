"""
RideGuard — Ride-Sharing Driver Safety Simulation
===================================================
Simulates how DrowsGuard would integrate into a ride-sharing platform.
Reads real drowsiness data from drowsiness.db (same DB as main.py).

Run on port 5001 — completely separate from dashboard.py (port 5000).

Usage:
    python ride_simulation.py
    Open: http://localhost:5001
"""

from flask import Flask, render_template_string, jsonify
import sqlite3
import random
from datetime import datetime, timedelta

app = Flask(__name__)
DB_PATH = "drowsiness.db"

# ── Simulated trip / driver data ──────────────────────────────────────────
DRIVER = {
    "name":    "Sadikshya K.",
    "id":      "DRV-7742",
    "rating":  4.87,
    "vehicle": "Toyota Corolla · BA 12 PA 3847",
    "photo":   "SK",          # initials for avatar
    "trips_today": 6,
    "hours_driving": 3.4,
}

PASSENGER = {
    "name":     "Rohan M.",
    "pickup":   "Thamel, Kathmandu",
    "dropoff":  "Patan Dhoka, Lalitpur",
    "eta_mins": 12,
    "fare":     "NPR 340",
}

INCIDENT_RESPONSES = [
    "⚠️  In-app warning sent to driver",
    "🔔  Passenger notified of safety check",
    "📍  Trip paused — driver asked to pull over",
    "🏢  Fleet safety team alerted",
    "📞  Emergency contact protocol initiated",
]

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def get_latest_session():
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute("SELECT * FROM sessions ORDER BY id DESC LIMIT 1")
        row = c.fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception:
        return None

def get_recent_alerts(session_id=None, limit=10):
    try:
        conn = get_db()
        c = conn.cursor()
        if session_id:
            c.execute("""SELECT * FROM alerts WHERE session_id=?
                         ORDER BY id DESC LIMIT ?""", (session_id, limit))
        else:
            c.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,))
        rows = [dict(r) for r in c.fetchall()]
        conn.close()
        return rows
    except Exception:
        return []

def get_safety_score(total_alerts, hours):
    """Calculate a driver safety score 0-100 based on alert frequency."""
    if hours == 0:
        return 100
    rate = total_alerts / hours
    if rate == 0:   return 100
    if rate < 1:    return 92
    if rate < 2:    return 80
    if rate < 4:    return 65
    return 45

def get_risk_level(total_alerts):
    if total_alerts == 0: return "SAFE",    "#00C896", "All systems normal"
    if total_alerts < 3:  return "CAUTION", "#F59E0B", "Mild fatigue detected"
    if total_alerts < 6:  return "WARNING", "#F97316", "Significant fatigue"
    return                        "DANGER",  "#EF4444", "Immediate action required"


# ── HTML Template ─────────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RideGuard — Driver Safety Portal</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:        #070B14;
  --surface:   #0D1220;
  --surface2:  #121929;
  --border:    #1E2D45;
  --border2:   #243350;
  --accent:    #3B82F6;
  --accent2:   #60A5FA;
  --green:     #00C896;
  --amber:     #F59E0B;
  --orange:    #F97316;
  --red:       #EF4444;
  --text:      #E8EDF5;
  --muted:     #64748B;
  --dim:       #334155;
  --font-head: 'Syne', sans-serif;
  --font-mono: 'DM Mono', monospace;
  --font-body: 'DM Sans', sans-serif;
}

body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--font-body);
  min-height: 100vh;
  overflow-x: hidden;
}

/* ── Background grid ── */
body::before {
  content: '';
  position: fixed; inset: 0;
  background-image:
    linear-gradient(rgba(59,130,246,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(59,130,246,0.03) 1px, transparent 1px);
  background-size: 40px 40px;
  pointer-events: none;
  z-index: 0;
}

/* ── Top navbar ── */
.navbar {
  position: sticky; top: 0; z-index: 100;
  background: rgba(7,11,20,0.92);
  backdrop-filter: blur(12px);
  border-bottom: 1px solid var(--border);
  padding: 0 32px;
  height: 60px;
  display: flex; align-items: center; justify-content: space-between;
}
.nav-brand {
  display: flex; align-items: center; gap: 10px;
}
.nav-logo {
  width: 32px; height: 32px; border-radius: 8px;
  background: var(--accent);
  display: flex; align-items: center; justify-content: center;
  font-family: var(--font-head);
  font-weight: 800; font-size: 14px; color: #fff;
}
.nav-title {
  font-family: var(--font-head);
  font-weight: 700; font-size: 18px; color: var(--text);
  letter-spacing: -0.3px;
}
.nav-sub { font-size: 11px; color: var(--muted); font-family: var(--font-mono); }
.nav-right { display: flex; align-items: center; gap: 16px; }
.live-badge {
  display: flex; align-items: center; gap: 6px;
  background: rgba(0,200,150,0.1);
  border: 1px solid rgba(0,200,150,0.3);
  border-radius: 20px; padding: 4px 12px;
  font-size: 11px; font-family: var(--font-mono);
  color: var(--green); font-weight: 500;
}
.live-dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--green);
  animation: pulse-dot 1.4s ease infinite;
}
@keyframes pulse-dot {
  0%,100% { opacity:1; transform:scale(1); }
  50%      { opacity:0.5; transform:scale(0.7); }
}
.nav-time {
  font-family: var(--font-mono); font-size: 13px; color: var(--muted);
}

/* ── Main layout ── */
.main { position: relative; z-index: 1; padding: 28px 32px; max-width: 1400px; margin: 0 auto; }

/* ── Alert banner ── */
.alert-banner {
  border-radius: 12px; padding: 16px 24px;
  display: flex; align-items: center; gap: 16px;
  margin-bottom: 24px;
  border: 1px solid;
  animation: fadeIn 0.4s ease;
}
.alert-banner.safe   { background: rgba(0,200,150,0.08); border-color: rgba(0,200,150,0.25); }
.alert-banner.caution{ background: rgba(245,158,11,0.08); border-color: rgba(245,158,11,0.25); }
.alert-banner.warning{ background: rgba(249,115,22,0.08); border-color: rgba(249,115,22,0.25); }
.alert-banner.danger { background: rgba(239,68,68,0.10); border-color: rgba(239,68,68,0.35);
                       animation: danger-pulse 2s ease infinite; }
@keyframes danger-pulse {
  0%,100% { box-shadow: 0 0 0 0 rgba(239,68,68,0); }
  50%      { box-shadow: 0 0 0 8px rgba(239,68,68,0.08); }
}
.banner-icon { font-size: 24px; }
.banner-text h3 { font-family: var(--font-head); font-weight: 700; font-size: 15px; }
.banner-text p  { font-size: 13px; color: var(--muted); margin-top: 2px; }
.banner-actions { margin-left: auto; display: flex; gap: 10px; }
.btn {
  padding: 8px 18px; border-radius: 8px; border: none; cursor: pointer;
  font-family: var(--font-body); font-size: 13px; font-weight: 500;
  transition: all 0.15s;
}
.btn-primary { background: var(--accent); color: #fff; }
.btn-primary:hover { background: var(--accent2); }
.btn-ghost { background: var(--surface2); color: var(--text); border: 1px solid var(--border); }
.btn-ghost:hover { border-color: var(--border2); }

/* ── Grid ── */
.grid-3 { display: grid; grid-template-columns: repeat(3,1fr); gap: 16px; margin-bottom: 20px; }
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; }
.grid-main { display: grid; grid-template-columns: 1fr 340px; gap: 16px; margin-bottom: 20px; }

/* ── Cards ── */
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 14px; padding: 20px;
  transition: border-color 0.2s;
}
.card:hover { border-color: var(--border2); }
.card-label {
  font-family: var(--font-mono); font-size: 10px;
  color: var(--muted); letter-spacing: 0.08em;
  text-transform: uppercase; margin-bottom: 10px;
}
.card-value {
  font-family: var(--font-head); font-weight: 700;
  font-size: 32px; line-height: 1;
}
.card-sub { font-size: 12px; color: var(--muted); margin-top: 6px; }

/* ── Safety score ring ── */
.score-ring-wrap { display:flex; align-items:center; gap:24px; }
.score-ring { position:relative; width:90px; height:90px; flex-shrink:0; }
.score-ring svg { transform: rotate(-90deg); }
.score-ring-val {
  position:absolute; inset:0;
  display:flex; flex-direction:column;
  align-items:center; justify-content:center;
  font-family: var(--font-head); font-weight:800;
}
.score-ring-val span:first-child { font-size:22px; }
.score-ring-val span:last-child  { font-size:9px; color:var(--muted); margin-top:1px; }
.score-details { flex:1; }
.score-bar-row { display:flex; align-items:center; gap:8px; margin-bottom:8px; }
.score-bar-label { font-size:11px; color:var(--muted); width:80px; font-family:var(--font-mono); }
.score-bar-track { flex:1; height:4px; background:var(--border); border-radius:2px; overflow:hidden; }
.score-bar-fill  { height:100%; border-radius:2px; transition: width 0.6s ease; }

/* ── Trip card ── */
.trip-header { display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:16px; }
.trip-status { font-family:var(--font-mono); font-size:10px; padding:3px 10px;
               border-radius:20px; border:1px solid; }
.trip-status.active { color:var(--green); border-color:rgba(0,200,150,0.3); background:rgba(0,200,150,0.08); }
.route-row { display:flex; align-items:center; gap:10px; padding:10px 0;
             border-bottom:1px solid var(--border); }
.route-row:last-child { border:none; }
.route-dot { width:8px; height:8px; border-radius:50%; flex-shrink:0; }
.route-dot.pickup  { background:var(--green); }
.route-dot.dropoff { background:var(--accent); }
.route-text h4 { font-size:13px; font-weight:500; }
.route-text p  { font-size:11px; color:var(--muted); margin-top:1px; }
.trip-meta { display:flex; justify-content:space-between; margin-top:14px; }
.trip-meta-item { text-align:center; }
.trip-meta-item span:first-child { display:block; font-family:var(--font-head); font-weight:700; font-size:18px; }
.trip-meta-item span:last-child  { font-size:10px; color:var(--muted); font-family:var(--font-mono); }

/* ── Driver card ── */
.driver-row { display:flex; align-items:center; gap:14px; margin-bottom:16px; }
.driver-avatar {
  width:48px; height:48px; border-radius:12px;
  background: linear-gradient(135deg, var(--accent), #7C3AED);
  display:flex; align-items:center; justify-content:center;
  font-family:var(--font-head); font-weight:700; font-size:16px; color:#fff;
  flex-shrink:0;
}
.driver-name { font-family:var(--font-head); font-weight:700; font-size:15px; }
.driver-id   { font-family:var(--font-mono); font-size:10px; color:var(--muted); margin-top:2px; }
.driver-stats { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
.dstat { background:var(--surface2); border-radius:8px; padding:10px 12px; }
.dstat-val { font-family:var(--font-head); font-weight:700; font-size:16px; }
.dstat-lbl { font-size:10px; color:var(--muted); font-family:var(--font-mono); margin-top:2px; }

/* ── Alert log ── */
.alert-log { display:flex; flex-direction:column; gap:8px; margin-top:8px; }
.log-item {
  display:flex; align-items:flex-start; gap:10px;
  background:var(--surface2); border-radius:10px;
  padding:10px 14px; border:1px solid var(--border);
  animation: slideIn 0.3s ease;
}
@keyframes slideIn { from { opacity:0; transform:translateY(-6px); } to { opacity:1; transform:none; } }
.log-dot { width:8px; height:8px; border-radius:50%; flex-shrink:0; margin-top:4px; }
.log-dot.ear { background:var(--red); }
.log-dot.mar { background:#A855F7; }
.log-text { flex:1; }
.log-text h4 { font-size:12px; font-weight:500; }
.log-text p  { font-size:11px; color:var(--muted); margin-top:2px; font-family:var(--font-mono); }
.log-time { font-family:var(--font-mono); font-size:10px; color:var(--dim); white-space:nowrap; }

/* ── Incident response ── */
.response-list { display:flex; flex-direction:column; gap:8px; margin-top:10px; }
.response-item {
  display:flex; align-items:center; gap:10px;
  padding:10px 14px; border-radius:10px;
  border:1px solid var(--border); font-size:12px;
  transition: all 0.3s;
}
.response-item.triggered {
  background:rgba(239,68,68,0.06);
  border-color:rgba(239,68,68,0.2);
}
.response-item.idle {
  opacity:0.4;
}
.response-check {
  width:18px; height:18px; border-radius:50%; flex-shrink:0;
  display:flex; align-items:center; justify-content:center;
  font-size:10px;
}
.response-check.done { background:var(--red); color:#fff; }
.response-check.wait { background:var(--border); color:var(--muted); }

/* ── Section title ── */
.section-title {
  font-family:var(--font-head); font-weight:700; font-size:13px;
  color:var(--muted); text-transform:uppercase; letter-spacing:0.06em;
  margin-bottom:14px;
}

/* ── No alerts empty ── */
.empty { text-align:center; padding:32px; color:var(--muted); font-size:13px; }
.empty-icon { font-size:32px; margin-bottom:8px; }

/* ── Footer ── */
.footer {
  border-top:1px solid var(--border); margin-top:32px; padding-top:16px;
  display:flex; justify-content:space-between; align-items:center;
  font-size:11px; color:var(--dim); font-family:var(--font-mono);
}

@keyframes fadeIn { from { opacity:0; transform:translateY(4px); } to { opacity:1; transform:none; } }

@media (max-width: 900px) {
  .grid-3, .grid-main, .grid-2 { grid-template-columns:1fr; }
  .main { padding:16px; }
}
</style>
</head>
<body>

<!-- Navbar -->
<nav class="navbar">
  <div class="nav-brand">
    <div class="nav-logo">RG</div>
    <div>
      <div class="nav-title">RideGuard</div>
      <div class="nav-sub">Driver Safety Portal — Simulation</div>
    </div>
  </div>
  <div class="nav-right">
    <div class="live-badge"><div class="live-dot"></div>LIVE MONITORING</div>
    <div class="nav-time" id="clock">--:--:--</div>
  </div>
</nav>

<div class="main">

  <!-- Alert Banner -->
  <div class="alert-banner {{ banner_class }}" id="alertBanner">
    <div class="banner-icon">{{ banner_icon }}</div>
    <div class="banner-text">
      <h3 style="color:{{ risk_color }}">{{ risk_level }} — {{ risk_msg }}</h3>
      <p>Driver {{ driver.name }} · Session #{{ session.id if session else '—' }} · {{ total_alerts }} alert(s) this session</p>
    </div>
    <div class="banner-actions">
      {% if risk_level == 'DANGER' %}
      <button class="btn btn-primary" onclick="triggerResponse()">Trigger Response</button>
      {% endif %}
      <button class="btn btn-ghost" onclick="location.reload()">Refresh</button>
    </div>
  </div>

  <!-- Top stat cards -->
  <div class="grid-3">

    <div class="card">
      <div class="card-label">Safety Score</div>
      <div class="score-ring-wrap">
        <div class="score-ring">
          <svg width="90" height="90" viewBox="0 0 90 90">
            <circle cx="45" cy="45" r="38" fill="none" stroke="#1E2D45" stroke-width="6"/>
            <circle cx="45" cy="45" r="38" fill="none"
              stroke="{{ score_color }}" stroke-width="6"
              stroke-linecap="round"
              stroke-dasharray="{{ score_dash }} 239"
              style="transition:stroke-dasharray 1s ease"/>
          </svg>
          <div class="score-ring-val">
            <span style="color:{{ score_color }}">{{ safety_score }}</span>
            <span>/100</span>
          </div>
        </div>
        <div class="score-details">
          <div class="score-bar-row">
            <span class="score-bar-label">ALERTNESS</span>
            <div class="score-bar-track">
              <div class="score-bar-fill" style="width:{{ safety_score }}%;background:{{ score_color }}"></div>
            </div>
          </div>
          <div class="score-bar-row">
            <span class="score-bar-label">TRIP HOURS</span>
            <div class="score-bar-track">
              <div class="score-bar-fill" style="width:{{ [driver.hours_driving/8*100,100]|min|int }}%;background:#3B82F6"></div>
            </div>
          </div>
          <div class="score-bar-row">
            <span class="score-bar-label">INCIDENTS</span>
            <div class="score-bar-track">
              <div class="score-bar-fill" style="width:{{ [total_alerts*10,100]|min }}%;background:#EF4444"></div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-label">Current Session Alerts</div>
      <div class="card-value" style="color:{{ 'var(--green)' if total_alerts==0 else ('var(--amber)' if total_alerts<4 else 'var(--red)') }}">
        {{ total_alerts }}
      </div>
      <div class="card-sub">
        {{ ear_alerts }} drowsiness &nbsp;·&nbsp; {{ mar_alerts }} yawn
      </div>
      <div style="margin-top:12px;display:flex;gap:8px;">
        <div style="flex:1;background:var(--surface2);border-radius:6px;padding:8px;text-align:center">
          <div style="font-family:var(--font-head);font-weight:700;color:var(--red)">{{ ear_alerts }}</div>
          <div style="font-size:10px;color:var(--muted);font-family:var(--font-mono)">EAR</div>
        </div>
        <div style="flex:1;background:var(--surface2);border-radius:6px;padding:8px;text-align:center">
          <div style="font-family:var(--font-head);font-weight:700;color:#A855F7">{{ mar_alerts }}</div>
          <div style="font-size:10px;color:var(--muted);font-family:var(--font-mono)">MAR</div>
        </div>
        <div style="flex:1;background:var(--surface2);border-radius:6px;padding:8px;text-align:center">
          <div style="font-family:var(--font-head);font-weight:700;color:var(--accent)">{{ driver.trips_today }}</div>
          <div style="font-size:10px;color:var(--muted);font-family:var(--font-mono)">TRIPS</div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-label">Session Duration</div>
      <div class="card-value" style="color:var(--accent)">{{ duration }}</div>
      <div class="card-sub">Started {{ session_start }}</div>
      <div style="margin-top:14px;font-size:12px;color:var(--muted);">
        <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
          <span>Hours on road today</span>
          <span style="color:var(--text)">{{ driver.hours_driving }}h</span>
        </div>
        <div style="display:flex;justify-content:space-between;">
          <span>Driver rating</span>
          <span style="color:var(--amber)">★ {{ driver.rating }}</span>
        </div>
      </div>
    </div>

  </div>

  <!-- Main section -->
  <div class="grid-main">

    <!-- Left: alerts + incident response -->
    <div>
      <!-- Recent alerts -->
      <div class="card" style="margin-bottom:16px">
        <div class="section-title">Recent Drowsiness Events</div>
        <div class="alert-log">
          {% if alerts %}
            {% for a in alerts %}
            <div class="log-item">
              <div class="log-dot {{ a.alert_type|lower }}"></div>
              <div class="log-text">
                <h4>
                  {% if a.alert_type == 'EAR' %}
                    👁  Drowsiness Detected — Eyes Closing
                  {% else %}
                    😮  Fatigue Detected — Yawning
                  {% endif %}
                </h4>
                <p>
                  EAR {{ "%.3f"|format(a.ear_value) if a.ear_value else '—' }}
                  &nbsp;·&nbsp;
                  MAR {{ "%.3f"|format(a.mar_value) if a.mar_value else '—' }}
                  &nbsp;·&nbsp;
                  {{ a.duration_frames }} frames
                </p>
              </div>
              <div class="log-time">{{ a.alert_time[11:19] if a.alert_time else '—' }}</div>
            </div>
            {% endfor %}
          {% else %}
            <div class="empty">
              <div class="empty-icon">✅</div>
              No drowsiness events this session
            </div>
          {% endif %}
        </div>
      </div>

      <!-- Incident response -->
      <div class="card">
        <div class="section-title">Automated Response Protocol</div>
        <p style="font-size:12px;color:var(--muted);margin-bottom:14px;">
          Actions triggered automatically when DANGER threshold is reached.
          {% if risk_level == 'DANGER' %}
          <strong style="color:var(--red)"> Protocol active.</strong>
          {% else %}
          Monitoring — not yet triggered.
          {% endif %}
        </p>
        <div class="response-list">
          {% set triggered = risk_level == 'DANGER' %}
          {% for i, resp in responses %}
          <div class="response-item {{ 'triggered' if triggered else 'idle' }}">
            <div class="response-check {{ 'done' if triggered else 'wait' }}">
              {{ '✓' if triggered else i+1 }}
            </div>
            <span>{{ resp }}</span>
          </div>
          {% endfor %}
        </div>
      </div>
    </div>

    <!-- Right: driver + trip info -->
    <div style="display:flex;flex-direction:column;gap:16px;">

      <!-- Driver -->
      <div class="card">
        <div class="section-title">Driver</div>
        <div class="driver-row">
          <div class="driver-avatar">{{ driver.photo }}</div>
          <div>
            <div class="driver-name">{{ driver.name }}</div>
            <div class="driver-id">{{ driver.id }}</div>
            <div style="font-size:11px;color:var(--muted);margin-top:4px;">{{ driver.vehicle }}</div>
          </div>
        </div>
        <div class="driver-stats">
          <div class="dstat">
            <div class="dstat-val" style="color:var(--amber)">★ {{ driver.rating }}</div>
            <div class="dstat-lbl">RATING</div>
          </div>
          <div class="dstat">
            <div class="dstat-val">{{ driver.trips_today }}</div>
            <div class="dstat-lbl">TRIPS TODAY</div>
          </div>
          <div class="dstat">
            <div class="dstat-val">{{ driver.hours_driving }}h</div>
            <div class="dstat-lbl">DRIVING</div>
          </div>
          <div class="dstat">
            <div class="dstat-val" style="color:{{ score_color }}">{{ safety_score }}</div>
            <div class="dstat-lbl">SAFETY SCORE</div>
          </div>
        </div>
      </div>

      <!-- Active Trip -->
      <div class="card">
        <div class="trip-header">
          <div class="section-title" style="margin:0">Active Trip</div>
          <div class="trip-status active">● ON TRIP</div>
        </div>
        <div class="route-row">
          <div class="route-dot pickup"></div>
          <div class="route-text">
            <h4>{{ passenger.pickup }}</h4>
            <p>Pickup · {{ passenger.name }}</p>
          </div>
        </div>
        <div class="route-row">
          <div class="route-dot dropoff"></div>
          <div class="route-text">
            <h4>{{ passenger.dropoff }}</h4>
            <p>Dropoff destination</p>
          </div>
        </div>
        <div class="trip-meta">
          <div class="trip-meta-item">
            <span>{{ passenger.eta_mins }}m</span>
            <span>ETA</span>
          </div>
          <div class="trip-meta-item">
            <span>{{ passenger.fare }}</span>
            <span>FARE</span>
          </div>
          <div class="trip-meta-item">
            <span style="color:{{ risk_color }}">{{ risk_level }}</span>
            <span>STATUS</span>
          </div>
        </div>
      </div>

      <!-- Deployment note -->
      <div class="card" style="border-color:rgba(59,130,246,0.2);background:rgba(59,130,246,0.04)">
        <div style="font-size:11px;color:var(--muted);font-family:var(--font-mono);line-height:1.7">
          <div style="color:var(--accent);font-weight:500;margin-bottom:6px;">📡 INTEGRATION NOTE</div>
          In production, DrowsGuard would be embedded as an SDK in the driver app.
          Drowsiness events would POST to the platform's REST API, triggering automated
          passenger notifications, fleet alerts, and trip suspension protocols —
          mirroring systems used by Seeing Machines and Mobileye in commercial fleets.
        </div>
      </div>

    </div>
  </div>

  <div class="footer">
    <span>RideGuard · DrowsGuard Integration Simulation · Port 5001</span>
    <span>Data source: drowsiness.db · Refreshes every 10s</span>
  </div>

</div>

<script>
// Clock
function updateClock(){
  const n = new Date();
  document.getElementById('clock').textContent =
    String(n.getHours()).padStart(2,'0')+':'+
    String(n.getMinutes()).padStart(2,'0')+':'+
    String(n.getSeconds()).padStart(2,'0');
}
setInterval(updateClock, 1000); updateClock();

// Auto-refresh every 10s to pick up new alerts from main.py
setTimeout(() => location.reload(), 10000);

function triggerResponse(){
  alert('🚨 Simulated Response Triggered\\n\\n✓ Driver warned via in-app alert\\n✓ Passenger notified\\n✓ Fleet safety team alerted\\n✓ Trip suspension recommended\\n\\n(In production this would call the platform REST API)');
}
</script>
</body>
</html>
"""

@app.route("/")
def index():
    session     = get_latest_session()
    total_alerts = 0
    ear_alerts   = 0
    mar_alerts   = 0
    duration     = "—"
    session_start = "—"

    if session:
        total_alerts  = session.get("total_alerts", 0) or 0
        alerts        = get_recent_alerts(session["id"], 8)
        ear_alerts    = sum(1 for a in alerts if a.get("alert_type") == "EAR")
        mar_alerts    = sum(1 for a in alerts if a.get("alert_type") == "MAR")

        # Duration
        try:
            start = datetime.fromisoformat(session["session_start"])
            end   = datetime.fromisoformat(session["session_end"]) if session.get("session_end") else datetime.now()
            secs  = int((end - start).total_seconds())
            duration = f"{secs//60}m {secs%60}s"
            session_start = start.strftime("%H:%M:%S")
        except Exception:
            pass
    else:
        alerts = []

    risk_level, risk_color, risk_msg = get_risk_level(total_alerts)
    safety_score = get_safety_score(total_alerts, DRIVER["hours_driving"])
    score_color  = "#00C896" if safety_score >= 85 else ("#F59E0B" if safety_score >= 65 else "#EF4444")
    score_dash   = int(safety_score / 100 * 238)

    banner_map = {"SAFE": "safe", "CAUTION": "caution", "WARNING": "warning", "DANGER": "danger"}
    banner_cls  = banner_map.get(risk_level, "safe")
    banner_icon = {"SAFE": "🟢", "CAUTION": "🟡", "WARNING": "🟠", "DANGER": "🔴"}.get(risk_level, "🟢")

    responses = list(enumerate(INCIDENT_RESPONSES))

    return render_template_string(HTML,
        session       = session,
        driver        = DRIVER,
        passenger     = PASSENGER,
        alerts        = alerts,
        total_alerts  = total_alerts,
        ear_alerts    = ear_alerts,
        mar_alerts    = mar_alerts,
        risk_level    = risk_level,
        risk_color    = risk_color,
        risk_msg      = risk_msg,
        safety_score  = safety_score,
        score_color   = score_color,
        score_dash    = score_dash,
        banner_class  = banner_cls,
        banner_icon   = banner_icon,
        duration      = duration,
        session_start = session_start,
        responses     = responses,
    )

if __name__ == "__main__":
    print("\n  RideGuard — Ride-Sharing Safety Simulation")
    print("  Open: http://localhost:5001")
    print("  Make sure main.py is running to see live data\n")
    app.run(debug=True, port=5001)