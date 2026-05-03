from flask import Flask, render_template, request, redirect, url_for
import sqlite3
from datetime import datetime

app = Flask(__name__)
DB_PATH = "drowsiness.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_archive_table():
    """Create archive tables if they don't exist yet."""
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS sessions_archive (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            original_id   INTEGER,
            session_start TEXT,
            session_end   TEXT,
            total_alerts  INTEGER DEFAULT 0,
            avg_ear       REAL,
            archived_at   TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS alerts_archive (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            original_id     INTEGER,
            session_id      INTEGER,
            alert_time      TEXT,
            alert_type      TEXT,
            ear_value       REAL,
            mar_value       REAL,
            duration_frames INTEGER,
            archived_at     TEXT
        )
    """)
    conn.commit()
    conn.close()

init_archive_table()

@app.route("/archive", methods=["POST"])
def archive():
    """Copy all live data to archive tables then wipe live tables."""
    conn = get_db()
    c = conn.cursor()
    now = datetime.now().isoformat()

    c.execute("""
        INSERT INTO sessions_archive
            (original_id, session_start, session_end, total_alerts, avg_ear, archived_at)
        SELECT id, session_start, session_end, total_alerts, avg_ear, ?
        FROM sessions
    """, (now,))

    c.execute("""
        INSERT INTO alerts_archive
            (original_id, session_id, alert_time, alert_type,
             ear_value, mar_value, duration_frames, archived_at)
        SELECT id, session_id, alert_time, alert_type,
               ear_value, mar_value, duration_frames, ?
        FROM alerts
    """, (now,))

    c.execute("DELETE FROM alerts")
    c.execute("DELETE FROM sessions")
    # Reset auto-increment counters so IDs start from 1 again
    c.execute("DELETE FROM sqlite_sequence WHERE name='sessions'")
    c.execute("DELETE FROM sqlite_sequence WHERE name='alerts'")

    conn.commit()
    conn.close()
    return redirect(url_for("index"))

@app.route("/")
def index():
    conn = get_db()
    c = conn.cursor()

    # ── Date filter ───────────────────────────────────────────────────────
    selected_date = request.args.get("date", "")
    if selected_date:
        date_filter       = f"DATE(session_start) = '{selected_date}'"
        alert_date_filter = f"DATE(alert_time) = '{selected_date}'"
    else:
        date_filter       = "1=1"
        alert_date_filter = "1=1"

    # ── Summary stats ─────────────────────────────────────────────────────
    c.execute(f"SELECT COUNT(*) as total FROM sessions WHERE {date_filter}")
    total_sessions = c.fetchone()["total"]

    c.execute(f"SELECT COALESCE(SUM(total_alerts), 0) as total FROM sessions WHERE {date_filter}")
    total_alerts = c.fetchone()["total"]

    c.execute(f"SELECT COUNT(*) as total FROM alerts WHERE alert_type='EAR' AND {alert_date_filter}")
    ear_alerts = c.fetchone()["total"]

    c.execute(f"SELECT COUNT(*) as total FROM alerts WHERE alert_type='MAR' AND {alert_date_filter}")
    mar_alerts = c.fetchone()["total"]

    # ── Sessions table ────────────────────────────────────────────────────
    c.execute(f"""
        SELECT id, session_start, session_end, total_alerts, avg_ear,
        ROUND(
            (JULIANDAY(COALESCE(session_end, datetime('now'))) - JULIANDAY(session_start)) * 86400
        ) as duration
        FROM sessions
        WHERE {date_filter}
        ORDER BY id DESC
    """)
    sessions = c.fetchall()

    # ── Alerts table ──────────────────────────────────────────────────────
    c.execute(f"""
        SELECT a.id, a.session_id, a.alert_time, a.alert_type,
               a.ear_value, a.mar_value, a.duration_frames
        FROM alerts a
        WHERE {alert_date_filter}
        ORDER BY a.id DESC
        LIMIT 50
    """)
    alerts = c.fetchall()

    # ── Chart data ────────────────────────────────────────────────────────
    c.execute(f"""
        SELECT id, total_alerts FROM sessions
        WHERE {date_filter}
        ORDER BY id ASC
    """)
    chart_data    = c.fetchall()
    chart_labels  = [f"S{r['id']}" for r in chart_data]
    chart_values  = [r['total_alerts'] for r in chart_data]

    # ── Dates available in DB for the picker ─────────────────────────────
    c.execute("""
        SELECT DISTINCT DATE(session_start) as day
        FROM sessions ORDER BY day DESC
    """)
    available_dates = [r["day"] for r in c.fetchall()]

    conn.close()

    # ── Risk level ────────────────────────────────────────────────────────
    if total_alerts == 0:
        risk_level    = "LOW"
        risk_color    = "#4ade80"
        driver_status = "SAFE"
    elif total_alerts < 5:
        risk_level    = "MEDIUM"
        risk_color    = "#fb923c"
        driver_status = "AT RISK"
    else:
        risk_level    = "HIGH"
        risk_color    = "#f87171"
        driver_status = "DROWSY"

    return render_template("index.html",
        total_sessions  = total_sessions,
        total_alerts    = total_alerts,
        ear_alerts      = ear_alerts,
        mar_alerts      = mar_alerts,
        sessions        = sessions,
        alerts          = alerts,
        risk_level      = risk_level,
        risk_color      = risk_color,
        driver_status   = driver_status,
        chart_labels    = chart_labels,
        chart_values    = chart_values,
        selected_date   = selected_date,
        available_dates = available_dates)

if __name__ == "__main__":
    app.run(debug=True)