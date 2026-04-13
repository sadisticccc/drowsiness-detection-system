from flask import Flask, render_template
import sqlite3

app = Flask(__name__)
DB_PATH = "drowsiness.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.route("/")
def index():
    conn = get_db()
    c = conn.cursor()

    # Summary stats
    c.execute("SELECT COUNT(*) as total FROM sessions")
    total_sessions = c.fetchone()["total"]

    c.execute("SELECT COALESCE(SUM(total_alerts), 0) as total FROM sessions")
    total_alerts = c.fetchone()["total"]

    c.execute("SELECT COUNT(*) as total FROM alerts WHERE alert_type='EAR'")
    ear_alerts = c.fetchone()["total"]

    c.execute("SELECT COUNT(*) as total FROM alerts WHERE alert_type='MAR'")
    mar_alerts = c.fetchone()["total"]

    # Sessions
    c.execute("""
        SELECT id, session_start, session_end, total_alerts,
        ROUND((JULIANDAY(session_end) - JULIANDAY(session_start)) * 86400) as duration
        FROM sessions ORDER BY id DESC
    """)
    sessions = c.fetchall()

    # Recent alerts
    c.execute("""
        SELECT a.id, a.alert_time, a.alert_type, a.ear_value,
               a.mar_value, a.duration_frames, a.session_id
        FROM alerts a ORDER BY a.id DESC LIMIT 50
    """)
    alerts = c.fetchall()

    conn.close()
    return render_template("index.html",
        total_sessions=total_sessions,
        total_alerts=total_alerts,
        ear_alerts=ear_alerts,
        mar_alerts=mar_alerts,
        sessions=sessions,
        alerts=alerts)

if __name__ == "__main__":
    app.run(debug=True)