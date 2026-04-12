import sqlite3

conn = sqlite3.connect('drowsiness.db')
c = conn.cursor()
c.execute('SELECT * FROM sessions')
sessions = c.fetchall()
c.execute('SELECT * FROM alerts')
alerts = c.fetchall()
conn.close()

with open('oracle_import.sql', 'w') as f:
    f.write('-- SESSIONS\n')
    for s in sessions:
        end = s[2] if s[2] else s[1]
        f.write(f"INSERT INTO drowsiness_sessions (id, session_start, session_end, total_alerts) VALUES ({s[0]}, TO_TIMESTAMP('{s[1][:19]}', 'YYYY-MM-DD HH24:MI:SS'), TO_TIMESTAMP('{end[:19]}', 'YYYY-MM-DD HH24:MI:SS'), {s[3]});\n")

    f.write('\n-- ALERTS\n')
    for a in alerts:
        f.write(f"INSERT INTO drowsiness_alerts (id, session_id, alert_time, ear_value, duration_frames) VALUES ({a[0]}, {a[1]}, TO_TIMESTAMP('{a[2][:19]}', 'YYYY-MM-DD HH24:MI:SS'), {a[3]}, {a[6]});\n")

print("oracle_import.sql created successfully!")