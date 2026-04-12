-- SESSIONS
INSERT INTO drowsiness_sessions (id, session_start, session_end, total_alerts) VALUES (1, TO_TIMESTAMP('2026-04-12T17:19:59', 'YYYY-MM-DD HH24:MI:SS'), TO_TIMESTAMP('2026-04-12T17:20:32', 'YYYY-MM-DD HH24:MI:SS'), 9);
INSERT INTO drowsiness_sessions (id, session_start, session_end, total_alerts) VALUES (2, TO_TIMESTAMP('2026-04-12T17:22:01', 'YYYY-MM-DD HH24:MI:SS'), TO_TIMESTAMP('2026-04-12T17:22:11', 'YYYY-MM-DD HH24:MI:SS'), 1);
INSERT INTO drowsiness_sessions (id, session_start, session_end, total_alerts) VALUES (3, TO_TIMESTAMP('2026-04-12T17:26:30', 'YYYY-MM-DD HH24:MI:SS'), TO_TIMESTAMP('2026-04-12T17:27:53', 'YYYY-MM-DD HH24:MI:SS'), 11);
INSERT INTO drowsiness_sessions (id, session_start, session_end, total_alerts) VALUES (4, TO_TIMESTAMP('2026-04-12T17:28:51', 'YYYY-MM-DD HH24:MI:SS'), TO_TIMESTAMP('2026-04-12T17:28:56', 'YYYY-MM-DD HH24:MI:SS'), 0);

-- ALERTS
INSERT INTO drowsiness_alerts (id, session_id, alert_time, ear_value, duration_frames) VALUES (1, 1, TO_TIMESTAMP('2026-04-12T17:20:04', 'YYYY-MM-DD HH24:MI:SS'), EAR, 20);
INSERT INTO drowsiness_alerts (id, session_id, alert_time, ear_value, duration_frames) VALUES (2, 1, TO_TIMESTAMP('2026-04-12T17:20:06', 'YYYY-MM-DD HH24:MI:SS'), EAR, 20);
INSERT INTO drowsiness_alerts (id, session_id, alert_time, ear_value, duration_frames) VALUES (3, 1, TO_TIMESTAMP('2026-04-12T17:20:07', 'YYYY-MM-DD HH24:MI:SS'), EAR, 20);
INSERT INTO drowsiness_alerts (id, session_id, alert_time, ear_value, duration_frames) VALUES (4, 1, TO_TIMESTAMP('2026-04-12T17:20:10', 'YYYY-MM-DD HH24:MI:SS'), EAR, 20);
INSERT INTO drowsiness_alerts (id, session_id, alert_time, ear_value, duration_frames) VALUES (5, 1, TO_TIMESTAMP('2026-04-12T17:20:11', 'YYYY-MM-DD HH24:MI:SS'), MAR, 15);
INSERT INTO drowsiness_alerts (id, session_id, alert_time, ear_value, duration_frames) VALUES (6, 1, TO_TIMESTAMP('2026-04-12T17:20:12', 'YYYY-MM-DD HH24:MI:SS'), EAR, 20);
INSERT INTO drowsiness_alerts (id, session_id, alert_time, ear_value, duration_frames) VALUES (7, 1, TO_TIMESTAMP('2026-04-12T17:20:16', 'YYYY-MM-DD HH24:MI:SS'), MAR, 15);
INSERT INTO drowsiness_alerts (id, session_id, alert_time, ear_value, duration_frames) VALUES (8, 1, TO_TIMESTAMP('2026-04-12T17:20:17', 'YYYY-MM-DD HH24:MI:SS'), EAR, 20);
INSERT INTO drowsiness_alerts (id, session_id, alert_time, ear_value, duration_frames) VALUES (9, 1, TO_TIMESTAMP('2026-04-12T17:20:22', 'YYYY-MM-DD HH24:MI:SS'), EAR, 20);
INSERT INTO drowsiness_alerts (id, session_id, alert_time, ear_value, duration_frames) VALUES (10, 2, TO_TIMESTAMP('2026-04-12T17:22:11', 'YYYY-MM-DD HH24:MI:SS'), EAR, 20);
INSERT INTO drowsiness_alerts (id, session_id, alert_time, ear_value, duration_frames) VALUES (11, 3, TO_TIMESTAMP('2026-04-12T17:26:39', 'YYYY-MM-DD HH24:MI:SS'), MAR, 15);
INSERT INTO drowsiness_alerts (id, session_id, alert_time, ear_value, duration_frames) VALUES (12, 3, TO_TIMESTAMP('2026-04-12T17:26:40', 'YYYY-MM-DD HH24:MI:SS'), EAR, 20);
INSERT INTO drowsiness_alerts (id, session_id, alert_time, ear_value, duration_frames) VALUES (13, 3, TO_TIMESTAMP('2026-04-12T17:26:43', 'YYYY-MM-DD HH24:MI:SS'), EAR, 20);
INSERT INTO drowsiness_alerts (id, session_id, alert_time, ear_value, duration_frames) VALUES (14, 3, TO_TIMESTAMP('2026-04-12T17:26:44', 'YYYY-MM-DD HH24:MI:SS'), EAR, 20);
INSERT INTO drowsiness_alerts (id, session_id, alert_time, ear_value, duration_frames) VALUES (15, 3, TO_TIMESTAMP('2026-04-12T17:26:46', 'YYYY-MM-DD HH24:MI:SS'), EAR, 20);
INSERT INTO drowsiness_alerts (id, session_id, alert_time, ear_value, duration_frames) VALUES (16, 3, TO_TIMESTAMP('2026-04-12T17:26:48', 'YYYY-MM-DD HH24:MI:SS'), EAR, 20);
INSERT INTO drowsiness_alerts (id, session_id, alert_time, ear_value, duration_frames) VALUES (17, 3, TO_TIMESTAMP('2026-04-12T17:26:50', 'YYYY-MM-DD HH24:MI:SS'), EAR, 20);
INSERT INTO drowsiness_alerts (id, session_id, alert_time, ear_value, duration_frames) VALUES (18, 3, TO_TIMESTAMP('2026-04-12T17:26:57', 'YYYY-MM-DD HH24:MI:SS'), EAR, 20);
INSERT INTO drowsiness_alerts (id, session_id, alert_time, ear_value, duration_frames) VALUES (19, 3, TO_TIMESTAMP('2026-04-12T17:27:00', 'YYYY-MM-DD HH24:MI:SS'), EAR, 20);
INSERT INTO drowsiness_alerts (id, session_id, alert_time, ear_value, duration_frames) VALUES (20, 3, TO_TIMESTAMP('2026-04-12T17:27:09', 'YYYY-MM-DD HH24:MI:SS'), EAR, 20);
INSERT INTO drowsiness_alerts (id, session_id, alert_time, ear_value, duration_frames) VALUES (21, 3, TO_TIMESTAMP('2026-04-12T17:27:47', 'YYYY-MM-DD HH24:MI:SS'), EAR, 20);
