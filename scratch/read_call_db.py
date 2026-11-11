import sqlite3

c = sqlite3.connect(r"C:\Users\Admin\.gemini\antigravity\brain\5b24f344-f49d-43f5-989c-2e5ba434c29c\scratch\receptionist_copy.db")
cur = c.cursor()
cid = "4f1029c8-24dc-4a16-bf50-2bf7c5e89ac9"

cur.execute("SELECT id, call_sid, caller_phone, caller_name, call_type, status, duration, recording_url, created_at FROM call_logs WHERE call_sid = ?", (cid,))
row = cur.fetchone()
print("CallLog row:", row)

cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [t[0] for t in cur.fetchall()]
print("Tables in DB:", tables)

for tbl in tables:
    cur.execute(f"PRAGMA table_info({tbl})")
    cols = [col[1] for col in cur.fetchall()]
    if any(c in cols for c in ["call_id", "call_sid", "session_id"]):
        print(f"Table {tbl} has call link: {cols}")
        for col in ["call_id", "call_sid"]:
            if col in cols:
                try:
                    cur.execute(f"SELECT * FROM {tbl} WHERE {col} = ?", (cid,))
                    rows = cur.fetchall()
                    if rows:
                        print(f"  Found {len(rows)} rows in {tbl}:")
                        for r in rows[:10]:
                            print("   ", r)
                except Exception as e:
                    pass

c.close()
