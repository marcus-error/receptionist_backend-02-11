import sqlite3
for db_path in ['receptionist.db', 'receptionist_backend-02-11/receptionist.db']:
    try:
        conn = sqlite3.connect(db_path)
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        print(f"=== {db_path} ===")
        print("Tables:", tables)
        for t in ["departments", "staff"]:
            if t in tables:
                cols = conn.execute(f"PRAGMA table_info({t})").fetchall()
                print(f"Columns for {t}:", [(c[1], c[2]) for c in cols])
        conn.close()
    except Exception as e:
        print(db_path, e)

