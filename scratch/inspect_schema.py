import sqlite3

conn = sqlite3.connect('receptionist.db')
cur = conn.cursor()

tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()]

print('--- ALL TABLES AND ORG COLUMNS ---')
for table in sorted(tables):
    cur.execute(f"PRAGMA table_info('{table}')")
    cols = cur.fetchall()
    col_names = [c[1] for c in cols]
    org_cols = [c for c in col_names if 'org' in c.lower()]
    cur.execute(f"PRAGMA foreign_key_list('{table}')")
    fks = cur.fetchall()
    fk_desc = [f"{fk[3]} -> {fk[2]}.{fk[4]}" for fk in fks]
    cur.execute(f"SELECT count(*) FROM '{table}'")
    cnt = cur.fetchone()[0]
    print(f"{table} (rows={cnt}): org_cols={org_cols}, fks={fk_desc}")

conn.close()

