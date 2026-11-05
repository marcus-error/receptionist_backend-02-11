import sqlite3

conn = sqlite3.connect('receptionist.db')
cur = conn.cursor()

for table in ['conversation_messages', 'reminders', 'gemini_accounts', 'user_permissions', 'calendar_configs']:
    cur.execute(f"PRAGMA table_info('{table}')")
    cols = cur.fetchall()
    print(f"\nTABLE {table}:")
    for c in cols:
        print(f"  {c[1]} ({c[2]})")
    cur.execute(f"PRAGMA foreign_key_list('{table}')")
    fks = cur.fetchall()
    print(f"  FKS: {fks}")

conn.close()

