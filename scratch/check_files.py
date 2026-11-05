import sqlite3

conn = sqlite3.connect('receptionist.db')
cur = conn.cursor()
cur.execute('SELECT id, organization_id, title, file_path FROM knowledge_documents')
print('KNOWLEDGE DOCS:')
for r in cur.fetchall():
    print(r)

cur.execute("SELECT id, organization_id, recording_url FROM call_logs WHERE recording_url IS NOT NULL AND recording_url != '' LIMIT 5")
print('\nCALL RECORDINGS:')
for r in cur.fetchall():
    print(r)
conn.close()

