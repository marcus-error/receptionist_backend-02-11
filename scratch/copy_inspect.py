import sqlite3, shutil, os

db_src = r"C:\projects\2-11 ai receptionist updated\receptionist_backend-02-11\receptionist.db"
db_copy = r"C:\projects\2-11 ai receptionist updated\receptionist_backend-02-11\scratch\temp_inspect.db"

# Copy db and wal/shm if present
shutil.copy2(db_src, db_copy)
if os.path.exists(db_src + "-wal"):
    shutil.copy2(db_src + "-wal", db_copy + "-wal")
if os.path.exists(db_src + "-shm"):
    shutil.copy2(db_src + "-shm", db_copy + "-shm")

conn = sqlite3.connect(db_copy)
cur = conn.cursor()

print("=== ORGANIZATIONS ===")
for row in cur.execute("SELECT id, name, slug FROM organizations"):
    print(row)

print("\n=== KNOWLEDGE DOCUMENTS (SCRIPTS) ===")
for row in cur.execute("SELECT id, organization_id, title, scope, document_type, ai_usage, status FROM knowledge_documents"):
    print(row)

print("\n=== DEPARTMENTS ===")
for row in cur.execute("SELECT id, organization_id, name FROM departments"):
    print(row)

print("\n=== STAFF ===")
for row in cur.execute("SELECT id, organization_id, name, department, services FROM staff"):
    print(row)

conn.close()

