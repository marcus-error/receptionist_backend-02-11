import sqlite3

create_table_sql = """
CREATE TABLE IF NOT EXISTS department_staff (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    department_id INTEGER NOT NULL REFERENCES departments(id) ON DELETE CASCADE,
    staff_id INTEGER NOT NULL REFERENCES staff(id) ON DELETE CASCADE,
    organization_id INTEGER NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_department_staff UNIQUE (department_id, staff_id)
);
"""

create_indices_sql = [
    "CREATE INDEX IF NOT EXISTS ix_department_staff_id ON department_staff (id);",
    "CREATE INDEX IF NOT EXISTS ix_department_staff_dept_id ON department_staff (department_id);",
    "CREATE INDEX IF NOT EXISTS ix_department_staff_staff_id ON department_staff (staff_id);",
    "CREATE INDEX IF NOT EXISTS ix_department_staff_org_id ON department_staff (organization_id);"
]

for db_path in ["receptionist.db", "receptionist_backend-02-11/receptionist.db"]:
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(create_table_sql)
        for idx_sql in create_indices_sql:
            cur.execute(idx_sql)
        
        # Backfill existing staff members based on staff.department matching department.name
        # To preserve any existing legacy relationships
        cur.execute("""
            INSERT OR IGNORE INTO department_staff (department_id, staff_id, organization_id, created_at)
            SELECT d.id, s.id, s.organization_id, CURRENT_TIMESTAMP
            FROM staff s
            JOIN departments d ON LOWER(TRIM(s.department)) = LOWER(TRIM(d.name)) 
                AND s.organization_id = d.organization_id
        """)
        
        conn.commit()
        rows = cur.execute("SELECT count(*) FROM department_staff").fetchone()[0]
        print(f"[{db_path}] department_staff table created/verified successfully. Existing mapped rows: {rows}")
        conn.close()
    except Exception as e:
        print(f"[{db_path}] Error: {e}")

