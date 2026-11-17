import sys, os
sys.path.insert(0, r"C:\projects\2-11 ai receptionist updated\receptionist_backend-02-11")

from src.config.database import SessionLocal
from src.models.organization import Organization
from src.models.knowledge_doc import KnowledgeDocument
from src.models.department import Department
from src.models.staff import Staff

db = SessionLocal()
try:
    print("=== ORGANIZATIONS ===")
    orgs = db.query(Organization).all()
    for o in orgs:
        print(f"ID={o.id}, Name='{o.name}', Slug='{o.slug}'")

    print("\n=== KNOWLEDGE DOCUMENTS (SCRIPTS) ===")
    docs = db.query(KnowledgeDocument).all()
    for d in docs:
        print(f"ID={d.id}, OrgID={d.organization_id}, Title='{d.title}', Scope='{d.scope}', Type='{d.document_type}', ai_usage='{d.ai_usage}', Status='{d.status}'")

    print("\n=== DEPARTMENTS ===")
    depts = db.query(Department).all()
    for dept in depts:
        print(f"ID={dept.id}, OrgID={dept.organization_id}, Name='{dept.name}'")

    print("\n=== STAFF ===")
    staff = db.query(Staff).all()
    for s in staff:
        print(f"ID={s.id}, OrgID={s.organization_id}, Name='{s.name}', Dept='{s.department}', Services='{s.services}'")

finally:
    db.close()

