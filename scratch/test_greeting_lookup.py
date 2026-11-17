import os
import sys
import shutil
import tempfile

sys.path.insert(0, r"c:\projects\2-11 ai receptionist updated\receptionist_backend-02-11")

td = tempfile.mkdtemp()
src_db = r"c:\projects\2-11 ai receptionist updated\receptionist_backend-02-11\receptionist.db"
target_db = os.path.join(td, "test.db")
shutil.copy(src_db, target_db)
for ext in ["-wal", "-shm"]:
    if os.path.exists(src_db + ext):
        shutil.copy(src_db + ext, target_db + ext)

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import src.config.database as db_mod

eng = create_engine(f"sqlite:///{target_db.replace('\\', '/')}")
db_mod.SessionLocal = sessionmaker(bind=eng)
db_mod.engine = eng

from src.services.rag_service import rag_service

greeting = rag_service.get_organization_greeting(1, document_id="6")
print(f"Aminoid AI (Doc 6) Greeting: {greeting}")

greeting_hosp = rag_service.get_organization_greeting(1, document_id="1")
print(f"Sarans Hospital (Doc 1) Greeting: {greeting_hosp}")

resolved_doc = rag_service._resolve_active_document_id(1)
print(f"Active doc resolved for Org 1: {resolved_doc}")

shutil.rmtree(td, ignore_errors=True)

