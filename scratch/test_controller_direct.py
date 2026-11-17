import os, sys
os.environ["APP_ENV"] = "test"
os.environ["PYTEST_CURRENT_TEST"] = "1"
BASE_DIR = r"c:\projects\2-11 ai receptionist updated\receptionist_backend-02-11"
sys.path.insert(0, BASE_DIR)
SCRATCH_DB = r"C:/Users/Admin/.gemini/antigravity/brain/5b24f344-f49d-43f5-989c-2e5ba434c29c/scratch/voice_rag_test.db"
os.environ["DATABASE_URL"] = f"sqlite:///{SCRATCH_DB}"

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import src.config.database as db_mod

test_engine = create_engine(f"sqlite:///{SCRATCH_DB}", connect_args={"check_same_thread": False})
db_mod.engine = test_engine
db_mod.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

from src.models.user import User
from src.controllers.knowledge_controller import get_active_doc

db = db_mod.SessionLocal()
user = db.query(User).filter(User.id == 2).first()
print("Found user:", user.id, user.email, user.organization_id)

res = get_active_doc(current_user=user, db=db)
print("Result of get_active_doc:", res)

# Now test voice_socket functions
from src.sockets.voice_socket import _resolve_and_validate_active_doc, _authenticate_ws_token
from src.config.jwt import create_access_token

token = create_access_token({"sub": "2", "organization_id": 1, "role": "admin"})
resolved_org = _authenticate_ws_token(token)
print("Resolved org from token:", resolved_org)

doc, meta = _resolve_and_validate_active_doc(resolved_org)
print("Resolved active doc:", doc.id if doc else None, doc.title if doc else None)
print("Script meta:", meta.get("business_type") if meta else None, meta.get("booking_type") if meta else None)

db.close()

