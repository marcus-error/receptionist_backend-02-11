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

print("Step 1: Database configured")

from fastapi import FastAPI
from fastapi.testclient import TestClient
from src.sockets import voice_socket
from src.routes import knowledge_routes
from src.config.jwt import create_access_token

print("Step 2: Imports complete")

test_app = FastAPI()
test_app.include_router(voice_socket.router)
test_app.include_router(knowledge_routes.router, prefix="/api/v1")

admin_token = create_access_token({"sub": "2", "organization_id": 1, "role": "admin"})
client = TestClient(test_app)

print("Step 3: TestClient initialized")

resp = client.get("/api/v1/knowledge/active", headers={"Authorization": f"Bearer {admin_token}"})
print("Step 4: GET /active returned:", resp.status_code, resp.json())
