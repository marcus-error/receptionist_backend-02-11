import os, sys, json, asyncio
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
db_mod.SessionLocal.configure(bind=test_engine)

from src.sockets.voice_socket import voice_endpoint
from src.config.jwt import create_access_token

class FakeWebSocket:
    def __init__(self, query_params=None, incoming_messages=None):
        self.query_params = query_params or {}
        self.incoming = list(incoming_messages or [])
        self.sent_messages = []
        self.closed = False

    async def accept(self):
        pass

    async def send_json(self, data):
        self.sent_messages.append(data)

    async def receive(self):
        if not self.incoming:
            raise asyncio.CancelledError("Test complete")
        msg = self.incoming.pop(0)
        return {"text": json.dumps(msg)} if isinstance(msg, dict) else msg

async def run_test():
    token = create_access_token({"sub": "2", "organization_id": 1, "role": "admin"})
    
    incoming = [
        {"type": "start_call", "token": token},
        {"type": "stop_call"}
    ]
    ws = FakeWebSocket(query_params={"token": token}, incoming_messages=incoming)
    try:
        await voice_endpoint(ws)
    except asyncio.CancelledError:
        pass

    print("--- Sent Messages ---")
    for m in ws.sent_messages:
        t = m.get("type")
        if t == "audio":
            print(f"Type: {t}, Audio bytes len: {len(m.get('audio', ''))}")
        else:
            print(f"Type: {t}, Data: {m}")

asyncio.run(run_test())
