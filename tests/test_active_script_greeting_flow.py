import asyncio
import logging
import pytest
import os
import sys

from src.config.database import SessionLocal, init_db
from src.models.knowledge_doc import KnowledgeDocument, KnowledgeDocStatus
from src.models.organization import Organization
from src.services.rag_service import rag_service
from src.services.sip_call_handler import SipCallSession
from src.services.knowledge_service import KnowledgeService

logger = logging.getLogger("test_active_script_greeting_flow")
logging.basicConfig(level=logging.INFO)


@pytest.fixture(autouse=True)
def setup_test_environment():
    init_db()
    db = SessionLocal()
    # Ensure test organizations exist
    org1 = db.query(Organization).filter(Organization.id == 1).first()
    if not org1:
        db.add(Organization(id=1, name="Sarans Healthcare"))
    org2 = db.query(Organization).filter(Organization.id == 2).first()
    if not org2:
        db.add(Organization(id=2, name="Chennai Speciality Clinic"))
    db.commit()
    db.close()


@pytest.mark.asyncio
async def test_three_script_activation_and_greeting_flow():
    """
    Test script activation and dynamic time-aware greeting structure:
    "Good [morning/afternoon/evening]. Welcome to [Organization Name]. How may I assist you?"
    1. Activate Script 1 (Sarans Hospitals) -> verify dynamic greeting spoken.
    2. Activate Script 2 (Apollo Heart Center) -> verify dynamic greeting spoken.
    3. Activate Script 3 (Dental Clinic) -> verify dynamic greeting spoken.
    """
    db = SessionLocal()
    spoken_greetings = []

    def mock_audio_send(data):
        pass

    try:
        # ── SCRIPT 1: Sarans Hospitals ────────────────────────────
        s1 = db.query(KnowledgeDocument).filter(KnowledgeDocument.title == "Test_Sarans_Script_1").first()
        if not s1:
            s1 = KnowledgeDocument(
                title="Test_Sarans_Script_1",
                organization_id=1,
                document_type="SCRIPT",
                scope="inbound",
                ai_usage="Active",
                status=KnowledgeDocStatus.INDEXED,
                content="""# SARANS HOSPITALS INBOUND SCRIPT
# 1. Greeting & Call Opening
"Good morning! Welcome to Sarans Hospitals. How may I help you today?"

# 2. Departments
- Cardiology
- Dental Care"""
            )
            db.add(s1)
            db.commit()
        else:
            s1.ai_usage = "Active"
            s1.content = """# SARANS HOSPITALS INBOUND SCRIPT
# 1. Greeting & Call Opening
"Good morning! Welcome to Sarans Hospitals. How may I help you today?"

# 2. Departments
- Cardiology
- Dental Care"""
            db.commit()

        # Deactivate any other script for org 1
        db.query(KnowledgeDocument).filter(
            KnowledgeDocument.organization_id == 1,
            KnowledgeDocument.id != s1.id
        ).update({"ai_usage": "Inactive"})
        db.commit()
        rag_service.invalidate_caches(1)

        # Place Call 1 for Org 1
        call1 = SipCallSession(
            caller_phone="+919876543210",
            organization_id=1,
            audio_send_callback=mock_audio_send
        )
        call1._speak = lambda text, intent="", metrics=None: spoken_greetings.append((1, text)) or asyncio.sleep(0)
        await call1._send_opening_greeting()

        assert len(spoken_greetings) == 1
        assert spoken_greetings[-1][0] == 1
        assert "Sarans Hospitals" in spoken_greetings[-1][1] and ("Thank you for calling" in spoken_greetings[-1][1] or "Welcome to" in spoken_greetings[-1][1])
        assert "How may I assist you" in spoken_greetings[-1][1] or "How may I help you" in spoken_greetings[-1][1]
        assert call1.active_script_id == str(s1.id)

        # ── SCRIPT 2: Apollo Heart Center (Switch Script for Org 1) ─
        s2 = db.query(KnowledgeDocument).filter(KnowledgeDocument.title == "Test_Apollo_Script_2").first()
        if not s2:
            s2 = KnowledgeDocument(
                title="Test_Apollo_Script_2",
                organization_id=1,
                document_type="SCRIPT",
                scope="inbound",
                ai_usage="Active",
                status=KnowledgeDocStatus.INDEXED,
                content="""Voice Greeting: "Good afternoon, welcome to Apollo Heart Center. How may I direct your call?"

# Doctor Directory
- Dr. Ramesh (Cardiology)"""
            )
            db.add(s2)
            db.commit()
        else:
            s2.ai_usage = "Active"
            db.commit()

        # Activate s2 and deactivate s1
        s1.ai_usage = "Inactive"
        db.commit()
        rag_service.invalidate_caches(1)

        # Place Call 2 for Org 1
        call2 = SipCallSession(
            caller_phone="+919876543211",
            organization_id=1,
            audio_send_callback=mock_audio_send
        )
        call2._speak = lambda text, intent="", metrics=None: spoken_greetings.append((2, text)) or asyncio.sleep(0)
        await call2._send_opening_greeting()

        assert len(spoken_greetings) == 2
        assert spoken_greetings[-1][0] == 2
        assert "Apollo Heart Center" in spoken_greetings[-1][1] and ("Thank you for calling" in spoken_greetings[-1][1] or "Welcome to" in spoken_greetings[-1][1])
        assert "How may I assist you" in spoken_greetings[-1][1] or "How may I direct your call" in spoken_greetings[-1][1]
        assert call2.active_script_id == str(s2.id)

        # ── SCRIPT 3: Dental Clinic for Org 2 ────────────────────────
        s3 = db.query(KnowledgeDocument).filter(KnowledgeDocument.title == "Test_Dental_Script_3").first()
        if not s3:
            s3 = KnowledgeDocument(
                title="Test_Dental_Script_3",
                organization_id=2,
                document_type="SCRIPT",
                scope="inbound",
                ai_usage="Active",
                status=KnowledgeDocStatus.INDEXED,
                content="""# 1. GREETING
"Good morning, welcome to City Dental Clinic. How may I assist you?"

# 2. Treatments
- Scaling
- Root Canal"""
            )
            db.add(s3)
            db.commit()
        else:
            s3.ai_usage = "Active"
            s3.content = """# 1. GREETING
"Good morning, welcome to City Dental Clinic. How may I assist you?"

# 2. Treatments
- Scaling
- Root Canal"""
            db.commit()

        rag_service.invalidate_caches(2)

        # Place Call 3 for Org 2
        call3 = SipCallSession(
            caller_phone="+919876543212",
            organization_id=2,
            audio_send_callback=mock_audio_send
        )
        call3._speak = lambda text, intent="", metrics=None: spoken_greetings.append((3, text)) or asyncio.sleep(0)
        await call3._send_opening_greeting()

        assert len(spoken_greetings) == 3
        assert spoken_greetings[-1][0] == 3
        assert "City Dental Clinic" in spoken_greetings[-1][1] and ("Thank you for calling" in spoken_greetings[-1][1] or "Welcome to" in spoken_greetings[-1][1])
        assert "How may I assist you" in spoken_greetings[-1][1]
        assert call3.active_script_id == str(s3.id)
        assert call3.current_lang == "en"

    finally:
        # Cleanup test scripts
        db.query(KnowledgeDocument).filter(
            KnowledgeDocument.title.in_([
                "Test_Sarans_Script_1", "Test_Apollo_Script_2",
                "Test_Dental_Script_3"
            ])
        ).delete(synchronize_session=False)
        db.commit()
        db.close()


@pytest.mark.asyncio
async def test_missing_greeting_section_in_active_script():
    """
    Verify that if an active script has NO greeting section (only doctors/info),
    the system does NOT silently speak a generic greeting or filename.
    """
    db = SessionLocal()
    spoken_greetings = []

    def mock_audio_send(data):
        pass

    try:
        s_no_greeting = KnowledgeDocument(
            title="Test_No_Greeting_Script",
            organization_id=1,
            document_type="SCRIPT",
            scope="inbound",
            ai_usage="Active",
            status=KnowledgeDocStatus.INDEXED,
            content="""# HOSPITAL INFORMATION
Hospital Name: ABC General Hospital
# DEPARTMENTS
- Cardiology
# DOCTOR DIRECTORY
Dr Priya"""
        )
        db.add(s_no_greeting)
        db.commit()

        # Deactivate all others for org 1
        db.query(KnowledgeDocument).filter(
            KnowledgeDocument.organization_id == 1,
            KnowledgeDocument.id != s_no_greeting.id
        ).update({"ai_usage": "Inactive"})
        db.commit()
        rag_service.invalidate_caches(1)

        # Extract greeting check directly
        greeting, doc_id, doc_title = rag_service.get_active_greeting(1)
        assert greeting is None or greeting == "", "Greeting must be None or empty when no greeting section is present"

        # Call session check
        call = SipCallSession(
            caller_phone="+919876543220",
            organization_id=1,
            audio_send_callback=mock_audio_send
        )
        call._speak = lambda text, intent="", metrics=None: spoken_greetings.append(text) or asyncio.sleep(0)
        await call._send_opening_greeting()

        # MUST NOT speak any generic greeting!
        assert len(spoken_greetings) == 0, "No spoken greeting must be delivered when script lacks greeting section"

    finally:
        db.query(KnowledgeDocument).filter(KnowledgeDocument.title == "Test_No_Greeting_Script").delete(synchronize_session=False)
        db.commit()
        db.close()


@pytest.mark.asyncio
async def test_empty_script_handling():
    """Verify that an empty script logs configuration error and does not speak a fallback greeting."""
    db = SessionLocal()
    spoken_greetings = []

    def mock_audio_send(data):
        pass

    try:
        s_empty = KnowledgeDocument(
            title="Test_Empty_Script",
            organization_id=1,
            document_type="SCRIPT",
            scope="inbound",
            ai_usage="Active",
            status=KnowledgeDocStatus.INDEXED,
            content=""
        )
        db.add(s_empty)
        db.commit()

        # Deactivate all others for org 1
        db.query(KnowledgeDocument).filter(
            KnowledgeDocument.organization_id == 1,
            KnowledgeDocument.id != s_empty.id
        ).update({"ai_usage": "Inactive"})
        db.commit()
        rag_service.invalidate_caches(1)

        greeting, doc_id, doc_title = rag_service.get_active_greeting(1)
        assert greeting is None or greeting == "", "Greeting must be None or empty for empty script"

        call = SipCallSession(
            caller_phone="+919876543221",
            organization_id=1,
            audio_send_callback=mock_audio_send
        )
        call._speak = lambda text, intent="", metrics=None: spoken_greetings.append(text) or asyncio.sleep(0)
        await call._send_opening_greeting()

        assert len(spoken_greetings) == 0, "No spoken greeting must be delivered for empty script"

    finally:
        db.query(KnowledgeDocument).filter(KnowledgeDocument.title == "Test_Empty_Script").delete(synchronize_session=False)
        db.commit()
        db.close()
