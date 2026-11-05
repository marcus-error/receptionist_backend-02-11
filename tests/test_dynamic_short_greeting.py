"""
Test Suite: Dynamic Short Greeting System
Features Verified:
1. Dynamic greeting generation using active organisation's uploaded receptionist script.
2. Short, natural, and professional greeting length (ideally 1 sentence, <= 15 words).
3. Stripping of lengthy service/department/doctor catalogues from the initial greeting.
4. Dynamic time-aware salutations (Good morning / Good afternoon / Good evening).
5. Dynamic script upload/update syncs AppSettings.default_greeting in SQLite DB.
6. Multi-tenant isolation: Each organisation uses only its own script and AppSettings.
7. Fallback greeting dynamically uses the DB organization name (zero hardcoding).
8. Outbound dynamic short greeting with customer confirmation.
9. Conversation continuation never repeats the initial greeting.
"""

import asyncio
import pytest
from datetime import datetime

from src.config.database import SessionLocal, init_db
from src.models.organization import Organization
from src.models.settings import AppSettings
from src.models.knowledge_doc import KnowledgeDocument, KnowledgeDocStatus
from src.services.rag_service import rag_service
from src.services.knowledge_service import KnowledgeService
from src.services.sip_call_handler import SipCallSession
from src.services.ai_service import ai_service
from src.validators.knowledge_validator import KnowledgeDocCreate


@pytest.fixture(autouse=True)
def setup_database():
    init_db()
    db = SessionLocal()
    # Ensure test organizations exist
    org1 = db.query(Organization).filter(Organization.id == 1).first()
    if not org1:
        db.add(Organization(id=1, name="Sarans Hospitals"))
    doc1 = db.query(KnowledgeDocument).filter(KnowledgeDocument.id == 1).first()
    if doc1:
        doc1.ai_usage = "Active"
    db.commit()
    db.close()
    rag_service.invalidate_caches(1)


def test_short_greeting_condensation_removes_catalogue():
    """
    Verify that an uploaded script containing a lengthy catalogue of services/doctors
    is distilled into a crisp, professional 1-sentence opening (<= 14 words).
    """
    raw_script_text = (
        "Good Morning! Welcome to Sarans Hospitals. Thank you for calling us. "
        "I can assist you with information about our treatments, doctors, consultations, "
        "services, branches, working hours and appointments. How may I help you today?"
    )

    short_greeting = rag_service.extract_short_greeting(
        raw_greeting=raw_script_text,
        org_name="Sarans Hospitals",
        organization_id=1,
        scope="inbound"
    )

    assert short_greeting is not None
    assert "Sarans Hospitals" in short_greeting
    assert "Thank you for calling Sarans Hospitals" in short_greeting
    assert "How may I assist you" in short_greeting
    # Verify catalogue is stripped
    assert "treatments, doctors, consultations" not in short_greeting
    assert "branches, working hours" not in short_greeting
    # Verify length is short and professional (ideally 1 sentence, <= 14 words)
    words = short_greeting.split()
    assert len(words) <= 14, f"Greeting too long: {len(words)} words ('{short_greeting}')"


def test_time_aware_dynamic_prefix():
    """Verify that the greeting prefix dynamically matches time of day."""
    morning = rag_service._condense_to_short_greeting("Welcome to Aminoid AI.", resolved_org="Aminoid AI", time_prefix="Good morning")
    assert morning == "Good morning! Thank you for calling Aminoid AI. How may I assist you?"

    afternoon = rag_service._condense_to_short_greeting("Welcome to Aminoid AI.", resolved_org="Aminoid AI", time_prefix="Good afternoon")
    assert afternoon == "Good afternoon! Thank you for calling Aminoid AI. How may I assist you?"

    evening = rag_service._condense_to_short_greeting("Welcome to Aminoid AI.", resolved_org="Aminoid AI", time_prefix="Good evening")
    assert evening == "Good evening! Thank you for calling Aminoid AI. How may I assist you?"


def test_script_upload_syncs_app_settings():
    """
    When an admin uploads/creates an active receptionist script,
    KnowledgeService extracts its short greeting and syncs it into AppSettings.default_greeting.
    """
    db = SessionLocal()
    try:
        # Create a test organization
        test_org = db.query(Organization).filter(Organization.id == 101).first()
        if not test_org:
            test_org = Organization(id=101, name="Apex Cardiology Center")
            db.add(test_org)
            db.commit()

        ks = KnowledgeService(db)
        script_payload = KnowledgeDocCreate(
            title="Apex_Cardiology_Receptionist_Script.pdf",
            organization_id=101,
            document_type="SCRIPT",
            scope="inbound",
            ai_usage="Active",
            content="""# APEX CARDIOLOGY RECEPTIONIST SCRIPT
## 1. Greeting & Call Opening
"Good morning! Thank you for calling Apex Cardiology Center. We provide cardiac diagnostics, ECG, and consultations. How may I assist you today?"
"""
        )

        doc = ks.create_doc(script_payload, organization_id=101)
        assert doc.id is not None

        # Verify AppSettings was updated for Org 101
        settings = db.query(AppSettings).filter(AppSettings.organization_id == 101).first()
        assert settings is not None
        assert "Apex Cardiology Center" in settings.default_greeting
        assert "treatments" not in settings.default_greeting
        assert len(settings.default_greeting.split()) <= 14

    finally:
        db.query(KnowledgeDocument).filter(KnowledgeDocument.organization_id == 101).delete()
        db.query(AppSettings).filter(AppSettings.organization_id == 101).delete()
        db.query(Organization).filter(Organization.id == 101).delete()
        db.commit()
        db.close()


def test_multi_tenant_script_isolation():
    """
    Verify multi-tenant isolation:
    Org 201 (Aminoid AI) and Org 202 (Sarans Hospitals) each have their own
    distinct dynamic short greeting saved and retrieved.
    """
    db = SessionLocal()
    try:
        org_a = Organization(id=201, name="Aminoid AI")
        org_b = Organization(id=202, name="Sarans Hospitals")
        db.add_all([org_a, org_b])
        db.commit()

        ks = KnowledgeService(db)

        # Upload script for Org A
        doc_a = ks.create_doc(KnowledgeDocCreate(
            title="Aminoid_Script.txt",
            organization_id=201,
            document_type="SCRIPT",
            scope="inbound",
            ai_usage="Active",
            content="""# AMINOID AI SCRIPT
## Greeting
"Good morning! Thank you for calling Aminoid AI. How may I assist you?"
"""
        ), organization_id=201)

        # Upload script for Org B
        doc_b = ks.create_doc(KnowledgeDocCreate(
            title="Sarans_Script.txt",
            organization_id=202,
            document_type="SCRIPT",
            scope="inbound",
            ai_usage="Active",
            content="""# SARANS SCRIPT
## Greeting
"Good afternoon! Thank you for calling Sarans Hospitals. How may I assist you?"
"""
        ), organization_id=202)

        g_a, id_a, _ = rag_service.get_active_greeting(201)
        g_b, id_b, _ = rag_service.get_active_greeting(202)

        assert "Aminoid AI" in g_a
        assert "Sarans Hospitals" not in g_a
        assert id_a == doc_a.id

        assert "Sarans Hospitals" in g_b
        assert "Aminoid AI" not in g_b
        assert id_b == doc_b.id

        # Verify AppSettings isolation
        settings_a = db.query(AppSettings).filter(AppSettings.organization_id == 201).first()
        settings_b = db.query(AppSettings).filter(AppSettings.organization_id == 202).first()
        assert "Aminoid AI" in settings_a.default_greeting
        assert "Sarans Hospitals" in settings_b.default_greeting

    finally:
        db.query(KnowledgeDocument).filter(KnowledgeDocument.organization_id.in_([201, 202])).delete()
        db.query(AppSettings).filter(AppSettings.organization_id.in_([201, 202])).delete()
        db.query(Organization).filter(Organization.id.in_([201, 202])).delete()
        db.commit()
        db.close()


def test_dynamic_fallback_using_db_organization_name():
    """
    If no greeting is provided in the script, use a short dynamic fallback
    using the organization name from the database (zero hardcoding).
    """
    db = SessionLocal()
    try:
        org = Organization(id=301, name="Sunrise Dental Clinic")
        db.add(org)
        db.commit()

        # Dynamic fallback generation
        fallback = rag_service.extract_short_greeting(
            raw_greeting="",
            org_name="",
            organization_id=301,
            scope="inbound"
        )
        assert "Sunrise Dental Clinic" in fallback
        assert "Thank you for calling Sunrise Dental Clinic" in fallback
        assert "How may I assist you" in fallback
        # Under NO circumstance uses generic 'Ava' or 'Hello! How can I help you today?'
        assert "Ava" not in fallback
        assert "our company" not in fallback

    finally:
        db.query(Organization).filter(Organization.id == 301).delete()
        db.commit()
        db.close()


def test_outbound_dynamic_short_greeting():
    """Verify outbound scripts produce a concise customer confirmation greeting."""
    raw_outbound = "Good morning. Am I speaking with Mr. Robert Smith regarding your appointment?"
    outbound_greeting = rag_service.extract_short_greeting(
        raw_greeting=raw_outbound,
        org_name="Sarans Hospitals",
        organization_id=1,
        scope="outbound"
    )

    assert "Am I speaking with" in outbound_greeting
    assert len(outbound_greeting.split()) <= 15


@pytest.mark.asyncio
async def test_conversation_flow_and_no_greeting_repetition():
    """
    Verify complete conversation flow:
    1. AI answers with short dynamic greeting.
    2. Customer speaks (asks a question).
    3. AI responds naturally without repeating the greeting.
    """
    spoken_outputs = []

    db = SessionLocal()
    org1 = db.query(Organization).filter(Organization.id == 1).first()
    if not org1:
        org1 = Organization(id=1, name="Sarans Hospitals")
        db.add(org1)
        db.commit()
    else:
        org1.name = "Sarans Hospitals"
        db.commit()

    doc1 = db.query(KnowledgeDocument).filter(KnowledgeDocument.organization_id == 1).first()
    if not doc1:
        doc1 = KnowledgeDocument(
            title="SARANS_HOSPITALS_RAG_Inbound_Script_Fine_Tuned.pdf",
            organization_id=1,
            document_type="SCRIPT",
            scope="inbound",
            ai_usage="Active",
            status=KnowledgeDocStatus.INDEXED,
            content="""# SARANS HOSPITALS - RAG INBOUND SCRIPT (FINE-TUNED)

## 1. Approved Spoken Greeting & Call Opening
"Good Morning! Welcome to Sarans Hospitals. Thank you for calling us. I can assist you with information about our treatments, doctors, consultations, services, branches, working hours and appointments. How may I help you today?"

## 2. Treatments & Services
We provide cardiology, orthopedics, neurology, dental care, and emergency treatments."""
        )
        db.add(doc1)
        db.commit()
    else:
        doc1.ai_usage = "Active"
        db.commit()
    db.close()
    rag_service.invalidate_caches(1)

    def mock_send(bytes_):
        pass

    session = SipCallSession(
        call_sid="sip_test_flow_123",
        organization_id=1,
        audio_send_callback=mock_send
    )
    session._speak = lambda text, intent="", metrics=None: spoken_outputs.append(text) or asyncio.sleep(0)

    # 1. Opening greeting
    await session._send_opening_greeting()
    assert len(spoken_outputs) == 1
    initial_greeting = spoken_outputs[0]
    assert "Sarans Hospitals" in initial_greeting
    assert len(initial_greeting.split()) <= 14

    # 2. Customer speaks: "What treatments do you offer?"
    response = ai_service.generate_response(
        text="What treatments do you offer?",
        session_history=session.session_history,
        organization_id=1
    )

    # 3. AI responds with information - must NOT repeat greeting
    assert response is not None
    assert not response.lower().startswith("good morning")
    assert not response.lower().startswith("good afternoon")
    assert not response.lower().startswith("welcome to sarans")
    assert "Thank you for calling Sarans Hospitals. How may I assist you" not in response
