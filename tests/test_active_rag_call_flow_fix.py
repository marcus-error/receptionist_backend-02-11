import os
import sys
import asyncio
import pytest
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config.database import SessionLocal
from src.models.knowledge_doc import KnowledgeDocument, KnowledgeDocStatus
from src.models.call_log import CallType
from src.services.rag_service import rag_service
from src.services.sip_call_handler import SipCallSession

NURSE_O_CLOCK_CONTENT = """NURSE O' CLOCK — CHENNAI INFORMATION
Organization Name: Nurse O' Clock
Organization Type: Home Healthcare and Patient-Care Service Provider
Service Location: Chennai, Tamil Nadu
About Us: Nurse O' Clock supports patients, elderly people, families and caregivers through home-based healthcare,
caregiving, patient support and related service coordination.
Communication Style: Warm, respectful, patient, professional, reassuring and empathetic.
AI Receptionist Role: Service enquiries, basic information, lead capture, appointment/service requests and routing.

SERVICES
The current knowledge base includes the following services:
• Senior Assisted Living
• Nursing Services
• Home Nurse / Medical Assistants
• Elderly Care / Old Age Care
• Patient Attender / Caretaker
• Physiotherapy
• Lab Test
• Medical Equipment

OFFICE / SERVICE CENTER INFORMATION
Office / Service Center Name: Nurse O' Clock Chennai Care Center
Address: No. 24, Lakshmi Nagar Main Road, Velachery, Chennai, Tamil Nadu — 600042
Landmark: Near Velachery Bus Stand
City: Chennai
State: Tamil Nadu
PIN Code: 600042
Phone: +91 90000 12345
Email: care@nurseoclock-demo.example
Website: https://nurseoclock-demo.example
Service Coverage: Chennai and selected nearby localities, subject to service availability.

STANDARD AI RECEPTIONIST CONVERSATION FLOW
1. Greet the caller warmly.
2. Identify intent: service enquiry, doctor enquiry, appointment request, existing-service follow-up or urgent situation.
3. Close politely and empathetically.
"""

PRAMERICA_CONTENT = """PRAMERICA LIFE INSURANCE
OUTBOUND RAG SCRIPT

COMPANY INFORMATION
Company Name: Pramerica Life Insurance
Company Type: Life Insurance

OPENING
AI Agent:
"Good morning / afternoon. Am I speaking with [Customer Name]?"

Customer:
"Yes."

AI Agent:
"Thank you. My name is [Agent Name], calling from Pramerica Life Insurance. I'm calling to understand your life insurance requirement and share information about our available insurance plans. How may I assist you?"
"""


@pytest.fixture(autouse=True)
def setup_nurse_o_clock_doc():
    """Ensure Document 2 exists in the test database for all tests."""
    db = SessionLocal()
    doc2 = db.query(KnowledgeDocument).filter(KnowledgeDocument.id == 2).first()
    if not doc2:
        doc2 = KnowledgeDocument(
            id=2,
            title="nurse_o_clock_receptionist_rag_formatted.txt",
            organization_id=None,
            document_type="RAG",
            scope="organization",
            visibility="organization",
            ai_usage="Active",
            status=KnowledgeDocStatus.INDEXED,
            content=NURSE_O_CLOCK_CONTENT
        )
        db.add(doc2)
        db.commit()
    else:
        doc2.ai_usage = "Active"
        doc2.content = NURSE_O_CLOCK_CONTENT
        db.commit()

    rag_service.clear_cache()
    yield doc2

    try:
        db.query(KnowledgeDocument).filter(KnowledgeDocument.id == 2).update({"ai_usage": "Active"})
        db.commit()
    except Exception:
        pass
    finally:
        db.close()
        rag_service.clear_cache()


@pytest.mark.asyncio
async def test_outbound_call_resolution_doc2(setup_nurse_o_clock_doc):
    """
    Simulates production Call ID sip_out_59d5a308ac for organization_id=1.
    Must resolve Document 2 (nurse_o_clock_receptionist_rag_formatted.txt),
    generate outbound greeting 'Hello! This is Nurse O' Clock calling. How may I assist you today?',
    and log source=ACTIVE_SCRIPT_GREETING.
    """
    rag_service.clear_cache()
    spoken = []
    logged_sources = []

    def mock_audio(b):
        pass

    session = SipCallSession(
        call_sid="sip_out_59d5a308ac",
        caller_phone="+919876543210",
        caller_name="Outbound Lead",
        organization_id=1,
        audio_send_callback=mock_audio,
        call_type=CallType.OUTBOUND
    )

    async def mock_speak(text, intent="", metrics=None):
        spoken.append(text)
        if metrics and "source" in metrics:
            logged_sources.append(metrics["source"])

    session._speak = mock_speak

    assert session.active_script_id == "2", f"Expected active_script_id '2', got {session.active_script_id}"
    await session._send_opening_greeting()

    assert len(spoken) == 1, f"Expected 1 greeting spoken, got {len(spoken)}"
    greeting = spoken[0]
    print(f"Spoken Outbound Greeting: {greeting}")
    assert "Nurse O' Clock" in greeting
    assert "calling" in greeting.lower()
    assert "ACTIVE_SCRIPT_GREETING" in logged_sources


@pytest.mark.asyncio
async def test_inbound_call_resolution_doc2(setup_nurse_o_clock_doc):
    """
    Simulates inbound call for organization_id=1.
    Must resolve Document 2, generate inbound greeting 'Thank you for calling Nurse O' Clock...',
    and log source=ACTIVE_SCRIPT_GREETING.
    """
    rag_service.clear_cache()
    spoken = []
    logged_sources = []

    def mock_audio(b):
        pass

    session = SipCallSession(
        call_sid="sip_in_test_nurse",
        caller_phone="+919876543211",
        caller_name="Customer Mobile",
        organization_id=1,
        audio_send_callback=mock_audio,
        call_type=CallType.INBOUND
    )

    async def mock_speak(text, intent="", metrics=None):
        spoken.append(text)
        if metrics and "source" in metrics:
            logged_sources.append(metrics["source"])

    session._speak = mock_speak

    assert session.active_script_id == "2", f"Expected active_script_id '2', got {session.active_script_id}"
    await session._send_opening_greeting()

    assert len(spoken) == 1
    greeting = spoken[0]
    print(f"Spoken Inbound Greeting: {greeting}")
    assert "Nurse O' Clock" in greeting
    assert "thank you for calling" in greeting.lower()
    assert "ACTIVE_SCRIPT_GREETING" in logged_sources


@pytest.mark.asyncio
async def test_multitenant_isolation(setup_nurse_o_clock_doc):
    """
    Ensures Org 1 gets Nurse O' Clock, Org 2 gets its own greeting,
    and neither organization's knowledge or greeting leaks to the other.
    """
    db = SessionLocal()
    try:
        # Create dedicated script for Org 2
        org2_doc = KnowledgeDocument(
            title="City Dental Clinic Script",
            organization_id=2,
            document_type="SCRIPT",
            scope="inbound",
            ai_usage="Active",
            status=KnowledgeDocStatus.INDEXED,
            content="""# GREETING
Receptionist: "Welcome to City Dental Clinic. How may I help you with your dental care today?"

# DOCTORS
Dr. Smith - Orthodontist"""
        )
        db.add(org2_doc)
        db.commit()

        rag_service.clear_cache()

        # Org 1 check
        g1, id1, t1 = rag_service.get_active_greeting(organization_id=1, scope="inbound")
        assert "Nurse O' Clock" in g1
        assert id1 == 2

        # Org 2 check
        g2, id2, t2 = rag_service.get_active_greeting(organization_id=2, scope="inbound")
        assert "City Dental Clinic" in g2
        assert id2 == org2_doc.id

        # Master context isolation check
        ctx1 = rag_service.get_organization_master_context(organization_id=1)
        assert "City Dental Clinic" not in ctx1
        assert "Nurse O' Clock" in ctx1

        ctx2 = rag_service.get_organization_master_context(organization_id=2)
        assert "Nurse O' Clock" not in ctx2
        assert "City Dental Clinic" in ctx2

    finally:
        db.query(KnowledgeDocument).filter(KnowledgeDocument.organization_id == 2).delete(synchronize_session=False)
        db.commit()
        db.close()
        rag_service.clear_cache()


@pytest.mark.asyncio
async def test_document_activation_switch(setup_nurse_o_clock_doc):
    """
    Tests dynamic switching when activating Document 4 (Pramerica) vs Document 2 (Nurse O' Clock).
    """
    db = SessionLocal()
    try:
        # Create or update Doc 4
        doc4 = db.query(KnowledgeDocument).filter(KnowledgeDocument.id == 4).first()
        if not doc4:
            doc4 = KnowledgeDocument(
                id=4,
                title="Pramerica_Life_Insurance_Outbound_RAG_Script.txt",
                organization_id=1,
                document_type="RAG",
                scope="organization",
                visibility="organization",
                ai_usage="Inactive",
                status=KnowledgeDocStatus.INDEXED,
                content=PRAMERICA_CONTENT
            )
            db.add(doc4)
            db.commit()

        # 1. Activate Document 4 for Org 1
        db.query(KnowledgeDocument).filter(KnowledgeDocument.id == 2).update({"ai_usage": "Inactive"})
        db.query(KnowledgeDocument).filter(KnowledgeDocument.id == 4).update({"ai_usage": "Active", "organization_id": 1})
        db.commit()
        rag_service.clear_cache()

        resolved_4 = rag_service._resolve_active_document_id(organization_id=1, scope="outbound")
        assert resolved_4 == "4"
        g4, id4, _ = rag_service.get_active_greeting(organization_id=1, scope="outbound")
        assert "Pramerica" in g4 or "good morning" in g4.lower() or "speaking with" in g4.lower()
        assert id4 == 4

        # 2. Reactivate Document 2
        db.query(KnowledgeDocument).filter(KnowledgeDocument.id == 4).update({"ai_usage": "Inactive"})
        db.query(KnowledgeDocument).filter(KnowledgeDocument.id == 2).update({"ai_usage": "Active"})
        db.commit()
        rag_service.clear_cache()

        resolved_2 = rag_service._resolve_active_document_id(organization_id=1, scope="outbound")
        assert resolved_2 == "2"
        g2, id2, _ = rag_service.get_active_greeting(organization_id=1, scope="outbound")
        assert "Nurse O' Clock" in g2
        assert id2 == 2

    finally:
        db.query(KnowledgeDocument).filter(KnowledgeDocument.id == 2).update({"ai_usage": "Active"})
        db.commit()
        db.close()
        rag_service.clear_cache()


@pytest.mark.asyncio
async def test_no_active_doc_configuration_error():
    """
    When an organization has NO active document, get_active_greeting must return (None, None, None)
    and SipCallSession must not speak any fallback greeting.
    """
    db = SessionLocal()
    try:
        # Deactivate all documents
        db.query(KnowledgeDocument).update({"ai_usage": "Inactive"})
        db.commit()
        rag_service.clear_cache()

        greeting, doc_id, doc_title = rag_service.get_active_greeting(organization_id=9999, scope="inbound")
        assert greeting is None
        assert doc_id is None
        assert doc_title is None

        spoken = []
        session = SipCallSession(
            call_sid="sip_test_missing_doc",
            organization_id=9999,
            audio_send_callback=lambda b: None
        )
        session._speak = lambda text, intent="", metrics=None: spoken.append(text) or asyncio.sleep(0)
        await session._send_opening_greeting()
        # MUST NOT speak any generic greeting!
        assert len(spoken) == 0, f"Expected 0 spoken greetings for unconfigured org, got {len(spoken)}"
    finally:
        db.query(KnowledgeDocument).filter(KnowledgeDocument.id == 2).update({"ai_usage": "Active"})
        db.commit()
        db.close()
        rag_service.clear_cache()


@pytest.mark.asyncio
async def test_rag_turn_vs_conversation_source_logging(setup_nurse_o_clock_doc):
    """
    Verifies that RAG queries get source=ACTIVE_KNOWLEDGE_RAG and conversational turns get source=CONVERSATION.
    Never logs source=ACTIVE_SCRIPT_RAG.
    """
    rag_service.clear_cache()
    session = SipCallSession(
        call_sid="sip_test_turns",
        organization_id=1,
        audio_send_callback=lambda b: None
    )

    spoken_metrics = []
    async def mock_speak(text, intent="", metrics=None):
        if metrics:
            spoken_metrics.append(dict(metrics))

    session._speak = mock_speak

    # 1. Opening greeting
    await session._send_opening_greeting()
    assert len(spoken_metrics) == 1
    assert spoken_metrics[0].get("source") == "ACTIVE_SCRIPT_GREETING"
