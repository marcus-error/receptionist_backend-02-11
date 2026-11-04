import pytest
from datetime import datetime, timezone
import json
from unittest.mock import MagicMock, patch

from src.models.call_log import CallLog, CallType
from src.models.knowledge_doc import KnowledgeDocument, KnowledgeDocStatus
from src.models.organization import Organization
from src.services.booking_state_service import booking_state_service, BookingSession, BookingState
from src.services.calendar_verification_service import calendar_verification_service, CalendarVerificationResult
from src.services.rag_service import rag_service
from src.services.agent_engine import agent_engine


@pytest.fixture
def test_org_and_script(db_session):
    """
    Creates an independent tenant with an active RAG receptionist script.
    """
    # Create Organization
    org = Organization(
        name="Zenith Dental Care",
        organization_type="Healthcare",
        status="active"
    )
    db_session.add(org)
    db_session.commit()
    db_session.refresh(org)

    script_content = (
        "# Zenith Dental Care - Receptionist Script\n\n"
        "**Greeting:** Good morning, thank you for calling Zenith Dental Care. How can I help you smile today?\n\n"
        "Working Hours: Monday to Saturday, 9:00 AM to 6:00 PM\n"
        "Location: 100 Main Street, Suite 400, Chennai\n"
        "Services: Teeth Whitening, Root Canal, Dental Implants, Routine Cleaning\n"
        "Root Canal Cost: 5000 INR\n"
    )

    doc = KnowledgeDocument(
        organization_id=org.id,
        title="Zenith_Dental_Receptionist_Script.txt",
        file_path="/docs/zenith_script.txt",
        file_type="txt",
        document_type="SCRIPT",
        scope="inbound",
        ai_usage="Active",
        status=KnowledgeDocStatus.INDEXED,
        content=script_content,
        created_at=datetime.now(timezone.utc)
    )
    db_session.add(doc)
    db_session.commit()
    db_session.refresh(doc)

    return org, doc


def test_script_based_greeting_retrieval(test_org_and_script):
    """
    1. AI speaks the greeting configured in the organization's active RAG script.
    Does not use generic greeting when script greeting exists.
    """
    org, doc = test_org_and_script

    greeting = rag_service.get_organization_greeting(org.id, document_id=str(doc.id), scope="inbound")
    assert "Good morning, thank you for calling Zenith Dental Care" in greeting
    assert "How can I help you smile today?" in greeting
    assert greeting != "How may I assist you today?"


def test_slot_extraction_from_conversational_turn(test_org_and_script):
    """
    2. Extract customer name, email, and phone even from natural compound introductions.
    """
    org, _ = test_org_and_script
    session = BookingSession(session_id="test_sess_conv_1", organization_id=org.id)

    # Natural greeting + name
    booking_state_service.extract_slots("Hi, my name is Sanjay Kumar", session)
    assert session.patient_name == "Sanjay Kumar"

    # Compound turn with email and phone
    booking_state_service.extract_slots(
        "My email is sanjay@example.com and my phone number is 9876543210", session
    )
    assert session.email == "sanjay@example.com"
    assert session.contact_number == "9876543210"
    # Name must NOT be overwritten or lost
    assert session.patient_name == "Sanjay Kumar"


@pytest.mark.asyncio
async def test_agent_engine_conversation_memory_and_no_repeated_questions(test_org_and_script, db_session):
    """
    Multi-turn conversation flow:
    Turn 1: Customer introduces name during general inquiry.
    Turn 2: Customer provides email and phone number.
    Turn 3: Customer requests booking for a service.
    Turn 4: AI must NOT ask for name, email, or phone again because they were already provided!
            AI should move directly to confirming the appointment with the collected details.
    """
    org, doc = test_org_and_script
    session_id = "test_agent_memory_1001"

    # Pre-create a CallLog for session persistence
    call = CallLog(
        call_sid=session_id,
        organization_id=org.id,
        caller_name="Unknown",
        caller_phone="Unknown",
        call_type=CallType.INBOUND,
        status="in-progress"
    )
    db_session.add(call)
    db_session.commit()

    # TURN 1: Customer introduces self & asks about hours
    res1 = await agent_engine.process_turn(
        user_text="Good morning, my name is Dr. Ramesh Patel. What are your working hours?",
        session_id=session_id,
        organization_id=org.id,
        channel="phone"
    )

    sess = booking_state_service.get_session(session_id)
    assert sess.patient_name == "Ramesh Patel"
    # Response should answer from RAG
    assert "9:00 AM" in res1["response"] or "working hours" in res1["response"].lower() or "6:00 PM" in res1["response"]

    # TURN 2: Customer shares contact info
    res2 = await agent_engine.process_turn(
        user_text="My email is ramesh.patel@example.com and phone is 9876543210",
        session_id=session_id,
        organization_id=org.id,
        channel="phone"
    )

    sess = booking_state_service.get_session(session_id)
    assert sess.patient_name == "Ramesh Patel"
    assert sess.email == "ramesh.patel@example.com"
    assert sess.contact_number == "9876543210"

    # Mock availability for turns 3 and 4
    with patch.object(
        calendar_verification_service,
        "verify_availability",
        return_value=CalendarVerificationResult(is_checked=True, is_available=True)
    ):
        # TURN 3: Customer asks to book an appointment
        res3 = await agent_engine.process_turn(
            user_text="I would like to book a Root Canal appointment for tomorrow at 10 AM",
            session_id=session_id,
            organization_id=org.id,
            channel="phone"
        )

        sess = booking_state_service.get_session(session_id)
        assert sess.patient_name == "Ramesh Patel"
        assert sess.email == "ramesh.patel@example.com"
        assert sess.contact_number == "9876543210"

        # AI should NOT ask "what is your name?", "what is your email?", or "what is your phone?"
        resp_lower = res3["response"].lower()
        assert "what is your name" not in resp_lower
        assert "may i have your full name" not in resp_lower
        assert "share your contact" not in resp_lower
        assert "share your email" not in resp_lower

        # The AI should read back the collected details for confirmation
        assert "ramesh patel" in resp_lower
        assert "9876543210" in resp_lower

        # TURN 4: Customer confirms
        res4 = await agent_engine.process_turn(
            user_text="Yes, that's correct, please confirm.",
            session_id=session_id,
            organization_id=org.id,
            channel="phone"
        )

        assert any(w in res4["response"].lower() for w in ["confirmed", "booked", "scheduled", "success"])
        assert sess.is_completed is True

    # TURN 5: Verify CallLog was updated with customer details and conversation state
    db_session.refresh(call)
    assert call.caller_name == "Ramesh Patel"
    assert call.caller_phone == "9876543210"
    assert call.receptionist_state is not None
    state_json = json.loads(call.receptionist_state)
    assert state_json["patient_name"] == "Ramesh Patel"
    assert state_json["email"] == "ramesh.patel@example.com"
    assert state_json["contact_number"] == "9876543210"


def test_rag_inbound_multiline_script_greeting_and_tenant_isolation(db_session):
    """
    Verify dynamic extraction of multi-line script greetings with unicode quotes,
    header without colons, organization name extraction, and strict tenant isolation.
    """
    # Create Organization A
    org_a = Organization(
        name="Sarans Wellness Care",
        organization_type="Healthcare",
        status="active"
    )
    db_session.add(org_a)
    db_session.commit()
    db_session.refresh(org_a)

    script_content = (
        "SARANS WELLNESS CARE INFORMATION\n"
        "Company / Organization Name\n"
        "Sarans Wellness Care\n\n"
        "PRODUCTION-READY INBOUND SCRIPTS\n"
        "GREETING & CALL OPENING\n"
        " \u201cGood Morning! Welcome to Sarans Wellness Care. Thank you for calling us. I can assist you with "
        "information about our treatments, doctors, working hours and appointments. How may I help you today?\u201d\n\n"
        "Working Hours: Monday to Saturday, 9:00 AM to 6:00 PM\n"
    )

    doc_a = KnowledgeDocument(
        organization_id=org_a.id,
        title="SARANS_WELLNESS_RAG_Inbound_Script.pdf",
        file_path="/docs/sarans_script.pdf",
        file_type="pdf",
        document_type="RAG",
        scope="inbound",
        ai_usage="Active",
        status=KnowledgeDocStatus.INDEXED,
        content=script_content,
        created_at=datetime.now(timezone.utc)
    )
    db_session.add(doc_a)
    db_session.commit()
    db_session.refresh(doc_a)

    # 1. Verify dynamic organization name extraction
    extracted_name = rag_service.get_organization_name(org_a.id, document_id=str(doc_a.id), scope="inbound")
    assert extracted_name == "Sarans Wellness Care"

    # 2. Verify exact greeting extraction from multi-line unicode quoted script
    greeting_a = rag_service.get_organization_greeting(org_a.id, document_id=str(doc_a.id), scope="inbound")
    assert "Good Morning! Welcome to Sarans Wellness Care" in greeting_a
    assert "How may I help you today?" in greeting_a
    assert "treatments, doctors, working hours and appointments" in greeting_a

    # 3. Verify strict tenant isolation: an unrelated organization does NOT get Org A's greeting
    greeting_other = rag_service.get_organization_greeting(99999)
    assert "Sarans Wellness Care" not in greeting_other
    assert greeting_other != greeting_a
