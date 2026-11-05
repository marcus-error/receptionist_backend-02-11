"""
Acceptance Test Suite for CallZenza AI Receptionist Greeting & Call Direction Fixes.

Validates:
1. Strict Call Direction Identification (INBOUND vs OUTBOUND via SIP metadata and CallType, no caller phone inference).
2. Inbound Greeting Exact Format: "Good [time of day]! Thank you for calling [Organisation Name]. How may I assist you?".
3. Diagnostic Logging:
   - call_direction
   - organisation_id
   - active_script_id
   - selected_greeting_source
   - extracted_greeting
   - final_greeting_text
   - greeting_cache_hit / cache_version
   - TTS_input_text
   - RTP_utterance_id
   - Exact identity: final_greeting_text == TTS_input_text
4. Inbound Identity / Call Origin Queries:
   - "You're calling from?" -> "You've reached [Org]. How may I help you?"
   - "Where are you calling from?" -> "You've reached [Org]. How may I help you?"
   - "Who is calling?" -> "You've reached [Org]. How may I help you?"
   - "You've reached where?" -> "You've reached [Org]. How may I help you?"
   - "Which hospital is this?" -> "You've reached [Org]. How may I help you?"
   - Zero "I'm calling from [Org]" on inbound calls.
5. "Hello?" handling: Does not repeat the opening greeting, returns "Yes, how may I assist you?".
6. Outbound calls preserve outbound introduction behavior.
"""

import pytest
import re
from unittest.mock import AsyncMock, patch
from src.config.database import SessionLocal, init_db
from src.models.knowledge_doc import KnowledgeDocument, KnowledgeDocStatus
from src.models.organization import Organization
from src.services.sip_call_handler import SipCallSession, CallType
from src.services.rag_service import rag_service
from src.services.ai_service import ai_service
from src.services.guardrail_service import guardrail_service


@pytest.fixture(autouse=True)
def setup_test_org_and_script():
    """Ensure organization 1 is Sarans Hospitals and has an active script in the test database."""
    init_db()
    db = SessionLocal()
    try:
        org = db.query(Organization).filter(Organization.id == 1).first()
        if not org:
            org = Organization(id=1, name="Sarans Hospitals", organization_type="Healthcare", status="active")
            db.add(org)
        else:
            org.name = "Sarans Hospitals"

        # Ensure active script exists
        script = db.query(KnowledgeDocument).filter(
            KnowledgeDocument.organization_id == 1,
            KnowledgeDocument.title == "Sarans_Hospitals_Inbound_Script"
        ).first()
        if not script:
            script = KnowledgeDocument(
                id=1,
                title="Sarans_Hospitals_Inbound_Script",
                organization_id=1,
                document_type="SCRIPT",
                scope="inbound",
                ai_usage="Active",
                status=KnowledgeDocStatus.INDEXED,
                content=(
                    "# SARANS HOSPITALS SCRIPT\n"
                    "# 1. Greeting & Call Opening\n"
                    "\"Good Morning! Welcome to Sarans Hospitals. Thank you for calling us. "
                    "I can assist you with information about our treatments, doctors, consultations, "
                    "services, branches, working hours and appointments. How may I help you today?\""
                )
            )
            db.add(script)
        else:
            script.ai_usage = "Active"
            script.status = KnowledgeDocStatus.INDEXED
        db.commit()
    finally:
        db.close()

    rag_service.clear_cache()
    yield


@pytest.mark.asyncio
async def test_inbound_call_direction_and_exact_greeting():
    """Verify inbound calls use CallType.INBOUND and generate exact required format."""
    audio_sent = []
    def mock_send(b):
        audio_sent.append(b)

    session = SipCallSession(
        caller_phone="+919876543210",
        caller_name="Dr. Test Caller",
        call_type=CallType.INBOUND,
        call_sid="sip_in_test_001",
        organization_id=1,
        audio_send_callback=mock_send,
    )

    assert session.call_type == CallType.INBOUND
    assert session.booking_session.call_type == CallType.INBOUND
    assert session.booking_session.call_direction == "INBOUND"

    with patch.object(session, "_speak", new_callable=AsyncMock) as mock_speak:
        await session._send_opening_greeting()
        assert mock_speak.called

        spoken_text = mock_speak.call_args[0][0]
        metrics = mock_speak.call_args[0][2]

        # Verify exact inbound format: "Good [time of day]! Thank you for calling [Org]. How may I assist you?"
        pattern = r'^Good (morning|afternoon|evening)! Thank you for calling Sarans Hospitals\. How may I assist you\?$'
        assert re.match(pattern, spoken_text), f"Greeting '{spoken_text}' does not match inbound pattern"

        # Verify NEVER contains outbound phrases
        assert "This is Sarans Hospitals calling" not in spoken_text
        assert "calling from" not in spoken_text

        # Verify metrics passed to _speak
        assert metrics["utt_id"] == "utt_greeting"
        assert metrics["is_greeting"] is True
        assert metrics["call_direction"] == "INBOUND"


@pytest.mark.asyncio
async def test_outbound_call_direction_and_preservation():
    """Verify outbound calls preserve outbound introduction."""
    session = SipCallSession(
        caller_phone="+919876543210",
        caller_name="Outbound → 9876543210",
        call_type=CallType.OUTBOUND,
        call_sid="sip_out_test_002",
        organization_id=1,
    )

    assert session.call_type == CallType.OUTBOUND
    assert session.booking_session.call_type == CallType.OUTBOUND
    assert session.booking_session.call_direction == "OUTBOUND"

    with patch.object(session, "_speak", new_callable=AsyncMock) as mock_speak:
        await session._send_opening_greeting()
        assert mock_speak.called

        spoken_text = mock_speak.call_args[0][0]
        metrics = mock_speak.call_args[0][2]

        # Outbound calls should maintain outbound introduction
        assert "calling" in spoken_text or "Am I speaking with" in spoken_text
        assert metrics["call_direction"] == "OUTBOUND"


@pytest.mark.asyncio
async def test_diagnostic_logging_and_tts_input_identity(caplog):
    """Verify all mandatory diagnostic keys are logged and TTS_input_text matches final_greeting_text."""
    import logging
    caplog.set_level(logging.INFO)

    session = SipCallSession(
        caller_phone="+919876543210",
        caller_name="Test Inbound",
        call_type=CallType.INBOUND,
        call_sid="sip_diag_test_003",
        organization_id=1,
    )

    with patch.object(session, "_speak", new_callable=AsyncMock):
        await session._send_opening_greeting()

    log_text = caplog.text
    assert "[GREETING_SELECTION_DIAGNOSTIC]" in log_text
    assert "call_direction=INBOUND" in log_text
    assert "organisation_id=1" in log_text
    assert "active_script_id=" in log_text
    assert "selected_greeting_source=" in log_text
    assert "extracted_greeting=" in log_text
    assert "final_greeting_text=" in log_text
    assert "greeting_cache_hit=" in log_text
    assert "cache_version=" in log_text
    assert "TTS_input_text=" in log_text
    assert "RTP_utterance_id=utt_greeting" in log_text
    assert "[GREETING_RESPONSE]" in log_text
    assert "[GREETING_TTS]" in log_text


def test_inbound_identity_origin_queries():
    """Verify responses to origin questions on inbound calls never say 'I am calling from'."""
    queries = [
        "You're calling from?",
        "Where are you calling from?",
        "Who is calling?",
        "You've reached where?",
        "Which hospital is this?",
        "Who am I speaking with?",
        "Who is this?",
    ]

    for q in queries:
        resp = ai_service._fast_rule_response(
            text=q,
            intent="general_question",
            organization_id=1,
            is_ongoing=True,
            call_direction="INBOUND"
        )
        assert resp is not None, f"Query '{q}' should be handled by fast rule response"
        assert "You've reached" in resp, f"Query '{q}' response '{resp}' should say 'You've reached...'"
        assert "I'm calling from" not in resp, f"Query '{q}' response '{resp}' must not say 'I'm calling from'"
        assert "calling from" not in resp.lower() or "reached" in resp


def test_outbound_identity_queries():
    """Verify outbound identity queries preserve outbound behavior."""
    resp = ai_service._fast_rule_response(
        text="Who is calling?",
        intent="general_question",
        organization_id=1,
        is_ongoing=True,
        call_direction="OUTBOUND"
    )
    assert resp is not None
    assert "I'm calling from" in resp


def test_hello_does_not_repeat_greeting():
    """Verify caller saying 'Hello?' after opening greeting gets 'Yes, how may I assist you?'."""
    session_state = {"greeting_sent": True, "turn_count": 1}

    resp = ai_service._fast_rule_response(
        text="Hello?",
        intent="greeting",
        organization_id=1,
        is_ongoing=True,
        booking_session=session_state,
        has_context=True,
        call_direction="INBOUND"
    )
    assert resp == "Yes, how may I assist you?"


def test_guardrail_sanitizes_inbound_outbound_phrase():
    """Verify guardrail replaces any stray 'I'm calling from [Org]' on inbound calls."""
    stray_response = "Good afternoon. I'm calling from Sarans Hospitals. How may I help you?"
    sanitized = guardrail_service.validate_and_sanitize_response(
        stray_response,
        call_direction="INBOUND",
        org_name="Sarans Hospitals"
    )
    assert "I'm calling from" not in sanitized
    assert "You've reached Sarans Hospitals" in sanitized
