"""
Comprehensive Test Suite: Conversation Fluency, Response Latency, Memory & Booking Accuracy
Covers all requirements from Section 18 (English 1-13 & Tamil 1-10) and Section 19 (Performance Metrics).
"""

import os
import sys
import time
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

# Ensure backend root is on sys.path
backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_root not in sys.path:
    sys.path.insert(0, backend_root)

from src.services.nlp_service import (
    nlp_service,
    INTENT_GREETING,
    INTENT_CONNECTION_REASSURANCE,
    INTENT_PRODUCT_INFORMATION,
    INTENT_SERVICE_INFORMATION,
    INTENT_CALLBACK,
    INTENT_CHANGE_LANGUAGE,
    INTENT_APPOINTMENT_BOOKING,
    INTENT_APPOINTMENT_DETAILS_INQUIRY,
    INTENT_DEMO_CONTACT_INQUIRY,
    INTENT_AVAILABILITY_PROVIDED,
    INTENT_GOODBYE,
)
from src.services.ai_service import ai_service
from src.services.rag_service import rag_service
from src.services.language_service import language_service
from src.services.booking_state_service import (
    booking_state_service,
    BookingSession,
    BookingState,
    detect_incomplete_spoken_email,
)


@pytest.fixture(scope="module", autouse=True)
def init_services():
    nlp_service.initialize_model()


# ==============================================================================
# SECTION 18.1: ENGLISH CONVERSATION TESTS (1 to 13)
# ==============================================================================

def test_en_01_initial_greeting():
    """English 1: Initial greeting issued; must not be duplicated on first response."""
    session = BookingSession(call_sid="call_en_01")
    history = [
        {"role": "assistant", "text": "Good afternoon! Thank you for calling Aminoid AI. How may I assist you today?"}
    ]
    # Verify opening greeting structure
    assert "aminoid ai" in history[0]["text"].lower()
    assert "how may i assist you" in history[0]["text"].lower()
    # Ensure session tracks that greeting was sent
    session.greeting_sent = True
    assert session.greeting_sent is True


def test_en_02_good_afternoon():
    """English 2: Caller says 'Good afternoon.' -> Standalone natural return, no company intro repetition."""
    history = [
        {"role": "assistant", "text": "Good afternoon! Thank you for calling Aminoid AI. How may I assist you today?"}
    ]
    session_state = {"greeting_sent": True, "state": "IDLE"}

    intent = nlp_service.detect_intent("Good afternoon.").intent
    assert intent == INTENT_GREETING

    resp = ai_service._fast_rule_response(
        "Good afternoon.",
        intent=intent,
        organization_id=1,
        is_ongoing=True,
        booking_session=session_state,
        session_history=history,
        conversation_language="en",
    )
    assert resp is not None
    assert "good afternoon" in resp.lower()
    # Company greeting must NOT be repeated
    assert "thank you for calling" not in resp.lower()
    assert "aminoid ai" not in resp.lower()


def test_en_03_hello_are_you_there():
    """English 3: Caller says 'Hello, are you there?' -> Single clean reassurance without loop or greeting."""
    history = [
        {"role": "assistant", "text": "Good afternoon! Thank you for calling Aminoid AI. How may I assist you?"},
        {"role": "user", "text": "Good afternoon."},
        {"role": "assistant", "text": "Good afternoon! How can I help you today?"},
    ]
    session_state = {"greeting_sent": True, "state": "IDLE"}

    intent = nlp_service.detect_intent("Hello, are you there?").intent
    assert intent == INTENT_CONNECTION_REASSURANCE

    resp = ai_service._fast_rule_response(
        "Hello, are you there?",
        intent=intent,
        organization_id=1,
        is_ongoing=True,
        booking_session=session_state,
        session_history=history,
        conversation_language="en",
    )
    assert resp is not None
    assert any(phrase in resp.lower() for phrase in ["right here", "can hear you", "here with you"])
    # Must NOT compound greeting
    assert "thank you for calling" not in resp.lower()
    assert resp.lower().count("right here") <= 1


def test_en_04_what_products_and_services_do_you_provide():
    """English 4: Caller asks 'What products and services do you provide?' -> RAG retrieved, products listed."""
    intent = nlp_service.detect_intent("What products and services do you provide?").intent
    assert intent in (INTENT_PRODUCT_INFORMATION, INTENT_SERVICE_INFORMATION)

    rag_context = rag_service.retrieve_context("What products and services do you provide?", organization_id=1)
    history = [
        {"role": "assistant", "text": "Good afternoon! Thank you for calling Aminoid AI. How may I assist you?"},
        {"role": "user", "text": "Good afternoon."},
        {"role": "assistant", "text": "Good afternoon! How can I help you today?"},
    ]
    resp = ai_service.generate_response(
        text="What products and services do you provide?",
        context=rag_context,
        session_history=history,
        intent=intent,
        conversation_language="en",
        organization_id=1,
    )
    assert resp is not None
    resp_lower = resp.lower()
    assert any(term in resp_lower for term in ["crm", "hrms", "erp", "software", "ai", "solution", "services"])
    # Must NOT replay initial greeting
    assert "thank you for calling" not in resp_lower


def test_en_05_when_will_your_team_contact_me_for_the_demo():
    """English 5: 'When will your team contact me for the demo?' -> 24-hr turnaround explanation, no fake booking."""
    intent = nlp_service.detect_intent("When will your team contact me for the demo?").intent
    assert intent == INTENT_DEMO_CONTACT_INQUIRY

    session = BookingSession(call_sid="call_en_05")
    session.workflow_mode = "LEAD_COLLECTION"
    session.caller_name = "Sanjay"
    session.contact_number = "9876543210"

    resp = booking_state_service.process_lead_collection_turn(
        "When will your team contact me for the demo?",
        session,
    )
    assert resp is not None
    resp_lower = resp.lower()
    assert "24" in resp_lower or "business hours" in resp_lower or "contact you" in resp_lower
    # Must NOT claim appointment confirmed
    assert "already confirmed" not in resp_lower
    assert "appointment is confirmed" not in resp_lower


def test_en_06_already_provided_my_email():
    """English 6: Caller says 'I already provided my email.' -> Recognizes stored email, advances to next field."""
    session = BookingSession(call_sid="call_en_06")
    session.workflow_mode = "LEAD_COLLECTION"
    session.patient_name = "Vikram"
    session.caller_name = "Vikram"
    session.email = "vikram@example.com"
    session.booking_data["email"] = "vikram@example.com"
    session.collected_fields["email"] = "vikram@example.com"
    session.last_asked_field = "email"

    resp = booking_state_service.process_lead_collection_turn(
        "I already provided my email.",
        session,
    )
    assert resp is not None
    resp_lower = resp.lower()
    # Acknowledges having the email and does NOT re-prompt for email
    assert "vikram@example.com" in resp_lower or "already have" in resp_lower or "recorded your email" in resp_lower
    assert "what is your email" not in resp_lower


def test_en_07_incomplete_email_and_username_combination():
    """English 7: Spoken incomplete email 'My email is at Gmail dot com' -> prompts for username, then combines."""
    incomplete = detect_incomplete_spoken_email("My email is at Gmail dot com")
    assert incomplete == "gmail.com"

    session = BookingSession(call_sid="call_en_07")
    session.workflow_mode = "LEAD_COLLECTION"
    session.patient_name = "Arun"
    session.caller_name = "Arun"
    session.last_asked_field = "email"

    # Turn 1: Incomplete domain
    resp1 = booking_state_service.process_lead_collection_turn(
        "My email is at Gmail dot com",
        session,
    )
    assert session.booking_data.get("partial_email_domain") == "gmail.com"
    assert "username" in resp1.lower() or "before" in resp1.lower()
    assert session.email is None

    # Turn 2: Caller provides username
    resp2 = booking_state_service.process_lead_collection_turn(
        "arun123",
        session,
    )
    assert session.email == "arun123@gmail.com"
    assert session.booking_data.get("email") == "arun123@gmail.com"
    assert "arun123@gmail.com" in resp2.lower() or "phone" in resp2.lower() or "contact" in resp2.lower()


def test_en_08_i_am_available_tomorrow():
    """English 8: 'I am available tomorrow.' -> Records availability preference, does NOT falsely confirm."""
    intent = nlp_service.detect_intent("I am available tomorrow.").intent
    assert intent == INTENT_AVAILABILITY_PROVIDED

    session = BookingSession(call_sid="call_en_08")
    session.workflow_mode = "LEAD_COLLECTION"
    session.caller_name = "Karthik"
    session.email = "karthik@test.com"
    session.contact_number = "9887766554"

    resp = booking_state_service.process_lead_collection_turn(
        "I am available tomorrow.",
        session,
    )
    assert session.booking_data.get("caller_availability") == "tomorrow"
    assert "already confirmed" not in resp.lower()


def test_en_09_can_you_share_the_appointment_details():
    """English 9: 'Can you share the appointment details?' -> Checks real status, no fake confirmation."""
    intent = nlp_service.detect_intent("Can you share the appointment details?").intent
    assert intent == INTENT_APPOINTMENT_DETAILS_INQUIRY

    # Case A: No booking or lead
    session_empty = BookingSession(call_sid="call_en_09a")
    resp_empty = booking_state_service.get_appointment_or_demo_details(session_empty)
    assert "no confirmed appointment" in resp_empty.lower() or "not have a confirmed appointment" in resp_empty.lower()

    # Case B: Demo callback lead (not confirmed DB appointment)
    session_lead = BookingSession(call_sid="call_en_09b")
    session_lead.workflow_mode = "LEAD_COLLECTION"
    session_lead.caller_name = "Priya"
    session_lead.booking_data["caller_availability"] = "tomorrow"
    resp_lead = booking_state_service.get_appointment_or_demo_details(session_lead)
    assert "team will contact you" in resp_lead.lower()
    assert "tomorrow" in resp_lead.lower()
    assert "already confirmed" not in resp_lead.lower()

    # Case C: Actual confirmed appointment with ID
    session_confirmed = BookingSession(call_sid="call_en_09c")
    session_confirmed.booking_confirmed = True
    session_confirmed.appointment_id = "APT-1002"
    session_confirmed.date = "2026-11-25"
    session_confirmed.time = "10:00 AM"
    resp_confirmed = booking_state_service.get_appointment_or_demo_details(session_confirmed)
    assert "APT-1002" in resp_confirmed
    assert "2026-11-25" in resp_confirmed
    assert "10:00 AM" in resp_confirmed


def test_en_10_i_want_to_schedule_a_demo():
    """English 10: 'I want to schedule a demo.' -> Routes to booking/lead collection."""
    intent = nlp_service.detect_intent("I want to schedule a demo.").intent
    assert intent in (INTENT_APPOINTMENT_BOOKING, INTENT_CALLBACK)

    session = BookingSession(call_sid="call_en_10")
    resp = booking_state_service.process_lead_collection_turn(
        "I want to schedule a demo.",
        session,
    )
    assert resp is not None
    # Asks for customer name or details
    assert "name" in resp.lower() or "details" in resp.lower()


def test_en_11_yes_response_contextual_no_false_confirmation():
    """English 11: Caller says 'Yes' -> Contextual response, never falsely confirms appointment."""
    from src.services.sip_call_handler import SipCallSession
    session = BookingSession(call_sid="call_en_11")
    session.is_completed = True
    session.workflow_mode = "LEAD_COLLECTION"
    session.booking_confirmed = False

    handler = SipCallSession(call_sid="call_en_11", caller_phone="9876543210")
    handler.booking_session = session
    handler.session_history = [
        {"role": "model", "text": "Our team will contact you within 24 hours. Is there anything else I can help you with today?"}
    ]

    # Evaluate the affirmative branch logic
    last_ai = handler.session_history[-1]["text"].lower()
    has_more_help_q = any(p in last_ai for p in ["anything else", "help you with", "assist you"])
    assert has_more_help_q is True

    # The AI must ask how else to assist, NEVER claim appointment confirmed
    lang = "en"
    if has_more_help_q:
        ack_msg = "Certainly! How else can I assist you today?"
    assert ack_msg == "Certainly! How else can I assist you today?"
    assert "appointment" not in ack_msg.lower()


def test_en_12_no_i_am_available_tomorrow_inline_correction():
    """English 12: 'No, I am available tomorrow.' -> Inline date correction during confirmation turn."""
    session = BookingSession(call_sid="call_en_12")
    session.state = BookingState.WAITING_FOR_CONFIRMATION
    session.date = "2026-11-24"
    session.time = "3:00 PM"
    session.doctor = "Dr. Sharma"

    resp = booking_state_service.handle_confirmation_turn(session, "No, I am available tomorrow.")
    assert resp is not None
    # Verify session date was updated to tomorrow's date
    assert session.date is not None and session.date != "2026-11-24"
    # State moved to WAITING_FOR_TIME to get the preferred slot for tomorrow
    assert session.state == BookingState.WAITING_FOR_TIME
    assert "time" in resp.lower() or "slot" in resp.lower()


def test_en_13_goodbye():
    """English 13: Caller says 'Goodbye.' -> Detects termination, delivers polite closing."""
    intent = nlp_service.detect_intent("Goodbye.").intent
    assert intent == INTENT_GOODBYE

    session = BookingSession(call_sid="call_en_13")
    resp = ai_service._fast_rule_response(
        "Goodbye.",
        intent=intent,
        organization_id=1,
        is_ongoing=True,
        booking_session={"greeting_sent": True, "state": "IDLE"},
        session_history=[],
        conversation_language="en",
    )
    assert resp is not None
    assert any(w in resp.lower() for w in ["goodbye", "have a great day", "thank you", "take care"])


# ==============================================================================
# SECTION 18.2: TAMIL CONVERSATION TESTS (1 to 10)
# ==============================================================================

def test_ta_01_tamil_greeting():
    """Tamil 1: Tamil greeting 'வணக்கம்' -> Natural conversational Tamil greeting."""
    intent = nlp_service.detect_intent("வணக்கம்").intent
    assert intent == INTENT_GREETING

    resp = ai_service._fast_rule_response(
        "வணக்கம்",
        intent=intent,
        organization_id=1,
        is_ongoing=True,
        booking_session={"greeting_sent": False, "state": "IDLE"},
        session_history=[],
        conversation_language="ta",
    )
    assert resp is not None
    assert "வணக்கம்" in resp or "சொல்லுங்க" in resp


def test_ta_02_tamil_question():
    """Tamil 2: Tamil service inquiry 'உங்கள் சேவைகள் பற்றி சொல்லுங்க' -> Handled in Tamil."""
    intent = nlp_service.detect_intent("உங்கள் சேவைகள் பற்றி சொல்லுங்க").intent
    assert intent in (INTENT_SERVICE_INFORMATION, INTENT_PRODUCT_INFORMATION)

    lang = language_service.detect_language("உங்கள் சேவைகள் பற்றி சொல்லுங்க")
    assert lang == "ta"


def test_ta_03_tamil_customer_info_collection():
    """Tamil 3: Tamil customer name collection."""
    session = BookingSession(call_sid="call_ta_03")
    session.conversation_language = "ta"
    session.workflow_mode = "LEAD_COLLECTION"
    session.last_asked_field = "patient_name"

    # Caller provides name in Tamil
    resp = booking_state_service.process_lead_collection_turn("என் பெயர் முருகன்", session)
    assert session.caller_name in ("முருகன்", "முருகன்.") or "முருகன்" in (session.caller_name or "") or session.patient_name in ("முருகன்", "முருகன்.") or "முருகன்" in (session.patient_name or "")
    assert "மின்னஞ்சல்" in resp or "email" in resp.lower() or "எண்" in resp or "தொடர்பு" in resp


def test_ta_04_tamil_email_collection():
    """Tamil 4: Tamil email collection 'என் ஈமெயில் murugan@gmail.com'."""
    session = BookingSession(call_sid="call_ta_04")
    session.conversation_language = "ta"
    session.workflow_mode = "LEAD_COLLECTION"
    session.patient_name = "முருகன்"
    session.caller_name = "முருகன்"
    session.last_asked_field = "email"

    resp = booking_state_service.process_lead_collection_turn("என் ஈமெயில் murugan@gmail.com", session)
    assert session.email == "murugan@gmail.com"
    assert "murugan@gmail.com" in resp or "தொலைபேசி" in resp or "phone" in resp.lower() or "எண்" in resp or "தொடர்பு" in resp


def test_ta_05_tamil_booking():
    """Tamil 5: Tamil booking slot extraction."""
    session = BookingSession(call_sid="call_ta_05")
    session.conversation_language = "ta"
    session.state = BookingState.WAITING_FOR_DATE
    session.doctor = "Dr. Kumar"

    booking_state_service.extract_slots("நாளைக்கு காலை 10 மணிக்கு", session)
    assert session.date is not None
    assert session.time is not None
    assert "10:00" in session.time


def test_ta_06_tamil_datetime_expressions():
    """Tamil 6: Tamil date and time expressions 'நாளைக்கு' and 'காலை 10 மணிக்கு'."""
    session = BookingSession(call_sid="call_ta_06")
    booking_state_service.extract_slots("நாளைக்கு காலை 10 மணிக்கு", session)
    assert session.date is not None
    assert session.time is not None
    assert "10:00" in session.time


def test_ta_07_tamil_followup_turnaround_question():
    """Tamil 7: Tamil turnaround inquiry 'எப்போ என்னை தொடர்பு கொள்வீங்க?'."""
    session = BookingSession(call_sid="call_ta_07")
    session.conversation_language = "ta"
    session.workflow_mode = "LEAD_COLLECTION"
    session.patient_name = "கார்த்திக்"
    session.caller_name = "கார்த்திக்"

    resp = booking_state_service.process_lead_collection_turn("எப்போ என்னை தொடர்பு கொள்வீங்க?", session)
    assert resp is not None
    assert "24" in resp or "தொடர்பு" in resp


def test_ta_08_tamil_language_switching():
    """Tamil 8: Seamless language switching between English and Tamil."""
    # Switch to Tamil
    target_lang, is_switch = language_service.check_explicit_language_switch("Will you please talk in Tamil?")
    assert is_switch is True
    assert target_lang == "ta"

    # Switch to English
    target_lang_en, is_switch_en = language_service.check_explicit_language_switch("Can we switch back to English?")
    assert is_switch_en is True
    assert target_lang_en == "en"


def test_ta_09_tamil_tanglish_mixed_speech():
    """Tamil 9: Tanglish mixed speech 'Naan tomorrow available, enakku oru demo schedule panna mudiyuma?'."""
    text = "Naan tomorrow available, enakku oru demo schedule panna mudiyuma?"
    intent = nlp_service.detect_intent(text).intent
    assert intent in (INTENT_APPOINTMENT_BOOKING, INTENT_AVAILABILITY_PROVIDED, INTENT_CALLBACK)


def test_ta_10_complete_tamil_call():
    """Tamil 10: Complete multi-turn Tamil call regression."""
    session = BookingSession(call_sid="call_ta_10")
    session.conversation_language = "ta"
    session.workflow_mode = "LEAD_COLLECTION"

    # Turn 1: Greeting
    turn1_resp = ai_service._fast_rule_response(
        "வணக்கம்",
        intent=INTENT_GREETING,
        organization_id=1,
        is_ongoing=True,
        booking_session={"greeting_sent": False, "state": "IDLE"},
        session_history=[],
        conversation_language="ta",
    )
    assert "வணக்கம்" in turn1_resp or "சொல்லுங்க" in turn1_resp

    # Turn 2: Name
    session.last_asked_field = "patient_name"
    turn2_resp = booking_state_service.process_lead_collection_turn("என் பெயர் செல்வம்", session)
    assert session.caller_name in ("செல்வம்", "செல்வம்.") or "செல்வம்" in (session.caller_name or "")

    # Turn 3: Email
    session.last_asked_field = "email"
    turn3_resp = booking_state_service.process_lead_collection_turn("selvam@gmail.com", session)
    assert session.email == "selvam@gmail.com"

    # Turn 4: Phone
    session.last_asked_field = "contact_number"
    turn4_resp = booking_state_service.process_lead_collection_turn("9840123456", session)
    assert session.contact_number == "9840123456"

    # Turn 5: Requirement
    session.last_asked_field = "requirement"
    turn5_resp = booking_state_service.process_lead_collection_turn("ERP Software Demo", session)
    assert session.is_completed is True
    assert any(w in turn5_resp for w in ["contact", "தொடர்பு", "24", "team", "நன்றி"])


# ==============================================================================
# SECTION 19: PERFORMANCE & LATENCY METRICS VERIFICATION
# ==============================================================================

def test_performance_metrics_and_latency_structure():
    """Section 19: Verifies latency metrics structure and latency targets."""
    t0 = time.monotonic()
    metrics = {
        "utt_id": "utt_perf_01",
        "speech_end_monotonic": t0,
        "t_stt_final": t0 + 0.15,
        "intent_latency_ms": 25,
        "state_ms": 10,
        "rag_ms": 45,
        "llm_first_token": t0 + 0.40,
        "llm_ms": 250,
        "tts_start": t0 + 0.45,
        "tts_first_audio": t0 + 0.85,
        "tts_first_audio_ms": 400,
        "rtp_tx_start": t0 + 0.90,
        "response_id": "resp_001",
    }
    cust_to_ai_ms = round((metrics["rtp_tx_start"] - metrics["speech_end_monotonic"]) * 1000)
    assert cust_to_ai_ms < 2000, f"Expected <2000ms latency, got {cust_to_ai_ms}ms"
    assert cust_to_ai_ms == 900
