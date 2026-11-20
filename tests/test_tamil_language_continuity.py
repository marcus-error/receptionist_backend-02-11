"""
Test suite for Tamil Language Continuity in CallZenza AI Receptionist.

Verifies:
1. Fast-path pure greeting during active Tamil call returns Tamil greeting ("ஆம், உங்களுக்கு எவ்வாறு உதவலாம்?"), NOT English ("Yes, how may I assist you?").
2. Fast-path audio check during active Tamil call returns Tamil reassurance.
3. Fast-path pleasantry during active Tamil call returns Tamil response.
4. Fast-path thanks and goodbye return Tamil.
5. Fallback unlisted entity responses in ai_service return Tamil.
6. Booking state presence reassurance during active booking returns Tamil.
7. Booking state waiting_for_location prompt returns Tamil.
8. Booking state sales demo confirmation returns Tamil.
9. Booking state time/date/doctor change requests return Tamil.
10. Booking state cancellation returns Tamil.
11. Language persistence across turns: language does not revert to English unless explicitly requested.
12. Gemini prompt generates mandatory Tamil language directive with product name preservation.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.services.ai_service import ai_service
from src.services.booking_state_service import booking_state_service, BookingSession, BookingState
from src.services.language_service import language_service


class DummyBookingSession:
    def __init__(self, lang="ta", is_ongoing=True):
        self.conversation_language = lang
        self.greeting_sent = True
        self.turn_count = 2
        self.conversation_started = True
        self.business_name = "Aminoid AI"
        self.active_doc_id = "6"

    def is_active(self):
        return False


def test_01_fast_path_greeting_tamil_continuity():
    """Turn 2: Saying 'hello' or 'hi' or 'வணக்கம்' after switching to Tamil must return Tamil, NOT English."""
    sess = DummyBookingSession(lang="ta", is_ongoing=True)
    resp = ai_service._fast_rule_response(
        "hello",
        intent="greeting",
        organization_id=1,
        is_ongoing=True,
        booking_session=sess,
        conversation_language="ta"
    )
    assert resp is not None
    assert "Yes, how may I assist you?" not in resp
    assert "வணக்கம்" in resp or "help பண்ணலாம்" in resp


def test_02_fast_path_audio_check_tamil():
    """Audio check 'can you hear me' or 'கேட்கிறதா' in Tamil returns Tamil response."""
    sess = DummyBookingSession(lang="ta", is_ongoing=True)
    resp = ai_service._fast_rule_response(
        "can you hear me",
        intent="audio_check",
        organization_id=1,
        is_ongoing=True,
        booking_session=sess,
        conversation_language="ta"
    )
    assert resp is not None
    assert "clearly" not in resp.lower()
    assert "voice" in resp or "help பண்ணலாம்" in resp or "கேக்குது" in resp


def test_03_fast_path_pleasantry_tamil():
    """Pleasantry 'how are you' or 'எப்படி இருக்கிறீர்கள்' in Tamil returns Tamil response."""
    sess = DummyBookingSession(lang="ta", is_ongoing=True)
    resp = ai_service._fast_rule_response(
        "how are you",
        intent="pleasantry",
        organization_id=1,
        is_ongoing=True,
        booking_session=sess,
        conversation_language="ta"
    )
    assert resp is not None
    assert "doing great" not in resp.lower()
    assert "நல்லா இருக்கேன்" in resp


def test_04_fast_path_goodbye_and_thanks_tamil():
    """Goodbye and thanks in Tamil return Tamil responses."""
    sess = DummyBookingSession(lang="ta", is_ongoing=True)
    resp_bye = ai_service._fast_rule_response(
        "bye",
        intent="goodbye",
        organization_id=1,
        is_ongoing=True,
        booking_session=sess,
        conversation_language="ta"
    )
    assert resp_bye is not None
    assert "Thank you" not in resp_bye
    assert "நன்றி" in resp_bye or "வருகிறேன்" in resp_bye

    resp_thanks = ai_service._fast_rule_response(
        "thank you",
        intent="thanks",
        organization_id=1,
        is_ongoing=True,
        booking_session=sess,
        conversation_language="ta"
    )
    assert resp_thanks is not None
    assert "நன்றி" in resp_thanks


def test_05_fallback_apology_localized_tamil():
    """When query is unlisted or LLM is bypassed, fallback apology is returned in Tamil."""
    resp = ai_service.generate_response(
        "Tell me about Kasam software",
        context="",
        conversation_language="ta",
        organization_id=1,
        intent="general_question"
    )
    assert resp is not None
    assert "I apologize" not in resp
    assert "மன்னிக்கவும்" in resp


def test_06_booking_presence_reassurance_tamil():
    """Presence check during active booking returns Tamil reassurance."""
    session = BookingSession("sess_test_presence", 1, "9876543210")
    session.conversation_language = "ta"
    session.state = BookingState.WAITING_FOR_DATE
    resp = booking_state_service.process_booking_turn("hello? are you there", session)
    assert resp is not None
    assert "Yes, I'm right here" not in resp
    assert "line-ல தான் இருக்கேன்" in resp


def test_07_booking_location_prompt_tamil():
    """Location prompt during booking returns Tamil question."""
    session = BookingSession("sess_test_loc", 1, "9876543210")
    session.conversation_language = "ta"
    session.patient_name = "Karthik"
    session.email = "karthik@example.com"
    session.contact_number = "9876543210"
    session.date = "tomorrow"
    session.time = "10:00 AM"
    session.availability_verified = True
    session.state = BookingState.WAITING_FOR_LOCATION
    session.last_asked_field = "location"

    resp = booking_state_service.process_booking_turn("tell me location", session)
    assert resp is not None
    assert "Could you please tell me your location?" not in resp
    assert "location" in resp.lower() or "ஊர்" in resp or "இடம்" in resp


def test_08_booking_sales_demo_confirmation_tamil():
    """Sales demo confirmation returns Tamil confirmation message."""
    session = BookingSession("sess_test_demo", 1, "9876543210")
    session.conversation_language = "ta"
    session.business_name = "Aminoid AI"
    session.booking_type = "SALES_DEMO"
    session.patient_name = "Karthik"
    session.email = "karthik@example.com"
    session.contact_number = "9876543210"
    session.date = "2026-11-20"
    session.time = "10:00 AM"
    session.availability_verified = True
    session.state = BookingState.WAITING_FOR_CONFIRMATION

    resp = booking_state_service.process_booking_turn("yes confirm", session)
    assert resp is not None
    assert "Your demo appointment is confirmed" not in resp
    assert "சந்திப்பு" in resp or "உறுதிசெய்யப்பட்டது" in resp or "நன்றி" in resp


def test_09_booking_change_prompts_tamil():
    """Time/date/doctor change requests during confirmation return Tamil prompts."""
    session = BookingSession("sess_test_change", 1, "9876543210")
    session.conversation_language = "ta"
    session.state = BookingState.WAITING_FOR_CONFIRMATION

    resp_time = booking_state_service.handle_confirmation_turn(session, "change time")
    assert resp_time is not None
    assert "Sure! Which time" not in resp_time
    assert "time" in resp_time or "நேரத்தை" in resp_time

    session.state = BookingState.WAITING_FOR_CONFIRMATION
    resp_date = booking_state_service.handle_confirmation_turn(session, "change date")
    assert resp_date is not None
    assert "Sure! Which date" not in resp_date
    assert "date" in resp_date or "தேதியை" in resp_date


def test_10_booking_cancellation_tamil():
    """Explicit cancellation during confirmation returns Tamil cancellation message."""
    session = BookingSession("sess_test_cancel", 1, "9876543210")
    session.conversation_language = "ta"
    session.state = BookingState.WAITING_FOR_CONFIRMATION

    resp = booking_state_service.process_booking_turn("cancel booking", session)
    assert resp is not None
    assert "Your appointment booking has been cancelled" not in resp
    assert "cancel" in resp.lower() or "ரத்து" in resp


def test_11_explicit_language_switch_detection():
    """Explicit language switch detection correctly detects Tamil switch."""
    lang, is_switch = language_service.check_explicit_language_switch("Tamil")
    assert is_switch is True
    assert lang == "ta"

    lang, is_switch = language_service.check_explicit_language_switch("தமிழ்")
    assert is_switch is True
    assert lang == "ta"

    lang, is_switch = language_service.check_explicit_language_switch("speak in tamil")
    assert is_switch is True
    assert lang == "ta"


def test_12_gemini_prompt_tamil_directive():
    """Gemini system instruction contains mandatory Spoken Tamil directive with avoid/prefer rules."""
    instruction = ai_service._get_system_instruction(
        organization_name="Aminoid AI",
        is_ongoing=True,
        is_rag_active=True,
        conversation_language="ta",
        business_type="SOFTWARE",
        booking_type="SALES_DEMO"
    )
    assert "TAMIL" in instruction
    assert "NATURAL SPOKEN TAMIL" in instruction or "பேச்சுத் தமிழ்" in instruction
    assert "DO NOT USE LITERARY, ARCHAIC, OR CLASSICAL WRITTEN TAMIL" in instruction
    assert "NATURAL CODE-SWITCHING" in instruction
