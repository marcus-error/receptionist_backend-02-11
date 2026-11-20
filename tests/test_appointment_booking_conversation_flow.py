import pytest
import re
from unittest.mock import MagicMock, patch
from src.services.booking_state_service import (
    BookingStateService,
    BookingSession,
    BookingState,
    CORRECTION_AND_CONFIRMATION_STATES,
)


@pytest.fixture
def service():
    return BookingStateService()


@pytest.fixture
def demo_session(service):
    sess = service.get_session("test_flow_session_1")
    sess.booking_type = "SALES_DEMO"
    sess.business_type = "SOFTWARE"
    sess.workflow_type = "SALES_DEMO"
    sess.business_name = "Aminoid AI"
    sess.date = "Tomorrow"
    sess.time = "10:00 AM"
    sess.patient_name = "Sanjay"
    sess.caller_name = "Sanjay"
    sess.contact_number = "9876543210"
    sess.email = "sanjay@example.com"
    sess.availability_verified = True
    return sess


# -----------------------------------------------------------------------------
# ISSUE 1 & 3: Name Extraction & Business Description Separation
# -----------------------------------------------------------------------------
def test_name_extraction_preserves_caller_name_over_business_description(service):
    """
    Caller: "My name is Hyundai."
    Caller: "Sanjay."
    Caller: "I'm starting a new business."
    The system must NOT set patient_name = "Starting New Business".
    It must preserve "Sanjay" and capture the business description separately.
    """
    sess = service.get_session("test_name_sess")
    sess.booking_type = "SALES_DEMO"
    sess.date = "Tomorrow"
    sess.time = "10:00 AM"

    # Turn 1: "My name is Hyundai."
    service.process_booking_turn("My name is Hyundai.", sess)
    assert sess.patient_name == "Hyundai"

    # Turn 2: Caller corrects name: "Sanjay."
    service.process_booking_turn("Sanjay.", sess)
    assert sess.patient_name == "Sanjay"

    # Turn 3: Caller gives business description: "I'm starting a new business."
    service.process_booking_turn("I'm starting a new business.", sess)
    # Full name must remain Sanjay, NOT "Starting New Business"!
    assert sess.patient_name == "Sanjay"
    assert "business" in (sess.business_description or sess.booking_data.get("business_description", "")).lower()


def test_business_tokens_rejected_as_person_names():
    """
    Verifies that business, action, and conversation phrases are strictly rejected as person names.
    """
    assert not BookingStateService.is_valid_person_name("Starting New Business")
    assert not BookingStateService.is_valid_person_name("Running A Company")
    assert not BookingStateService.is_valid_person_name("Continue The Last Conversation")
    assert not BookingStateService.is_valid_person_name("Starting A Business")
    assert not BookingStateService.is_valid_person_name("New Startup")
    assert not BookingStateService.is_valid_person_name("Software Solution")

    # Genuine person names must pass
    assert BookingStateService.is_valid_person_name("Sanjay")
    assert BookingStateService.is_valid_person_name("Hyundai")
    assert BookingStateService.is_valid_person_name("Sanjay Kumar")
    assert BookingStateService.is_valid_person_name("Angie")


def test_unclear_name_asks_for_repeat(service):
    """
    When the system asks for patient name and the caller gives an unclear input (not a name),
    it must ask 'Could you please repeat your name?'.
    """
    sess = service.get_session("test_unclear_name_sess")
    sess.booking_type = "SALES_DEMO"
    sess.date = "Tomorrow"
    sess.time = "10:00 AM"
    sess.state = BookingState.COLLECTING_PATIENT_NAME
    sess.last_asked_field = "patient_name"

    resp = service.process_booking_turn("The model.", sess)
    assert "repeat your name" in resp.lower()
    assert sess.patient_name is None


def test_explicit_name_correction(service):
    """
    Explicit corrections like "No, my name is Sanjay" or "I said Sanjay" update the name immediately.
    """
    sess = service.get_session("test_exp_name_sess")
    sess.patient_name = "How"

    service.process_booking_turn("No, my name is Sanjay.", sess)
    assert sess.patient_name == "Sanjay"


# -----------------------------------------------------------------------------
# ISSUE 2: Resume / Context-Aware Conversation
# -----------------------------------------------------------------------------
def test_resume_conversation_intent(service):
    """
    When caller says "Continue the last conversation.", "Continue.", "Let's proceed.", "Okay, go ahead.",
    the active booking is resumed from its current state without restarting or misidentifying "continue" as a name.
    """
    sess = service.get_session("test_resume_sess")
    sess.booking_type = "SALES_DEMO"
    sess.date = "Tomorrow"
    sess.time = "10:00 AM"
    sess.state = BookingState.COLLECTING_PATIENT_NAME
    sess.last_asked_field = "patient_name"

    resp = service.process_booking_turn("Continue the last conversation.", sess)
    # Must NOT set patient_name = "Continue" or "Continue The Last Conversation"
    assert sess.patient_name is None
    # Must ask for the next missing field (patient name)
    assert "name" in resp.lower()

    # If caller has all fields and says "Continue."
    sess.patient_name = "Sanjay"
    sess.contact_number = "9876543210"
    sess.email = "sanjay@example.com"
    sess.state = BookingState.WAITING_FOR_CONFIRMATION

    resp2 = service.process_booking_turn("Continue.", sess)
    # Should route to confirmation or confirmation prompt
    assert any(w in resp2.lower() for w in ["confirm", "details", "proceed", "sanjay"])


# -----------------------------------------------------------------------------
# ISSUE 4 & 5: Appointment Correction Workflow (Field Selection & Value)
# -----------------------------------------------------------------------------
def test_correction_workflow_multi_step(service, demo_session):
    """
    Step 1: In WAITING_FOR_CONFIRMATION, caller says "No" / "Make changes" -> enters WAITING_FOR_CORRECTION_FIELD.
    Step 2: Caller says "Email." -> enters WAITING_FOR_CORRECTION_VALUE.
    Step 3: Caller gives new email -> updates ONLY email, preserves name/phone/date/time, displays updated summary.
    """
    demo_session.state = BookingState.WAITING_FOR_CONFIRMATION
    orig_name = demo_session.patient_name
    orig_phone = demo_session.contact_number
    orig_date = demo_session.date
    orig_time = demo_session.time

    # Step 1: Caller says "No"
    resp1 = service.handle_confirmation_turn(demo_session, "No, I want to make changes.")
    assert demo_session.state == BookingState.WAITING_FOR_CORRECTION_FIELD
    assert "which detail" in resp1.lower()

    # Step 2: Caller specifies the field: "Email."
    resp2 = service.handle_confirmation_turn(demo_session, "Email.")
    assert demo_session.state == BookingState.WAITING_FOR_CORRECTION_VALUE
    assert "email address" in resp2.lower()

    # Step 3: Caller provides new email: "angie@example.com"
    resp3 = service.handle_confirmation_turn(demo_session, "angie@example.com")
    assert demo_session.state == BookingState.WAITING_FOR_FINAL_CONFIRMATION
    assert demo_session.email == "angie@example.com"
    # Preserved all other fields!
    assert demo_session.patient_name == orig_name
    assert demo_session.contact_number == orig_phone
    assert demo_session.date == orig_date
    assert demo_session.time == orig_time
    assert "angie@example.com" in resp3
    assert "correct" in resp3.lower()


def test_correction_workflow_inline_field_and_value(service, demo_session):
    """
    Caller in WAITING_FOR_CONFIRMATION says: "Change my email to angie@example.com".
    AI immediately extracts new email, updates email, preserves other fields, and presents updated summary.
    """
    demo_session.state = BookingState.WAITING_FOR_CONFIRMATION
    orig_name = demo_session.patient_name
    orig_phone = demo_session.contact_number

    resp = service.handle_confirmation_turn(demo_session, "Change my email to angie@example.com")
    assert demo_session.email == "angie@example.com"
    assert demo_session.patient_name == orig_name
    assert demo_session.contact_number == orig_phone
    assert demo_session.state == BookingState.WAITING_FOR_FINAL_CONFIRMATION
    assert "angie@example.com" in resp


def test_correction_workflow_multiple_fields(service, demo_session):
    """
    Caller says "Change my email and appointment time".
    AI prompts for first field (email), then next field (time), without resetting.
    """
    demo_session.state = BookingState.WAITING_FOR_CONFIRMATION
    resp1 = service.handle_confirmation_turn(demo_session, "Change my email and appointment time")
    assert demo_session.state == BookingState.WAITING_FOR_CORRECTION_VALUE
    assert "email" in resp1.lower()
    assert "time" in demo_session.booking_data.get("pending_correction_fields", [])

    # Caller provides email
    resp2 = service.handle_confirmation_turn(demo_session, "newmail@example.com")
    assert demo_session.email == "newmail@example.com"
    # AI prompts for time next
    assert "time" in resp2.lower()
    assert demo_session.state == BookingState.WAITING_FOR_CORRECTION_VALUE

    # Caller provides time
    resp3 = service.handle_confirmation_turn(demo_session, "3:00 PM")
    assert demo_session.time == "3:00 PM"
    assert demo_session.state == BookingState.WAITING_FOR_FINAL_CONFIRMATION
    assert "newmail@example.com" in resp3
    assert "3:00 PM" in resp3


# -----------------------------------------------------------------------------
# ISSUE 6 & 8: Confirmation Intent Detection & Booking Execution Idempotency
# -----------------------------------------------------------------------------
def test_confirmation_routes_correctly(service, demo_session):
    """
    In WAITING_FOR_CONFIRMATION:
    - Negations / Changes route to correction mode
    - Cancellations route to BOOKING_CANCELLED
    - Affirmations route to execution
    """
    # Test Cancellation
    demo_session.state = BookingState.WAITING_FOR_CONFIRMATION
    resp_cancel = service.handle_confirmation_turn(demo_session, "Cancel the appointment")
    assert demo_session.state == BookingState.BOOKING_CANCELLED
    assert "cancelled" in resp_cancel.lower()

    # Reset session for affirmation test
    demo_session.state = BookingState.WAITING_FOR_CONFIRMATION
    demo_session.date = "Tomorrow"
    demo_session.time = "10:00 AM"
    demo_session.patient_name = "Sanjay"
    demo_session.contact_number = "9876543210"

    with patch.object(service, "_execute_db_booking", return_value="Your demo appointment is confirmed! Appt ID #101.") as mock_exec:
        resp_affirm = service.handle_confirmation_turn(demo_session, "Yes, please confirm.")
        assert mock_exec.called
        assert "confirmed" in resp_affirm.lower()


def test_idempotent_confirmation_no_duplicates(service, demo_session):
    """
    If caller repeats confirmation after appointment is already confirmed,
    reuse the confirmation and appointment ID without creating a duplicate booking.
    """
    demo_session.state = BookingState.BOOKING_CONFIRMED
    demo_session.is_completed = True
    demo_session.appointment_id = 999

    resp = service.handle_confirmation_turn(demo_session, "Yes, confirm it.")
    assert "#999" in resp
    assert "already confirmed" in resp.lower()


# -----------------------------------------------------------------------------
# ISSUE 7: Date & Time Validation
# -----------------------------------------------------------------------------
def test_unclear_date_asks_for_clarification(service):
    """
    When asked for date, if caller gives an unclear phrase like "The model.",
    the AI does NOT guess a date; it asks "Sorry, I didn't catch the date. Would tomorrow or another date work for you?".
    """
    sess = service.get_session("test_unclear_date_sess")
    sess.booking_type = "SALES_DEMO"
    sess.state = BookingState.WAITING_FOR_DATE
    sess.last_asked_field = "date"

    resp = service.process_booking_turn("The model.", sess)
    assert sess.date is None
    assert "didn't catch the date" in resp.lower()
    assert "tomorrow or another date" in resp.lower()


# -----------------------------------------------------------------------------
# ISSUE 9 & 10: PII Masking & Notification Status Tracking
# -----------------------------------------------------------------------------
def test_pii_masking():
    """
    Verifies that _mask_pii correctly masks emails, phone numbers, and names.
    """
    assert BookingStateService._mask_pii("sanjay@example.com") == "s****y@example.com"
    assert BookingStateService._mask_pii("9876543210") == "******3210"
    assert BookingStateService._mask_pii("+919876543210") == "********3210"
    assert BookingStateService._mask_pii("Sanjay") == "S****y"
    assert BookingStateService._mask_pii(None) == "None"


def test_correction_and_confirmation_states_membership():
    """
    Verifies all new correction and confirmation states are in CORRECTION_AND_CONFIRMATION_STATES.
    """
    assert BookingState.WAITING_FOR_CONFIRMATION in CORRECTION_AND_CONFIRMATION_STATES
    assert BookingState.WAITING_FOR_CORRECTION_FIELD in CORRECTION_AND_CONFIRMATION_STATES
    assert BookingState.WAITING_FOR_CORRECTION_VALUE in CORRECTION_AND_CONFIRMATION_STATES
    assert BookingState.WAITING_FOR_FINAL_CONFIRMATION in CORRECTION_AND_CONFIRMATION_STATES
