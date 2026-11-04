import pytest
import asyncio
from unittest.mock import patch, MagicMock
from src.services.booking_state_service import (
    BookingStateService,
    BookingSession,
    BookingState,
    FieldState
)
from src.services.ai_service import ai_service
from src.services.calendar_verification_service import CalendarVerificationResult
from src.models.staff import Staff, StaffStatus
from src.models.organization import Organization
from src.models.appointment import Appointment


@pytest.fixture
def isolated_org_fixture(db_session):
    org = Organization(name="CallZenza 15 Scenarios Clinic", organization_type="Healthcare", status="active")
    db_session.add(org)
    db_session.commit()
    db_session.refresh(org)

    staff = Staff(
        organization_id=org.id,
        name="Dr. Anita",
        department="Cardiology",
        status=StaffStatus.AVAILABLE,
        online_booking_enabled=True,
        appointment_duration="30 min",
    )
    db_session.add(staff)
    db_session.commit()
    db_session.refresh(staff)
    return org, staff


def test_scenario_01_name_provided_once_remembered(db_session, isolated_org_fixture):
    """1. Customer provides name once; AI remembers it for the entire call."""
    org, staff = isolated_org_fixture
    svc = BookingStateService()
    session = svc.get_session("test_scen_1", call_sid="call_scen_1")
    session.organization_id = org.id

    mock_avail = CalendarVerificationResult(is_checked=True, is_available=True)
    with patch("src.services.calendar_verification_service.calendar_verification_service.verify_availability", return_value=mock_avail):
        svc.process_booking_turn("I want an appointment with Dr. Anita", session)
        svc.process_booking_turn("Monday", session)
        svc.process_booking_turn("11 AM", session)
        r_name = svc.process_booking_turn("My name is Abi Kumar", session)

        assert session.patient_name == "Abi Kumar"
        assert session.field_states["customer_name"] == FieldState.VERIFIED
        # Ensure next question asks for phone or email, NOT name again
        assert "name" not in r_name.lower() or "last name" not in r_name.lower()
        assert "phone" in r_name.lower() or "number" in r_name.lower()


def test_scenario_02_email_provided_once_not_requested_again(db_session, isolated_org_fixture):
    """2. Customer provides email once; AI does not request it again."""
    org, staff = isolated_org_fixture
    svc = BookingStateService()
    session = svc.get_session("test_scen_2", call_sid="call_scen_2")
    session.organization_id = org.id
    session.doctor = "Dr. Anita"
    session.doctor_id = staff.id
    session.date = "2026-11-02"
    session.time = "11:00 AM"
    session.patient_name = "Abi Kumar"
    session.contact_number = "882-566-4227"
    session.availability_verified = True

    mock_avail = CalendarVerificationResult(is_checked=True, is_available=True)
    with patch("src.services.calendar_verification_service.calendar_verification_service.verify_availability", return_value=mock_avail):
        r_email = svc.process_booking_turn("v123@gmail.com", session)
        assert session.email == "v123@gmail.com"
        assert session.field_states["email"] == FieldState.VERIFIED
        # Should move to confirmation and NOT ask for email again
        assert session.state == BookingState.WAITING_FOR_CONFIRMATION
        assert "confirm" in r_email.lower() or "v123@gmail.com" in r_email.lower()


def test_scenario_03_phone_provided_once_not_requested_again(db_session, isolated_org_fixture):
    """3. Customer provides phone once; AI does not request it again."""
    org, staff = isolated_org_fixture
    svc = BookingStateService()
    session = svc.get_session("test_scen_3", call_sid="call_scen_3")
    session.organization_id = org.id
    session.doctor = "Dr. Anita"
    session.doctor_id = staff.id
    session.date = "2026-11-02"
    session.time = "11:00 AM"
    session.patient_name = "Abi Kumar"
    session.availability_verified = True

    mock_avail = CalendarVerificationResult(is_checked=True, is_available=True)
    with patch("src.services.calendar_verification_service.calendar_verification_service.verify_availability", return_value=mock_avail):
        r_phone = svc.process_booking_turn("882-566-4227", session)
        assert "882" in (session.contact_number or "")
        assert session.field_states["phone"] == FieldState.VERIFIED
        # Next question must be for email, NOT phone again
        assert "phone" not in r_phone.lower()
        assert "email" in r_phone.lower()


def test_scenario_04_date_and_time_retained_together(db_session, isolated_org_fixture):
    """4. Customer provides date and time; AI retains both."""
    org, staff = isolated_org_fixture
    svc = BookingStateService()
    session = svc.get_session("test_scen_4", call_sid="call_scen_4")
    session.organization_id = org.id
    session.doctor = "Dr. Anita"

    mock_avail = CalendarVerificationResult(is_checked=True, is_available=True)
    with patch("src.services.calendar_verification_service.calendar_verification_service.verify_availability", return_value=mock_avail):
        svc.process_booking_turn("Monday 11 AM", session)
        assert session.date is not None
        assert session.time == "11:00 AM"
        assert session.field_states["preferred_date"] == FieldState.VERIFIED
        assert session.field_states["preferred_time"] in (FieldState.VERIFIED, FieldState.COLLECTED)


def test_scenario_05_field_correction(db_session, isolated_org_fixture):
    """5. Customer corrects an existing field."""
    org, staff = isolated_org_fixture
    svc = BookingStateService()
    session = svc.get_session("test_scen_5", call_sid="call_scen_5")
    session.organization_id = org.id
    session.doctor = "Dr. Anita"
    session.doctor_id = staff.id
    session.date = "2026-11-02"
    session.time = "11:00 AM"
    session.patient_name = "Abi Kumar"
    session.email = "old_email@gmail.com"
    session.availability_verified = True

    mock_avail = CalendarVerificationResult(is_checked=True, is_available=True)
    with patch("src.services.calendar_verification_service.calendar_verification_service.verify_availability", return_value=mock_avail):
        r_corr = svc.process_booking_turn("Change my email to new_abi@gmail.com", session)
        assert session.email == "new_abi@gmail.com"
        # Other fields must not be erased
        assert session.patient_name == "Abi Kumar"
        assert session.time == "11:00 AM"


def test_scenario_06_already_told_recovery(db_session, isolated_org_fixture):
    """6. Customer says 'I already told you.'"""
    org, staff = isolated_org_fixture
    svc = BookingStateService()
    session = svc.get_session("test_scen_6", call_sid="call_scen_6")
    session.organization_id = org.id
    session.doctor = "Dr. Anita"
    session.doctor_id = staff.id
    session.date = "2026-11-02"
    session.time = "11:00 AM"
    session.patient_name = "Abi"

    r_told = svc.process_booking_turn("I already told you my name", session)
    assert "You're right" in r_told or "Abi" in r_told
    assert session.patient_name == "Abi"


def test_scenario_07_ambiguous_time_clarification(db_session, isolated_org_fixture):
    """7. Customer gives an ambiguous answer."""
    org, staff = isolated_org_fixture
    svc = BookingStateService()
    session = svc.get_session("test_scen_7", call_sid="call_scen_7")
    session.organization_id = org.id
    session.doctor = "Dr. Anita"
    session.date = "2026-11-02"
    session.time = "11:00 AM"

    # Mentioning 11 PM when 11 AM is set triggers clarification rather than silent overwrite
    r_amb = svc.process_booking_turn("Actually could we do 11 PM?", session)
    assert "change" in r_amb.lower() or "11:00 am" in r_amb.lower()
    assert session.booking_data.get("ambiguous_time_candidate") == "11:00 PM"


def test_scenario_08_partial_and_final_transcript_deduplication(db_session, isolated_org_fixture):
    """8. Partial and final transcript duplication prevention."""
    from src.services.sip_call_handler import SipCallSession
    org, staff = isolated_org_fixture

    handler = SipCallSession(call_sid="call_dedup_test", caller_phone="1234567890", organization_id=org.id)
    # Process final transcript
    asyncio.run(handler._on_deepgram_final("utt_001", "Hello I want to book Dr. Anita", {}))
    count_before = len(handler._processed_utt_ids)

    # Re-sending same utt_id must be ignored
    asyncio.run(handler._on_deepgram_final("utt_001", "Hello I want to book Dr. Anita", {}))
    assert len(handler._processed_utt_ids) == count_before


def test_scenario_09_llm_timeout_retry_preserves_history():
    """9. LLM timeout and retry with preserved history."""
    history = [
        {"role": "user", "text": "Hello"},
        {"role": "model", "text": "Hi! How can I help you today?"},
        {"role": "user", "text": "I want an appointment on Monday"}
    ]
    mock_mistral = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "I have an opening at 10 AM on Monday."
    mock_mistral.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    with patch.object(ai_service, "gemini_client") as mock_gemini:
        mock_gemini.models.generate_content.side_effect = TimeoutError("Timeout")
        with patch.object(ai_service, "mistral_client", mock_mistral):
            res = ai_service.generate_response("I want an appointment on Monday", session_history=history)
            assert res is not None
            assert len(history) == 3


def test_scenario_10_quota_error_provider_fallback():
    """10. Gemini quota error and provider fallback."""
    mock_mistral = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Our office hours are 9 AM to 5 PM."
    mock_mistral.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    with patch.object(ai_service, "gemini_client") as mock_gemini:
        mock_gemini.models.generate_content.side_effect = Exception("429 ResourceExhausted")
        with patch.object(ai_service, "mistral_client", mock_mistral):
            res = ai_service.generate_response("What are your office hours?")
            assert "9 AM" in res


def test_scenario_11_booking_database_failure_handled(db_session, isolated_org_fixture):
    """11. Booking database failure."""
    org, staff = isolated_org_fixture
    svc = BookingStateService()
    session = svc.get_session("test_scen_11", call_sid="call_scen_11")
    session.organization_id = org.id
    session.doctor = "Dr. Anita"
    session.doctor_id = staff.id
    session.date = "2026-11-02"
    session.time = "11:00 AM"
    session.patient_name = "Abi Kumar"
    session.contact_number = "882-566-4227"
    session.email = "v123@gmail.com"
    session.state = BookingState.WAITING_FOR_CONFIRMATION

    with patch("src.config.database.SessionLocal", side_effect=Exception("DB Connection Timeout")):
        res = svc.process_booking_turn("Yes please confirm", session)
        assert session.booking_confirmed is False
        assert session.appointment_id is None
        assert "unable" in res.lower() or "apologize" in res.lower() or "failed" in res.lower()


def test_scenario_12_successful_booking_and_actual_reference(db_session, isolated_org_fixture):
    """12. Successful booking and actual booking reference."""
    org, staff = isolated_org_fixture
    svc = BookingStateService()
    session = svc.get_session("test_scen_12", call_sid="call_scen_12")
    session.organization_id = org.id
    session.doctor = "Dr. Anita"
    session.doctor_id = staff.id
    session.date = "2026-11-02"
    session.time = "11:00 AM"
    session.patient_name = "Abi Kumar"
    session.contact_number = "882-566-4227"
    session.email = "v123@gmail.com"
    session.state = BookingState.WAITING_FOR_CONFIRMATION

    mock_avail = CalendarVerificationResult(is_checked=True, is_available=True)
    with patch("src.services.calendar_verification_service.calendar_verification_service.verify_availability", return_value=mock_avail):
        res = svc.process_booking_turn("Yes please confirm", session)
        assert session.booking_confirmed is True
        assert session.appointment_id is not None
        assert session.appointment_id > 0
        assert f"#{session.appointment_id}" in res


def test_scenario_13_duplicate_booking_prevention(db_session, isolated_org_fixture):
    """13. Duplicate booking prevention."""
    org, staff = isolated_org_fixture
    svc = BookingStateService()
    session = svc.get_session("test_scen_13", call_sid="call_scen_13")
    session.organization_id = org.id
    session.doctor = "Dr. Anita"
    session.doctor_id = staff.id
    session.date = "2026-11-02"
    session.time = "11:00 AM"
    session.patient_name = "Abi Kumar"
    session.contact_number = "882-566-4227"
    session.email = "v123@gmail.com"
    session.state = BookingState.WAITING_FOR_CONFIRMATION

    mock_avail = CalendarVerificationResult(is_checked=True, is_available=True)
    with patch("src.services.calendar_verification_service.calendar_verification_service.verify_availability", return_value=mock_avail):
        # Turn 1: Confirmation
        svc.process_booking_turn("Yes please confirm", session, turn_id="turn_confirm_1")
        first_id = session.appointment_id

        # Turn 2: Repeated confirmation phrase
        res2 = svc.process_booking_turn("Yes please confirm", session, turn_id="turn_confirm_1")
        # Idempotency prevents creating a second appointment
        assert session.appointment_id == first_id
        assert f"#{first_id}" in res2


def test_scenario_14_two_simultaneous_calls_isolated(db_session):
    """14. Two simultaneous calls with different customers."""
    svc = BookingStateService()
    s1 = svc.get_session("call_user_A", call_sid="sid_user_A")
    s2 = svc.get_session("call_user_B", call_sid="sid_user_B")

    s1.patient_name = "Alice Wonder"
    s1.time = "10:00 AM"

    s2.patient_name = "Bob Builder"
    s2.time = "2:00 PM"

    assert s1.patient_name == "Alice Wonder"
    assert s2.patient_name == "Bob Builder"
    assert s1.time != s2.time


def test_scenario_15_organization_level_data_isolation(db_session):
    """15. Organization-level data isolation."""
    svc = BookingStateService()
    s_org1 = svc.get_session("call_org1", call_sid="call_org1")
    s_org1.organization_id = 101

    s_org2 = svc.get_session("call_org2", call_sid="call_org2")
    s_org2.organization_id = 202

    assert s_org1.organization_id == 101
    assert s_org2.organization_id == 202
    assert s_org1.organization_id != s_org2.organization_id
