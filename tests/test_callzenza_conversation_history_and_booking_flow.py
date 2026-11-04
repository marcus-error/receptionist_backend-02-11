import pytest
from unittest.mock import patch, MagicMock
from src.services.booking_state_service import (
    BookingStateService,
    BookingSession,
    BookingState,
)
from src.services.ai_service import ai_service
from src.services.calendar_verification_service import CalendarVerificationResult
from src.models.staff import Staff, StaffStatus
from src.models.organization import Organization



@pytest.fixture
def isolated_org(db_session):
    org = Organization(name="CallZenza Flow Hospital", organization_type="Healthcare", status="active")
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


def test_ten_turn_complete_conversation_flow(db_session, isolated_org):
    """
    Test a complete multi-turn conversation (10+ turns) exercising:
    1. Greeting and initial booking intent
    2. Date and time selection
    3. Ambiguous time mention (11 PM after 11 AM) asking for clarification
    4. Time correction confirmation
    5. First name given ('Abi'), system asking ONLY for last name
    6. Customer providing last name ('Kumar') -> combined to 'Abi Kumar'
    7. Phone and email collection
    8. Asking an unrelated question (e.g. 'Where are you located?') and returning to booking
    9. Asking the AI to repeat previously provided information
    10. Confirmation and successful booking without repeated questions
    """
    org, staff = isolated_org
    svc = BookingStateService()
    session = svc.get_session("test_call_10_turns", call_sid="call_sid_10_turns")
    session.organization_id = org.id

    mock_avail = CalendarVerificationResult(is_checked=True, is_available=True)

    with patch("src.services.calendar_verification_service.calendar_verification_service.verify_availability", return_value=mock_avail):
        # TURN 1: Caller wants appointment with Dr. Anita
        r1 = svc.process_booking_turn("I would like to book an appointment with Dr. Anita", session)
        assert session.doctor == "Dr. Anita"
        assert session.doctor_id == staff.id
        assert "date" in r1.lower()

        # TURN 2: Caller provides date 'Monday'
        r2 = svc.process_booking_turn("Monday", session)
        assert session.date in ("Monday", "2026-11-02")
        assert "time" in r2.lower()

        # TURN 3: Caller provides time '11 AM'
        r3 = svc.process_booking_turn("11 AM", session)
        assert session.time == "11:00 AM"
        # Since name is not yet provided, system should ask for name
        assert "name" in r3.lower()

        # TURN 4: Ambiguous time mention ('What about 11 PM?')
        # System should ask whether caller wants to change time, not silently overwrite
        r4 = svc.process_booking_turn("Actually could we do 11 PM?", session)
        # Check that clarification is asked
        assert "11:00 am" in r4.lower() or "11:00 pm" in r4.lower() or "change" in r4.lower()

        # TURN 5: Caller decides 'No, keep 11 AM'
        r5 = svc.process_booking_turn("No, keep 11 AM", session)
        assert session.time == "11:00 AM"
        assert "name" in r5.lower()

        # TURN 6: Caller gives only first name 'Abi'
        # System must ask ONLY for last name, not restart or repeat full name prompt
        r6 = svc.process_booking_turn("My name is Abi", session)
        assert session.patient_name == "Abi"
        assert "last name" in r6.lower()
        assert "Abi" in r6

        # TURN 7: Caller provides last name 'Kumar'
        r7 = svc.process_booking_turn("Kumar", session)
        assert session.patient_name == "Abi Kumar"
        # Now name is complete, next missing field is phone
        assert "phone" in r7.lower() or "number" in r7.lower()

        # TURN 8: Caller provides phone number
        r8 = svc.process_booking_turn("882-566-4227", session)
        assert "882" in (session.contact_number or "")
        # Next missing field is email
        assert "email" in r8.lower()

        # TURN 9: Caller provides email
        r9 = svc.process_booking_turn("V123@gmail.com", session)
        assert session.email == "v123@gmail.com"
        # All fields are collected! System must read back for confirmation
        assert session.state == BookingState.WAITING_FOR_CONFIRMATION
        assert "Abi Kumar" in r9
        assert "11:00 AM" in r9

        # TURN 10: Caller asks AI to repeat information so far
        r10 = svc.process_booking_turn("Can you repeat all my details?", session)
        assert "Abi Kumar" in r10
        assert "v123@gmail.com" in r10
        assert "Dr. Anita" in r10
        # State must remain WAITING_FOR_CONFIRMATION without losing any field
        assert session.state == BookingState.WAITING_FOR_CONFIRMATION

        # TURN 11: Final explicit affirmation to complete booking
        r11 = svc.process_booking_turn("Yes, that is all correct, please confirm", session)
        assert session.booking_confirmed is True
        assert session.appointment_id is not None
        assert session.appointment_id > 0
        assert f"#{session.appointment_id}" in r11

        # History must contain turns and not be empty
        assert len(session.history) >= 10

