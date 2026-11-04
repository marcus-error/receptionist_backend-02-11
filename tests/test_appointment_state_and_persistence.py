import pytest
from src.services.booking_state_service import booking_state_service, BookingState
from src.config.database import SessionLocal
from src.models.conversation_message import ConversationMessage
from src.models.appointment import Appointment


def test_nov3_appointment_conversation_flow_and_persistence():
    session_id = "test_live_nov3_flow"
    conv_id = "conv_live_nov3_flow"
    call_sid = "sid_live_nov3_flow"

    # Initialize session with caller ID like live call
    session = booking_state_service.get_session(session_id, conv_id, call_sid)
    session.caller_phone = "+919361701329"
    session.organization_id = 1

    # Turn 1: Customer specifies service
    r1 = booking_state_service.process_booking_turn("I want to book an appointment for Custom Software", session)
    assert session.service == "Custom Software"
    assert session.contact_number == "+919361701329"
    assert "Custom Software" in r1

    # Turn 2: Customer specifies date and time
    r2 = booking_state_service.process_booking_turn("Tomorrow at 11:30 AM", session)
    assert session.date is not None
    assert session.time is not None
    assert "full name" in r2.lower()

    # Turn 3: Customer provides single name "Hariharan"
    r3 = booking_state_service.process_booking_turn("Hariharan", session)
    assert session.patient_name == "Hariharan"
    assert "email" in r3.lower()

    # Turn 4: Customer provides email
    r4 = booking_state_service.process_booking_turn("harryhern23@gmail.com", session)
    assert session.email == "harryhern23@gmail.com"
    assert session.state == BookingState.WAITING_FOR_CONFIRMATION
    assert "Hariharan" in r4
    assert "+919361701329" in r4
    assert "harryhern23@gmail.com" in r4

    # Turn 5: Customer confirms
    r5 = booking_state_service.process_booking_turn("Yes", session)
    assert session.state == BookingState.BOOKING_CONFIRMED
    assert session.appointment_id is not None
    assert "confirmed" in r5.lower()
    assert f"#{session.appointment_id}" in r5

    # Verify conversation messages persisted in database
    db = SessionLocal()
    try:
        msgs = db.query(ConversationMessage).filter(ConversationMessage.session_id == conv_id).all()
        assert len(msgs) >= 10
        user_msgs = [m for m in msgs if m.role == "user"]
        assistant_msgs = [m for m in msgs if m.role == "assistant"]
        assert len(user_msgs) == 5
        assert len(assistant_msgs) == 5

        # Verify DB appointment was created
        appt = db.query(Appointment).filter(Appointment.id == session.appointment_id).first()
        assert appt is not None
        assert appt.customer == "Hariharan"
        assert appt.phone == "+919361701329"
        assert appt.email == "harryhern23@gmail.com"
        assert appt.service == "Custom Software"
    finally:
        db.close()

