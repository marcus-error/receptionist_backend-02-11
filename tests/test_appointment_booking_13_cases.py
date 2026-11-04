import pytest
import re
import asyncio
from unittest.mock import patch, MagicMock
from datetime import date, datetime, timedelta
from src.config.database import SessionLocal
from src.models.staff import Staff, StaffStatus
from src.models.appointment import Appointment, AppointmentStatus
from src.models.call_log import CallLog, CallType
from src.services.booking_state_service import (
    booking_state_service,
    BookingSession,
    BookingState,
    parse_spoken_phone_number,
    detect_incomplete_phone,
)
from src.services.calendar_verification_service import calendar_verification_service
from src.services.appointment_service import AppointmentService
from src.validators.appointment_validator import AppointmentCreate


@pytest.fixture
def clean_db():
    db = SessionLocal()
    try:
        # Ensure test staff exists in org 1
        staff = db.query(Staff).filter(Staff.name == "Dr. M. Anitha", Staff.organization_id == 1).first()
        if not staff:
            staff = Staff(
                name="Dr. M. Anitha",
                department="Siddha Consultant",
                organization_id=1,
                status=StaffStatus.AVAILABLE,
                working_days='["Monday","Tuesday","Wednesday","Thursday","Friday"]',
                start_time="09:00",
                end_time="17:00",
                working_hours="09:00-17:00",
                online_booking_enabled=True,
            )
            db.add(staff)
            db.commit()
            db.refresh(staff)

        # Clean existing test appointments for this customer
        db.query(Appointment).filter(
            Appointment.customer.in_(["Abi Kumar", "Ravi Kumar", "Suresh Raina", "Test Patient", "Concurrent User 1", "Concurrent User 2"])
        ).delete(synchronize_session=False)
        db.commit()
        yield db
    finally:
        db.close()


def create_pre_confirmation_session(session_id="test_sess_conf", doctor="Dr. M. Anitha"):
    session = booking_state_service.get_session(session_id)
    session.reset()
    session.organization_id = 1
    session.doctor = doctor
    session.doctor_id = 8
    session.department = "Siddha Consultant"
    session.date = (datetime.now() + timedelta(days=2)).strftime("%A")
    session.time = "11:00 AM"
    session.patient_name = "Abi Kumar"
    session.contact_number = "9876543210"
    session.email = "test@example.com"
    session.state = BookingState.WAITING_FOR_CONFIRMATION
    session.last_asked_field = "confirmation"
    session.booking_data["details_confirmed"] = False
    return session


def test_case_01_customer_says_yes(clean_db):
    """1. Customer says 'Yes.' -> creates appointment with proper state transitions."""
    session = create_pre_confirmation_session("sess_01")
    resp = booking_state_service.handle_confirmation_turn(session, "Yes.")

    assert session.state in (BookingState.BOOKING_CONFIRMED, BookingState.BOOKED)
    assert session.is_completed is True
    assert session.appointment_id is not None
    assert f"#{session.appointment_id}" in resp


def test_case_02_customer_says_yeah_go_ahead(clean_db):
    """2. Customer says 'Yeah, go ahead.' -> creates appointment."""
    session = create_pre_confirmation_session("sess_02")
    resp = booking_state_service.handle_confirmation_turn(session, "Yeah, go ahead.")

    assert session.state in (BookingState.BOOKING_CONFIRMED, BookingState.BOOKED)
    assert session.is_completed is True
    assert session.appointment_id is not None
    assert f"#{session.appointment_id}" in resp


def test_case_03_customer_says_okay(clean_db):
    """3. Customer says 'Okay.' -> creates appointment."""
    session = create_pre_confirmation_session("sess_03")
    resp = booking_state_service.handle_confirmation_turn(session, "Okay.")

    assert session.state in (BookingState.BOOKING_CONFIRMED, BookingState.BOOKED)
    assert session.is_completed is True
    assert session.appointment_id is not None
    assert f"#{session.appointment_id}" in resp


def test_case_04_customer_says_proceed(clean_db):
    """4. Customer says 'Proceed.' -> creates appointment."""
    session = create_pre_confirmation_session("sess_04")
    resp = booking_state_service.handle_confirmation_turn(session, "Proceed.")

    assert session.state in (BookingState.BOOKING_CONFIRMED, BookingState.BOOKED)
    assert session.is_completed is True
    assert session.appointment_id is not None
    assert f"#{session.appointment_id}" in resp


def test_case_05_customer_says_no(clean_db):
    """5. Customer says 'No.' -> does NOT create appointment."""
    session = create_pre_confirmation_session("sess_05")
    resp = booking_state_service.handle_confirmation_turn(session, "No, cancel.")

    assert session.state != BookingState.BOOKING_CONFIRMED
    assert session.state != BookingState.BOOKED
    assert session.appointment_id is None
    assert "cancelled" in resp.lower()


def test_case_06_hello_during_confirmation(clean_db):
    """6. Customer says 'Hello?' during confirmation -> respond briefly, details preserved, state WAITING_FOR_CONFIRMATION."""
    session = create_pre_confirmation_session("sess_06")
    resp = booking_state_service.handle_confirmation_turn(session, "Hello?")

    assert session.state == BookingState.WAITING_FOR_CONFIRMATION
    assert session.patient_name == "Abi Kumar"
    assert session.contact_number == "9876543210"
    assert session.appointment_id is None
    assert "i'm here" in resp.lower() or "proceed" in resp.lower()


def test_case_07_unclear_answer(clean_db):
    """7. Customer gives an unclear answer -> ask single short clarification, no appointment created."""
    session = create_pre_confirmation_session("sess_07")
    resp = booking_state_service.handle_confirmation_turn(session, "maybe bananas Tuesday")

    assert session.state == BookingState.WAITING_FOR_CONFIRMATION
    assert session.appointment_id is None
    assert "would you like me to proceed" in resp.lower() or "please say 'yes'" in resp.lower()


def test_case_08_customer_interrupts_ai_confirmation(clean_db):
    """8. Customer interrupts AI confirmation -> barge-in flag set and details remain intact."""
    session = create_pre_confirmation_session("sess_08")
    # Simulate barge-in interrupt event
    session.booking_data["barge_in_occurred"] = True

    # Caller affirms after interrupting
    resp = booking_state_service.handle_confirmation_turn(session, "Go ahead, book it.")
    assert session.state in (BookingState.BOOKING_CONFIRMED, BookingState.BOOKED)
    assert session.appointment_id is not None


def test_case_09_customer_repeats_yeah_multiple_times(clean_db):
    """9. Customer repeats 'Yeah' multiple times -> exactly one appointment created, duplicate prevented."""
    session = create_pre_confirmation_session("sess_09")
    resp1 = booking_state_service.handle_confirmation_turn(session, "Yeah yeah yeah, go ahead.")
    first_id = session.appointment_id
    assert first_id is not None
    assert session.state in (BookingState.BOOKING_CONFIRMED, BookingState.BOOKED)

    # Next turn caller says "Yeah" again
    resp2 = booking_state_service.handle_confirmation_turn(session, "Yeah.")
    assert session.appointment_id == first_id
    assert "already confirmed" in resp2.lower()

    # Query DB: exactly 1 appointment exists for this customer
    appts = clean_db.query(Appointment).filter(
        Appointment.customer == "Abi Kumar",
        Appointment.staff == "Dr. M. Anitha"
    ).all()
    assert len(appts) == 1


def test_case_10_database_booking_insertion_fails(clean_db):
    """10. Database booking insertion fails -> explain failure, allow retry, state remains WAITING_FOR_CONFIRMATION."""
    session = create_pre_confirmation_session("sess_10")

    with patch.object(AppointmentService, "create_appointment", side_effect=Exception("Database lock timeout")):
        resp = booking_state_service.handle_confirmation_turn(session, "Yes, confirm.")

    assert session.state == BookingState.WAITING_FOR_CONFIRMATION
    assert session.is_completed is False
    assert session.appointment_id is None
    assert "issue" in resp.lower() or "error" in resp.lower() or "try" in resp.lower()


def test_case_11_invalid_phone_number(clean_db):
    """11. Customer provides an invalid phone number (e.g. 9 digits) -> requests correction, does not confirm."""
    session = booking_state_service.get_session("sess_11")
    session.reset()
    session.organization_id = 1
    session.doctor = "Dr. M. Anitha"
    session.doctor_id = 8
    session.department = "Siddha Consultant"
    session.date = "tomorrow"
    session.time = "10:00 AM"
    session.patient_name = "Abi Kumar"
    session.state = BookingState.COLLECTING_PHONE
    session.last_asked_field = "contact_number"

    # Customer provides 9 digits: "882564223"
    resp = booking_state_service.process_booking_turn("882564223", session)
    assert session.contact_number is None
    assert session.state != BookingState.WAITING_FOR_CONFIRMATION
    assert "10" in resp or "phone" in resp.lower() or "mobile" in resp.lower()
    assert "phone_number" in session.missing_fields


def test_case_12_doctor_or_dept_does_not_exist(clean_db):
    """12. Department or doctor does not exist in DB (e.g. Phone) -> rejected, requests clarification."""
    session = booking_state_service.get_session("sess_12")
    session.reset()
    session.organization_id = 1

    # Extract with "Phone" as doctor/service
    booking_state_service.extract_slots("Can you tell about your Appointment, Phone specialist?", session)
    assert session.doctor is None
    assert session.doctor_id is None
    assert session.department != "Phone"
    assert session.service != "Phone"
    assert "department_or_doctor" in session.missing_fields

    # Extract with totally fake doctor
    booking_state_service.extract_slots("I want to see Dr. FakeUnregisteredPerson", session)
    assert session.doctor is None
    assert session.doctor_id is None
    assert "department_or_doctor" in session.missing_fields


def test_case_13_two_utterances_close_together(clean_db):
    """13. Two customer utterances arrive close together -> exactly one appointment created."""
    session = create_pre_confirmation_session("sess_13")

    # Turn 1: "Yeah, go ahead"
    resp1 = booking_state_service.handle_confirmation_turn(session, "Yeah, go ahead.")
    first_id = session.appointment_id
    assert first_id is not None
    assert session.state in (BookingState.BOOKING_CONFIRMED, BookingState.BOOKED)

    # Turn 2: "Confirm it." arriving right after
    resp2 = booking_state_service.handle_confirmation_turn(session, "Confirm it.")
    assert session.appointment_id == first_id

    # Verify only 1 record created in database
    count = clean_db.query(Appointment).filter(
        Appointment.customer == "Abi Kumar",
        Appointment.staff == "Dr. M. Anitha"
    ).count()
    assert count == 1


def test_case_14_instant_booking_and_status_tracking(clean_db):
    """14. Instant booking execution (< 1s) with sync status tracking columns."""
    import time
    session = create_pre_confirmation_session("sess_14")
    t0 = time.time()
    resp = booking_state_service.handle_confirmation_turn(session, "Yes, confirm booking.")
    elapsed = time.time() - t0

    assert elapsed < 1.0, f"Booking execution took {elapsed}s, expected < 1s"
    assert session.state == BookingState.BOOKING_CONFIRMED
    assert session.appointment_id is not None
    assert f"#{session.appointment_id}" in resp

    # Verify DB record and sync tracking columns
    appt = clean_db.query(Appointment).filter(Appointment.id == session.appointment_id).first()
    assert appt is not None
    assert appt.status == AppointmentStatus.CONFIRMED
    assert appt.calendar_sync_status in ("SYNCED", "PENDING", "NOT_CONFIGURED", "FAILED")
    assert appt.email_status in ("SENT", "PENDING", "SKIPPED", "FAILED")
    assert appt.whatsapp_status in ("SENT", "PENDING", "SKIPPED", "FAILED")


def test_case_15_deterministic_template_format(clean_db):
    """15. Deterministic confirmation text without LLM latency."""
    session = create_pre_confirmation_session("sess_15")
    resp = booking_state_service.handle_confirmation_turn(session, "Please book it.")
    
    assert "Your appointment is confirmed with Dr. M. Anitha" in resp
    assert f"Your appointment ID is #{session.appointment_id}." in resp
    assert "I've sent your confirmation details." in resp
