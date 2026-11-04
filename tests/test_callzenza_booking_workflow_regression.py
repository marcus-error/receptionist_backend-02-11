import pytest
from unittest.mock import patch, MagicMock
from src.services.booking_state_service import (
    BookingStateService,
    BookingSession,
    BookingState,
    parse_spoken_email,
)
from src.services.entity_validation_service import entity_validation_service
from src.models.staff import Staff, StaffStatus
from src.models.appointment import Appointment
from src.models.organization import Organization


@pytest.fixture
def test_org(db_session):
    org = Organization(name="CallZenza Regression Org", organization_type="Healthcare", status="active")
    db_session.add(org)
    db_session.commit()
    db_session.refresh(org)
    return org


@pytest.fixture
def staff_anita(db_session, test_org):
    staff = Staff(
        organization_id=test_org.id,
        name="Dr. Anita",
        department="Cardiology",
        status=StaffStatus.AVAILABLE,
        online_booking_enabled=True,
        appointment_duration="30 min",
    )
    db_session.add(staff)
    db_session.commit()
    db_session.refresh(staff)
    return staff


def test_spoken_email_correction_not_contaminated():
    """Verify spoken email corrections never prefix 'changemymail' to domain or username."""
    # Test 1: Spoken 'change my mail at...'
    res1 = parse_spoken_email("change my mail at one two three gmail.com")
    assert res1 is not None
    assert "changemymail" not in res1
    assert res1.endswith("@gmail.com")

    # Test 2: 'change my email to sanjay at gmail dot com'
    res2 = parse_spoken_email("change my email to sanjay at gmail dot com")
    assert res2 == "sanjay@gmail.com"

    # Test 3: 'update my email to john dot doe at gmail dot com'
    res3 = parse_spoken_email("update my email to john dot doe at gmail dot com")
    assert res3 == "john.doe@gmail.com"


def test_number_words_in_email_convert_correctly():
    """Verify spoken number words in email addresses are converted to digits or kept cleanly."""
    res = parse_spoken_email("one two three at gmail dot com")
    assert res in ("123@gmail.com", "onetwothree@gmail.com")
    assert "changemymail" not in res


def test_doctor_verification_anissa_vs_anita(db_session, staff_anita):
    """Verify that ambiguous doctor name 'Dr. Anissa' closely matching 'Dr. Anita' is flagged as ambiguous."""
    val_res = entity_validation_service.validate_entity("Dr. Anissa", organization_id=staff_anita.organization_id, db_session=db_session)
    assert val_res.is_verified is False
    assert val_res.is_ambiguous is True
    assert val_res.suggested_entity == "Dr. Anita"


def test_doctor_correction_updates_staff_id(db_session, staff_anita):
    """Verify that saying 'Not Dr. Anissa, Dr. Anita' or 'Change to Dr. Anita' correctly selects Dr. Anita."""
    svc = BookingStateService()
    session = BookingSession("sess_correction_test")
    session.organization_id = staff_anita.organization_id

    # Turn 1: Caller corrects to Dr. Anita
    svc.extract_slots("Not Dr. Anissa, Dr. Anita", session)
    assert session.doctor == "Dr. Anita"
    assert session.doctor_id == staff_anita.id


def test_booking_only_confirmed_after_db_commit(db_session, staff_anita):
    """Verify that an appointment is NEVER marked confirmed if the database commit fails."""
    svc = BookingStateService()
    session = BookingSession("sess_fail_test")
    session.organization_id = staff_anita.organization_id
    session.patient_name = "Ramesh Kumar"
    session.contact_number = "9876543210"
    session.email = "ramesh@example.com"
    session.doctor = "Dr. Anita"
    session.doctor_id = staff_anita.id
    session.date = "25 Nov 2026"
    session.time = "10:00 AM"
    session.booking_data["details_confirmed"] = True

    with patch("src.services.appointment_service.AppointmentService.create_appointment", side_effect=RuntimeError("Database disk full")):
        resp = svc._execute_db_booking(session)
        assert session.booking_confirmed is False
        assert session.appointment_id is None
        assert "unable to confirm" in resp.lower() or "issue" in resp.lower() or "follow up" in resp.lower()
        assert "successfully confirmed" not in resp.lower()


def test_calendar_failure_handled_gracefully(db_session, staff_anita):
    """Verify that when calendar availability check raises an error, AI explains issue instead of fake confirmation."""
    svc = BookingStateService()
    session = BookingSession("sess_cal_fail_test")
    session.organization_id = staff_anita.organization_id
    session.doctor = "Dr. Anita"
    session.doctor_id = staff_anita.id
    session.date = "25 Nov 2026"
    session.time = "10:00 AM"

    with patch("src.services.calendar_verification_service.calendar_verification_service.verify_availability", side_effect=Exception("Calendar API Timeout")):
        resp = svc.process_booking_turn("10:00 AM", session)
        assert session.availability_verified is False
        assert "unable to verify" in resp.lower() or "issue" in resp.lower() or "check" in resp.lower()


def test_duplicate_booking_prevention(db_session, staff_anita):
    """Verify duplicate booking for same customer, doctor, date, time returns existing appointment ID."""
    svc = BookingStateService()
    session = BookingSession("sess_dup_test")
    session.organization_id = staff_anita.organization_id
    session.patient_name = "Karthik Raja"
    session.contact_number = "9840123456"
    session.email = "karthik@example.com"
    session.doctor = "Dr. Anita"
    session.doctor_id = staff_anita.id
    session.date = "28 Nov 2026"
    session.time = "11:00 AM"
    session.booking_data["details_confirmed"] = True

    # Pre-create appointment in DB
    existing_appt = Appointment(
        organization_id=staff_anita.organization_id,
        customer="Karthik Raja",
        phone="9840123456",
        email="karthik@example.com",
        staff="Dr. Anita",
        department="Cardiology",
        appointment_date="28 Nov 2026",
        appointment_time="11:00 AM",
        status="Confirmed",
    )
    db_session.add(existing_appt)
    db_session.commit()
    db_session.refresh(existing_appt)

    resp = svc._execute_db_booking(session)
    assert session.appointment_id == existing_appt.id
    assert session.booking_confirmed is True
    assert f"#{existing_appt.id}" in resp


def test_successful_booking_returns_real_appointment_id(db_session, staff_anita):
    """Verify that a successful booking commits to DB and returns the actual appointment ID."""
    svc = BookingStateService()
    session = BookingSession("sess_success_test")
    session.organization_id = staff_anita.organization_id
    session.patient_name = "Priya Dharshini"
    session.contact_number = "9812345678"
    session.email = "priya@example.com"
    session.doctor = "Dr. Anita"
    session.doctor_id = staff_anita.id
    session.date = "30 Nov 2026"
    session.time = "02:00 PM"
    session.booking_data["details_confirmed"] = True

    resp = svc._execute_db_booking(session)
    assert session.booking_confirmed is True
    assert session.appointment_id is not None
    assert session.appointment_id > 0
    assert f"#{session.appointment_id}" in resp

    # Verify directly from DB
    in_db = db_session.query(Appointment).filter(Appointment.id == session.appointment_id).first()
    assert in_db is not None
    assert in_db.customer == "Priya Dharshini"
    assert in_db.staff == "Dr. Anita"


def test_email_change_confirmed_before_update(db_session, staff_anita):
    """When a customer changes email during confirmation, system updates email and reads it back before booking."""
    svc = BookingStateService()
    session = BookingSession("sess_email_change_test")
    session.organization_id = staff_anita.organization_id
    session.patient_name = "Sanjay Kumar"
    session.contact_number = "9876543210"
    session.email = "oldmail@example.com"
    session.doctor = "Dr. Anita"
    session.doctor_id = staff_anita.id
    session.date = "29 Nov 2026"
    session.time = "10:30 AM"
    session.availability_verified = True
    session.state = BookingState.WAITING_FOR_CONFIRMATION

    resp = svc.process_booking_turn("Change my email to sanjay at gmail dot com", session)
    assert session.email == "sanjay@gmail.com"
    assert session.state == BookingState.WAITING_FOR_CONFIRMATION
    assert session.booking_confirmed is False
    assert "sanjay@gmail.com" in resp
    assert "confirm" in resp.lower() or "correct" in resp.lower()


def test_explicit_affirmation_required_before_booking_commit(db_session, staff_anita):
    """Random questions or non-affirmative phrases in WAITING_FOR_CONFIRMATION must NOT trigger booking."""
    svc = BookingStateService()
    session = BookingSession("sess_affirm_req_test")
    session.organization_id = staff_anita.organization_id
    session.patient_name = "Vijay Antony"
    session.contact_number = "9876543211"
    session.email = "vijay@example.com"
    session.doctor = "Dr. Anita"
    session.doctor_id = staff_anita.id
    session.date = "29 Nov 2026"
    session.time = "10:30 AM"
    session.availability_verified = True
    session.state = BookingState.WAITING_FOR_CONFIRMATION

    # Customer asks a random question instead of affirming
    resp = svc.process_booking_turn("Is there parking available at the clinic?", session)
    assert session.booking_confirmed is False
    assert session.appointment_id is None
    assert "confirm" in resp.lower()
