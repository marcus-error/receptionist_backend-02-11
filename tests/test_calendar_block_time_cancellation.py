import pytest
from datetime import datetime, date, time, timedelta
from fastapi import HTTPException
from sqlalchemy.orm import Session

from src.config.database import SessionLocal, init_db
from src.models.appointment import Appointment, AppointmentStatus
from src.models.staff import Staff
from src.models.block_time import BlockTime
from src.routes.availability_routes import get_available_slots, get_current_business_datetime
from src.services.appointment_service import AppointmentService
from src.validators.appointment_validator import AppointmentCreate
from src.services.cancellation_service import cancellation_service
from src.services.nlp_service import nlp_service, INTENT_APPOINTMENT_CANCELLATION
from src.services.booking_state_service import booking_state_service, BookingSession, BookingState


@pytest.fixture(scope="module")
def setup_db():
    init_db()
    db = SessionLocal()
    # Ensure test staff exists
    staff = db.query(Staff).filter(Staff.name == "Dr. Calendar Tester").first()
    if not staff:
        staff = Staff(
            name="Dr. Calendar Tester",
            department="General Medicine",
            working_hours="09:00 - 17:00",
            appointment_duration="30 Minutes",
            organization_id=1
        )
        db.add(staff)
        db.commit()
        db.refresh(staff)
    yield db, staff
    # Cleanup test staff and any associated appointments/blocks
    db.query(Appointment).filter(Appointment.staff == "Dr. Calendar Tester").delete()
    db.query(BlockTime).filter(BlockTime.staff_name == "Dr. Calendar Tester").delete()
    db.commit()
    db.close()


def test_block_time_crud_persistence(setup_db):
    db, staff = setup_db
    tomorrow = date.today() + timedelta(days=2)
    start_dt = datetime.combine(tomorrow, time(10, 0))
    end_dt = datetime.combine(tomorrow, time(12, 0))

    # 1. Create BlockTime
    block = BlockTime(
        organization_id=1,
        staff_id=staff.id,
        staff_name=staff.name,
        start_datetime=start_dt,
        end_datetime=end_dt,
        reason="Department Meeting"
    )
    db.add(block)
    db.commit()
    db.refresh(block)

    assert block.id is not None
    assert block.reason == "Department Meeting"

    # 2. Query BlockTime
    persisted = db.query(BlockTime).filter(BlockTime.id == block.id).first()
    assert persisted is not None
    assert persisted.staff_name == staff.name
    assert persisted.start_datetime == start_dt
    assert persisted.end_datetime == end_dt

    # Clean up
    db.delete(persisted)
    db.commit()


def test_availability_excludes_blocked_time(setup_db):
    db, staff = setup_db
    target_date = date.today() + timedelta(days=3)

    # First check availability without block
    res_before = get_available_slots(
        staff_id=staff.id,
        selected_date=target_date,
        db=db
    )
    all_slots_before = [s["start"] for s in res_before["available_slots"]]
    assert "10:00" in all_slots_before
    assert "10:30" in all_slots_before
    assert "11:00" in all_slots_before

    # Now block 10:00 to 11:00
    block = BlockTime(
        organization_id=1,
        staff_id=staff.id,
        staff_name=staff.name,
        start_datetime=datetime.combine(target_date, time(10, 0)),
        end_datetime=datetime.combine(target_date, time(11, 0)),
        reason="Surgery"
    )
    db.add(block)
    db.commit()

    try:
        # Check availability with block
        res_after = get_available_slots(
            staff_id=staff.id,
            selected_date=target_date,
            db=db
        )
        slots_after = [s["start"] for s in res_after["available_slots"]]
        # 10:00 and 10:30 should be blocked
        assert "10:00" not in slots_after
        assert "10:30" not in slots_after
        # 11:00 onwards should be available
        assert "11:00" in slots_after
    finally:
        db.delete(block)
        db.commit()


def test_booking_rejection_for_blocked_time(setup_db):
    db, staff = setup_db
    target_date = date.today() + timedelta(days=4)
    target_date_str = target_date.strftime("%d %b %Y")

    # Block 14:00 to 15:00
    block = BlockTime(
        organization_id=1,
        staff_id=staff.id,
        staff_name=staff.name,
        start_datetime=datetime.combine(target_date, time(14, 0)),
        end_datetime=datetime.combine(target_date, time(15, 0)),
        reason="Doctor Personal Leave"
    )
    db.add(block)
    db.commit()

    svc = AppointmentService(db)
    try:
        # Attempt to book 02:00 PM (14:00) which is blocked
        with pytest.raises(HTTPException) as exc_info:
            svc.create_appointment(
                AppointmentCreate(
                    customer="Blocked Slot Tester",
                    phone="9876543210",
                    department="General Medicine",
                    staff=staff.name,
                    appointment_date=target_date_str,
                    appointment_time="02:00 PM",
                    duration="30 Minutes"
                ),
                organization_id=1
            )
        assert exc_info.value.status_code == 409
        assert "blocked" in exc_info.value.detail.lower()
    finally:
        db.delete(block)
        db.commit()


def test_past_date_booking_rejection(setup_db):
    db, staff = setup_db
    past_date = date.today() - timedelta(days=2)
    past_date_str = past_date.strftime("%d %b %Y")

    # 1. get_available_slots on past date returns empty with is_past_date=True
    slots_res = get_available_slots(
        staff_id=staff.id,
        selected_date=past_date,
        db=db
    )
    assert slots_res["available_slots"] == []
    assert slots_res.get("is_past_date") is True

    # 2. AppointmentService.create_appointment rejects past date with 400
    svc = AppointmentService(db)
    with pytest.raises(HTTPException) as exc_info:
        svc.create_appointment(
            AppointmentCreate(
                customer="Past Date Tester",
                phone="9876543210",
                department="General Medicine",
                staff=staff.name,
                appointment_date=past_date_str,
                appointment_time="10:00 AM",
                duration="30 Minutes"
            ),
            organization_id=1
        )
    assert exc_info.value.status_code == 400
    assert "past dates" in exc_info.value.detail.lower()


def test_booking_turn_past_date_immediate_rejection():
    session = BookingSession("past-date-turn-test")
    session.patient_name = "Alex"
    session.doctor = "Dr. Calendar Tester"
    session.state = BookingState.WAITING_FOR_DATE
    session.last_asked_field = "date"

    # User proposes a past date
    past_d = date.today() - timedelta(days=3)
    user_utterance = f"I want an appointment on {past_d.strftime('%d %B %Y')}"

    response = booking_state_service.process_booking_turn(user_utterance, session)
    assert "already passed" in response.lower()
    # Session date must remain None and state must stay WAITING_FOR_DATE
    assert session.date is None
    assert session.state == BookingState.WAITING_FOR_DATE


def test_past_time_on_today_rejection(setup_db):
    db, staff = setup_db
    today = date.today()
    today_str = today.strftime("%d %b %Y")

    # A time earlier than 08:00 AM (e.g. 05:00 AM or a fixed reference time in the past)
    now_dt = get_current_business_datetime(org_id=1, db=db)
    past_hour = max(0, now_dt.hour - 2)
    past_time_str = f"{past_hour:02d}:00"
    past_time_12h = datetime.strptime(past_time_str, "%H:%M").strftime("%I:%M %p")

    # If past_hour is during or before business hours, create_appointment must reject if it has passed
    if now_dt.hour >= 9:
        svc = AppointmentService(db)
        with pytest.raises(HTTPException) as exc_info:
            svc.create_appointment(
                AppointmentCreate(
                    customer="Past Time Tester",
                    phone="9876543210",
                    department="General Medicine",
                    staff=staff.name,
                    appointment_date=today_str,
                    appointment_time=past_time_12h,
                    duration="30 Minutes"
                ),
                organization_id=1
            )
        assert exc_info.value.status_code == 400
        assert "past" in exc_info.value.detail.lower()


def test_nlp_cancellation_intent_detection():
    phrases = [
        "Cancel my appointment",
        "I want to cancel my booking",
        "Cancel tomorrow's appointment",
        "I need to cancel the appointment I booked",
        "Please drop my appointment",
        "I won't be able to make it to my appointment",
        "Can you cancel my booking for tomorrow?",
        "Cancel booking",
    ]
    for p in phrases:
        res = nlp_service.detect_intent(p)
        assert res.intent == INTENT_APPOINTMENT_CANCELLATION, f"Failed for phrase: '{p}', got intent '{res.intent}'"


def test_cancellation_service_single_appointment(setup_db):
    db, staff = setup_db
    future_date = date.today() + timedelta(days=5)
    future_date_str = future_date.strftime("%d %b %Y")

    # Create active appointment
    appt = Appointment(
        organization_id=1,
        customer="Cancel Me Test User",
        phone="9988776655",
        department="General Medicine",
        staff=staff.name,
        appointment_date=future_date_str,
        appointment_time="11:30 AM",
        status=AppointmentStatus.CONFIRMED
    )
    db.add(appt)
    db.commit()
    db.refresh(appt)

    appt_id = appt.id
    try:
        # Caller calls asking to cancel
        res = cancellation_service.handle_cancellation(
            utterance="I want to cancel my appointment",
            caller_phone="9988776655",
            caller_name="Cancel Me Test User",
            organization_id=1,
            language="en"
        )
        assert res["cancelled"] is True
        assert res["appointment_id"] == appt_id
        assert "successfully cancelled" in res["response"].lower()

        # Verify DB status is Cancelled
        db.refresh(appt)
        assert appt.status == AppointmentStatus.CANCELLED

        # Verify duplicate cancellation request handling
        dup_res = cancellation_service.handle_cancellation(
            utterance="Cancel my appointment please",
            caller_phone="9988776655",
            caller_name="Cancel Me Test User",
            organization_id=1,
            language="en"
        )
        assert dup_res["cancelled"] is False
        assert dup_res.get("already_cancelled") is True
        assert "already been cancelled" in dup_res["response"].lower()
    finally:
        db.delete(appt)
        db.commit()


def test_cancellation_service_multiple_appointments_clarification(setup_db):
    db, staff = setup_db
    d1 = (date.today() + timedelta(days=6)).strftime("%d %b %Y")
    d2 = (date.today() + timedelta(days=7)).strftime("%d %b %Y")

    appt1 = Appointment(
        organization_id=1,
        customer="Multi Appt User",
        phone="9123456780",
        department="General Medicine",
        staff=staff.name,
        appointment_date=d1,
        appointment_time="10:00 AM",
        status=AppointmentStatus.CONFIRMED
    )
    appt2 = Appointment(
        organization_id=1,
        customer="Multi Appt User",
        phone="9123456780",
        department="General Medicine",
        staff=staff.name,
        appointment_date=d2,
        appointment_time="03:00 PM",
        status=AppointmentStatus.CONFIRMED
    )
    db.add_all([appt1, appt2])
    db.commit()
    db.refresh(appt1)
    db.refresh(appt2)

    session = BookingSession("multi-cancel-session")
    session.contact_number = "9123456780"
    session.patient_name = "Multi Appt User"

    try:
        # Caller asks to cancel without specifying which one
        res = cancellation_service.handle_cancellation(
            utterance="Cancel my booking",
            caller_phone="9123456780",
            caller_name="Multi Appt User",
            session=session,
            organization_id=1,
            language="en"
        )
        assert res["cancelled"] is False
        assert res.get("multiple_found") is True
        assert "Which appointment would you like to cancel?" in res["response"]
        assert session.state == "WAITING_FOR_CANCELLATION_CHOICE"

        # Caller answers "the first one"
        choice_res = cancellation_service.handle_cancellation(
            utterance="Please cancel the first one",
            caller_phone="9123456780",
            session=session,
            organization_id=1,
            language="en"
        )
        assert choice_res["cancelled"] is True
        assert choice_res["appointment_id"] == appt1.id
        assert "successfully cancelled" in choice_res["response"].lower()

        db.refresh(appt1)
        assert appt1.status == AppointmentStatus.CANCELLED
    finally:
        db.delete(appt1)
        db.delete(appt2)
        db.commit()


def test_cancellation_service_no_appointments_found(setup_db):
    res = cancellation_service.handle_cancellation(
        utterance="Cancel my appointment please",
        caller_phone="0000000000",
        caller_name="Nonexistent Person",
        organization_id=1,
        language="en"
    )
    assert res["cancelled"] is False
    assert res.get("not_found") is True
    assert "could not find any active appointments" in res["response"].lower()
