import pytest
from datetime import date, datetime, timedelta, time
from unittest.mock import MagicMock, patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.config.database import Base
from src.models.staff import Staff, StaffStatus
from src.models.appointment import Appointment, AppointmentStatus
from src.routes.availability_routes import (
    get_available_slots,
    get_current_business_datetime,
    find_next_available_slots,
    check_slot_availability,
    SlotCheckRequest,
)
from src.services.calendar_verification_service import calendar_verification_service
from src.services.booking_state_service import booking_state_service, BookingSession, BookingState


@pytest.fixture
def test_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestingSessionLocal()

    # Seed staff
    doc = Staff(
        id=1,
        organization_id=1,
        name="Dr. Sarah Jenkins",
        department="General Medicine",
        working_hours="09:00 - 18:00",
        start_time="09:00",
        end_time="18:00",
        appointment_duration="30 Minutes",
        break_start="13:00",
        break_end="14:00",
        status=StaffStatus.AVAILABLE,
        online_booking_enabled=True,
    )
    db.add(doc)
    db.commit()

    yield db
    db.close()


def test_past_time_slots_excluded_from_available_slots_for_today(test_db):
    """
    Scenario 1 & 2:
    Current time is 3:30 PM (15:30) on current date.
    get_available_slots for today must NOT include past slots (e.g. 9:00 AM to 3:30 PM).
    Must only return future slots (4:00 PM, 4:30 PM, 5:00 PM, 5:30 PM).
    """
    ref_dt = datetime(2026, 11, 19, 15, 30)  # 3:30 PM today
    today = ref_dt.date()

    res = get_available_slots(
        staff_id=1,
        selected_date=today,
        db=test_db,
        reference_now=ref_dt,
    )

    avail_slots = [s["start"] for s in res["available_slots"]]
    # All slots <= 15:30 must be excluded
    assert "09:00" not in avail_slots
    assert "10:00" not in avail_slots
    assert "15:00" not in avail_slots
    assert "15:30" not in avail_slots

    # Only remaining slots strictly in the future are returned
    assert "16:00" in avail_slots
    assert "16:30" in avail_slots
    assert "17:00" in avail_slots
    assert "17:30" in avail_slots


def test_future_date_includes_all_working_slots(test_db):
    """
    Scenario:
    Customer requests tomorrow. All working slots for tomorrow (excluding lunch break)
    must be returned without any past-time filtering.
    """
    ref_dt = datetime(2026, 11, 19, 15, 30)
    tomorrow = ref_dt.date() + timedelta(days=1)

    res = get_available_slots(
        staff_id=1,
        selected_date=tomorrow,
        db=test_db,
        reference_now=ref_dt,
    )

    avail_slots = [s["start"] for s in res["available_slots"]]
    assert "09:00" in avail_slots
    assert "10:00" in avail_slots
    assert "14:00" in avail_slots
    # Break time (13:00 - 14:00) must still be blocked
    assert "13:00" not in avail_slots
    assert "13:30" not in avail_slots


def test_past_date_returns_zero_slots(test_db):
    """
    Scenario:
    If customer requests a date in the past, get_available_slots returns 0 slots
    with clear reason.
    """
    ref_dt = datetime(2026, 11, 19, 15, 30)
    yesterday = ref_dt.date() - timedelta(days=1)

    res = get_available_slots(
        staff_id=1,
        selected_date=yesterday,
        db=test_db,
        reference_now=ref_dt,
    )

    assert len(res["available_slots"]) == 0
    assert "past" in res["reason"].lower()


def test_check_slot_availability_endpoint_identifies_past_time(test_db):
    """
    Scenario:
    POST /availability/check-slot with Today 3:00 PM when now is 3:30 PM.
    Must return is_available=False, is_past_time=True, and list available alternative slots.
    """
    ref_dt = datetime(2026, 11, 19, 15, 30)
    req = SlotCheckRequest(
        staff_id=1,
        date="2026-11-19",
        time="3:00 PM",
    )

    res = check_slot_availability(req, db=test_db, reference_now=ref_dt)

    assert res["is_available"] is False
    assert res["is_past_time"] is True
    assert "passed" in res["reason"]
    # Suggestions should only be future slots
    for alt in res["available_slots"]:
        assert alt in ["4:00 PM", "4:30 PM", "5:00 PM", "5:30 PM"]


def test_calendar_verification_service_detects_past_time(test_db):
    """
    Scenario:
    calendar_verification_service.verify_availability() called with today 3:00 PM when now is 3:30 PM.
    Result must have is_available=False, is_past_time=True, and alternative_slots for today.
    """
    ref_dt = datetime(2026, 11, 19, 15, 30)

    with patch("src.config.database.SessionLocal", return_value=test_db):
        res = calendar_verification_service.verify_availability(
            "I want an appointment with Dr. Sarah Jenkins today at 3:00 PM",
            reference_now=ref_dt,
        )

        assert res.is_checked is True
        assert res.is_available is False
        assert res.is_past_time is True
        assert "passed" in res.reason.lower()
        assert len(res.alternative_slots) > 0
        assert "4:00 PM" in res.alternative_slots


def test_step4_booking_turn_rejects_past_time_and_offers_available(test_db):
    """
    Scenario:
    Customer requests today at 3:00 PM when current time is 3:30 PM.
    The response must explicitly state:
    "Sorry, 3:00 PM is no longer available because that time has passed. Today's available slots are 4:00 PM, 4:30 PM, 5:00 PM, and 5:30 PM. Which time would you prefer?"
    """
    ref_dt = datetime(2026, 11, 19, 15, 30)

    session = BookingSession(
        session_id="test_past_time_01",
        organization_id=1,
    )
    session.doctor = "Dr. Sarah Jenkins"
    session.doctor_id = 1
    session.date = "Today"
    session.time = "3:00 PM"
    session.state = BookingState.WAITING_FOR_TIME

    with patch("src.config.database.SessionLocal", return_value=test_db), \
         patch("src.routes.availability_routes.get_current_business_datetime", return_value=ref_dt):
        reply = booking_state_service.process_booking_turn(
            session=session,
            text="Today at 3:00 PM",
        )

        assert "passed" in reply.lower()
        assert "3:00 PM" in reply
        assert "4:00 PM" in reply
        assert session.time is None  # Cleared so user chooses valid time
        assert session.state == BookingState.WAITING_FOR_TIME


def test_step4_booking_turn_accepts_valid_future_time(test_db):
    """
    Scenario:
    Customer then replies with "4:00 PM".
    Since 4:00 PM is in the future (> 3:30 PM) and free, it is accepted!
    """
    ref_dt = datetime(2026, 11, 19, 15, 30)

    session = BookingSession(
        session_id="test_future_time_02",
        organization_id=1,
    )
    session.doctor = "Dr. Sarah Jenkins"
    session.doctor_id = 1
    session.date = "Today"
    session.time = "4:00 PM"
    session.state = BookingState.WAITING_FOR_TIME

    with patch("src.config.database.SessionLocal", return_value=test_db), \
         patch("src.routes.availability_routes.get_current_business_datetime", return_value=ref_dt):
        reply = booking_state_service.process_booking_turn(
            session=session,
            text="4:00 PM",
        )

        assert session.availability_verified is True
        assert session.time == "4:00 PM"
        # Transitions to patient name
        assert session.state == BookingState.COLLECTING_PATIENT_NAME


def test_all_slots_today_exhausted_offers_next_available_date(test_db):
    """
    Scenario:
    Current time is 6:30 PM (18:30), after working hours end at 18:00.
    All slots today have passed.
    AI must dynamically fetch and offer tomorrow's available slots!
    """
    ref_dt = datetime(2026, 11, 19, 18, 30)  # After 6:00 PM
    today = ref_dt.date()

    with patch("src.config.database.SessionLocal", return_value=test_db), \
         patch("src.routes.availability_routes.get_current_business_datetime", return_value=ref_dt):
        session = BookingSession(
            session_id="test_exhausted_03",
            organization_id=1,
        )
        session.doctor = "Dr. Sarah Jenkins"
        session.doctor_id = 1
        session.date = "Today"
        session.time = "5:00 PM"  # already passed
        session.state = BookingState.WAITING_FOR_TIME

        reply = booking_state_service.process_booking_turn(
            session=session,
            text="Today at 5:00 PM",
        )

        assert "passed" in reply.lower()
        assert "no remaining slots available today" in reply.lower()
        assert "next available date" in reply.lower()
        assert "9:00 AM" in reply or "10:00 AM" in reply


def test_cancelled_appointment_frees_slot_again(test_db):
    """
    Scenario:
    An appointment at 16:00 is created. Slot 16:00 is blocked.
    When the appointment is cancelled (status='Cancelled'), slot 16:00 becomes free again.
    """
    ref_dt = datetime(2026, 11, 19, 15, 30)
    today = ref_dt.date()
    today_str = today.strftime("%d %b %Y")

    # Add active appointment
    appt = Appointment(
        organization_id=1,
        department="General Medicine",
        staff="Dr. Sarah Jenkins",
        customer="John Doe",
        appointment_date=today_str,
        appointment_time="04:00 PM",
        duration="30 Minutes",
        status=AppointmentStatus.CONFIRMED,
    )
    test_db.add(appt)
    test_db.commit()

    # Verify 16:00 is blocked
    res1 = get_available_slots(staff_id=1, selected_date=today, db=test_db, reference_now=ref_dt)
    avail1 = [s["start"] for s in res1["available_slots"]]
    assert "16:00" not in avail1

    # Cancel the appointment
    appt.status = AppointmentStatus.CANCELLED
    test_db.commit()

    # Verify 16:00 is now free again!
    res2 = get_available_slots(staff_id=1, selected_date=today, db=test_db, reference_now=ref_dt)
    avail2 = [s["start"] for s in res2["available_slots"]]
    assert "16:00" in avail2


def test_pre_commit_past_time_rejection(test_db):
    """
    Scenario:
    Session enters _execute_db_booking with a past time for today.
    _execute_db_booking must reject commit and return past time prompt.
    """
    ref_dt = datetime(2026, 11, 19, 15, 30)

    session = BookingSession(
        session_id="test_pre_commit_past",
        organization_id=1,
    )
    session.doctor = "Dr. Sarah Jenkins"
    session.doctor_id = 1
    session.patient_name = "Alice Green"
    session.contact_number = "9876543210"
    session.date = "Today"
    session.time = "2:00 PM"  # in past

    with patch("src.config.database.SessionLocal", return_value=test_db), \
         patch("src.routes.availability_routes.get_current_business_datetime", return_value=ref_dt):
        reply = booking_state_service._execute_db_booking(session)

        assert "passed" in reply.lower()
        assert "no longer available" in reply.lower()
        # Appointment must NOT have been committed
        saved = test_db.query(Appointment).filter(Appointment.customer == "Alice Green").first()
        assert saved is None


def test_multilingual_past_time_tamil_and_hindi(test_db):
    """
    Scenario:
    Past time rejection in Tamil and Hindi.
    """
    ref_dt = datetime(2026, 11, 19, 15, 30)

    # Tamil
    session_ta = BookingSession(
        session_id="test_ta_past",
        organization_id=1,
    )
    session_ta.conversation_language = "ta"
    session_ta.doctor = "Dr. Sarah Jenkins"
    session_ta.doctor_id = 1
    session_ta.date = "Today"
    session_ta.time = "3:00 PM"
    session_ta.state = BookingState.WAITING_FOR_TIME

    with patch("src.config.database.SessionLocal", return_value=test_db), \
         patch("src.routes.availability_routes.get_current_business_datetime", return_value=ref_dt):
        reply_ta = booking_state_service.process_booking_turn(
            session=session_ta,
            text="Today 3:00 PM",
        )
        assert "மன்னிக்கவும்" in reply_ta
        assert "முடிந்துவிட்டது" in reply_ta

    # Hindi
    session_hi = BookingSession(
        session_id="test_hi_past",
        organization_id=1,
    )
    session_hi.conversation_language = "hi"
    session_hi.doctor = "Dr. Sarah Jenkins"
    session_hi.doctor_id = 1
    session_hi.date = "Today"
    session_hi.time = "3:00 PM"
    session_hi.state = BookingState.WAITING_FOR_TIME

    with patch("src.config.database.SessionLocal", return_value=test_db), \
         patch("src.routes.availability_routes.get_current_business_datetime", return_value=ref_dt):
        reply_hi = booking_state_service.process_booking_turn(
            session=session_hi,
            text="Today 3:00 PM",
        )
        assert "Kshama kijiye" in reply_hi
        assert "nikal chuka hai" in reply_hi


def test_natural_language_time_parsing():
    """
    Scenario:
    Natural language relative phrases convert to appropriate 24-hour times.
    """
    svc = calendar_verification_service
    assert svc.parse_time_to_24h("after lunch") == "14:00"
    assert svc.parse_time_to_24h("post lunch") == "14:00"
    assert svc.parse_time_to_24h("this afternoon") == "14:00"
    assert svc.parse_time_to_24h("this evening") == "17:00"
    assert svc.parse_time_to_24h("morning") == "10:00"
    assert svc.parse_time_to_24h("noon") == "12:00"
    assert svc.parse_time_to_24h("3 o'clock") == "15:00"
    assert svc.parse_time_to_24h("4 o clock") == "16:00"
    assert svc.parse_time_to_24h("3 in the afternoon") == "15:00"
    assert svc.parse_time_to_24h("10 in the morning") == "10:00"
