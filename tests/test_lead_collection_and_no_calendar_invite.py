"""
Unit & Integration Test Suite for Lead Collection & No Calendar Invite Guarantee.

Verifies:
1. Customer asks about services -> AI explains -> customer requests team contact -> collect details -> callback lead created.
2. Customer says "Go ahead" -> continue current context.
3. Customer says "Continue previous conversation" -> resume from missing field.
4. Customer gives name and email -> AI must not ask them again.
5. Customer requests appointment explicitly -> appointment workflow activated.
6. Customer requests callback only -> no calendar event created.
7. Customer does not want a calendar invite -> no invite generated or mentioned.
8. Customer corrects email -> update email only.
9. Customer says "Thank you" -> polite closure.
10. Duplicate customer details -> prevent duplicate lead creation.
"""

import os
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.config.database import Base
from src.models.organization import Organization
from src.models.lead import Lead
from src.models.appointment import Appointment
from src.models.staff import Staff, StaffStatus
from src.services.booking_state_service import (
    BookingSession,
    BookingState,
    SessionWorkflowMode,
    booking_state_service
)
from src.services.lead_service import LeadService
from src.services.nlp_service import nlp_service, INTENT_CALLBACK, INTENT_APPOINTMENT_BOOKING, INTENT_THANKS


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Seed organization
    org = Organization(id=1, name="Aminoid AI")
    session.add(org)
    session.commit()

    # Seed staff
    staff = Staff(
        id=1,
        organization_id=1,
        name="Harini",
        department="General Medicine",
        status=StaffStatus.AVAILABLE
    )
    session.add(staff)
    session.commit()

    yield session
    session.close()


def test_01_services_inquiry_to_callback_lead_creation(db_session, monkeypatch):
    """
    Case 1: Customer asks about services -> AI explains -> customer requests team contact ->
    collect details -> callback lead created.
    """
    # Verify NLP detects callback request
    nlp_res = nlp_service.detect_intent("I want to speak with your team")
    assert nlp_res.intent == INTENT_CALLBACK

    # Create session
    session = BookingSession(
        session_id="test_sess_01",
        organization_id=1,
        business_type="SOFTWARE",
        booking_type="SALES_DEMO"
    )
    session.service = "AI Voice Agents"
    session.workflow_mode = SessionWorkflowMode.LEAD_COLLECTION

    # Turn 1: Ask for name
    resp1 = booking_state_service.process_booking_turn("I want to speak with your team", session)
    assert "name" in resp1.lower()
    assert "calendar invite" not in resp1.lower()

    # Turn 2: Provide name -> Ask for email
    resp2 = booking_state_service.process_booking_turn("My name is Sanjay Kumar", session)
    assert session.patient_name == "Sanjay Kumar"
    assert "email" in resp2.lower()
    assert "calendar invite" not in resp2.lower()

    # Turn 3: Provide email -> Ask for phone
    resp3 = booking_state_service.process_booking_turn("My email is sanjay@example.com", session)
    assert session.email == "sanjay@example.com"
    assert "phone" in resp3.lower() or "number" in resp3.lower()
    assert "calendar invite" not in resp3.lower()

    # Turn 4: Provide phone -> Ask for industry
    resp4 = booking_state_service.process_booking_turn("My phone is 9876543210", session)
    assert session.contact_number == "9876543210"
    assert "industry" in resp4.lower() or "business" in resp4.lower() or "requirement" in resp4.lower() or "noted" in resp4.lower()

    # Turn 5: Provide industry -> complete or requirement
    resp5 = booking_state_service.process_booking_turn("We are in Retail industry", session)
    assert "Retail" in (session.industry or "")

    # Final response must NOT mention calendar invite
    assert "calendar invite" not in resp5.lower()
    assert "calendar invitation" not in resp5.lower()


def test_02_customer_says_go_ahead_continues_context(db_session):
    """
    Case 2: Customer says "Go ahead" -> continue current context without restart.
    """
    session = BookingSession(
        session_id="test_sess_02",
        organization_id=1,
        business_type="SOFTWARE",
        booking_type="SALES_DEMO"
    )
    session.workflow_mode = SessionWorkflowMode.LEAD_COLLECTION
    session.patient_name = "Alex"

    # Last asked field was email
    session.state = BookingState.COLLECTING_EMAIL
    session.last_asked_field = "email"

    # Customer says "Go ahead"
    resp = booking_state_service.process_booking_turn("Go ahead", session)
    # Must NOT restart greeting or ask name again!
    assert "email" in resp.lower()
    assert session.patient_name == "Alex"
    assert "calendar invite" not in resp.lower()


def test_03_customer_says_continue_previous_conversation_resumes_missing_field(db_session):
    """
    Case 3: Customer says "Continue previous conversation" -> resume from missing field.
    """
    session = BookingSession(
        session_id="test_sess_03",
        organization_id=1,
        business_type="SOFTWARE",
        booking_type="SALES_DEMO"
    )
    session.workflow_mode = SessionWorkflowMode.LEAD_COLLECTION
    session.patient_name = "Ramesh"
    session.email = "ramesh@example.com"

    # Session missing phone, industry, requirement
    resp = booking_state_service.process_booking_turn("Continue previous conversation", session)

    # Must ask for next missing field (phone) and NOT re-ask name or email
    assert "phone" in resp.lower() or "contact" in resp.lower()
    assert "what is your name" not in resp.lower()
    assert "what is your email" not in resp.lower()


def test_04_customer_gives_name_and_email_together_never_asked_again(db_session):
    """
    Case 4: Customer gives name and email -> AI must not ask them again.
    """
    session = BookingSession(
        session_id="test_sess_04",
        organization_id=1,
        business_type="SOFTWARE",
        booking_type="SALES_DEMO"
    )
    session.workflow_mode = SessionWorkflowMode.LEAD_COLLECTION

    # Utterance provides both name and email
    resp = booking_state_service.process_booking_turn(
        "My name is Priya and my email is priya@company.com", session
    )

    assert session.patient_name == "Priya"
    assert session.email == "priya@company.com"

    # AI proceeds directly to phone number, never asking for name or email again
    assert "phone" in resp.lower() or "number" in resp.lower()
    assert "what is your name" not in resp.lower()
    assert "what is your email" not in resp.lower()


def test_05_customer_requests_appointment_explicitly_activates_appointment_flow(db_session):
    """
    Case 5: Customer requests appointment explicitly -> appointment workflow activated with date/time.
    """
    nlp_res = nlp_service.detect_intent("Book an appointment tomorrow at 10 AM")
    assert nlp_res.intent == INTENT_APPOINTMENT_BOOKING

    session = BookingSession(
        session_id="test_sess_05",
        organization_id=1,
        business_type="SOFTWARE",
        booking_type="SALES_DEMO"
    )

    # Customer explicitly requests appointment
    resp = booking_state_service.process_booking_turn(
        "I want to book an appointment for tomorrow at 10 AM", session
    )

    assert session.workflow_mode == SessionWorkflowMode.APPOINTMENT_BOOKING
    assert session.date == "Tomorrow"


def test_06_customer_requests_callback_only_no_calendar_event_created(db_session, monkeypatch):
    """
    Case 6: Customer requests callback only -> no calendar event created.
    """
    session = BookingSession(
        session_id="test_sess_06",
        organization_id=1,
        business_type="SOFTWARE",
        booking_type="SALES_DEMO"
    )
    session.workflow_mode = SessionWorkflowMode.LEAD_COLLECTION
    session.patient_name = "Karthik"
    session.email = "karthik@test.com"
    session.contact_number = "9123456780"
    session.industry = "Manufacturing"
    session.requirement = "ERP Software"

    # Turn completes lead collection
    resp = booking_state_service.process_booking_turn("That is all my requirements", session)

    assert "thank you" in resp.lower()
    assert "calendar invite" not in resp.lower()
    assert "calendar invitation" not in resp.lower()
    assert "scheduled a meeting" not in resp.lower()

    # Verify via LeadService in DB
    lead_svc = LeadService(db_session)
    lead = lead_svc.upsert_lead_from_session(
        session,
        lead_status="New",
        appointment_status="Not Booked",
        follow_up_required=True,
        notes="Callback requested: ERP Software"
    )
    db_session.commit()

    assert lead is not None
    assert lead.lead_status == "New"
    assert lead.appointment_status == "Not Booked"
    assert lead.follow_up_required is True
    assert lead.calendar_event_id is None
    assert lead.appointment_id is None

    # Verify no appointment record was created
    appts = db_session.query(Appointment).filter(Appointment.organization_id == 1).all()
    assert len(appts) == 0


def test_07_customer_does_not_want_calendar_invite_no_mention(db_session):
    """
    Case 7: Customer does not want a calendar invite -> no invite generated or mentioned.
    """
    session = BookingSession(
        session_id="test_sess_07",
        organization_id=1,
        business_type="SOFTWARE",
        booking_type="SALES_DEMO"
    )
    session.workflow_mode = SessionWorkflowMode.LEAD_COLLECTION
    session.patient_name = "Vijay"
    session.contact_number = "9988776655"
    session.industry = "Logistics"
    session.requirement = "Voice Agents"

    resp = booking_state_service.process_booking_turn(
        "My email is vijay@logistics.com and I don't want a calendar invite", session
    )

    assert "calendar invite" not in resp.lower()
    assert "calendar invitation" not in resp.lower()
    assert "we will send you a calendar invite" not in resp.lower()
    assert session.email == "vijay@logistics.com"


def test_08_customer_corrects_email_updates_email_only(db_session):
    """
    Case 8: Customer corrects email -> update email only.
    """
    session = BookingSession(
        session_id="test_sess_08",
        organization_id=1,
        business_type="SOFTWARE",
        booking_type="SALES_DEMO"
    )
    session.workflow_mode = SessionWorkflowMode.LEAD_COLLECTION
    session.patient_name = "Deepak"
    session.email = "old_email@gmail.com"
    session.contact_number = "9876543210"
    session.industry = "Fintech"

    # Customer corrects email
    resp = booking_state_service.process_booking_turn(
        "Actually, change my email to deepak.new@fintech.com", session
    )

    assert session.email == "deepak.new@fintech.com"
    assert session.patient_name == "Deepak"
    assert session.contact_number == "9876543210"
    assert session.industry == "Fintech"
    assert "calendar invite" not in resp.lower()


def test_09_customer_says_thank_you_polite_closure(db_session):
    """
    Case 9: Customer says "Thank you" -> polite closure.
    """
    # Verify NLP detects thanks
    nlp_res = nlp_service.detect_intent("Thank you very much, that's all")
    assert nlp_res.intent == INTENT_THANKS

    session = BookingSession(
        session_id="test_sess_09",
        organization_id=1,
        business_type="SOFTWARE",
        booking_type="SALES_DEMO"
    )
    session.workflow_mode = SessionWorkflowMode.LEAD_COLLECTION
    session.patient_name = "Ananya"
    session.is_completed = True

    resp = booking_state_service.process_booking_turn("Thank you so much", session)
    assert "welcome" in resp.lower()
    assert "calendar invite" not in resp.lower()


def test_10_duplicate_customer_details_prevent_duplicate_lead_creation(db_session):
    """
    Case 10: Duplicate customer details -> prevent duplicate lead creation.
    """
    session = BookingSession(
        session_id="test_sess_10",
        organization_id=1,
        business_type="SOFTWARE",
        booking_type="SALES_DEMO"
    )
    session.workflow_mode = SessionWorkflowMode.LEAD_COLLECTION
    session.patient_name = "Suresh"
    session.contact_number = "9840123456"
    session.email = "suresh@company.com"
    session.industry = "Education"
    session.requirement = "Galaxy Chat"

    lead_svc = LeadService(db_session)

    # First upsert
    lead1 = lead_svc.upsert_lead_from_session(
        session,
        lead_status="New",
        appointment_status="Not Booked",
        follow_up_required=True,
        notes="Callback requested: Galaxy Chat"
    )
    db_session.commit()
    assert lead1 is not None

    # Count total leads for org 1
    total_leads = db_session.query(Lead).filter(Lead.organization_id == 1).count()
    assert total_leads == 1

    # Second upsert with identical phone number & email
    lead2 = lead_svc.upsert_lead_from_session(
        session,
        lead_status="New",
        appointment_status="Not Booked",
        follow_up_required=True,
        notes="Customer called again regarding Galaxy Chat"
    )
    db_session.commit()

    # Must update existing lead, not create duplicate
    assert lead2.id == lead1.id
    total_leads_after = db_session.query(Lead).filter(Lead.organization_id == 1).count()
    assert total_leads_after == 1
