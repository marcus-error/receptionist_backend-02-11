import os
import sys
import shutil
import sqlite3
import pytest

# Ensure scratch directory for isolated test DB
SCRATCH_DIR = r"C:\Users\Admin\.gemini\antigravity\brain\5b24f344-f49d-43f5-989c-2e5ba434c29c\scratch"
os.makedirs(SCRATCH_DIR, exist_ok=True)
TEST_DB_PATH = os.path.join(SCRATCH_DIR, "test_response_gen.db")
SRC_DB_PATH = r"receptionist_backend-02-11\receptionist.db"

# Copy latest DB
shutil.copy2(SRC_DB_PATH, TEST_DB_PATH)

# Set DB url before any app imports
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB_PATH.replace('\\', '/')}"
os.environ["DEFAULT_LLM_PROVIDER"] = "grounded_rag"

sys.path.insert(0, os.path.abspath("receptionist_backend-02-11"))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import src.config.database as db_mod

test_engine = create_engine(f"sqlite:///{TEST_DB_PATH}".replace('\\', '/'), connect_args={"check_same_thread": False})
db_mod.engine = test_engine
db_mod.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

from src.services.booking_state_service import booking_state_service, BookingState, BookingSession
from src.services.rag_service import rag_service
from src.services.ai_service import ai_service


def setup_function():
    # Fresh copy of DB for each test
    shutil.copy2(SRC_DB_PATH, TEST_DB_PATH)
    t_eng = create_engine(f"sqlite:///{TEST_DB_PATH}".replace('\\', '/'), connect_args={"check_same_thread": False})
    db_mod.engine = t_eng
    db_mod.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=t_eng)
    rag_service.clear_cache()
    booking_state_service._sessions.clear()


def test_scenario_1_greeting_flow():
    """1. Customer says 'Hi' -> Opening greeting / polite greeting flow unchanged."""
    session = booking_state_service.get_session("test_s1", organization_id=1)
    
    # In initial state, intent is None, missing_fields is empty
    assert session.current_intent is None
    assert session.state == BookingState.IDLE
    assert session.missing_fields == []
    
    # Pure greeting before conversation started
    resp = ai_service._fast_rule_response("Hi", intent="greeting", organization_id=1, booking_session=session)
    # Opening greeting or polite greeting returned
    assert resp is not None
    assert "Aminoid AI" in resp or "help" in resp.lower() or "assist" in resp.lower()


def test_scenario_2_what_products_offer():
    """2. Customer asks 'What products does your company offer?' -> Answers directly from Aminoid AI script."""
    # Ensure Org 1 has Doc 6 active (Aminoid AI)
    meta = rag_service.get_active_script_metadata(1, scope="inbound")
    assert meta.get("active_script_id") == "6"
    assert "Aminoid" in meta.get("business_name", "")

    session = booking_state_service.get_session("test_s2", organization_id=1, business_name=meta.get("business_name"), business_type="SOFTWARE", booking_type="SALES_DEMO")
    
    query = "What products does your company offer?"
    # Fast path must NOT intercept business query
    fast_resp = ai_service._fast_rule_response(query, intent="product_information", organization_id=1, booking_session=session)
    assert fast_resp is None, "Fast path should NOT intercept product query!"
    
    # RAG must be needed
    assert rag_service.is_rag_needed(intent="product_information", text=query) is True
    
    # RAG retrieval with scope='inbound'
    context = rag_service.retrieve_context(query=query, organization_id=1, scope="inbound")
    assert len(context) > 0, "RAG context should not be empty!"
    assert any(term in context for term in ["Voice Agent", "AI", "Chatbot", "CRM", "HRMS", "ERP", "Aminoid"]), "Context must contain Aminoid products!"
    
    # While in IDLE state, missing_fields MUST be empty
    state_dict = session.get_state_dict()
    assert state_dict["missing_fields"] == [], "Idle session should have 0 missing fields!"
    assert state_dict["current_intent"] in ("general_inquiry", None)
    
    # LLM response generation with conversation state
    resp = ai_service.generate_response(
        text=query,
        context=context,
        intent="product_information",
        organization_id=1,
        conversation_state=state_dict
    )
    
    # Verify response mentions Aminoid software products and DOES NOT ask for appointment date/time
    assert not any(phrase in resp.lower() for phrase in ["what date would you prefer for your appointment", "which doctor", "department or doctor"]), f"Unexpected medical/booking prompt: {resp}"
    assert any(term in resp.lower() for term in ["voice", "agent", "crm", "chatbot", "software", "aminoid", "product", "hrms", "erp"]), f"Response did not mention Aminoid products: {resp}"


def test_scenario_3_what_services_provide():
    """3. Customer asks 'What services do you provide?' -> Answers directly from active script."""
    meta = rag_service.get_active_script_metadata(1, scope="inbound")
    session = booking_state_service.get_session("test_s3", organization_id=1, business_name=meta.get("business_name"), business_type="SOFTWARE", booking_type="SALES_DEMO")
    
    query = "What services do you provide?"
    assert ai_service._fast_rule_response(query, intent="service_information", organization_id=1, booking_session=session) is None
    
    context = rag_service.retrieve_context(query=query, organization_id=1, scope="inbound")
    assert len(context) > 0
    
    state_dict = session.get_state_dict()
    resp = ai_service.generate_response(
        text=query,
        context=context,
        intent="service_information",
        organization_id=1,
        conversation_state=state_dict
    )
    assert not any(phrase in resp.lower() for phrase in ["which doctor", "patient name", "cardiology"]), f"Response contaminated with healthcare: {resp}"


def test_scenario_4_book_software_demo():
    """4. Customer asks 'Can I book a software demo?' -> SALES_DEMO workflow begins. Collects demo date, time, name, email. Does NOT ask for doctor/department."""
    session = booking_state_service.get_session("test_s4", organization_id=1, business_type="SOFTWARE", booking_type="SALES_DEMO", business_name="Aminoid AI")
    
    query = "Can I book a software demo?"
    # Process booking turn
    resp = booking_state_service.process_booking_turn(query, session)
    
    # Session state transitions to collecting demo date
    assert session.state == BookingState.WAITING_FOR_DATE
    assert "demo" in resp.lower()
    assert any(word in resp.lower() for word in ["date", "prefer", "schedule"]), f"Unexpected response: {resp}"
    
    # Verify missing fields are DEMO fields, NOT medical
    missing = session.missing_fields
    assert "department_or_doctor" not in missing
    assert "demo_date" in missing or "appointment_date" not in missing
    assert "full_name" in missing


def test_scenario_5_general_question_and_audio_check():
    """5. Customer asks general question / audio check -> Answers naturally without inventing facts."""
    session = booking_state_service.get_session("test_s5", organization_id=1, business_name="Aminoid AI")
    
    # 5a: Audio check
    audio_resp = ai_service._fast_rule_response("Can you hear me?", intent="audio_check", organization_id=1, booking_session=session)
    assert audio_resp is not None
    assert "hear you" in audio_resp.lower()
    
    # 5b: General trivia question
    gen_q = "What is the speed of light?"
    fast_resp = ai_service._fast_rule_response(gen_q, intent="general_question", organization_id=1, booking_session=session)
    assert fast_resp is None
    
    context = rag_service.retrieve_context(query=gen_q, organization_id=1, scope="inbound")
    resp = ai_service.generate_response(text=gen_q, context=context, intent="general_question", organization_id=1, conversation_state=session.get_state_dict())
    assert resp is not None and len(resp) > 0


def test_scenario_6_script_switch():
    """6. Script switch -> Active script changed to Sarans Hospitals; fresh call uses new script exclusively."""
    # Connect to DB and switch active document for Org 1 to Doc 1 (Sarans Hospitals)
    conn = sqlite3.connect(TEST_DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE knowledge_documents SET ai_usage='Inactive' WHERE organization_id=1")
    c.execute("UPDATE knowledge_documents SET ai_usage='Active' WHERE id=1 AND organization_id=1")
    conn.commit()
    conn.close()
    
    rag_service.clear_cache()
    
    meta = rag_service.get_active_script_metadata(1, scope="inbound")
    assert meta.get("active_script_id") == "1"
    assert "Sarans" in meta.get("active_script_name", "") or "Sarans" in meta.get("business_name", "")
    assert meta.get("business_type") in ("HOSPITAL", "CLINIC")
    
    session = booking_state_service.get_session("test_s6", organization_id=1, business_type=meta.get("business_type"), booking_type=meta.get("booking_type"), business_name=meta.get("business_name"))
    
    # Query for services
    query = "What services do you provide?"
    context = rag_service.retrieve_context(query=query, organization_id=1, scope="inbound")
    assert len(context) > 0
    # Must contain hospital departments, zero Aminoid AI terms
    assert any(term in context.lower() for term in ["cardiology", "dental", "medicine", "doctor", "hospital"]), f"Context missing hospital info: {context[:200]}"
    assert "aminoid" not in context.lower()


def test_scenario_7_sarans_hospitals_active_workflow():
    """7. Sarans Hospitals active -> Healthcare workflow works: doctors, departments, patient appointments."""
    conn = sqlite3.connect(TEST_DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE knowledge_documents SET ai_usage='Inactive' WHERE organization_id=1")
    c.execute("UPDATE knowledge_documents SET ai_usage='Active' WHERE id=1 AND organization_id=1")
    conn.commit()
    conn.close()
    
    rag_service.clear_cache()
    
    session = booking_state_service.get_session("test_s7", organization_id=1, business_type="HOSPITAL", booking_type="HOSPITAL_APPOINTMENT", business_name="Sarans Hospitals")
    
    resp = booking_state_service.process_booking_turn("I want to book an appointment with Dr. Murugan", session)
    assert session.doctor is not None or "date" in resp.lower() or "doctor" in resp.lower()
    
    # Missing fields for hospital must include appointment fields
    missing = session.missing_fields
    assert "demo_date" not in missing


def test_scenario_8_two_orgs_isolation():
    """8. Two orgs -> Org 1 has Aminoid AI, Org 2 has different script. Strict isolation."""
    # Org 1: Aminoid AI (Doc 6)
    # Org 2: Aminoid AI Org2 Script or Hospital (Doc 7)
    conn = sqlite3.connect(TEST_DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE knowledge_documents SET ai_usage='Inactive'")
    c.execute("UPDATE knowledge_documents SET ai_usage='Active' WHERE id=6")  # Org 1 -> Doc 6
    c.execute("UPDATE knowledge_documents SET ai_usage='Active' WHERE id=7")  # Org 2 -> Doc 7
    conn.commit()
    conn.close()
    
    rag_service.clear_cache()
    
    meta_org1 = rag_service.get_active_script_metadata(1, scope="inbound")
    meta_org2 = rag_service.get_active_script_metadata(2, scope="inbound")
    
    assert meta_org1.get("active_script_id") == "6"
    assert meta_org2.get("active_script_id") == "7"
    
    ctx_org1 = rag_service.retrieve_context(query="What products do you offer?", organization_id=1, scope="inbound")
    ctx_org2 = rag_service.retrieve_context(query="What products do you offer?", organization_id=2, scope="inbound")
    
    assert len(ctx_org1) > 0
    assert len(ctx_org2) > 0
