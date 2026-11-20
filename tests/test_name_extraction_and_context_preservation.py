import pytest
from src.services.booking_state_service import BookingStateService, BookingSession, BookingState
from src.services.ai_service import AIService


def test_clean_person_name_and_validity():
    bss = BookingStateService()

    # Test "How much name is Sanjay."
    assert bss.clean_person_name("How much name is Sanjay.") == "Sanjay"
    assert bss.clean_person_name("No. My name is Sanjay.") == "Sanjay"
    assert bss.clean_person_name("Call me Sanjay") == "Sanjay"
    assert bss.clean_person_name("I'm Sanjay") == "Sanjay"
    assert bss.clean_person_name("This is Sanjay") == "Sanjay"
    assert bss.clean_person_name("I said Sanjay") == "Sanjay"
    assert bss.clean_person_name("Actually my name is Sanjay") == "Sanjay"

    # Conversational tokens must NOT be valid names
    assert bss.is_valid_person_name("How") is False
    assert bss.is_valid_person_name("How Much") is False
    assert bss.is_valid_person_name("What") is False
    assert bss.is_valid_person_name("Can") is False
    assert bss.is_valid_person_name("No") is False
    assert bss.is_valid_person_name("You") is False
    assert bss.is_valid_person_name("Software") is False
    assert bss.is_valid_person_name("Appointment") is False
    assert bss.is_valid_person_name("Sanjay") is True


def test_extract_slots_name_correction_and_variations():
    bss = BookingStateService()
    session = BookingSession("test-sess-1", business_type="SOFTWARE", booking_type="SALES_DEMO")
    
    # 1. User says "How much name is Sanjay."
    bss.extract_slots("How much name is Sanjay.", session)
    assert session.patient_name == "Sanjay"

    # Simulate an erroneous name being in session
    session.patient_name = "WrongName"
    session.state = BookingState.COLLECTING_PHONE
    session.last_asked_field = "contact_number"

    # 2. Caller corrects: "No. My name is Sanjay."
    bss.extract_slots("No. My name is Sanjay.", session)
    assert session.patient_name == "Sanjay"
    assert session.booking_data.get("name_just_corrected") is True

    # 3. Caller says "Call me Sanjay."
    session.patient_name = "OldName"
    bss.extract_slots("Call me Sanjay.", session)
    assert session.patient_name == "Sanjay"

    # 4. Caller says "I said Sanjay."
    session.patient_name = "OldName"
    bss.extract_slots("I said Sanjay.", session)
    assert session.patient_name == "Sanjay"

    # 5. Caller says "I'm Sanjay."
    session.patient_name = "OldName"
    bss.extract_slots("I'm Sanjay.", session)
    assert session.patient_name == "Sanjay"

    # 6. Action verb after I'm should NOT extract as name
    session.patient_name = "Sanjay"
    bss.extract_slots("I'm looking for a software demo", session)
    assert session.patient_name == "Sanjay"


def test_process_booking_turn_name_correction_flow():
    bss = BookingStateService()
    session = BookingSession("test-sess-2", business_type="SOFTWARE", booking_type="SALES_DEMO")
    session.service = "Website Development"
    session.date = "Tomorrow"
    session.time = "9:00 AM"
    session.state = BookingState.COLLECTING_PHONE
    session.last_asked_field = "contact_number"
    session.patient_name = "How"  # Previously erroneous name

    # Caller corrects: "No. My name is Sanjay."
    resp = bss.process_booking_turn("No. My name is Sanjay.", session)
    
    # AI must address caller as Sanjay, never "How"
    assert "Sanjay" in resp
    assert "How" not in resp.split()
    assert session.patient_name == "Sanjay"
    assert "phone" in resp.lower()


def test_context_preservation_across_turns():
    bss = BookingStateService()
    session = BookingSession("test-sess-3", business_type="SOFTWARE", booking_type="SALES_DEMO")

    # Turn 1: Industry and Requirement
    bss.extract_slots("We are in manufacturing industry and need website development", session)
    assert session.industry == "Manufacturing"
    assert session.requirement == "Website Development"
    assert session.service == "Website Development"

    # Turn 2: Interested area / team
    bss.extract_slots("I want to speak to the sales team for demo booking", session)
    assert session.interested_area == "Sales"

    # Turn 3: Date and Time slot
    bss.extract_slots("Let's do tomorrow at 9 AM", session)
    assert session.date is not None
    assert "9:00 AM" in session.time

    # Turn 4: Name
    bss.extract_slots("How much name is Sanjay.", session)
    assert session.patient_name == "Sanjay"

    # Turn 5: Phone
    bss.extract_slots("My phone number is 9876543210", session)
    assert session.contact_number == "9876543210"

    # Turn 6: Email
    bss.extract_slots("My email is sanjay@example.com", session)
    assert session.email == "sanjay@example.com"

    # Verify all collected fields preserved
    collected = session.collected_fields
    assert collected["industry"] == "Manufacturing"
    assert collected["requirement"] == "Website Development"
    assert collected["interested_area"] == "Sales"
    assert collected["team"] == "Sales"
    assert collected["full_name"] == "Sanjay"
    assert collected["contact_number"] == "9876543210"
    assert collected["email"] == "sanjay@example.com"
    assert "9:00 AM" in collected["time"]


def test_ai_service_help_inquiry_fast_rule():
    ai = AIService()
    session = BookingSession("test-sess-4", business_name="Aminoid AI")
    
    # "Can I help you?"
    resp1 = ai._fast_rule_response("Can I help you?", booking_session=session)
    assert resp1 is not None
    assert "receptionist" in resp1.lower()

    # "Can you help me?"
    resp2 = ai._fast_rule_response("Can you help me?", booking_session=session)
    assert resp2 is not None
    assert "receptionist" in resp2.lower()
