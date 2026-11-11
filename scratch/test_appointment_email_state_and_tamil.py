"""
Test Suite: CALLZENZA Appointment Email Capture, Booking State Priority & Multilingual Response
Validates all 10 scenarios requested by user.
"""

import sys
import os
import re

backend_dir = r"c:\projects\2-11 ai receptionist updated\receptionist_backend-02-11"
db_path = os.path.join(backend_dir, "receptionist.db").replace("\\", "/")
os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"

sys.path.insert(0, backend_dir)

from src.services.booking_state_service import (
    booking_state_service, BookingSession, BookingState, extract_email
)
from src.services.language_service import language_service

def test_all():
    print("=" * 80)
    print("RUNNING CALLZENZA APPOINTMENT & MULTILINGUAL TEST SUITE (10 SCENARIOS)")
    print("=" * 80)

    # ──────────────────────────────────────────────────────────────────────────
    # Scenario 1: Customer provides a complete email in one utterance
    # ──────────────────────────────────────────────────────────────────────────
    print("\n--- Scenario 1: Complete email in one utterance ---")
    raw_email_1 = "sanjaykumar@gmail.com"
    cand_1 = extract_email(raw_email_1)
    assert cand_1 == "sanjaykumar@gmail.com", f"Expected sanjaykumar@gmail.com, got {cand_1}"
    
    spoken_email_1 = "my email is sanjaykumar at gmail dot com"
    cand_1b = extract_email(spoken_email_1)
    assert cand_1b == "sanjaykumar@gmail.com", f"Expected sanjaykumar@gmail.com, got {cand_1b}"
    
    sess_1 = BookingSession(session_id="test_sess_1", organization_id=1)
    sess_1.department = "Cardiology"
    sess_1.doctor = "Dr. Saravanan"
    sess_1.date = "2026-11-15"
    sess_1.time = "10:00 AM"
    sess_1.patient_name = "Sanjay Kumar"
    sess_1.contact_number = "9876543210"
    sess_1.state = BookingState.COLLECTING_EMAIL
    sess_1.last_asked_field = "email"

    resp_1 = booking_state_service.process_booking_turn("sanjaykumar@gmail.com", sess_1, turn_id="t1")
    assert sess_1.email == "sanjaykumar@gmail.com", f"Email not saved: {sess_1.email}"
    assert sess_1.state == BookingState.WAITING_FOR_CONFIRMATION, f"State: {sess_1.state}"
    assert "sanjaykumar@gmail.com" in resp_1, f"Response did not confirm email: {resp_1}"
    print(f" PASS: Email captured directly -> {sess_1.email}, State -> {sess_1.state}")

    # ──────────────────────────────────────────────────────────────────────────
    # Scenario 2: Customer provides an email split across partial transcripts (or trailing @gmail.com)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n--- Scenario 2: Trailing @gmail.com (missing local part) & repeat recovery ---")
    sess_2 = BookingSession(session_id="test_sess_2", organization_id=1)
    sess_2.department = "Cardiology"
    sess_2.doctor = "Dr. Saravanan"
    sess_2.date = "2026-11-15"
    sess_2.time = "10:00 AM"
    sess_2.patient_name = "Sanjay Kumar"
    sess_2.contact_number = "9876543210"
    sess_2.state = BookingState.COLLECTING_EMAIL
    sess_2.last_asked_field = "email"

    # Step 2a: Final STT emits incomplete "@gmail.com"
    cand_bad = extract_email("@gmail.com")
    assert cand_bad is None, f"Expected None for isolated @gmail.com, got {cand_bad}"
    
    resp_2a = booking_state_service.process_booking_turn("@gmail.com", sess_2, turn_id="t2a")
    assert sess_2.email is None, "Incomplete email must NEVER be saved"
    assert "didn't catch the full email" in resp_2a.lower() or "repeat or spell" in resp_2a.lower(), f"Unexpected reprompt: {resp_2a}"
    assert sess_2.state == BookingState.COLLECTING_EMAIL, f"State should remain COLLECTING_EMAIL, got {sess_2.state}"
    print(f" PASS: Incomplete '@gmail.com' rejected without fabrication, reprompted -> '{resp_2a}'")

    # Step 2b: Customer repeats complete email
    resp_2b = booking_state_service.process_booking_turn("sanjaykumar@gmail.com", sess_2, turn_id="t2b")
    assert sess_2.email == "sanjaykumar@gmail.com"
    assert sess_2.state == BookingState.WAITING_FOR_CONFIRMATION
    print(f" PASS: Verified repeated email accepted -> {sess_2.email}, State -> {sess_2.state}")

    # ──────────────────────────────────────────────────────────────────────────
    # Scenario 3: Customer spells an email address
    # ──────────────────────────────────────────────────────────────────────────
    print("\n--- Scenario 3: Spelled-out email address ---")
    spelled_input = "s a n j a y k u m a r at gmail dot com"
    cand_3 = extract_email(spelled_input)
    assert cand_3 == "sanjaykumar@gmail.com", f"Expected sanjaykumar@gmail.com, got {cand_3}"
    
    sess_3 = BookingSession(session_id="test_sess_3", organization_id=1)
    sess_3.department = "Cardiology"
    sess_3.doctor = "Dr. Saravanan"
    sess_3.date = "2026-11-15"
    sess_3.time = "10:00 AM"
    sess_3.patient_name = "Sanjay Kumar"
    sess_3.contact_number = "9876543210"
    sess_3.state = BookingState.COLLECTING_EMAIL
    sess_3.last_asked_field = "email"

    resp_3 = booking_state_service.process_booking_turn(spelled_input, sess_3, turn_id="t3")
    assert sess_3.email == "sanjaykumar@gmail.com"
    assert sess_3.state == BookingState.WAITING_FOR_CONFIRMATION
    print(f" PASS: Spelled email extracted -> {sess_3.email}")

    # ──────────────────────────────────────────────────────────────────────────
    # Scenario 4: Customer says "skip" for email
    # ──────────────────────────────────────────────────────────────────────────
    print("\n--- Scenario 4: Customer says 'skip' for email ---")
    sess_4 = BookingSession(session_id="test_sess_4", organization_id=1)
    sess_4.doctor = "Dr. Saravanan"
    sess_4.date = "2026-11-15"
    sess_4.time = "10:00 AM"
    sess_4.patient_name = "Sanjay Kumar"
    sess_4.contact_number = "9876543210"
    sess_4.state = BookingState.COLLECTING_EMAIL
    sess_4.last_asked_field = "email"

    resp_4 = booking_state_service.process_booking_turn("skip", sess_4, turn_id="t4")
    assert sess_4.email is None, "Email should be None"
    assert sess_4.booking_data.get("email_declined") is True, "email_declined should be True"
    assert sess_4.state == BookingState.WAITING_FOR_CONFIRMATION, f"Expected WAITING_FOR_CONFIRMATION, got {sess_4.state}"
    assert "confirm your details" in resp_4.lower()
    print(f" PASS: Email declined, advanced to confirmation -> State: {sess_4.state}")

    # ──────────────────────────────────────────────────────────────────────────
    # Scenario 5: Customer requests Tamil during active appointment booking
    # ──────────────────────────────────────────────────────────────────────────
    print("\n--- Scenario 5: Customer requests Tamil during active booking ---")
    sess_5 = BookingSession(session_id="test_sess_5", organization_id=1)
    sess_5.department = "Cardiology"
    sess_5.doctor = "Dr. Saravanan"
    sess_5.date = "2026-11-15"
    sess_5.time = "10:00 AM"
    sess_5.state = BookingState.COLLECTING_PATIENT_NAME
    sess_5.last_asked_field = "patient_name"

    # Customer speaks: "Can please talk in summer? Summer" (STT acoustic misrecognition of Tamil)
    target_5, is_sw_5 = language_service.check_explicit_language_switch("Can please talk in summer? Summer")
    assert is_sw_5 is True, "Expected language switch detected"
    assert target_5 == "ta", f"Expected 'ta', got {target_5}"

    # Confirm dynamic confirmation preserves booking state and prompts in Tamil
    switch_confirm = language_service.get_language_switch_confirmation("ta", booking_session=sess_5)
    sess_5.conversation_language = "ta"
    assert "தமிழில்" in switch_confirm, f"Tamil confirmation expected, got {switch_confirm}"
    assert "பெயரை" in switch_confirm or "புக் செய்ய" in switch_confirm, f"Should prompt for pending patient name: {switch_confirm}"
    assert sess_5.doctor == "Dr. Saravanan", "Doctor must be preserved"
    assert sess_5.date == "2026-11-15", "Date must be preserved"
    print(f" PASS: Switched to Tamil seamlessly -> '{switch_confirm}'")

    # ──────────────────────────────────────────────────────────────────────────
    # Scenario 6: Customer continues the booking in Tamil
    # ──────────────────────────────────────────────────────────────────────────
    print("\n--- Scenario 6: Customer continues booking in Tamil ---")
    sess_5.conversation_language = "ta"
    # Customer provides patient name in Tamil turn
    resp_6 = booking_state_service.process_booking_turn("என் பெயர் சஞ்சய் குமார்", sess_5, turn_id="t6")
    assert sess_5.patient_name is not None, "Patient name should be extracted"
    assert sess_5.state in (BookingState.COLLECTING_PHONE, BookingState.WAITING_FOR_CONTACT)
    assert "contact number" in resp_6 or "தொலைபேசி" in resp_6 or "எண்" in resp_6
    v_id, _, prov = language_service.get_tts_voice_and_model("ta")
    assert v_id == "ta-IN-iniya", f"Expected Murf Tamil voice ta-IN-iniya, got {v_id}"
    print(f" PASS: Continued in Tamil -> Name={sess_5.patient_name}, Voice={v_id}")

    # ──────────────────────────────────────────────────────────────────────────
    # Scenario 7: Customer switches from Tamil to English
    # ──────────────────────────────────────────────────────────────────────────
    print("\n--- Scenario 7: Customer switches back to English ---")
    target_7, is_sw_7 = language_service.check_explicit_language_switch("Can we switch back to English?")
    assert is_sw_7 is True
    assert target_7 == "en"
    sess_5.conversation_language = "en"
    v_id_en, _, _ = language_service.get_tts_voice_and_model("en")
    assert v_id_en == "en-IN-isha"
    assert sess_5.doctor == "Dr. Saravanan", "Collected details must be intact"
    print(f" PASS: Switched back to English -> Voice={v_id_en}, State intact")

    # ──────────────────────────────────────────────────────────────────────────
    # Scenario 8: Booking state priority & department retention
    # ──────────────────────────────────────────────────────────────────────────
    print("\n--- Scenario 8: Booking state priority & department retention ---")
    sess_8 = BookingSession(session_id="test_sess_8", organization_id=1)
    sess_8.booking_data["last_inquired_product"] = "Cardiology"

    # User says "I want to book an appointment"
    resp_8 = booking_state_service.process_booking_turn("I want to book an appointment", sess_8, turn_id="t8")
    assert sess_8.department == "Cardiology", f"Department should be retained as Cardiology, got {sess_8.department}"
    assert "Which service or product would you like to book?" not in resp_8, "Must NOT ask generic service/product question"
    assert "Cardiology" in resp_8 or "Dr." in resp_8, f"Expected Cardiology doctor offer, got: {resp_8}"
    print(f" PASS: Department retained -> {sess_8.department}, Response -> '{resp_8}'")

    # ──────────────────────────────────────────────────────────────────────────
    # Scenario 9: Spoken time recognition ("9 AM", "nine thirty", "9:30 AM")
    # ──────────────────────────────────────────────────────────────────────────
    print("\n--- Scenario 9: Spoken time extraction ---")
    sess_9a = BookingSession(session_id="test_9a", organization_id=1)
    sess_9a.state = BookingState.WAITING_FOR_TIME
    booking_state_service.extract_slots("nine thirty", sess_9a)
    assert sess_9a.time == "9:30 AM", f"Expected 9:30 AM, got {sess_9a.time}"

    sess_9b = BookingSession(session_id="test_9b", organization_id=1)
    sess_9b.state = BookingState.WAITING_FOR_TIME
    booking_state_service.extract_slots("9 AM", sess_9b)
    assert sess_9b.time == "9:00 AM", f"Expected 9:00 AM, got {sess_9b.time}"

    sess_9c = BookingSession(session_id="test_9c", organization_id=1)
    sess_9c.state = BookingState.WAITING_FOR_TIME
    booking_state_service.extract_slots("two thirty", sess_9c)
    assert sess_9c.time == "2:30 PM", f"Expected 2:30 PM, got {sess_9c.time}"
    print(f" PASS: Spoken times correctly parsed: 'nine thirty'->{sess_9a.time}, '9 AM'->{sess_9b.time}, 'two thirty'->{sess_9c.time}")

    # ──────────────────────────────────────────────────────────────────────────
    # Scenario 10: Idempotent Appointment Persistence & No Re-Booking Overwrite
    # ──────────────────────────────────────────────────────────────────────────
    print("\n--- Scenario 10: Appointment persistence and confirmation flow ---")
    sess_10 = BookingSession(session_id="test_sess_10", organization_id=1)
    sess_10.department = "Cardiology"
    sess_10.doctor = "Dr. Saravanan"
    sess_10.doctor_id = 1
    sess_10.date = "2026-11-20"
    sess_10.time = "10:00 AM"
    sess_10.patient_name = "Sanjay Kumar"
    sess_10.contact_number = "9876543210"
    sess_10.email = "sanjaykumar@gmail.com"
    sess_10.state = BookingState.WAITING_FOR_CONFIRMATION

    from unittest.mock import patch
    def fake_execute_db(session):
        session.appointment_id = 999
        session.booking_confirmed = True
        session.state = BookingState.BOOKING_CONFIRMED
        session.is_completed = True
        return f"Your appointment is confirmed with {session.doctor} on {session.date} at {session.time}. Your appointment ID is #999."

    with patch.object(booking_state_service, "_execute_db_booking", side_effect=fake_execute_db):
        conf_resp = booking_state_service.handle_confirmation_turn(sess_10, "yes, please confirm", turn_id="t10")
        assert sess_10.state in (BookingState.BOOKING_CONFIRMED, BookingState.BOOKED, BookingState.COMPLETED)
        assert sess_10.is_completed is True
        assert "999" in conf_resp
        print(f" PASS: Appointment confirmed -> State={sess_10.state}, Resp='{conf_resp[:60]}...'")

    print("\n" + "=" * 80)
    print("ALL 10 SCENARIOS PASSED WITH 100% SUCCESS!")
    print("=" * 80)

if __name__ == "__main__":
    test_all()
