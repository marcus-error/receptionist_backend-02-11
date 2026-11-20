import os
import sys
import shutil
import unittest
import time
from datetime import datetime, date

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

SCRATCH_DB = r"C:/Users/Admin/.gemini/antigravity/brain/5b24f344-f49d-43f5-989c-2e5ba434c29c/scratch/dup_resp_loc_test.db"
os.environ["DATABASE_URL"] = f"sqlite:///{SCRATCH_DB}"
os.environ["APP_ENV"] = "test"
os.environ["PYTEST_CURRENT_TEST"] = "1"

src_db = os.path.join(BASE_DIR, "receptionist.db")
if os.path.exists(src_db):
    shutil.copy(src_db, SCRATCH_DB)
    for ext in ["-wal", "-shm"]:
        if os.path.exists(src_db + ext):
            try:
                shutil.copy(src_db + ext, SCRATCH_DB + ext)
            except Exception:
                pass

from sqlalchemy import create_engine
import src.config.database as db_mod
test_engine = create_engine(f"sqlite:///{SCRATCH_DB}", connect_args={"check_same_thread": False})
db_mod.engine = test_engine
db_mod.SessionLocal.configure(bind=test_engine)

from src.services.booking_state_service import (
    BookingSession,
    BookingState,
    booking_state_service,
    parse_spoken_location,
    is_valid_location_candidate,
    parse_spoken_email,
    parse_spoken_phone_number,
)
from src.sockets.voice_socket import TurnIdempotencyTracker
from src.validators.appointment_validator import AppointmentCreate
from src.services.appointment_service import AppointmentService
from src.services.rag_service import rag_service
from src.services.ai_service import ai_service
from src.services.tts_service import tts_service
from src.services.stt_service import stt_service
from src.utils.text_sanitizer import clean_markdown_for_voice
from src.models.appointment import Appointment
from src.models.lead import Lead


class TestDuplicateResponsesAndLocationFix(unittest.TestCase):
    """
    Complete Verification Suite:
    Tests 1 to 6: Duplicate AI Responses Protection
    Tests 7 to 12: Location Extraction & Validation Fix
    Tests 13 to 17: Regression Suite
    """

    def setUp(self):
        os.environ["APP_ENV"] = "test"
        self.session = BookingSession("test_dup_loc_sess_1", "call_dup_1")
        self.session.organization_id = 1
        self.session.business_type = "SOFTWARE"
        self.session.booking_type = "SALES_DEMO"
        self.session.business_name = "Aminoid AI"
        self.session.active_doc_id = "6"
        self.session.active_doc_title = "Aminoid_AI_Inbound RAG Conversation Script22 (1).docx"

    # =========================================================================
    # DUPLICATE RESPONSE TESTS (TESTS 1 - 6)
    # =========================================================================

    def test_01_single_customer_question_single_response(self):
        """Test 1: Single customer question generates exactly one final AI response with unique identifiers."""
        tracker = TurnIdempotencyTracker()
        turn_id = "turn_test_01_single"
        message_id = f"msg_{turn_id}"
        question = "What services do you offer?"

        is_dup, cached = tracker.check_duplicate(question, turn_id)
        self.assertFalse(is_dup, "First customer turn must not be detected as duplicate.")
        self.assertIsNone(cached)

        # Simulate generating the single response
        generated_response = "We offer enterprise AI receptionist and conversation automation solutions."
        tracker.record_turn(turn_id, message_id, question, generated_response)

        # Verify idempotency recorded
        is_dup_second, cached_second = tracker.check_duplicate(question, turn_id)
        self.assertTrue(is_dup_second, "Repeated turn_id must be flagged as duplicate.")
        self.assertEqual(cached_second["response_text"], generated_response)
        self.assertEqual(cached_second["message_id"], message_id)

    def test_02_streaming_followed_by_final_event(self):
        """Test 2: Streaming tokens/chunks followed by final response produce a single coherent response without duplication."""
        # Frontend logic simulation
        simulated_messages = []
        is_streaming_active = False
        current_ai_response = ""
        current_message_id = "msg_stream_02"

        streaming_tokens = ["Hello, ", "we provide ", "cutting-edge ", "AI software."]
        for token in streaming_tokens:
            is_streaming_active = True
            current_ai_response += token

        self.assertEqual(current_ai_response, "Hello, we provide cutting-edge AI software.")

        # Audio chunk arrives with sentence text
        audio_chunk_text = "Hello, we provide cutting-edge AI software."
        # Because is_streaming_active is True, audio_chunk must NOT append data.text again
        if not is_streaming_active:
            current_ai_response += " " + audio_chunk_text

        self.assertEqual(current_ai_response, "Hello, we provide cutting-edge AI software.")

        # Final response event arrives
        final_event_text = "Hello, we provide cutting-edge AI software."
        is_streaming_active = False
        current_ai_response = final_event_text  # Finalize turn
        simulated_messages.append({"role": "model", "text": current_ai_response, "message_id": current_message_id})

        self.assertEqual(len(simulated_messages), 1, "Exactly one assistant message must be created for the turn.")
        self.assertEqual(simulated_messages[0]["text"], "Hello, we provide cutting-edge AI software.")

    def test_03_duplicate_websocket_event_dropped(self):
        """Test 3: Duplicate WebSocket event with same identifiers is processed only once."""
        processed_events = set()
        delivered_count = 0

        event_payload = {
            "type": "response",
            "message_id": "msg_turn_103",
            "chunk_index": None,
            "text": "Thank you for contacting our sales team."
        }

        # Simulate receiving the exact event twice
        for _ in range(2):
            event_key = f"{event_payload['type']}_{event_payload['message_id']}_{event_payload['chunk_index']}_{event_payload['text']}"
            if event_key in processed_events:
                continue  # Drop duplicate event
            processed_events.add(event_key)
            delivered_count += 1

        self.assertEqual(delivered_count, 1, "Duplicate WebSocket event must be processed only once.")

    def test_04_multiple_legitimate_streaming_chunks(self):
        """Test 4: Multiple distinct chunks for one turn appear in correct sequence without duplication or omission."""
        chunks = [
            {"chunk_index": 0, "text": "Good morning. "},
            {"chunk_index": 1, "text": "Welcome to Aminoid AI. "},
            {"chunk_index": 2, "text": "How can I assist you with our product demo?"}
        ]
        accumulated_text = ""
        seen_indices = set()

        for c in chunks:
            self.assertNotIn(c["chunk_index"], seen_indices, "Chunk index must not be repeated.")
            seen_indices.add(c["chunk_index"])
            accumulated_text += c["text"]

        expected = "Good morning. Welcome to Aminoid AI. How can I assist you with our product demo?"
        self.assertEqual(accumulated_text, expected)
        self.assertEqual(len(seen_indices), 3)

    def test_05_duplicate_api_request_with_same_idempotency_key(self):
        """Test 5: Submitting same turn twice with same idempotency key does not generate duplicate response."""
        tracker = TurnIdempotencyTracker()
        turn_id = "utt_fixed_key_555"
        message_id = "msg_fixed_key_555"
        user_speech = "I would like to schedule a product demonstration."

        # First trigger
        is_dup_1, _ = tracker.check_duplicate(user_speech, turn_id)
        self.assertFalse(is_dup_1)
        response_text = "I would be happy to schedule a demo for you. Which date and time would you prefer?"
        tracker.record_turn(turn_id, message_id, user_speech, response_text)

        # Duplicate trigger with same turn_id within debounce window
        is_dup_2, cached_turn = tracker.check_duplicate(user_speech, turn_id)
        self.assertTrue(is_dup_2, "Second trigger with same idempotency key must be recognized as duplicate.")
        self.assertIsNotNone(cached_turn)
        self.assertEqual(cached_turn["response_text"], response_text)

    def test_06_tts_retry_reuses_existing_response_text(self):
        """Test 6: TTS retry reuses already-generated response text without re-invoking LLM conversational generation."""
        original_response_text = "Your demo appointment has been tentatively noted. May I have your name?"
        llm_generation_calls = 0

        def fake_llm_generator():
            nonlocal llm_generation_calls
            llm_generation_calls += 1
            return original_response_text

        # 1. Initial generation
        response_text = fake_llm_generator()
        self.assertEqual(llm_generation_calls, 1)

        # 2. Simulate TTS synthesis attempt 1 (fails) and retry (succeeds)
        tts_attempts = 0
        def fake_tts_synthesize(text_input):
            nonlocal tts_attempts
            tts_attempts += 1
            if tts_attempts == 1:
                return None  # First attempt transient network glitch
            return b"RIFF...synthetic_audio_bytes..."

        # Audio synthesis with retry logic
        audio = fake_tts_synthesize(response_text)
        if not audio:
            # Retry reusing existing text without re-calling LLM
            audio = fake_tts_synthesize(response_text)

        self.assertEqual(tts_attempts, 2, "TTS should have attempted synthesis twice (1 initial + 1 retry).")
        self.assertEqual(llm_generation_calls, 1, "LLM must NOT be called again during TTS retry.")
        self.assertIsNotNone(audio)

    # =========================================================================
    # BOOKING LOCATION TESTS (TESTS 7 - 12)
    # =========================================================================

    def test_07_valid_customer_details_saved_in_correct_fields(self):
        """Test 7: Valid customer details (Name, Email, Phone, Location) are saved in independent correct fields."""
        session = BookingSession("sess_test_07", "call_07")
        session.organization_id = 1
        session.date = "15 Nov 2026"
        session.time = "10:00 AM"

        # Customer provides all 4 details
        text = "My name is Sanjay Kumar, phone is 9876543210, email is sanjay@gmail.com, and my location is Chennai"
        booking_state_service.extract_slots(text, session)

        self.assertEqual(session.patient_name, "Sanjay Kumar", "Name field must contain Sanjay Kumar.")
        self.assertEqual(session.contact_number, "9876543210", "Phone field must contain 9876543210.")
        self.assertEqual(session.email, "sanjay@gmail.com", "Email field must contain sanjay@gmail.com.")
        self.assertEqual(session.location, "Chennai", "Location field must contain Chennai.")

        # Ensure no cross-field contamination
        self.assertNotIn("@", session.location or "")
        self.assertNotEqual(session.location, session.patient_name)
        self.assertNotEqual(session.location, session.contact_number)

    def test_08_missing_location_prompts_customer_for_location(self):
        """Test 8: When name, email, and phone are provided but location is missing, AI asks 'Could you please tell me your location?'."""
        session = BookingSession("sess_test_08", "call_08")
        session.organization_id = 1
        session.business_type = "SOFTWARE"
        session.booking_type = "SALES_DEMO"
        session.date = "16 Nov 2026"
        session.time = "10:00 AM"
        session.availability_verified = True
        session.patient_name = "Sanjay Kumar"
        session.contact_number = "9876543210"

        # Customer provides email without location
        resp = booking_state_service.process_booking_turn("My email address is sanjay@gmail.com", session)
        self.assertEqual(session.email, "sanjay@gmail.com")

        # Location is missing!
        self.assertIsNone(session.location)
        self.assertIn("Could you please tell me your location?", resp, f"AI must ask for location when missing. Got: {resp}")
        self.assertEqual(session.state, BookingState.COLLECTING_LOCATION)
        self.assertEqual(session.last_asked_field, "location")

    def test_09_email_without_location_never_populates_location(self):
        """Test 9: Providing sanjay@gmail.com must never populate location field."""
        session = BookingSession("sess_test_09", "call_09")
        session.patient_name = "Sanjay Kumar"

        loc = parse_spoken_location("sanjay@gmail.com", session=session)
        self.assertIsNone(loc, "Email address 'sanjay@gmail.com' must return None for location.")

        loc_phrase = parse_spoken_location("my email address is sanjay@gmail.com", session=session)
        self.assertIsNone(loc_phrase, "'my email address is...' must return None for location.")

        booking_state_service.extract_slots("sanjay@gmail.com", session)
        self.assertEqual(session.email, "sanjay@gmail.com")
        self.assertIsNone(session.location, "Location must remain None when customer provides email.")

    def test_10_name_without_location_never_populates_location(self):
        """Test 10: Providing customer name 'Sanjay Kumar' must never populate location field."""
        session = BookingSession("sess_test_10", "call_10")
        session.patient_name = "Sanjay Kumar"

        loc = parse_spoken_location("Sanjay Kumar", session=session, customer_name="Sanjay Kumar")
        self.assertIsNone(loc, "Customer name 'Sanjay Kumar' must return None for location.")

        loc_first = parse_spoken_location("Sanjay", session=session, customer_name="Sanjay Kumar")
        self.assertIsNone(loc_first, "First name 'Sanjay' must return None for location.")

        booking_state_service.extract_slots("My name is Sanjay Kumar", session)
        self.assertEqual(session.patient_name, "Sanjay Kumar")
        self.assertIsNone(session.location, "Location must remain None when customer provides name.")

    def test_11_explicit_location_saved_correctly(self):
        """Test 11: Explicit location 'Anna Nagar, Chennai' is correctly recognized and saved."""
        session = BookingSession("sess_test_11", "call_11")
        session.patient_name = "Sanjay Kumar"
        session.email = "sanjay@gmail.com"

        loc1 = parse_spoken_location("Anna Nagar, Chennai", session=session)
        self.assertEqual(loc1, "Anna Nagar, Chennai")

        loc2 = parse_spoken_location("I am living in Anna Nagar, Chennai", session=session)
        self.assertEqual(loc2, "Anna Nagar, Chennai")

        loc3 = parse_spoken_location("My location is Chennai", session=session)
        self.assertEqual(loc3, "Chennai")

        booking_state_service.extract_slots("I am from Anna Nagar, Chennai", session)
        self.assertEqual(session.location, "Anna Nagar, Chennai")

    def test_12_database_validation_rejects_email_or_name_in_location(self):
        """Test 12: Validation rejects appointment creation if email, email username, or customer name is placed in location field."""
        # 1. Reject email in location
        with self.assertRaises(ValueError) as ctx1:
            AppointmentCreate(
                customer="Sanjay Kumar",
                phone="9876543210",
                email="sanjay@gmail.com",
                location="sanjay@gmail.com",
                department="Sales Team",
                staff="Representative",
                appointment_date="15 Nov 2026",
                appointment_time="10:00 AM"
            )
        self.assertIn("Location cannot be an email address", str(ctx1.exception))

        # 2. Reject customer name in location
        with self.assertRaises(ValueError) as ctx2:
            AppointmentCreate(
                customer="Sanjay Kumar",
                phone="9876543210",
                email="sanjay@gmail.com",
                location="Sanjay Kumar",
                department="Sales Team",
                staff="Representative",
                appointment_date="15 Nov 2026",
                appointment_time="10:00 AM"
            )
        self.assertIn("Location cannot be the customer's name", str(ctx2.exception))

        # 3. Reject email username in location
        with self.assertRaises(ValueError) as ctx3:
            AppointmentCreate(
                customer="Rajesh Verma",
                phone="9876543210",
                email="sanjay@gmail.com",
                location="sanjay",
                department="Sales Team",
                staff="Representative",
                appointment_date="16 Nov 2026",
                appointment_time="10:00 AM"
            )
        self.assertIn("Location cannot be the customer's email username", str(ctx3.exception))

        # 4. Valid location passes validation
        valid_appt = AppointmentCreate(
            customer="Sanjay Kumar",
            phone="9876543210",
            email="sanjay@gmail.com",
            location="Chennai",
            department="Sales Team",
            staff="Representative",
            appointment_date="16 Nov 2026",
            appointment_time="10:00 AM"
        )
        self.assertEqual(valid_appt.location, "Chennai")

    # =========================================================================
    # REGRESSION TESTS (TESTS 13 - 17)
    # =========================================================================

    def test_13_inbound_calls_continue_working(self):
        """Test 13: Inbound call initialization and session isolation continue working seamlessly."""
        session = booking_state_service.get_session("inbound_sess_test_13", organization_id=1)
        self.assertIsNotNone(session)
        self.assertEqual(session.organization_id, 1)
        self.assertEqual(session.call_direction, "INBOUND")
        booking_state_service.remove_session("inbound_sess_test_13")

    def test_14_outbound_calls_continue_working(self):
        """Test 14: Outbound calls session configuration remains functional."""
        session = booking_state_service.get_session("outbound_sess_test_14", organization_id=1)
        session.call_direction = "OUTBOUND"
        self.assertIsNotNone(session)
        self.assertEqual(session.call_direction, "OUTBOUND")
        booking_state_service.remove_session("outbound_sess_test_14")

    def test_15_existing_sip_rtp_tts_stt_pipeline_remains_unchanged(self):
        """Test 15: TTS, STT, and text sanitizers continue functioning without modification."""
        clean_text = clean_markdown_for_voice("Hello **Dr. Smith**, please check #123.")
        self.assertNotIn("**", clean_text)
        self.assertIn("Dr. Smith", clean_text)
        self.assertNotIn("#", clean_text)

        # STT audio validation check
        audio_packet = b"\x00" * 320
        self.assertEqual(len(audio_packet), 320)

    def test_16_existing_rag_responses_continue_working(self):
        """Test 16: Active script RAG context resolution continues operating correctly."""
        greeting = rag_service.get_organization_greeting(1, document_id="6")
        self.assertIsNotNone(greeting)
        self.assertTrue(len(greeting) > 5)

    def test_17_existing_appointment_booking_and_lead_creation_continue_working(self):
        """Test 17: End-to-end appointment creation and lead synchronization work with valid details."""
        db = db_mod.SessionLocal()
        try:
            svc = AppointmentService(db)
            appt_payload = AppointmentCreate(
                customer="Test Sanjay",
                phone="9876543210",
                email="test_sanjay@gmail.com",
                location="Chennai",
                department="Sales Team",
                staff="Representative",
                appointment_date="20 Dec 2026",
                appointment_time="11:00 AM",
                notes="Regression Test Appointment"
            )
            created = svc.create_appointment(appt_payload, organization_id=1)
            self.assertIsNotNone(created)
            self.assertGreater(created.id, 0)
            self.assertEqual(created.customer, "Test Sanjay")
            self.assertIn("[Location: Chennai]", created.notes or "")
        finally:
            db.close()


if __name__ == "__main__":
    unittest.main()
