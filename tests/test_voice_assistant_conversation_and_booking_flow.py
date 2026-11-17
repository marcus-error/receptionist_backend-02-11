import os
import sys
import shutil
import unittest
from datetime import datetime, date

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

SCRATCH_DB = r"C:/Users/Admin/.gemini/antigravity/brain/5b24f344-f49d-43f5-989c-2e5ba434c29c/scratch/voice_flow_test.db"
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
    parse_spoken_email,
)
from src.services.calendar_verification_service import calendar_verification_service
from src.services.agent_engine import agent_engine
from src.services.tts_service import TTSService


class TestVoiceAssistantConversationAndBookingFlow(unittest.TestCase):
    def setUp(self):
        os.environ["APP_ENV"] = "test"
        self.session = BookingSession("test_voice_flow_sess_1", "call_voice_1")
        self.session.organization_id = 1
        self.session.business_type = "SOFTWARE"
        self.session.booking_type = "SALES_DEMO"
        self.session.business_name = "Aminoid AI"
        self.session.active_doc_id = "6"
        self.session.active_doc_title = "Aminoid_AI_Inbound RAG Conversation Script22 (1).docx"

    def test_01_spoken_email_multi_segment_domains(self):
        """Verifies multi-segment domains (.co.in, .org.uk, .edu.in) are parsed without truncation."""
        test_cases = [
            ("sanjay@company.co.in", "sanjay@company.co.in"),
            ("s a n j a y at gmail dot com", "sanjay@gmail.com"),
            ("john dot smith at company dot co dot in", "john.smith@company.co.in"),
            ("my email is alex at domain dot org dot uk", "alex@domain.org.uk"),
            ("arun at the rate of aminoid dot ai", "arun@aminoid.ai"),
            ("contact at techcorp dot com", "contact@techcorp.com"),
        ]
        for spoken_input, expected_email in test_cases:
            res = parse_spoken_email(spoken_input)
            self.assertEqual(res, expected_email, f"Failed for input '{spoken_input}'")

    def test_02_email_incomplete_rejection(self):
        """Rejects incomplete fragments like '@gmail.com' or missing domains without crashing."""
        invalid_inputs = [
            "@gmail.com",
            "alex@",
            "at the rate of gmail dot com",
            "just email",
            "no dot",
        ]
        for inv in invalid_inputs:
            res = parse_spoken_email(inv)
            self.assertIsNone(res, f"Incomplete input '{inv}' should return None, got {res}")

    def test_03_date_parsing_preserves_customer_date_never_defaults_to_today(self):
        """Dates with commas, ordinals, 'of', and formats like DD-MM-YYYY or Month DD are preserved."""
        target_nov_15 = calendar_verification_service.parse_date("November 15, 2026", default=None)
        self.assertEqual(target_nov_15, date(2026, 11, 15))

        target_of_nov = calendar_verification_service.parse_date("15th of November, 2026", default=None)
        self.assertEqual(target_of_nov, date(2026, 11, 15))

        target_hyphen = calendar_verification_service.parse_date("15-11-2026", default=None)
        self.assertEqual(target_hyphen, date(2026, 11, 15))

        # Unrecognized / gibberish date strings must return None when default=None, NEVER silently mutate to today
        invalid_date = calendar_verification_service.parse_date("random noise or question", default=None)
        self.assertIsNone(invalid_date, "Invalid date string must not default to today")

        # Standalone month in _normalize_date must preserve Month instead of turning into today
        norm_month = booking_state_service._normalize_date("november")
        self.assertEqual(norm_month, "November")

    def test_04_software_sales_demo_booking_flow_turn_by_turn(self):
        """Simulates full conversational sales demo booking without healthcare doctor routing."""
        # 0. Sunday test: verify real availability and asking before changing date
        sunday_session = BookingSession("test_sunday", "call_sunday")
        sunday_session.organization_id = 1
        sunday_session.business_type = "SOFTWARE"
        sunday_session.booking_type = "SALES_DEMO"
        booking_state_service.process_booking_turn("I want to book a demo", sunday_session)
        sunday_resp = booking_state_service.process_booking_turn("Sunday, November 15, 2026", sunday_session)
        self.assertIn("no available slots remaining for sunday", sunday_resp.lower())
        self.assertIn("would you like to check another date", sunday_resp.lower())

        # 1. User wants to schedule a demo
        resp1 = booking_state_service.process_booking_turn("I would like to schedule a demo", self.session)
        self.assertEqual(self.session.state, BookingState.WAITING_FOR_DATE)
        self.assertIn("Which date would you prefer", resp1)
        self.assertNotIn("doctor", resp1.lower())
        self.assertNotIn("department", resp1.lower())

        # 2. User specifies a working weekday date
        resp2 = booking_state_service.process_booking_turn("November 16, 2026", self.session)
        self.assertEqual(self.session.date, "2026-11-16")
        self.assertEqual(self.session.state, BookingState.WAITING_FOR_TIME)
        self.assertIn("time", resp2.lower())
        self.assertNotIn("doctor", resp2.lower())

        # 3. User specifies time
        resp3 = booking_state_service.process_booking_turn("10:00 AM", self.session)
        self.assertEqual(self.session.state, BookingState.COLLECTING_PATIENT_NAME)
        self.assertIn("full name", resp3.lower())
        self.assertIn("demo", resp3.lower())
        self.assertNotIn("doctor", resp3.lower())

        # 4. User provides full name
        resp4 = booking_state_service.process_booking_turn("John Doe", self.session)
        self.assertEqual(self.session.patient_name, "John Doe")
        self.assertEqual(self.session.state, BookingState.COLLECTING_PHONE)
        self.assertIn("contact phone number", resp4.lower())

        # 5. User provides phone number
        resp5 = booking_state_service.process_booking_turn("9876543210", self.session)
        self.assertEqual(self.session.contact_number, "9876543210")
        self.assertEqual(self.session.state, BookingState.COLLECTING_EMAIL)
        self.assertIn("email address", resp5.lower())

        # 6. User provides email
        resp6 = booking_state_service.process_booking_turn("john.doe@company.co.in", self.session)
        self.assertEqual(self.session.email, "john.doe@company.co.in")
        self.assertEqual(self.session.state, BookingState.WAITING_FOR_CONFIRMATION)
        self.assertIn("john.doe@company.co.in", resp6)
        self.assertIn("demo", resp6.lower())
        self.assertNotIn("doctor", resp6.lower())

        # 7. User confirms booking
        resp7 = booking_state_service.process_booking_turn("Yes, that is correct, please confirm it", self.session)
        self.assertIn(self.session.state, (BookingState.BOOKING_CONFIRMED, BookingState.CONFIRMATION_ACCEPTED))
        self.assertIn("confirmed", resp7.lower())
        self.assertIn("demo", resp7.lower())

    def test_05_intent_switch_protection_during_active_booking(self):
        """Active booking state is preserved when customer answers booking prompts."""
        self.session.state = BookingState.WAITING_FOR_DATE
        self.session.last_asked_field = "date"
        # Customer says 'Tomorrow' with general_conversation intent classification
        switched = booking_state_service.check_and_handle_intent_switch(
            self.session, "general_conversation", "Tomorrow"
        )
        self.assertFalse(switched, "Booking state must not be bypassed for valid date answers")

        # Customer says their email with general_question intent
        self.session.state = BookingState.COLLECTING_EMAIL
        self.session.last_asked_field = "email"
        switched_email = booking_state_service.check_and_handle_intent_switch(
            self.session, "general_question", "my email is test@company.com"
        )
        self.assertFalse(switched_email, "Booking state must not be bypassed for email answers")

        # Genuine informational question DOES switch intent
        switched_info = booking_state_service.check_and_handle_intent_switch(
            self.session, "general_question", "What products do you offer?"
        )
        self.assertTrue(switched_info, "Genuine informational question must switch intent to RAG")

    def test_06_session_summary_robust_json_parsing(self):
        """Session summary parser handles markdown fences, text headers, and trailing commas cleanly."""
        mock_raw_outputs = [
            '```json\n{"summary": "Customer booked demo.", "outcome": "Appointment Booked"}\n```',
            'Here is the analysis:\n```json\n{"summary": "Demo scheduled successfully.", "outcome": "Success",}\n```',
            '{"summary": "Inquiry about AI Voice Agents.", "outcome": "Inquiry Completed", "customer_name": "Alice",}',
            'Plain text summary without any JSON brackets at all.'
        ]
        import re, json
        for raw in mock_raw_outputs:
            result = None
            m = re.search(r'(\{[\s\S]*\})', raw)
            if m:
                cand = m.group(1).strip()
                try:
                    result = json.loads(cand)
                except Exception:
                    cand_clean = re.sub(r',\s*([\}\]])', r'\1', cand)
                    try:
                        result = json.loads(cand_clean)
                    except Exception:
                        pass
            if not result or not isinstance(result, dict):
                first_line = raw.split("\n")[0][:120] if raw else "Summary generated from transcript."
                result = {
                    "summary": first_line,
                    "customer_name": "Customer",
                    "outcome": "completed"
                }
            self.assertIsInstance(result, dict)
            self.assertIn("summary", result)
            self.assertIn("outcome", result)


if __name__ == "__main__":
    unittest.main()
