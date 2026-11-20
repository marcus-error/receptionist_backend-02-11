import unittest
from unittest.mock import MagicMock, patch
import pytest

from src.services.booking_state_service import (
    BookingSession,
    BookingState,
    booking_state_service,
    parse_spoken_location,
    is_valid_location_candidate
)
from src.utils.text_sanitizer import deduplicate_repeated_sentences
from src.services.ai_service import ai_service


class TestRepeatedResponsesAndActiveScriptBooking(unittest.TestCase):

    def test_01_stated_time_does_not_ask_for_doctor(self):
        """Verify that saying 'My available time is 09:30 AM.' extracts time and does NOT trigger doctor question."""
        session = BookingSession("sess_demo_time", "call_demo_time")
        session.business_type = "SOFTWARE"
        session.booking_type = "SALES_DEMO"
        session.workflow_type = "SALES_DEMO"
        session.service = "Demo"
        session.date = "Tomorrow"
        session.patient_name = "Sanjay"
        session.contact_number = "9876543210"

        # Turn where caller states time with 'available time' in phrasing
        resp = booking_state_service.process_booking_turn("My available time is 09:30 AM.", session)

        # Must NOT ask for doctor or department
        self.assertNotIn("doctor", resp.lower())
        self.assertNotIn("department", resp.lower())
        self.assertIn(session.time, ("9:30 AM", "09:30 AM"))

    def test_02_asking_slots_in_software_demo_presents_demo_slots(self):
        """Verify that asking for available slots in software demo flow presents demo slots, not doctor inquiry."""
        session = BookingSession("sess_demo_slots", "call_demo_slots")
        session.business_type = "SOFTWARE"
        session.booking_type = "SALES_DEMO"
        session.workflow_type = "SALES_DEMO"
        session.service = "Demo"
        session.date = "Tomorrow"

        resp = booking_state_service.process_booking_turn("What time slots do you have available?", session)

        self.assertIn("demo slots are available", resp.lower())
        self.assertNotIn("doctor", resp.lower())
        self.assertNotIn("department", resp.lower())

    def test_03_confirmation_affirmation_not_treated_as_location(self):
        """Verify that utterances like 'Yeah. Everything is correct.' and 'Confirm my appointment.' execute booking and are never stored as location."""
        session = BookingSession("sess_confirm", "call_confirm")
        session.patient_name = "Sanjay"
        session.contact_number = "9876543210"
        session.email = "sanjay@gmail.com"
        session.date = "Tomorrow"
        session.time = "10:00 AM"
        session.location = "Chennai"
        session.service = "Demo"
        session.state = BookingState.WAITING_FOR_CONFIRMATION
        session.last_asked_field = "confirmation"

        # Rejection from parse_spoken_location
        loc = parse_spoken_location("Yeah. Confirm my appointment.", session=session)
        self.assertIsNone(loc, "Affirmation should never be parsed as location")

        loc2 = parse_spoken_location("Am I audible?", session=session)
        self.assertIsNone(loc2, "Microphone test should never be parsed as location")

        loc3 = parse_spoken_location("Yeah. Everything is correct.", session=session)
        self.assertIsNone(loc3, "Confirmation statement should never be parsed as location")

    def test_04_deduplicate_repeated_sentences_s1(self):
        """Verify that interleaved / fragmented duplicate text from Screenshot 1 is collapsed."""
        s1 = (
            "I apologize, but that specific information is not available in our active records. "
            "At Aminoid AI, I apologize, but that specific information is not available in our active records. "
            "we build websites, web applications, AI voice agents, chatbots, ERP systems, project management tools, "
            "and custom software. How else may At Aminoid AI, we build websites, web applications, AI voice agents, "
            "chatbots, ERP systems, project management tools, and custom software. I assist you today? "
            "How else may I assist you today?"
        )
        cleaned = deduplicate_repeated_sentences(s1)
        self.assertEqual(cleaned.count("I apologize, but that specific information is not available in our active records"), 1)
        self.assertEqual(cleaned.count("we build websites"), 1)
        self.assertEqual(cleaned.count("How else may I assist you today"), 1)

    def test_05_deduplicate_repeated_sentences_s2(self):
        """Verify that back-to-back duplicate response blocks from Screenshot 2 are collapsed."""
        s2 = (
            "Aminoid AI offers several products including AI Voice Agents, Galaxy Chat for omnichannel assistance, "
            "an ERP application for operations, a Project Management application, an Employee Activity Tracking application, "
            "and a Survey Platform, as well as website design, development, and custom software integrations. "
            "Would you like to know more about any specific product? "
            "Aminoid AI offers several products including AI Voice Agents, Galaxy Chat for omnichannel assistance, "
            "an ERP application for operations, a Project Management application, an Employee Activity Tracking application, "
            "and a Survey Platform, as well as website design, development, and custom software integrations. "
            "Would you like to know more about any specific product?"
        )
        cleaned = deduplicate_repeated_sentences(s2)
        self.assertEqual(cleaned.count("Aminoid AI offers several products"), 1)
        self.assertEqual(cleaned.count("Would you like to know more about any specific product?"), 1)

    def test_06_generate_response_stream_no_duplicate_on_chunk_exception(self):
        """Verify that generate_response_stream does not fail over to another model after yielding tokens."""
        # Mock gemini client returning chunks, then raising exception at end
        mock_chunk1 = MagicMock()
        mock_chunk1.text = "Hello! Aminoid AI offers AI Voice Agents. "

        def stream_generator():
            yield mock_chunk1
            raise ValueError("Stream interrupted or finished")

        mock_client = MagicMock()
        mock_client.models.generate_content_stream.side_effect = lambda **kwargs: stream_generator()

        with patch.object(ai_service, "gemini_client", mock_client), \
             patch.object(ai_service, "_get_provider", return_value="gemini"), \
             patch.object(ai_service, "generate_response", return_value="FALLBACK CALLED DUPLICATE") as mock_gen_resp:

            tokens = list(ai_service.generate_response_stream("Tell me about Aminoid AI", context="Aminoid AI"))

            # Must have yielded mock_chunk1
            self.assertIn("Hello! Aminoid AI offers AI Voice Agents. ", tokens)
            # Must NOT have called non-stream generate_response fallback because tokens were already yielded
            mock_gen_resp.assert_not_called()
            # Total yielded chunks should only be 1
            self.assertEqual(len(tokens), 1)


if __name__ == "__main__":
    unittest.main()
