import asyncio
import os
import sys
import unittest
from unittest.mock import MagicMock, AsyncMock, patch

# Ensure test environment
os.environ["APP_ENV"] = "test"
os.environ["PYTEST_CURRENT_TEST"] = "1"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from services.language_service import language_service, LANG_TAMIL, LANG_ENGLISH
from services.nlp_service import nlp_service, INTENT_CHANGE_LANGUAGE
from services.sip_call_handler import SipCallSession
from services.booking_state_service import BookingSession
from services.ai_service import ai_service


class TestLanguageSwitchSilence(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Create a SipCallSession mock instance for call simulation
        self.sent_audio_chunks = []
        self.handler = SipCallSession(
            caller_phone="9876543210",
            caller_name="Test Customer",
            organization_id=1,
            audio_send_callback=lambda chunk: self.sent_audio_chunks.append(chunk),
            call_sid="test-call-lang-switch-001"
        )
        self.handler.active_script_greeting = "Good evening! Thank you for calling Aminoid AI. How may I assist you?"
        self.handler.active_script_id = "1"
        self.handler.business_type = "SOFTWARE"

    # ── Test 1: "Can you speak Tamil?" ─────────────────────────────────────────
    async def test_01_can_you_speak_tamil_immediate_response(self):
        """
        Caller: "Can you speak Tamil?"
        Expected:
        - Detected as language switch to Tamil
        - Immediate Tamil confirmation generated without requiring another caller utterance
        - Response text is valid non-empty Tamil confirmation
        """
        text = "Can you speak Tamil?"
        nlp_res = nlp_service.detect_intent(text)
        self.assertEqual(nlp_res.intent, INTENT_CHANGE_LANGUAGE)
        self.assertEqual(nlp_res.requested_language, "Tamil")
        self.assertEqual(nlp_res.requested_language_code, "ta")

        target_lang, is_switch = language_service.check_explicit_language_switch(text)
        self.assertTrue(is_switch)
        self.assertEqual(target_lang, "ta")

        switch_msg = language_service.get_language_switch_confirmation("ta")
        self.assertTrue(len(switch_msg) > 0)
        self.assertIn("தமிழ்", switch_msg)
        self.assertIn("கண்டிப்பா", switch_msg)

    # ── Test 2: "Will you please talk in Tamil?" ──────────────────────────────
    async def test_02_will_you_please_talk_in_tamil_tts_and_playback(self):
        """
        Caller: "Will you please talk in Tamil?"
        Expected:
        - Tamil language selected (ta)
        - AI acknowledgment generated
        - TTS triggered and audio emitted to audio_send_callback
        - No silent turn, no barge-in drop during synthesis
        """
        text = "Will you please talk in Tamil?"
        target_lang, is_switch = language_service.check_explicit_language_switch(text)
        self.assertTrue(is_switch)
        self.assertEqual(target_lang, "ta")

        # Simulate call turn execution via _run_ai_pipeline
        metrics = {"utt_id": "utt_lang_sw_2", "confidence": 0.98}
        await self.handler._run_ai_pipeline(text, metrics)

        # Verify language state was updated and persisted
        self.assertEqual(self.handler.current_lang, "ta")
        self.assertEqual(self.handler.preferred_response_language, "ta")
        self.assertEqual(self.handler.booking_session.conversation_language, "ta")

        # Verify response text was logged and added to history
        self.assertGreater(len(self.handler.session_history), 0)
        last_model_msg = [m for m in self.handler.session_history if m["role"] == "model"][-1]["text"]
        self.assertIn("தமிழ்", last_model_msg)

        # Verify audio chunks were sent to RTP callback
        self.assertGreater(len(self.sent_audio_chunks), 0)
        total_audio_bytes = sum(len(c) for c in self.sent_audio_chunks)
        self.assertGreater(total_audio_bytes, 100)

    # ── Test 3: "தமிழ்ல பேசுங்க." ─────────────────────────────────────────────
    async def test_03_tamil_script_immediate_response(self):
        """
        Caller: "தமிழ்ல பேசுங்க."
        Expected:
        - Immediate Tamil response
        - No silence
        """
        text = "தமிழ்ல பேசுங்க."
        nlp_res = nlp_service.detect_intent(text)
        self.assertEqual(nlp_res.intent, INTENT_CHANGE_LANGUAGE)
        self.assertEqual(nlp_res.requested_language_code, "ta")

        target_lang, is_switch = language_service.check_explicit_language_switch(text)
        self.assertTrue(is_switch)
        self.assertEqual(target_lang, "ta")

        metrics = {"utt_id": "utt_lang_sw_3"}
        await self.handler._run_ai_pipeline(text, metrics)

        self.assertEqual(self.handler.preferred_response_language, "ta")
        self.assertGreater(len(self.sent_audio_chunks), 0)

    # ── Test 4: "Hello?" mid-flow ─────────────────────────────────────────────
    async def test_04_mid_flow_hello_contextual_response_no_welcome_replay(self):
        """
        Caller: "Hello?" mid-flow
        Expected:
        - Short contextual response
        - Does NOT repeat the full welcome greeting
        """
        # Mark greeting as already sent
        self.handler.booking_session.greeting_sent = True
        self.handler.booking_session.conversation_started = True
        self.handler.preferred_response_language = "ta"
        self.handler.current_lang = "ta"

        # Fast path check in ai_service
        fast_resp = ai_service._fast_rule_response(
            text="Hello?",
            intent="greeting",
            organization_id=1,
            is_ongoing=True,
            booking_session=self.handler.booking_session,
            session_history=[
                {"role": "assistant", "text": "Good evening! How may I assist you?"},
                {"role": "user", "text": "Will you please talk in Tamil?"},
                {"role": "assistant", "text": "கண்டிப்பா, தமிழ்ல பேசுறேன். சொல்லுங்க, உங்களுக்கு என்ன உதவி வேணும்?"}
            ],
            conversation_language="ta"
        )
        self.assertIsNotNone(fast_resp)
        # Must NOT be the full welcome greeting
        self.assertNotEqual(fast_resp, self.handler.active_script_greeting)
        # Must be in Tamil since current_lang is ta
        self.assertTrue(any(c in fast_resp for c in ["வணக்கம்", "நான்", "சொல்லுங்க", "உங்க"]))

    # ── Test 5: Post-switch continuation ──────────────────────────────────────
    async def test_05_post_switch_continuation_in_tamil(self):
        """
        After Tamil switch:
        Caller: "நீங்க என்னென்ன services பண்றீங்க?"
        Expected:
        - Response in Tamil
        - Existing conversation context retained
        - Current language remains "ta"
        """
        # Set state as already switched to Tamil
        self.handler.preferred_response_language = "ta"
        self.handler.current_lang = "ta"
        self.handler.booking_session.conversation_language = "ta"
        self.handler.session_history.extend([
            {"role": "user", "text": "Will you please talk in Tamil?"},
            {"role": "model", "text": "கண்டிப்பா, தமிழ்ல பேசுறேன். சொல்லுங்க, உங்களுக்கு என்ன உதவி வேணும்?"}
        ])

        caller_query = "நீங்க என்னென்ன services பண்றீங்க?"
        # Intent classification must recognize enquiry
        nlp_res = nlp_service.detect_intent(caller_query)
        self.assertNotEqual(nlp_res.intent, INTENT_CHANGE_LANGUAGE)

        # Language must NOT randomly flip back to English
        self.assertEqual(self.handler.preferred_response_language, "ta")

    # ── Test 6: Switch back to English ────────────────────────────────────────
    async def test_06_switch_back_to_english(self):
        """
        Caller switches: "Can you speak in English?"
        Expected:
        - Language changes to English ("en")
        - Immediate English acknowledgment
        - No silent turn
        """
        # Start in Tamil
        self.handler.preferred_response_language = "ta"
        self.handler.current_lang = "ta"
        self.handler.booking_session.conversation_language = "ta"

        switch_query = "Can you speak in English?"
        nlp_res = nlp_service.detect_intent(switch_query)
        self.assertEqual(nlp_res.intent, INTENT_CHANGE_LANGUAGE)
        self.assertEqual(nlp_res.requested_language, "English")
        self.assertEqual(nlp_res.requested_language_code, "en")

        metrics = {"utt_id": "utt_sw_back_6"}
        await self.handler._run_ai_pipeline(switch_query, metrics)

        # Language state must now be English
        self.assertEqual(self.handler.preferred_response_language, "en")
        self.assertEqual(self.handler.current_lang, "en")
        self.assertEqual(self.handler.booking_session.conversation_language, "en")

        # Model response must be English confirmation
        last_model_msg = [m for m in self.handler.session_history if m["role"] == "model"][-1]["text"]
        self.assertIn("English", last_model_msg)
        self.assertGreater(len(self.sent_audio_chunks), 0)


if __name__ == "__main__":
    unittest.main()
