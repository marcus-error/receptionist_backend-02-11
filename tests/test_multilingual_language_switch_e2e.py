import asyncio
import os
import sys
import unittest
from unittest.mock import MagicMock, AsyncMock, patch

# Ensure paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.services.language_service import language_service, SUPPORTED_LANGUAGES, TTS_VOICE_CONFIG
from src.services.deepgram_stt_service import DeepgramStreamingSession
from src.services.booking_state_service import BookingSession, BookingState


class TestMultilingualLanguageSwitchE2E(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def test_scenario_1_tamil_switch(self):
        """Test 1: Customer requests Tamil: 'Will you please talk in Tamil?'"""
        phrase = "Will you please talk in Tamil?"
        target_lang, is_switch, switch_reason = language_service.check_language_switch_detailed(phrase)
        self.assertTrue(is_switch)
        self.assertEqual(target_lang, "ta")
        self.assertEqual(switch_reason, "explicit")

        # Voice routing
        v_id, model, prov = language_service.get_tts_voice_and_model("ta")
        self.assertEqual(v_id, "ta-IN-iniya")
        self.assertEqual(prov, "murf")

        # Confirmation response
        conf = language_service.get_language_switch_confirmation("ta")
        self.assertIn("வணக்கம்", conf)

        # STT switch configuration
        stt = DeepgramStreamingSession(api_key="dummy_key", call_id="test_ta_call", language="en")
        stt._running = True
        stt._connected = True
        stt._connect_ws = AsyncMock(return_value=True)
        stt._send_loop = AsyncMock()
        stt._recv_loop = AsyncMock()
        res = self.loop.run_until_complete(stt.switch_language("ta"))
        self.assertTrue(res)
        self.assertEqual(stt.language, "ta")

    def test_scenario_2_hindi_switch(self):
        """Test 2: Customer requests Hindi: 'Can you speak in Hindi?'"""
        phrase = "Can you speak in Hindi?"
        target_lang, is_switch, switch_reason = language_service.check_language_switch_detailed(phrase)
        self.assertTrue(is_switch)
        self.assertEqual(target_lang, "hi")

        v_id, model, prov = language_service.get_tts_voice_and_model("hi")
        self.assertEqual(v_id, "hi-IN-shweta")
        self.assertEqual(prov, "murf")

        conf = language_service.get_language_switch_confirmation("hi")
        self.assertIn("नमस्ते", conf)

        stt = DeepgramStreamingSession(api_key="dummy_key", call_id="test_hi_call", language="en")
        stt._running = True
        stt._connected = True
        stt._connect_ws = AsyncMock(return_value=True)
        stt._send_loop = AsyncMock()
        stt._recv_loop = AsyncMock()
        res = self.loop.run_until_complete(stt.switch_language("hi"))
        self.assertTrue(res)
        self.assertEqual(stt.language, "hi")

    def test_scenario_3_kannada_switch(self):
        """Test 3: Customer requests Kannada: 'Please speak in Kannada.'"""
        phrase = "Please speak in Kannada."
        target_lang, is_switch, switch_reason = language_service.check_language_switch_detailed(phrase)
        self.assertTrue(is_switch)
        self.assertEqual(target_lang, "kn")

        v_id, model, prov = language_service.get_tts_voice_and_model("kn")
        self.assertEqual(v_id, "kn")
        self.assertEqual(prov, "gtts")

        conf = language_service.get_language_switch_confirmation("kn")
        self.assertIn("ನಮಸ್ಕಾರ", conf)

        stt = DeepgramStreamingSession(api_key="dummy_key", call_id="test_kn_call", language="en")
        stt._running = True
        stt._connected = True
        stt._connect_ws = AsyncMock(return_value=True)
        stt._send_loop = AsyncMock()
        stt._recv_loop = AsyncMock()
        res = self.loop.run_until_complete(stt.switch_language("kn"))
        self.assertTrue(res)
        self.assertEqual(stt.language, "kn")

    def test_scenario_4_telugu_switch(self):
        """Test 4: Customer requests Telugu: 'Can you talk in Telugu?'"""
        phrase = "Can you talk in Telugu?"
        target_lang, is_switch, switch_reason = language_service.check_language_switch_detailed(phrase)
        self.assertTrue(is_switch)
        self.assertEqual(target_lang, "te")

        v_id, model, prov = language_service.get_tts_voice_and_model("te")
        self.assertEqual(v_id, "te")
        self.assertEqual(prov, "gtts")

        conf = language_service.get_language_switch_confirmation("te")
        self.assertIn("నమస్కారం", conf)

        stt = DeepgramStreamingSession(api_key="dummy_key", call_id="test_te_call", language="en")
        stt._running = True
        stt._connected = True
        stt._connect_ws = AsyncMock(return_value=True)
        stt._send_loop = AsyncMock()
        stt._recv_loop = AsyncMock()
        res = self.loop.run_until_complete(stt.switch_language("te"))
        self.assertTrue(res)
        self.assertEqual(stt.language, "te")

    def test_scenario_5_english_return(self):
        """Test 5: Customer requests return to English: 'Please continue in English.'"""
        phrase = "Please continue in English."
        target_lang, is_switch, switch_reason = language_service.check_language_switch_detailed(phrase)
        self.assertTrue(is_switch)
        self.assertEqual(target_lang, "en")

        v_id, model, prov = language_service.get_tts_voice_and_model("en")
        self.assertEqual(v_id, "en-IN-isha")
        self.assertEqual(prov, "murf")

        conf = language_service.get_language_switch_confirmation("en")
        self.assertIn("English", conf)

    def test_scenario_6_multi_switch_in_single_call(self):
        """Test 6: Multiple switches in a single call: en -> ta -> hi -> ta -> en"""
        switches = [
            ("Will you please talk in Tamil?", "ta", "ta-IN-iniya"),
            ("Can you speak in Hindi?", "hi", "hi-IN-shweta"),
            ("Please speak in Tamil again", "ta", "ta-IN-iniya"),
            ("Please continue in English.", "en", "en-IN-isha"),
        ]

        stt = DeepgramStreamingSession(api_key="dummy_key", call_id="test_multi_switch", language="en")
        stt._running = True
        stt._connected = True
        stt._connect_ws = AsyncMock(return_value=True)
        stt._send_loop = AsyncMock()
        stt._recv_loop = AsyncMock()

        current_lang = "en"
        for utterance, expected_lang, expected_voice in switches:
            tgt_lang, is_sw, _ = language_service.check_language_switch_detailed(utterance)
            self.assertTrue(is_sw)
            self.assertEqual(tgt_lang, expected_lang)
            res = self.loop.run_until_complete(stt.switch_language(tgt_lang))
            self.assertTrue(res)
            self.assertEqual(stt.language, expected_lang)
            v_id, _, _ = language_service.get_tts_voice_and_model(tgt_lang)
            self.assertEqual(v_id, expected_voice)
            current_lang = tgt_lang

        self.assertEqual(current_lang, "en")

    def test_scenario_7_provider_reconnect_persistence(self):
        """Test 7: Language persistence across non-switch turns and transliterated speech"""
        current_preferred = "ta"
        
        # Subsequent turn is numbers (e.g. phone number) or transliterated words
        subsequent_utterances = [
            "9092649453",
            "hospital address",
            "appointment doctor",
            "yes tomorrow 10 am"
        ]
        for utt in subsequent_utterances:
            turn_res = language_service.resolve_call_language_turn(
                utt, current_preferred_lang=current_preferred
            )
            # The preferred_response_language must stay 'ta'
            self.assertEqual(turn_res["preferred_response_language"], "ta")
            self.assertFalse(turn_res["language_switch_requested"])

    def test_scenario_8_business_workflow_preservation(self):
        """Test 8: Appointment booking workflow remains in selected language"""
        session = BookingSession()
        session.conversation_language = "ta"
        self.assertEqual(session.conversation_language, "ta")

        # Confirmation prompt during booking in Tamil asks for department/doctor
        conf_booking = language_service.get_language_switch_confirmation("ta", booking_session=session)
        self.assertTrue(len(conf_booking) > 0)
        self.assertIn("வணக்கம்", conf_booking)


if __name__ == "__main__":
    unittest.main()
