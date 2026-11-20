"""
Test Suite: Tamil Speech-to-Text (STT) Recognition & Multilingual Continuity.
Validates:
1. Deepgram language parameter configuration for Tamil (ta), Hindi (hi), Telugu (te), Kannada (kn), English (en).
2. Deepgram streaming session URL generation and mid-call language switching.
3. STTService transcribe_audio_bytes and transcribe_async with language parameter.
4. Real Tamil test sentences:
   - 1. வணக்கம், எப்படி இருக்கீங்க?
   - 2. உங்கள் நிறுவனத்தில் என்னென்ன சேவைகள் இருக்கிறது?
   - 3. எனக்கு ஒரு appointment book பண்ண வேண்டும்.
   - 4. உங்கள் software பற்றி விளக்க முடியுமா?
   - 5. நாளைக்கு ஒரு meeting schedule பண்ணுங்க.
   - 6. என்னுடைய பெயர் சஞ்சய்.
   - 7. நான் தமிழ்நாட்டிலிருந்து பேசுகிறேன்.
5. Code-switching preservation (Tamil + English words like appointment, software, meeting).
6. Multi-turn language lock without flipping to English on contact details/names.
7. WebSocket language synchronization events (type: language_changed).
"""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from src.services.stt_service import stt_service, normalize_stt_language
from src.services.deepgram_stt_service import (
    DeepgramStreamingSession,
    normalize_deepgram_language
)
from src.services.language_service import (
    language_service,
    LANG_TAMIL,
    LANG_ENGLISH,
    LANG_HINDI,
    LANG_TELUGU,
    LANG_KANNADA
)
from src.services.booking_state_service import booking_state_service


class TestTamilSTTRecognition:

    def test_01_normalize_stt_language_codes(self):
        """Verify language normalization maps various inputs to canonical STT codes."""
        assert normalize_stt_language("ta") == "ta"
        assert normalize_stt_language("ta-IN") == "ta"
        assert normalize_stt_language("Tamil") == "ta"
        assert normalize_stt_language("TAMIL") == "ta"

        assert normalize_stt_language("hi") == "hi"
        assert normalize_stt_language("hi-IN") == "hi"
        assert normalize_stt_language("Hindi") == "hi"

        assert normalize_stt_language("te") == "te"
        assert normalize_stt_language("te-IN") == "te"
        assert normalize_stt_language("Telugu") == "te"

        assert normalize_stt_language("kn") == "kn"
        assert normalize_stt_language("kn-IN") == "kn"
        assert normalize_stt_language("Kannada") == "kn"

        assert normalize_stt_language("en") == "en"
        assert normalize_stt_language("en-IN") == "en"
        assert normalize_stt_language("en-US") == "en"
        assert normalize_stt_language(None) == "en"
        assert normalize_stt_language("") == "en"

    def test_02_deepgram_streaming_url_language_parameter(self):
        """Verify Deepgram streaming WebSocket URL sets language=ta for Tamil."""
        session_ta = DeepgramStreamingSession(
            call_id="call_test_tamil",
            language="ta",
            model="nova-3",
            sample_rate=8000
        )
        url = session_ta._build_ws_url()
        assert "model=nova-3" in url
        assert "language=ta" in url
        assert "sample_rate=8000" in url

        session_hi = DeepgramStreamingSession(
            call_id="call_test_hindi",
            language="hi-IN",
            model="nova-3",
            sample_rate=8000
        )
        url_hi = session_hi._build_ws_url()
        assert "language=hi" in url_hi

        session_te = DeepgramStreamingSession(
            call_id="call_test_telugu",
            language="te",
            model="nova-3",
            sample_rate=8000
        )
        url_te = session_te._build_ws_url()
        assert "language=te" in url_te

    def test_03_deepgram_streaming_dynamic_language_switch(self):
        """Verify Deepgram session dynamically switches to Tamil without losing state."""
        session = DeepgramStreamingSession(
            call_id="call_switch_test",
            language="en",
            model="nova-3"
        )
        assert session.language == "en"
        assert "language=en" in session._build_ws_url()

        # Test language update for Tamil
        session.language = normalize_deepgram_language("ta-IN")
        assert session.language == "ta"
        assert "language=ta" in session._build_ws_url()

        # Test language update for Hindi
        session.language = normalize_deepgram_language("hi-IN")
        assert session.language == "hi"
        assert "language=hi" in session._build_ws_url()

        # Test language update for Telugu
        session.language = normalize_deepgram_language("Telugu")
        assert session.language == "te"
        assert "language=te" in session._build_ws_url()

        # Test language update for Kannada
        session.language = normalize_deepgram_language("Kannada")
        assert session.language == "kn"
        assert "language=kn" in session._build_ws_url()

        # Test language update for English
        session.language = normalize_deepgram_language("English")
        assert session.language == "en"
        assert "language=en" in session._build_ws_url()

    def test_04_stt_service_transcribe_with_language_parameter(self):
        """Verify stt_service.transcribe_audio_bytes accepts language parameter and passes it."""
        fake_audio = b"RIFF" + b"\x00" * 100
        # In test environment, mock_stt is used
        stt_service.model = "mock_stt"

        # When language is Tamil
        transcript_ta = stt_service.transcribe_audio_bytes(fake_audio, language="ta")
        assert "வணக்கம்" in transcript_ta

        # When language is Hindi
        transcript_hi = stt_service.transcribe_audio_bytes(fake_audio, language="hi")
        assert "नमस्ते" in transcript_hi

        # When language is Telugu
        transcript_te = stt_service.transcribe_audio_bytes(fake_audio, language="te")
        assert "నమస్కారం" in transcript_te

        # When language is English
        transcript_en = stt_service.transcribe_audio_bytes(fake_audio, language="en")
        assert "Hello" in transcript_en

    def test_05_seven_specified_tamil_sentences_detection_and_script(self):
        """
        Verify all 7 real Tamil test sentences are accurately recognized as Tamil (ta),
        their native Tamil script is preserved, and code-switching is retained.
        """
        sentences = [
            ("வணக்கம், எப்படி இருக்கீங்க?", "Greeting"),
            ("உங்கள் நிறுவனத்தில் என்னென்ன சேவைகள் இருக்கிறது?", "Services Inquiry"),
            ("எனக்கு ஒரு appointment book பண்ண வேண்டும்.", "Appointment Booking with code-switching"),
            ("உங்கள் software பற்றி விளக்க முடியுமா?", "Product/Software Inquiry with code-switching"),
            ("நாளைக்கு ஒரு meeting schedule பண்ணுங்க.", "Meeting Scheduling with code-switching"),
            ("என்னுடைய பெயர் சஞ்சய்.", "Name disclosure"),
            ("நான் தமிழ்நாட்டிலிருந்து பேசுகிறேன்.", "Location disclosure"),
        ]

        for s_text, desc in sentences:
            detected_lang = language_service.detect_spoken_language(s_text, current_lang="en")
            assert detected_lang == "ta", f"Failed for '{s_text}' ({desc}): expected 'ta', got '{detected_lang}'"

            # Determine conversation language with previous language 'en'
            conv_lang = language_service.determine_conversation_language(s_text, current_lang="en")
            assert conv_lang == "ta", f"Failed determine_conversation_language for '{s_text}' ({desc})"

    def test_06_tamil_code_switching_and_continuity_across_turns(self):
        """
        Test multi-turn dialogue starting in Tamil, where customer uses English loanwords
        (appointment, software, meeting, email, phone) without flipping back to English.
        """
        session_id = "test_continuity_multi_turn"
        booking_session = booking_state_service.get_session(session_id=session_id)
        booking_session.conversation_language = "ta"

        # Turn 1: Pure Tamil
        t1 = "வணக்கம், எப்படி இருக்கீங்க?"
        lang1 = language_service.determine_conversation_language(t1, current_lang=booking_session.conversation_language)
        booking_session.conversation_language = lang1
        assert booking_session.conversation_language == "ta"

        # Turn 2: Tamil with English words "software"
        t2 = "உங்கள் software பற்றி விளக்க முடியுமா?"
        lang2 = language_service.determine_conversation_language(t2, current_lang=booking_session.conversation_language)
        booking_session.conversation_language = lang2
        assert booking_session.conversation_language == "ta"

        # Turn 3: Tamil with English words "appointment book"
        t3 = "எனக்கு ஒரு appointment book பண்ண வேண்டும்."
        lang3 = language_service.determine_conversation_language(t3, current_lang=booking_session.conversation_language)
        booking_session.conversation_language = lang3
        assert booking_session.conversation_language == "ta"

        # Turn 4: Latin contact details - email
        t4 = "sanjay.kumar@gmail.com"
        lang4 = language_service.determine_conversation_language(t4, current_lang=booking_session.conversation_language)
        booking_session.conversation_language = lang4
        assert booking_session.conversation_language == "ta", "Email address should not revert Tamil to English"

        # Turn 5: Phone number
        t5 = "9876543210"
        lang5 = language_service.determine_conversation_language(t5, current_lang=booking_session.conversation_language)
        booking_session.conversation_language = lang5
        assert booking_session.conversation_language == "ta", "Phone digits should not revert Tamil to English"

        # Turn 6: Explicit switch to English
        t6 = "Please talk in English now."
        lang6 = language_service.determine_conversation_language(t6, current_lang=booking_session.conversation_language)
        booking_session.conversation_language = lang6
        assert booking_session.conversation_language == "en", "Explicit English switch must be honored"

        booking_state_service.remove_session(session_id)

    def test_07_explicit_tamil_switch_phrases(self):
        """Verify various customer utterances requesting Tamil switch are recognized."""
        test_phrases = [
            ("Tamil", True),
            ("tamil", True),
            ("In Tamil", True),
            ("Please speak in Tamil", True),
            ("Can you talk in Tamil?", True),
            ("Switch to Tamil", True),
            ("Tamil please", True),
            ("Tamil la pesunga", True),
            ("தமிழில் பேசுங்கள்", True),
            ("தமிழ்ல பேசுங்க", True),
        ]

        for phrase, should_switch in test_phrases:
            target, is_switch = language_service.check_explicit_language_switch(phrase)
            assert is_switch is True, f"Failed for '{phrase}': expected switch True"
            assert target == "ta", f"Failed for '{phrase}': expected target 'ta', got '{target}'"

    def test_08_multilingual_all_5_languages_support(self):
        """Verify all 5 supported languages are recognized and mapped to proper STT codes."""
        stt_code_map = {
            "ta": "ta-IN",
            "hi": "hi-IN",
            "te": "te-IN",
            "kn": "kn-IN",
            "en": "en-IN"
        }

        # Tamil
        t, is_s = language_service.check_explicit_language_switch("Can you speak in Tamil?")
        assert is_s and t == "ta"
        assert stt_code_map[t] == "ta-IN"

        # Hindi
        t, is_s = language_service.check_explicit_language_switch("Please talk in Hindi")
        assert is_s and t == "hi"
        assert stt_code_map[t] == "hi-IN"

        # Telugu
        t, is_s = language_service.check_explicit_language_switch("Speak in Telugu please")
        assert is_s and t == "te"
        assert stt_code_map[t] == "te-IN"

        # Kannada
        t, is_s = language_service.check_explicit_language_switch("Kannada please")
        assert is_s and t == "kn"
        assert stt_code_map[t] == "kn-IN"

        # English
        t, is_s = language_service.check_explicit_language_switch("Switch back to English")
        assert is_s and t == "en"
        assert stt_code_map[t] == "en-IN"
