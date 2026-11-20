"""
Test Suite: Tamil Voice Consistency and Natural Spoken Tamil
Validates:
1. Voice Selection Consistency:
   - ta-IN-iniya (GEN1) for Tamil
   - en-IN-isha (GEN2) for English
   - hi-IN-shweta (GEN1) for Hindi
2. TTS Circuit Breaker & Fallback:
   - Timeout increased to 7.0s, does not trip on a single transient delay.
   - Requires 3 consecutive failures before tripping 60s circuit breaker.
   - Preserves language voice selection across mixed English-Tamil text.
3. Spoken Conversational Tamil (பேச்சுத் தமிழ்):
   - Fast-path responses use natural spoken Tamil (e.g. "நான் உங்களுக்கு எப்படி help பண்ணலாம்?").
   - Language switch responses use natural spoken Tamil.
   - Booking state prompts preserve English technical loanwords (service, appointment, date, time, confirm, cancel, email, phone).
   - Avoids classical/literary archaic Tamil (தூய தமிழ்).
"""

import pytest
import time
import os
from unittest.mock import patch, MagicMock

from src.services.murf_tts import murf_tts, parse_voice_and_model, MURF_LANGUAGE_VOICES
from src.services.tts_service import TTSService, tts_service
from src.services.language_service import language_service, LANG_TAMIL, LANG_ENGLISH
from src.services.ai_service import ai_service
from src.services.booking_state_service import booking_state_service, BookingSession, BookingState


def test_murf_voice_and_model_mapping():
    """Verify Murf voice and model mapping for Tamil, English, and Hindi."""
    # Tamil -> ta-IN-iniya (GEN1)
    v_ta, m_ta = parse_voice_and_model("ta-IN-iniya", "GEN1")
    assert v_ta == "ta-IN-iniya"
    assert m_ta == "GEN1"
    assert MURF_LANGUAGE_VOICES["ta"] == "ta-IN-iniya"

    # English -> en-IN-isha (GEN2)
    v_en, m_en = parse_voice_and_model("en-IN-isha", "GEN2")
    assert v_en == "en-IN-isha"
    assert m_en == "GEN2"
    assert MURF_LANGUAGE_VOICES["en"] == "en-IN-isha"

    # Hindi -> hi-IN-shweta (GEN1)
    assert MURF_LANGUAGE_VOICES["hi"] == "hi-IN-shweta"


def test_language_service_tts_voice_and_model():
    """Verify language_service returns correct voice, model, and provider for each language."""
    v_ta, m_ta, prov_ta = language_service.get_tts_voice_and_model("ta")
    assert v_ta == "ta-IN-iniya"
    assert m_ta == "GEN1"
    assert prov_ta == "murf"

    v_en, m_en, prov_en = language_service.get_tts_voice_and_model("en")
    assert v_en == "en-IN-isha"
    assert m_en == "GEN2"
    assert prov_en == "murf"


def test_tts_service_preserves_tamil_voice_for_mixed_code_switching():
    """Verify that during Tamil conversation, mixed text (Tamil + English words) uses Tamil voice."""
    mixed_text = "Aminoid AI-ல AI voice agents மற்றும் ERP software இருக்கு."
    
    # Temporarily bypass pytest environment guard to test actual routing
    env_clean = {k: v for k, v in os.environ.items() if k not in ("PYTEST_CURRENT_TEST", "APP_ENV")}
    with patch.dict(os.environ, env_clean, clear=True):
        with patch.object(murf_tts, "synthesize", return_value=b"RIFF_FAKE_AUDIO_DATA_FOR_TEST_ONLY_123456789") as mock_synth:
            with patch.object(TTSService, "_murf_disabled_until", 0.0):
                tts_service.synthesize_speech(mixed_text, language="ta")
                mock_synth.assert_called_once()
                call_kwargs = mock_synth.call_args[1]
                assert call_kwargs.get("voice_id") == "ta-IN-iniya"
                assert call_kwargs.get("language") == "ta"
                assert call_kwargs.get("model") == "GEN1"


def test_tts_circuit_breaker_requires_three_failures():
    """Verify that a single transient timeout does NOT trip the 60s circuit breaker."""
    TTSService._murf_consecutive_failures = 0
    TTSService._murf_disabled_until = 0.0

    env_clean = {k: v for k, v in os.environ.items() if k not in ("PYTEST_CURRENT_TEST", "APP_ENV")}
    with patch.dict(os.environ, env_clean, clear=True):
        with patch.object(murf_tts, "synthesize", side_effect=Exception("Transient network glitch")):
            with patch.object(tts_service, "_synthesize_gtts", return_value=b"GTTS_AUDIO"):
                # Turn 1: failure 1
                tts_service.synthesize_speech("Test turn 1", language="ta")
                assert TTSService._murf_consecutive_failures == 1
                assert TTSService._murf_disabled_until == 0.0  # Not tripped yet

                # Turn 2: failure 2
                tts_service.synthesize_speech("Test turn 2", language="ta")
                assert TTSService._murf_consecutive_failures == 2
                assert TTSService._murf_disabled_until == 0.0  # Still not tripped

                # Turn 3: failure 3 -> trips breaker
                tts_service.synthesize_speech("Test turn 3", language="ta")
                assert TTSService._murf_consecutive_failures == 3
                assert TTSService._murf_disabled_until > time.monotonic()  # Now tripped

    # Reset
    TTSService._murf_consecutive_failures = 0
    TTSService._murf_disabled_until = 0.0


def test_spoken_conversational_tamil_fast_paths():
    """Verify that fast-path responses return natural spoken Tamil instead of literary classical Tamil."""
    session = BookingSession("test_session_ta")
    session.conversation_language = "ta"
    session.greeting_sent = True

    # 1. Audible check ("கேட்கிறதா")
    resp_audible = ai_service._fast_rule_response("உங்களுக்கு கேட்கிறதா?", "general_conversation", booking_session=session, conversation_language="ta")
    assert resp_audible is not None
    assert "help பண்ணலாம்" in resp_audible
    assert "எவ்வாறு உதவலாம்" not in resp_audible

    # 2. Greeting when already greeted ("வணக்கம்")
    resp_greet = ai_service._fast_rule_response("வணக்கம்", "greeting", booking_session=session, is_ongoing=True, conversation_language="ta")
    assert resp_greet is not None
    assert "help பண்ணலாம்" in resp_greet

    # 3. Pleasantry ("how are you" / "நலமா")
    resp_how = ai_service._fast_rule_response("நலமா?", "general_conversation", booking_session=session, conversation_language="ta")
    assert resp_how is not None
    assert "நல்லா இருக்கேன்" in resp_how
    assert "நலமாக இருக்கிறேன்" not in resp_how

    # 4. Thanks ("நன்றி")
    resp_thanks = ai_service._fast_rule_response("நன்றி", "general_conversation", booking_session=session, conversation_language="ta")
    assert resp_thanks is not None
    assert "call பண்ணதுக்கு" in resp_thanks
    assert "தொடர்பு கொண்டதற்கு" not in resp_thanks

    # 5. Goodbye ("bye" / "போய் வருகிறேன்")
    resp_bye = ai_service._fast_rule_response("bye", "general_conversation", booking_session=session, conversation_language="ta")
    assert resp_bye is not None
    assert "Have a great day" in resp_bye or "Bye" in resp_bye
    assert "போய் வருகிறேன்" not in resp_bye


def test_language_service_switch_spoken_tamil():
    """Verify that language switch prompts use natural spoken Tamil with English loanwords."""
    # General language switch to Tamil
    conf = language_service.get_language_switch_confirmation("ta")
    assert "கண்டிப்பா! இனிமேல் தமிழ்ல பேசலாம்" in conf
    assert "help பண்ணலாம்" in conf
    assert "நிச்சயமாக! இனிமேல் தமிழில் பேசுகிறேன்" not in conf

    # Pending field: date/time
    mock_session = MagicMock()
    mock_session.is_active.return_value = True
    mock_session.last_asked_field = "date"
    conf_date = language_service.get_language_switch_confirmation("ta", booking_session=mock_session)
    assert "date" in conf_date
    assert "appointment book பண்ணணும்" in conf_date

    # Pending field: phone
    mock_session.last_asked_field = "phone_number"
    conf_phone = language_service.get_language_switch_confirmation("ta", booking_session=mock_session)
    assert "phone number" in conf_phone

    # Pending field: email
    mock_session.last_asked_field = "email"
    conf_email = language_service.get_language_switch_confirmation("ta", booking_session=mock_session)
    assert "email id" in conf_email


def test_booking_state_prompts_spoken_tamil():
    """Verify booking state service prompts use natural spoken Tamil and retain English technical terms."""
    # 1. Cancellation prompt
    session1 = BookingSession("test_bk_ta_1")
    session1.conversation_language = "ta"
    session1.patient_name = "Karthik"
    session1.contact_number = "9876543210"
    session1.date = "Tomorrow"
    session1.time = "10:00 AM"
    session1.state = BookingState.WAITING_FOR_TIME
    cancel_resp = booking_state_service.process_booking_turn("cancel booking", session1)
    assert "Booking cancel பண்ணியாச்சு" in cancel_resp
    assert "முன்பதிவு ரத்து செய்யப்பட்டது" not in cancel_resp

    # 2. Change time prompt
    session2 = BookingSession("test_bk_ta_2")
    session2.conversation_language = "ta"
    session2.patient_name = "Karthik"
    session2.contact_number = "9876543210"
    session2.date = "Tomorrow"
    session2.time = "10:00 AM"
    session2.state = BookingState.WAITING_FOR_CONFIRMATION
    change_time_resp = booking_state_service.process_booking_turn("change time", session2)
    assert "time உங்களுக்கு வசதியா இருக்கும்" in change_time_resp


def test_ai_service_system_instruction_spoken_tamil():
    """Verify ai_service._get_system_instruction contains Spoken Tamil directives and avoid/prefer examples."""
    sys_prompt = ai_service._get_system_instruction(
        organization_name="Aminoid AI",
        conversation_language="ta",
        business_type="SOFTWARE",
        booking_type="SALES_DEMO"
    )
    assert "MANDATORY NATURAL SPOKEN TAMIL (பேச்சுத் தமிழ்) DIRECTIVE" in sys_prompt
    assert "DO NOT USE LITERARY, ARCHAIC, OR CLASSICAL WRITTEN TAMIL" in sys_prompt
    assert "NATURAL CODE-SWITCHING" in sys_prompt
    assert "service" in sys_prompt
    assert "appointment" in sys_prompt
    assert "விரும்புறீங்க" in sys_prompt
