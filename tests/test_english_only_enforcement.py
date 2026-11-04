import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.services.language_service import language_service, ENGLISH_ONLY_REDIRECT_MESSAGE
from src.services.agent_engine import agent_engine
from src.services.booking_state_service import booking_state_service
from src.services.ai_service import ai_service
from src.services.murf_tts import murf_tts, MURF_LANGUAGE_VOICES
from src.services.deepgram_stt_service import DeepgramStreamingSession
from src.config.env import settings


def test_deepgram_stt_language_configuration():
    """Verify STT configuration is strictly English."""
    assert getattr(settings, "DEEPGRAM_LANGUAGE", "en") == "en"
    session = DeepgramStreamingSession(call_id="test_stt_en", sample_rate=8000)
    assert session.language == "en"
    ws_url = session._build_ws_url()
    assert "language=en" in ws_url


def test_language_service_always_returns_english():
    """Verify language_service detect and determine methods always lock to 'en'."""
    assert language_service.detect_spoken_language("Hello") == "en"
    assert language_service.detect_spoken_language("வணக்கம்") == "en"
    assert language_service.detect_spoken_language("Hola doctor") == "en"
    assert language_service.detect_language("Mujhe appointment chahiye") == "en"
    assert language_service.determine_conversation_language("Tamil please", current_lang="en") == "en"


@pytest.mark.asyncio
async def test_spanish_input_does_not_switch_and_asks_for_english():
    """Test that Spanish customer input is intercepted and redirected in English."""
    spanish_inputs = [
        "Hola, ¿puedo hablar con un médico por favor?",
        "Buenos días, necesito una cita para mañana",
        "Por favor habla en español"
    ]
    for text in spanish_inputs:
        assert language_service.is_non_english_input(text) or language_service.is_language_switch_request(text)
        _, is_switch = language_service.check_explicit_language_switch(text)
        assert is_switch is True

        # Test agent engine response
        session_id = f"test_es_{abs(hash(text))}"
        res = await agent_engine.process_turn(user_text=text, session_id=session_id, channel="phone")
        assert res["language"] == "en"
        assert res["response"] == ENGLISH_ONLY_REDIRECT_MESSAGE
        assert "English" in res["response"]


@pytest.mark.asyncio
async def test_hindi_input_does_not_switch_and_asks_for_english():
    """Test that Hindi customer input is intercepted and redirected in English."""
    hindi_inputs = [
        "Mujhe doctor se appointment lena hai",
        "Namaste, kya doctor available hain?",
        "Kal subah appointment mil sakta hai kya?",
        "Hindi mein baat karo please"
    ]
    for text in hindi_inputs:
        assert language_service.is_non_english_input(text) or language_service.is_language_switch_request(text)
        _, is_switch = language_service.check_explicit_language_switch(text)
        assert is_switch is True

        # Test agent engine response
        session_id = f"test_hi_{abs(hash(text))}"
        res = await agent_engine.process_turn(user_text=text, session_id=session_id, channel="phone")
        assert res["language"] == "en"
        assert res["response"] == ENGLISH_ONLY_REDIRECT_MESSAGE
        assert "English" in res["response"]


@pytest.mark.asyncio
async def test_tamil_input_does_not_switch_and_asks_for_english():
    """Test that Tamil customer input is intercepted and redirected in English."""
    tamil_inputs = [
        "வணக்கம் எனக்கு ஒரு அப்பாயின்ட்மென்ட் வேண்டும்",
        "Doctor appointment book panna mudiyuma?",
        "Tamil la pesunga please"
    ]
    for text in tamil_inputs:
        assert language_service.is_non_english_input(text) or language_service.is_language_switch_request(text)
        _, is_switch = language_service.check_explicit_language_switch(text)
        assert is_switch is True

        # Test agent engine response
        session_id = f"test_ta_{abs(hash(text))}"
        res = await agent_engine.process_turn(user_text=text, session_id=session_id, channel="phone")
        assert res["language"] == "en"
        assert res["response"] == ENGLISH_ONLY_REDIRECT_MESSAGE
        assert "English" in res["response"]


@pytest.mark.asyncio
async def test_english_conversations_flow_naturally():
    """Test that English customer inputs process normally without false redirection."""
    english_inputs = [
        "Hello, I would like to book an appointment with Dr. Sharma tomorrow morning.",
        "What are your hospital working hours?",
        "My name is John Doe and my phone number is 9876543210."
    ]
    for text in english_inputs:
        assert language_service.is_non_english_input(text) is False
        assert language_service.is_language_switch_request(text) is False

        session_id = f"test_en_{abs(hash(text))}"
        res = await agent_engine.process_turn(user_text=text, session_id=session_id, channel="phone")
        assert res["language"] == "en"
        assert res["response"] != ENGLISH_ONLY_REDIRECT_MESSAGE
        assert len(res["response"]) > 0


def test_tts_voice_mapping_enforces_english():
    """Verify TTS voice mapping is locked to English (en-IN-isha)."""
    assert MURF_LANGUAGE_VOICES == {"en": "en-IN-isha"}
    api_key, voice_id, model = murf_tts.get_credentials()
    assert voice_id == "en-IN-isha"
    assert model == "GEN2"


def test_llm_system_prompt_contains_strict_english_directive():
    """Verify LLM prompt includes strict English-only directive and zero multilingual locks."""
    prompt = ai_service._get_system_instruction(organization_name="Default Organization")
    assert "STRICT ENGLISH-ONLY CONVERSATION ENFORCEMENT" in prompt
    assert "NEVER respond in Spanish, Hindi, Tamil" in prompt
    assert "LANGUAGE LOCK: The caller is speaking in Tamil" not in prompt
    assert "LANGUAGE LOCK: The caller is speaking in Hindi" not in prompt
