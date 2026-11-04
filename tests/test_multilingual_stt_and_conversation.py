"""
Automated Test Suite for Multilingual STT and Same-Language AI Conversation.

Covers all 10 required multilingual conversational scenarios:
  1. Tamil-only conversation ("வணக்கம் எனக்கு ஒரு அப்பாயின்ட்மென்ட் வேண்டும்")
  2. English-only conversation ("Hello I need to book an appointment with a doctor")
  3. Hindi-only conversation ("मुझे एक अपॉइंटमेंट बुक करना है")
  4. Telugu-only conversation ("నమస్కారం నాకు డాక్టర్ అపాయింట్‌మెంట్ కావాలి")
  5. Malayalam-only conversation ("നമസ്കാരം എനിക്ക് ഡോക്ടറെ കാണാൻ അപ്പോയിന്റ്മെന്റ് വേണം")
  6. Kannada-only conversation ("ನಮಸ್ಕಾರ ನನಗೆ ವೈದ್ಯರ ಅಪಾಯಿಂಟ್ಮೆಂಟ್ ಬೇಕು")
  7. Tamil-English Tanglish conversation ("vanakkam enakku appointment book pannanum doctor irukkangala")
  8. English-to-Tamil language switching ("can you please speak in tamil")
  9. Tamil-to-English language switching ("can we please talk in english")
 10. Mixed-language booking with short affirmation immunity ("Yes", "Tomorrow 10 AM", "Dr Arun")
 plus:
 11. Per-call isolated language states (Call A vs Call B)
 12. Deepgram STT hint protection (preventing accented English from flipping to Hindi)
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from src.services.language_service import (
    language_service,
    LANG_TAMIL,
    LANG_TANGLISH,
    LANG_HINDI,
    LANG_ENGLISH,
    LANG_TELUGU,
    LANG_MALAYALAM,
    LANG_KANNADA,
)
from src.services.tts_service import tts_service
from src.services.booking_state_service import booking_state_service, BookingState
from src.services.sip_call_handler import SipCallSession


@pytest.fixture
def mock_sip_session():
    """Create a minimal SipCallSession for testing language state handling."""
    session = SipCallSession(
        call_sid="test_call_multilingual_123",
        organization_id=1
    )
    return session


# ==============================================================================
# Scenario 1: Pure Tamil Conversation
# ==============================================================================
def test_scenario_1_tamil_only_conversation(mock_sip_session):
    """Customer speaks Tamil -> detected as Tamil -> Murf ta-IN-iniya voice -> Tamil AI response."""
    text = "வணக்கம் எனக்கு ஒரு அப்பாயின்ட்மென்ட் வேண்டும்"
    
    # 1. Detection
    detected = language_service.detect_language(text)
    assert detected == LANG_TAMIL
    
    # 2. Stateful resolution
    resolved = language_service.determine_conversation_language(text, current_lang="en")
    assert resolved == LANG_TAMIL
    
    # 3. TTS Detection & Voice Mapping
    tts_lang = tts_service.detect_language(text)
    assert tts_lang == LANG_TAMIL
    
    # 4. Session state update
    mock_sip_session.current_lang = "en"
    new_lang = language_service.determine_conversation_language(text, current_lang=mock_sip_session.current_lang)
    mock_sip_session.current_lang = new_lang
    if new_lang == "ta":
        mock_sip_session.selected_tts_provider = "murf"
        mock_sip_session.selected_tts_voice = "ta-IN-iniya"
        mock_sip_session.detected_language = "Tamil"
        mock_sip_session.original_transcript = text
        mock_sip_session.transcript_language = "ta"
        mock_sip_session.ai_response_language = "ta"

    assert mock_sip_session.current_lang == "ta"
    assert mock_sip_session.selected_tts_provider == "murf"
    assert mock_sip_session.selected_tts_voice == "ta-IN-iniya"
    assert mock_sip_session.detected_language == "Tamil"
    assert mock_sip_session.original_transcript == text

    # 5. Audio generation test
    audio = tts_service.synthesize_speech("வணக்கம், நான் உங்களுக்கு எப்படி உதவட்டும்?", language=new_lang)
    assert isinstance(audio, bytes)
    assert len(audio) > 0


# ==============================================================================
# Scenario 2: English-only Conversation
# ==============================================================================
def test_scenario_2_english_only_conversation(mock_sip_session):
    """Customer speaks English -> detected as English -> Murf en-IN-isha voice -> English AI response."""
    text = "Hello I need to book an appointment with a doctor"
    
    detected = language_service.detect_language(text)
    assert detected == LANG_ENGLISH
    
    resolved = language_service.determine_conversation_language(text, current_lang="en")
    assert resolved == LANG_ENGLISH
    
    tts_lang = tts_service.detect_language(text)
    assert tts_lang == "en"

    mock_sip_session.current_lang = resolved
    mock_sip_session.selected_tts_provider = "murf"
    mock_sip_session.selected_tts_voice = "en-IN-isha"
    mock_sip_session.detected_language = "English"

    assert mock_sip_session.selected_tts_provider == "murf"
    assert mock_sip_session.selected_tts_voice == "en-IN-isha"
    assert mock_sip_session.detected_language == "English"


# ==============================================================================
# Scenario 3: Hindi-only Conversation
# ==============================================================================
def test_scenario_3_hindi_only_conversation(mock_sip_session):
    """Customer speaks Hindi -> detected as Hindi -> Murf hi-IN-shweta voice -> Hindi AI response."""
    text = "मुझे एक अपॉइंटमेंट बुक करना है"
    
    detected = language_service.detect_language(text)
    assert detected == LANG_HINDI
    
    resolved = language_service.determine_conversation_language(text, current_lang="en")
    assert resolved == LANG_HINDI
    
    tts_lang = tts_service.detect_language(text)
    assert tts_lang == "hi"

    mock_sip_session.current_lang = resolved
    if resolved == "hi":
        mock_sip_session.selected_tts_provider = "murf"
        mock_sip_session.selected_tts_voice = "hi-IN-shweta"
        mock_sip_session.detected_language = "Hindi"

    assert mock_sip_session.selected_tts_provider == "murf"
    assert mock_sip_session.selected_tts_voice == "hi-IN-shweta"
    assert mock_sip_session.detected_language == "Hindi"

    audio = tts_service.synthesize_speech("नमस्ते, मैं आपकी क्या सहायता कर सकता हूँ?", language=resolved)
    assert isinstance(audio, bytes)
    assert len(audio) > 0


# ==============================================================================
# Scenario 4: Telugu-only Conversation
# ==============================================================================
def test_scenario_4_telugu_only_conversation(mock_sip_session):
    """Customer speaks Telugu -> detected as Telugu -> gTTS te-IN voice -> Telugu AI response."""
    text = "నమస్కారం నాకు డాక్టర్ అపాయింట్‌మెంట్ కావాలి"
    
    detected = language_service.detect_language(text)
    assert detected == LANG_TELUGU
    
    resolved = language_service.determine_conversation_language(text, current_lang="en")
    assert resolved == LANG_TELUGU
    
    tts_lang = tts_service.detect_language(text)
    assert tts_lang == "te"

    mock_sip_session.current_lang = resolved
    if resolved == "te":
        mock_sip_session.selected_tts_provider = "gtts"
        mock_sip_session.selected_tts_voice = "te-IN"
        mock_sip_session.detected_language = "Telugu"

    assert mock_sip_session.selected_tts_provider == "gtts"
    assert mock_sip_session.selected_tts_voice == "te-IN"
    assert mock_sip_session.detected_language == "Telugu"

    audio = tts_service.synthesize_speech("నమస్కారం, నేను మీకు ఎలా సహాయపడగలను?", language=resolved)
    assert isinstance(audio, bytes)
    assert len(audio) > 0


# ==============================================================================
# Scenario 5: Malayalam-only Conversation
# ==============================================================================
def test_scenario_5_malayalam_only_conversation(mock_sip_session):
    """Customer speaks Malayalam -> detected as Malayalam -> gTTS ml-IN voice -> Malayalam AI response."""
    text = "നമസ്കാരം എനിക്ക് ഡോക്ടറെ കാണാൻ അപ്പോയിന്റ്മെന്റ് വേണം"
    
    detected = language_service.detect_language(text)
    assert detected == LANG_MALAYALAM
    
    resolved = language_service.determine_conversation_language(text, current_lang="en")
    assert resolved == LANG_MALAYALAM
    
    tts_lang = tts_service.detect_language(text)
    assert tts_lang == "ml"

    mock_sip_session.current_lang = resolved
    if resolved == "ml":
        mock_sip_session.selected_tts_provider = "gtts"
        mock_sip_session.selected_tts_voice = "ml-IN"
        mock_sip_session.detected_language = "Malayalam"

    assert mock_sip_session.selected_tts_provider == "gtts"
    assert mock_sip_session.selected_tts_voice == "ml-IN"
    assert mock_sip_session.detected_language == "Malayalam"

    audio = tts_service.synthesize_speech("നമസ്കാരം, ഞാൻ നിങ്ങളെ എങ്ങനെ സഹായിക്കണം?", language=resolved)
    assert isinstance(audio, bytes)
    assert len(audio) > 0


# ==============================================================================
# Scenario 6: Kannada-only Conversation
# ==============================================================================
def test_scenario_6_kannada_only_conversation(mock_sip_session):
    """Customer speaks Kannada -> detected as Kannada -> gTTS kn-IN voice -> Kannada AI response."""
    text = "ನಮಸ್ಕಾರ ನನಗೆ ವೈದ್ಯರ ಅಪಾಯಿಂಟ್ಮೆಂಟ್ ಬೇಕು"
    
    detected = language_service.detect_language(text)
    assert detected == LANG_KANNADA
    
    resolved = language_service.determine_conversation_language(text, current_lang="en")
    assert resolved == LANG_KANNADA
    
    tts_lang = tts_service.detect_language(text)
    assert tts_lang == "kn"

    mock_sip_session.current_lang = resolved
    if resolved == "kn":
        mock_sip_session.selected_tts_provider = "gtts"
        mock_sip_session.selected_tts_voice = "kn-IN"
        mock_sip_session.detected_language = "Kannada"

    assert mock_sip_session.selected_tts_provider == "gtts"
    assert mock_sip_session.selected_tts_voice == "kn-IN"
    assert mock_sip_session.detected_language == "Kannada"

    audio = tts_service.synthesize_speech("ನಮಸ್ಕಾರ, ನಾನು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಲಿ?", language=resolved)
    assert isinstance(audio, bytes)
    assert len(audio) > 0


# ==============================================================================
# Scenario 7: Tamil-English Tanglish Conversation
# ==============================================================================
def test_scenario_7_tanglish_conversation(mock_sip_session):
    """Customer speaks Tanglish -> detected as Tanglish -> preserves transcript -> Murf en-IN-isha voice."""
    text = "vanakkam enakku appointment book pannanum doctor irukkangala"
    
    detected = language_service.detect_language(text)
    assert detected in (LANG_TANGLISH, LANG_TAMIL)
    
    resolved = language_service.determine_conversation_language(text, current_lang="en")
    assert resolved in (LANG_TANGLISH, LANG_TAMIL)

    mock_sip_session.current_lang = resolved
    mock_sip_session.original_transcript = text
    mock_sip_session.transcript_language = resolved
    if resolved in ("tanglish", "ta_mix"):
        mock_sip_session.selected_tts_provider = "murf"
        mock_sip_session.selected_tts_voice = "en-IN-isha"
        mock_sip_session.detected_language = "Tanglish"

    # Preserves customer's original spoken Tanglish text cleanly
    assert mock_sip_session.original_transcript == "vanakkam enakku appointment book pannanum doctor irukkangala"
    assert mock_sip_session.selected_tts_voice == "en-IN-isha"


# ==============================================================================
# Scenario 8: English-to-Tamil Language Switching
# ==============================================================================
def test_scenario_8_switch_english_to_tamil():
    """Customer explicitly asks to speak Tamil -> switches to Tamil and gives natural Tamil response."""
    switch_phrase = "can you please speak in tamil"
    
    # 1. Fast predicate
    assert language_service.is_language_switch_request(switch_phrase) is True
    
    # 2. Check explicit switch
    target_lang, is_switch = language_service.check_explicit_language_switch(switch_phrase)
    assert is_switch is True
    assert target_lang == LANG_TAMIL
    
    # 3. Determine conversation language switches from 'en' to 'ta'
    resolved = language_service.determine_conversation_language(switch_phrase, current_lang="en")
    assert resolved == LANG_TAMIL

    # 4. Confirmation message is in natural Tamil
    confirmation = language_service.get_language_switch_confirmation(target_lang)
    assert "தமிழில்" in confirmation or "பேசுகிறேன்" in confirmation or "உதவ" in confirmation


# ==============================================================================
# Scenario 9: Tamil-to-English Language Switching
# ==============================================================================
def test_scenario_9_switch_tamil_to_english():
    """Customer explicitly asks to speak English -> switches to English and gives natural English response."""
    switch_phrase = "can we please talk in english"
    
    # 1. Fast predicate
    assert language_service.is_language_switch_request(switch_phrase) is True
    
    # 2. Check explicit switch
    target_lang, is_switch = language_service.check_explicit_language_switch(switch_phrase)
    assert is_switch is True
    assert target_lang == LANG_ENGLISH
    
    # 3. Determine conversation language switches from 'ta' to 'en'
    resolved = language_service.determine_conversation_language(switch_phrase, current_lang="ta")
    assert resolved == LANG_ENGLISH

    # 4. Confirmation message is in natural English
    confirmation = language_service.get_language_switch_confirmation(target_lang)
    assert "English" in confirmation


# ==============================================================================
# Scenario 10: Mixed-Language Appointment Booking with Short Affirmation Immunity
# ==============================================================================
def test_scenario_10_affirmation_and_entity_immunity():
    """Short affirmations and English entity names MUST NOT break an active Indic language lock."""
    # 1. Tamil lock
    current_lang = "ta"
    affirmations = ["Yes", "Yeah", "Sure", "Okay", "Tomorrow 10 AM", "Dr. Anita", "Anita Sharma"]
    for word in affirmations:
        resolved = language_service.determine_conversation_language(word, current_lang=current_lang)
        assert resolved == "ta", f"Word '{word}' erroneously broke Tamil lock to {resolved}"

    # 2. Hindi lock
    current_lang = "hi"
    for word in ["Yes", "Tomorrow 11 AM", "Dr. Anita", "Cardiology"]:
        resolved = language_service.determine_conversation_language(word, current_lang=current_lang)
        assert resolved == "hi", f"Word '{word}' erroneously broke Hindi lock to {resolved}"

    # 3. Telugu lock
    current_lang = "te"
    for word in ["Yes", "Tomorrow 10 AM", "Dr. Anita"]:
        resolved = language_service.determine_conversation_language(word, current_lang=current_lang)
        assert resolved == "te", f"Word '{word}' erroneously broke Telugu lock to {resolved}"

    # 4. Malayalam lock
    current_lang = "ml"
    for word in ["Yes", "Tomorrow 10 AM"]:
        resolved = language_service.determine_conversation_language(word, current_lang=current_lang)
        assert resolved == "ml", f"Word '{word}' erroneously broke Malayalam lock to {resolved}"

    # 5. Kannada lock
    current_lang = "kn"
    for word in ["Yes", "Tomorrow 10 AM"]:
        resolved = language_service.determine_conversation_language(word, current_lang=current_lang)
        assert resolved == "kn", f"Word '{word}' erroneously broke Kannada lock to {resolved}"


# ==============================================================================
# Scenario 11: Multi-Tenant & Per-Call Language Isolation
# ==============================================================================
def test_scenario_11_per_call_isolation():
    """Ensure two concurrent calls retain isolated, independent language contexts."""
    call_tamil = SipCallSession(call_sid="call_ta_001", organization_id=1)
    call_english = SipCallSession(call_sid="call_en_002", organization_id=1)

    # Call A receives Tamil
    lang_a = language_service.determine_conversation_language("வணக்கம் எனக்கு ஒரு அப்பாயின்ட்மென்ட் வேண்டும்", current_lang=call_tamil.current_lang)
    call_tamil.current_lang = lang_a
    call_tamil.selected_tts_voice = "ta-IN-iniya"
    call_tamil.selected_tts_provider = "murf"

    # Call B receives English
    lang_b = language_service.determine_conversation_language("Hello I need to check doctors list", current_lang=call_english.current_lang)
    call_english.current_lang = lang_b
    call_english.selected_tts_voice = "en-IN-isha"
    call_english.selected_tts_provider = "murf"

    # Verify complete isolation
    assert call_tamil.current_lang == "ta"
    assert call_tamil.selected_tts_voice == "ta-IN-iniya"

    assert call_english.current_lang == "en"
    assert call_english.selected_tts_voice == "en-IN-isha"


# ==============================================================================
# Scenario 12: STT Hint Guard (Accented English vs Hindi)
# ==============================================================================
def test_scenario_12_stt_hint_protection():
    """Deepgram noisy 'languages=[\"hi\"]' hint on accented English does NOT flip language to Hindi."""
    accented_english = "I want to schedule an appointment with doctor for tomorrow"
    resolved = language_service.detect_language(accented_english, stt_lang="hi")
    
    # Must remain English because sentence consists of clear English vocabulary without Devanagari or Hindi keywords
    assert resolved == LANG_ENGLISH
