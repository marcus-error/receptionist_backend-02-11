import sys, os

backend_path = os.path.abspath(r"C:\projects\2-11 ai receptionist updated\receptionist_backend-02-11")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from src.services.tts_service import tts_service
from src.services.language_service import language_service

print("Testing TTS Language Detection & Routing...")

samples = [
    ("வணக்கம், நான் உங்களுக்கு எப்படி உதவ முடியும்?", "ta"),
    ("नमस्ते, मैं आपकी क्या मदद कर सकता हूँ?", "hi"),
    ("ನಮಸ್ಕಾರ, ನಾನು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಬಹುದು?", "kn"),
    ("నమస్కారం, నేను మీకు ఎలా సహాయపడగలను?", "te"),
    ("Hello, how may I help you today?", "en"),
]

for text, exp_lang in samples:
    detected = tts_service.detect_language(text, fallback="en")
    print(f"[TTS LANG DETECT] '{text[:25]}...' -> detected={detected} (expected={exp_lang})")
    assert detected == exp_lang, f"Expected {exp_lang}, got {detected}"

    voice_id, model, provider = language_service.get_tts_voice_and_model(detected)
    print(f"  Routing: voice={voice_id}, model={model}, provider={provider}")
    if detected == "ta":
        assert voice_id == "ta-IN-iniya" and provider == "murf"
    elif detected == "hi":
        assert voice_id == "hi-IN-shweta" and provider == "murf"
    elif detected == "kn":
        assert voice_id == "kn" and provider == "gtts"
    elif detected == "te":
        assert voice_id == "te" and provider == "gtts"
    elif detected == "en":
        assert voice_id == "en-IN-isha" and provider == "murf"

print("\nALL TTS ROUTING TESTS PASSED 100%!")

