with open('src/services/language_service.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''    def get_language_name(self, lang: str) -> str:
        """Returns the English name for a language code."""
        return LANGUAGE_NAMES.get(lang, "English")


language_service = LanguageDetectionService()'''

replacement = '''    def get_language_name(self, lang: str) -> str:
        """Returns the English name for a language code."""
        return LANGUAGE_NAMES.get(lang, "English")

    def get_language_fallback_prompt(self, lang: str, org_name: Optional[str] = None) -> str:
        """Returns a polite fallback response in the specified language."""
        target = lang if lang in SUPPORTED_LANGUAGES else LANG_ENGLISH
        name = org_name or "our office"
        fallbacks = {
            LANG_TAMIL: f"நான் {name} நிறுவனத்திற்கான AI வரவேற்பாளர். உங்களுக்கு இன்று எவ்வாறு உதவலாம்?",
            LANG_ENGLISH: f"I am here to assist you with inquiries and appointments for {name}. How can I help you today?",
            LANG_HINDI: f"मैं {name} का AI स्वागतकर्ता हूँ। आज मैं आपकी क्या सहायता कर सकता हूँ?",
            LANG_KANNADA: f"ನಾನು {name} ಸಂಸ್ಥೆಯ AI ಸ್ವಾಗತಕಾರ. ಇಂದು ನಾನು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಲಿ?",
            LANG_TELUGU: f"నేను {name} సంస్థ యొక్క AI రిసెప్షనిస్ట్‌ని. ఈరోజు నేను మీకు ఎలా సహాయపడగలను?",
        }
        return fallbacks.get(target, fallbacks[LANG_ENGLISH])


language_service = LanguageDetectionService()'''

if target in text:
    text = text.replace(target, replacement, 1)
    with open('src/services/language_service.py', 'w', encoding='utf-8', newline='') as f:
        f.write(text)
    print("SUCCESS")
else:
    print("TARGET NOT FOUND")

