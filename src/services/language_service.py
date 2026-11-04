"""
English-Only Enforcement & Language Guard Service.

Enforces strict English-only speech recognition, conversation management,
and response generation.
Automatic language detection and switching are completely disabled.
If a customer speaks another language or requests a language switch, the service
redirects the conversation politely in English:
"I can only assist you in English. Please could you continue in English?"
"""

import logging
import re
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)

LANG_ENGLISH = "en"
ENGLISH_ONLY_REDIRECT_MESSAGE = "I can only assist you in English. Please could you continue in English?"

# Non-English detection patterns (Spanish, Hindi, Tamil, Telugu, Malayalam, Kannada, French, German, etc.)
_NON_ENGLISH_PATTERNS = [
    # Non-Latin scripts: Tamil, Devanagari, Telugu, Malayalam, Kannada, Arabic, Cyrillic, CJK
    r'[\u0b80-\u0bff]',  # Tamil
    r'[\u0900-\u097f]',  # Devanagari (Hindi, Marathi)
    r'[\u0c00-\u0c7f]',  # Telugu
    r'[\u0d00-\u0d7f]',  # Malayalam
    r'[\u0c80-\u0cff]',  # Kannada
    r'[\u0600-\u06ff]',  # Arabic
    r'[\u0400-\u04ff]',  # Cyrillic
    r'[\u4e00-\u9fff]',  # Chinese

    # Spanish phrases and keywords (excluding English shared words like hospital/clinic)
    r'\b(?:hola|buenos\s+dias|buenos\s+d[ií]as|buenas\s+tardes|buenas\s+noches|por\s+favor|gracias|muchas\s+gracias|de\s+nada)\b',
    r'\b(?:habla|hablas|hablar|hable|hablan|espanol|espa[nñ]ol|cita|medico|m[eé]dico|doctora|enfermera)\b',
    r'\b(?:quiero|necesito|puedo|podria|podr[ií]a|donde|d[oó]nde|cuando|cu[aá]ndo|como|c[oó]mo|cuanto|cu[aá]nto|para\s+ma[nñ]ana)\b',

    # Hindi / Hinglish phrases and keywords
    r'\b(?:namaste|namaskar|kya\s+hal|kaise\s+ho|shukriya|dhanyawad|alvida)\b',
    r'\b(?:mujhe|mera|meri|mere|aap|aapka|aapke|aapki|hum|humara|kripya|kripia|chahiye|lena|dena|karo|karna|karenge|batao|batayein|boliye|samajh|milna|kitna|hoga|hogi|theek|accha|haan|nahi|subah|shaam|baje|samay|hai|hain|hoon)\b',
    r'\b(?:baat\s+karo|mein\s+baat|se\s+appointment|ke\s+saath|doctor\s+se)\b',

    # Tamil / Tanglish phrases and keywords
    r'\b(?:vanakkam|tamizh|tamil|pannanum|venum|vendum|iruka|irukku|irukkangala|kitta|naalai|inniku|ennodaya|peru)\b',
    r'\b(?:pesunga|pesanum|pesalam|mudiyuma|mudiyum|varuma|kudunga|sollunga|solren|enga|eppadi|romba|aama|illa|seri|sari)\b',
    r'\b(?:la\s+pesu|la\s+pesunga)\b',

    # French phrases
    r'\b(?:bonjour|bonsoir|s\'il\s+vous\s+pla[iî]t|merci|au\s+revoir|parlez-vous|fran[cç]ais)\b',

    # German phrases
    r'\b(?:guten\s+tag|guten\s+morgen|bitte|danke|auf\s+wiedersehen|sprechen\s+sie|deutsch)\b',
]

# Explicit language switch requests (e.g. "Can you talk in Spanish?", "Speak in Hindi", "Switch to Tamil")
_LANGUAGE_SWITCH_PATTERNS = [
    r'\b(?:talk|speak|chat|converse|respond|reply|answer|communicate)\s+(?:to\s+me\s+)?in\s+([a-zA-Z\u0b80-\u0bff\u0900-\u097f]+)',
    r'\bcan\s+you\s+(?:please\s+)?(?:talk|speak|chat|converse|respond)\s+(?:to\s+me\s+)?in\s+([a-zA-Z\u0b80-\u0bff\u0900-\u097f]+)',
    r'\bcould\s+you\s+(?:please\s+)?(?:talk|speak|chat)\s+(?:to\s+me\s+)?in\s+([a-zA-Z\u0b80-\u0bff\u0900-\u097f]+)',
    r'\bplease\s+(?:talk|speak|chat)\s+(?:to\s+me\s+)?in\s+([a-zA-Z\u0b80-\u0bff\u0900-\u097f]+)',
    r'\bswitch\s+to\s+([a-zA-Z\u0b80-\u0bff\u0900-\u097f]+)',
    r'\bchange\s+(?:the\s+)?language\s+to\s+([a-zA-Z\u0b80-\u0bff\u0900-\u097f]+)',
    r'\b(?:talk|speak)\s+(?:spanish|hindi|tamil|telugu|malayalam|kannada|french|german)(?:\s+please)?\b',
    r'\b(?:spanish|hindi|tamil|telugu|malayalam|kannada|french|german)\s+please\b',
]


class LanguageDetectionService:
    """
    English-Only Language Enforcement Service.
    Enforces English across STT, AgentEngine, LLM, and TTS pipelines.
    """

    def __init__(self):
        self.default_language = LANG_ENGLISH

    def is_non_english_input(self, text: str) -> bool:
        """
        Detects if input text is non-English (Spanish, Hindi, Tamil, etc.).
        """
        txt = (text or "").strip()
        if not txt:
            return False

        txt_lower = txt.lower()
        for pattern in _NON_ENGLISH_PATTERNS:
            if re.search(pattern, txt_lower, re.IGNORECASE):
                return True
        return False

    def is_language_switch_request(self, text: str) -> bool:
        """
        Returns True if caller asks to switch to another language.
        """
        txt = (text or "").strip().lower()
        if not txt:
            return False

        for pattern in _LANGUAGE_SWITCH_PATTERNS:
            if re.search(pattern, txt, re.IGNORECASE):
                return True
        return False

    def check_explicit_language_switch(self, text: str) -> Tuple[Optional[str], bool]:
        """
        Checks if the caller requested another language.
        Instead of returning another language code, flags the switch request
        so the system can deliver the English-only redirection.
        Returns:
            ("en", True) if a non-English switch was requested.
            ("en", False) otherwise.
        """
        if self.is_language_switch_request(text) or self.is_non_english_input(text):
            logger.info(f"[ENGLISH_ONLY_GUARD] Detected foreign language or switch request in: '{text[:60]}'")
            return LANG_ENGLISH, True
        return LANG_ENGLISH, False

    def detect_spoken_language(self, text: str, stt_lang: Optional[str] = None) -> str:
        """
        Always returns English ('en') to enforce English STT and dialogue state.
        """
        return LANG_ENGLISH

    def detect_language(self, text: str, stt_lang: Optional[str] = None) -> str:
        """
        Always returns English ('en').
        """
        return LANG_ENGLISH

    def resolve_call_language_turn(
        self,
        text: str,
        current_preferred_lang: Optional[str] = None,
        stt_lang: Optional[str] = None,
        confidence: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Resolves per-call language state, strictly locking to English ('en').
        """
        is_switch = self.is_language_switch_request(text) or self.is_non_english_input(text)
        return {
            "detected_utterance_language": LANG_ENGLISH,
            "preferred_response_language": LANG_ENGLISH,
            "language_switch_requested": is_switch,
            "requested_switch_language": None,
            "stt_language_confidence": float(confidence or 1.0),
        }

    def determine_conversation_language(
        self,
        text: str,
        current_lang: Optional[str] = None,
        stt_lang: Optional[str] = None
    ) -> str:
        """
        Always returns English ('en').
        """
        return LANG_ENGLISH

    def get_language_switch_confirmation(
        self,
        target_lang: str = "en",
        booking_session=None
    ) -> str:
        """
        Returns polite English-only redirection message when another language is requested.
        "I can only assist you in English. Please could you continue in English?"
        """
        return ENGLISH_ONLY_REDIRECT_MESSAGE


language_service = LanguageDetectionService()
