"""
Multilingual Language Service — CallZenza AI Receptionist.

Supports strictly 5 languages:
1. Tamil (ta)
2. English (en)
3. Hindi (hi)
4. Kannada (kn)
5. Telugu (te)

Provides:
- Explicit language switch detection (English, transliterated, and native script).
- Per-call isolated language state management.
- Dynamic confirmation generation in the target language.
- STT language code resolution.
- TTS voice mapping (Murf / gTTS).
- Script detection without breaking on English names, emails, or technical terms.
"""

import logging
import re
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)

LANG_TAMIL = "ta"
LANG_ENGLISH = "en"
LANG_HINDI = "hi"
LANG_KANNADA = "kn"
LANG_TELUGU = "te"

SUPPORTED_LANGUAGES = {LANG_TAMIL, LANG_ENGLISH, LANG_HINDI, LANG_KANNADA, LANG_TELUGU}

LANGUAGE_NAMES = {
    LANG_TAMIL: "Tamil",
    LANG_ENGLISH: "English",
    LANG_HINDI: "Hindi",
    LANG_KANNADA: "Kannada",
    LANG_TELUGU: "Telugu",
}

# Provider voice mapping for the 5 supported languages
# Murf natively supports en-IN, ta-IN, hi-IN.
# Kannada (kn) and Telugu (te) route cleanly via native gTTS / fallback.
TTS_VOICE_CONFIG = {
    LANG_ENGLISH: {"voice_id": "en-IN-isha", "model": "GEN2", "provider": "murf"},
    LANG_TAMIL: {"voice_id": "ta-IN-iniya", "model": "GEN1", "provider": "murf"},
    LANG_HINDI: {"voice_id": "hi-IN-shweta", "model": "GEN1", "provider": "murf"},
    LANG_KANNADA: {"voice_id": "kn", "model": "default", "provider": "gtts"},
    LANG_TELUGU: {"voice_id": "te", "model": "default", "provider": "gtts"},
}

# Deepgram Nova-3 supported language codes
STT_LANGUAGE_MAP = {
    LANG_ENGLISH: "en",
    LANG_TAMIL: "ta",
    LANG_HINDI: "hi",
    LANG_KANNADA: "kn",
    LANG_TELUGU: "te",
}

# Name to ISO code map for parsing
_NAME_TO_CODE = {
    "tamil": LANG_TAMIL,
    "tamizh": LANG_TAMIL,
    "thamil": LANG_TAMIL,
    "thamizh": LANG_TAMIL,
    "tamilan": LANG_TAMIL,
    "tamizha": LANG_TAMIL,
    "தமிழ்": LANG_TAMIL,
    "தமிழு": LANG_TAMIL,
    "தமிழில்": LANG_TAMIL,
    "தமிழ்ல": LANG_TAMIL,
    "english": LANG_ENGLISH,
    "angrezi": LANG_ENGLISH,
    "ஆங்கிலம்": LANG_ENGLISH,
    "अंग्रेजी": LANG_ENGLISH,
    "ಇಂಗ್ಲಿಷ್": LANG_ENGLISH,
    "ఇంగ్లీష్": LANG_ENGLISH,
    "hindi": LANG_HINDI,
    "हिंदी": LANG_HINDI,
    "ஹிந்தி": LANG_HINDI,
    "இந்தி": LANG_HINDI,
    "kannada": LANG_KANNADA,
    "kanada": LANG_KANNADA,
    "கன்னடம்": LANG_KANNADA,
    "कन्नड़": LANG_KANNADA,
    "ಕನ್ನಡ": LANG_KANNADA,
    "ಕನ್ನಡದಲ್ಲಿ": LANG_KANNADA,
    "కన్నడ": LANG_KANNADA,
    "telugu": LANG_TELUGU,
    "telgu": LANG_TELUGU,
    "தெலுங்கு": LANG_TELUGU,
    "तेलुगु": LANG_TELUGU,
    "ತೆಲುಗು": LANG_TELUGU,
    "తెలుగు": LANG_TELUGU,
    "తెలుగులో": LANG_TELUGU,
    # Acoustic misrecognitions for Tamil strictly within language-intent phrases
    "summer": LANG_TAMIL,
    "samar": LANG_TAMIL,
    "tamir": LANG_TAMIL,
    "thamer": LANG_TAMIL,
    "tumbler": LANG_TAMIL,
    "tumble": LANG_TAMIL,
    "inglish": LANG_ENGLISH,
    "hindee": LANG_HINDI,
    "kannad": LANG_KANNADA,
    "telegu": LANG_TELUGU,
    "thelugu": LANG_TELUGU,
}

_CANONICAL_LANG_OR_REGEX = r'(?:tamil|tamizh|thamil|thamizh|tamilan|tamizha|english|angrezi|inglish|hindi|hindee|kannada|kanada|kannad|telugu|telgu|telegu|thelugu|தமிழ்|தமிழு|தமிழில்|தமிழ்ல|ஆங்கிலம்|हिंदी|இந்தி|ஹிந்தி|ಕನ್ನಡ|ಕನ್ನಡದಲ್ಲಿ|తెలుగు|తెలుగులో)'
_ACOUSTIC_TAMIL_REGEX = r'(?:summer|samar|tamir|thamer|tumbler|tumble)'

# Explicit language switch regex patterns
_SWITCH_PATTERNS = [
    # 1. Acoustic misrecognitions for Tamil — STRICTLY requires talk/speak/chat/switch verb
    (rf'\b(?:can|could|will|would)\s+(?:you\s+)?(?:please\s+)?(?:talk|speak|chat|converse|respond)\s+(?:to\s+me\s+)?(?:in\s+)?({_ACOUSTIC_TAMIL_REGEX})\b', 1),
    (rf'\b(?:please\s+)?(?:talk|speak|respond|chat|converse|switch(?:\s+to)?)\s+(?:in\s+)?({_ACOUSTIC_TAMIL_REGEX})\b', 1),
    (rf'\b(?:talk|speak)\s+in\s+({_ACOUSTIC_TAMIL_REGEX})\b', 1),

    # 2. Canonical language patterns with explicit verbs
    (rf'\b(?:change\s+(?:the\s+)?language\s+(?:back\s+)?to\s+)({_CANONICAL_LANG_OR_REGEX})\b', 1),
    (rf'\b(?:talk|speak|chat|converse|switch|change|respond)\s+(?:back\s+to\s+|to\s+|in\s+)?(?:the\s+language\s+to\s+)?({_CANONICAL_LANG_OR_REGEX})\b', 1),
    (rf'\b(?:can|could|will|would)\s+(?:you\s+|we\s+)?(?:please\s+)?(?:talk|speak|chat|converse|respond|switch)\s+(?:back\s+to\s+|to\s+me\s+in\s+|to\s+|in\s+)?({_CANONICAL_LANG_OR_REGEX})\b', 1),
    (rf'\bplease\s+(?:talk|speak|switch(?:\s+(?:back\s+)?to)?|respond)\s+(?:in\s+)?({_CANONICAL_LANG_OR_REGEX})\b', 1),
    (rf'\bi\s+(?:want|prefer|need|would\s+like)(?:\s+to\s+(?:talk|speak|hear|switch))?\s+(?:in\s+|to\s+)?({_CANONICAL_LANG_OR_REGEX})[\.\!\?]?\s*$', 1),
    (rf'\b({_CANONICAL_LANG_OR_REGEX})\s+(?:please|language)\b', 1),
    (rf'\b(?:switch|change)\s+(?:back\s+)?to\s+({_CANONICAL_LANG_OR_REGEX})\b', 1),
    (rf'^\s*({_CANONICAL_LANG_OR_REGEX})[\.\!\?]?\s*$', 1),
    (rf'\bin\s+({_CANONICAL_LANG_OR_REGEX})[\.\!\?]?\s*$', 1),

    # Transliterated Indic phrases
    (r'\b(?:tamil|tamizh|thamil)\s*(?:la|il)\s*(?:pesunga|pesalam|pesu|pesa\s*mudiyuma)\b', LANG_TAMIL),
    (r'\b(?:pesunga|pesalam)\s*(?:tamil|tamizh|thamil)\s*(?:la|il)?\b', LANG_TAMIL),
    (r'\b(?:hindi)\s*(?:me|mein)\s*(?:baat\s*karo|baat\s*kijiye|boliye|baat\s*karein)\b', LANG_HINDI),
    (r'\b(?:kannada|kanada)\s*(?:dalli|alli)?\s*(?:matadi|mathadi|matanadi|mathanadi)\b', LANG_KANNADA),
    (r'\b(?:telugu|telgu)\s*(?:lo)?\s*(?:matladandi|matladu|matladavacha)\b', LANG_TELUGU),

    # Native script phrases
    (r'[\u0b80-\u0bff].*(?:பேசு|பேசுங்கள்|பேச|பேசலாம்)', LANG_TAMIL),
    (r'தமிழில்\s+(?:பேசுங்கள்|பேசு|பேசலாம்|பேச\s*முடியுமா)', LANG_TAMIL),
    (r'தமிழ்ல\s+(?:பேசுங்க|பேசு)', LANG_TAMIL),
    (r'[\u0900-\u097f].*(?:बात\s+करो|बोलिए|बात\s+कीजिए|बोलो)', LANG_HINDI),
    (r'हिंदी\s+में\s+(?:बात\s+करो|बोलिए|बात\s+कीजिए)', LANG_HINDI),
    (r'[\u0c80-\u0cff].*(?:ಮಾತನಾಡಿ|ಮಾತಾಡಿ|ಮಾತನಾಡಬಹುದಾ)', LANG_KANNADA),
    (r'ಕನ್ನಡದಲ್ಲಿ\s+(?:ಮಾತನಾಡಿ|ಮಾತಾಡಿ)', LANG_KANNADA),
    (r'[\u0c00-\u0c7f].*(?:మాట్లాడండి|మాట్లాడు|మాట్లాడవచ్చా)', LANG_TELUGU),
    (r'తెలుగులో\s+(?:మాట్లాడండి|మాట్లాడు)', LANG_TELUGU),
]

_INCOMPLETE_SWITCH_PATTERNS = [
    r'\b(?:can\s+(?:you\s+)?(?:please\s+)?)?(?:talk|speak|switch)\s+(?:in|to)\b\s*$',
    r'\b(?:can\s+(?:you\s+)?(?:please\s+)?)?(?:change|switch)\s+(?:the\s+)?language(?:\s+to)?\b\s*$',
    r'\b(?:will\s+you\s+)?(?:please\s+)?(?:talk|speak)\s+(?:to\s+me\s+)?in\b\s*$',
    r'\b(?:could\s+you\s+)?(?:please\s+)?(?:talk|speak)\s+in\b\s*$',
    r'\b(?:please\s+)?(?:switch|change)\s+to\b\s*$',
    r'\b(?:can\s+we\s+)?(?:talk|speak)\s+in\b\s*$',
    r'\b(?:can\s+you\s+)?(?:please\s+)?speak\s+in\b\s*$',
    r'\b(?:can\s+you\s+)?(?:please\s+)?talk\s+in\b\s*$',
]

_AMBIGUOUS_PATTERNS = [
    r'\b(?:can\s+(?:you\s+)?(?:please\s+)?)?(?:change|switch)\s+(?:the\s+)?language\b',
    r'\b(?:can\s+(?:you\s+)?(?:please\s+)?)?(?:talk|speak)\s+in\s+(?:another|other|different)\s+language\b',
    r'^\s*(?:change|switch)\s+language(?:\s+please)?\s*$',
    r'\bwhich\s+languages\s+(?:do\s+you\s+speak|can\s+you\s+speak)\b',
    r'\bcan\s+you\s+speak\s+other\s+languages\b',
]

# Patterns for unsupported language requests (to give polite feedback)
_UNSUPPORTED_LANGUAGE_PATTERN = re.compile(
    r'\b(?:talk|speak|switch(?:\s+to)?|change\s+to)\s+(?:in\s+)?(spanish|french|german|arabic|chinese|japanese|russian|bengali|marathi|malayalam|gujarati|punjabi|urdu)\b',
    re.IGNORECASE
)


class LanguageDetectionService:
    """
    Multilingual Language Service managing 5 supported languages:
    Tamil (ta), English (en), Hindi (hi), Kannada (kn), Telugu (te).
    """

    def __init__(self):
        self.default_language = LANG_ENGLISH

    def check_language_switch_detailed(self, text: str) -> Tuple[Optional[str], bool, str]:
        """
        Inspects text for an explicit customer request to switch languages.
        Returns:
            (target_lang, True, "explicit") if an explicit switch to a supported language was requested.
            (None, True, "ambiguous") if caller asked to change language but didn't specify which one.
            (None, True, "unsupported") if an unsupported language was requested.
            (None, False, "none") if no language switch was requested.
        """
        txt = (text or "").strip()
        if not txt:
            return None, False, "none"

        # Normalize duplicated consecutive words (e.g. 'summer? Summer' -> 'summer')
        norm = re.sub(r'(\b\w+\b)[\s\?\.\!\,-]+\1\b', r'\1', txt, flags=re.IGNORECASE)

        # 1. Check supported switch patterns
        for pattern, target in _SWITCH_PATTERNS:
            m = re.search(pattern, norm, re.IGNORECASE)
            if m:
                if isinstance(target, int):
                    lang_word = m.group(target).lower().strip()
                    resolved = _NAME_TO_CODE.get(lang_word)
                    if resolved in SUPPORTED_LANGUAGES:
                        logger.info(
                            f"\n[LANGUAGE_INTENT]"
                            f"\n  original_transcript='{txt}'"
                            f"\n  normalized_transcript='{norm}'"
                            f"\n  detected_requested_language={resolved}"
                            f"\n  confidence_reason=explicit_pattern_match"
                            f"\n  result=ACCEPTED"
                        )
                        return resolved, True, "explicit"
                elif target in SUPPORTED_LANGUAGES:
                    logger.info(
                        f"\n[LANGUAGE_INTENT]"
                        f"\n  original_transcript='{txt}'"
                        f"\n  normalized_transcript='{norm}'"
                        f"\n  detected_requested_language={target}"
                        f"\n  confidence_reason=explicit_target_match"
                        f"\n  result=ACCEPTED"
                    )
                    return target, True, "explicit"

        # 2. Check ambiguous language switch request (e.g. 'Can you change the language?')
        for pat in _AMBIGUOUS_PATTERNS:
            if re.search(pat, norm, re.IGNORECASE):
                logger.info(
                    f"\n[LANGUAGE_INTENT]"
                    f"\n  original_transcript='{txt}'"
                    f"\n  normalized_transcript='{norm}'"
                    f"\n  detected_requested_language=None"
                    f"\n  confidence_reason=ambiguous_language_request"
                    f"\n  result=CLARIFICATION_REQUIRED"
                )
                return None, True, "ambiguous"

        # 3. Check if user requested an unsupported language
        unsupported_match = _UNSUPPORTED_LANGUAGE_PATTERN.search(txt)
        if unsupported_match:
            req_lang = unsupported_match.group(1)
            logger.info(
                f"\n[LANGUAGE_INTENT]"
                f"\n  original_transcript='{txt}'"
                f"\n  normalized_transcript='{norm}'"
                f"\n  detected_requested_language={req_lang}"
                f"\n  confidence_reason=unsupported_language"
                f"\n  result=DECLINED_UNSUPPORTED"
            )
            return None, True, "unsupported"

        return None, False, "none"

    def check_explicit_language_switch(self, text: str) -> Tuple[Optional[str], bool]:
        """
        Backwards-compatible wrapper returning (target_lang, is_switch).
        """
        target_lang, is_switch, _ = self.check_language_switch_detailed(text)
        return target_lang, is_switch

    def is_ambiguous_language_switch(self, text: str) -> bool:
        """Returns True if the utterance is an ambiguous language switch request."""
        _, is_switch, reason = self.check_language_switch_detailed(text)
        return is_switch and reason == "ambiguous"

    def is_language_switch_request(self, text: str) -> bool:
        """Returns True if the utterance requests a language change."""
        _, is_switch = self.check_explicit_language_switch(text)
        return is_switch

    def is_incomplete_language_switch(self, text: str) -> bool:
        """
        Detects if the customer uttered a language switch request without specifying the language yet.
        E.g. 'Can you please talk in', 'Please switch to', 'Talk in'.
        """
        txt = (text or "").strip().lower()
        if not txt:
            return False
        target_lang, is_switch = self.check_explicit_language_switch(text)
        if is_switch:
            return False
        for pat in _INCOMPLETE_SWITCH_PATTERNS:
            if re.search(pat, txt, re.IGNORECASE):
                logger.info(f"[LANG_SWITCH_INCOMPLETE] Incomplete request detected in: '{text}'")
                return True
        return False

    def get_language_clarification_prompt(self, current_lang: str = LANG_ENGLISH) -> str:
        """
        Returns a clarification question asking the customer to select one of the five supported languages.
        """
        prompts = {
            LANG_ENGLISH: "Certainly! Which language would you like me to speak in: Tamil, English, Hindi, Kannada, or Telugu?",
            LANG_TAMIL: "நிச்சயமாக! தமிழ், ஆங்கிலம், இந்தி, கன்னடம் அல்லது தெலுங்கு - இதில் எந்த மொழியில் பேச விரும்புகிறீர்கள்?",
            LANG_HINDI: "ज़रूर! आप किस भाषा में बात करना चाहेंगे: हिंदी, अंग्रेज़ी, तमिल, कन्नड़ या तेलुगु?",
            LANG_KANNADA: "ಖಂಡಿತ! ನೀವು ಯಾವ ಭಾಷೆಯಲ್ಲಿ ಮಾತನಾಡಲು ಬಯಸುತ್ತೀರಿ: ಕನ್ನಡ, ಇಂಗ್ಲಿಷ್, ತಮಿಳು, ಹಿಂದಿ ಅಥವಾ ತೆಲುಗು?",
            LANG_TELUGU: "తప్పకుండా! మీరు ఏ భాషలో మాట్లాడాలనుకుంటున్నారు: తెలుగు, ఇంగ్లీష్, తమిళం, హిందీ లేదా కన్నడ?",
        }
        return prompts.get(current_lang, prompts[LANG_ENGLISH])

    def is_non_english_input(self, text: str) -> bool:
        """
        Checks if text contains non-English Indic script (Tamil, Devanagari, Telugu, Kannada).
        """
        txt = (text or "").strip()
        if not txt:
            return False
        # Matches Tamil, Devanagari, Telugu, Kannada scripts
        return bool(re.search(r'[\u0b80-\u0bff\u0900-\u097f\u0c80-\u0cff\u0c00-\u0c7f]', txt))

    def detect_spoken_language(self, text: str, current_lang: str = LANG_ENGLISH) -> str:
        """
        Detects language from script without flipping away on English words (loanwords/emails/names).
        If no non-Latin script is found, preserves current_lang.
        """
        txt = (text or "").strip()
        if not txt:
            return current_lang or LANG_ENGLISH

        if re.search(r'[\u0b80-\u0bff]', txt):
            return LANG_TAMIL
        elif re.search(r'[\u0900-\u097f]', txt):
            return LANG_HINDI
        elif re.search(r'[\u0c80-\u0cff]', txt):
            return LANG_KANNADA
        elif re.search(r'[\u0c00-\u0c7f]', txt):
            return LANG_TELUGU

        # Return active conversation language to prevent corruption on technical / contact details
        return current_lang if current_lang in SUPPORTED_LANGUAGES else LANG_ENGLISH

    def detect_language(self, text: str, stt_lang: Optional[str] = None) -> str:
        """Wrapper for detect_spoken_language."""
        return self.detect_spoken_language(text, current_lang=stt_lang or LANG_ENGLISH)

    def determine_conversation_language(
        self,
        text: str,
        current_lang: Optional[str] = None,
        stt_lang: Optional[str] = None
    ) -> str:
        """
        Determines current conversation language, respecting explicit switches
        and preserving the selected language across turns.
        """
        target_lang, is_switch = self.check_explicit_language_switch(text)
        if is_switch and target_lang in SUPPORTED_LANGUAGES:
            return target_lang

        curr = current_lang or stt_lang or self.default_language
        return self.detect_spoken_language(text, current_lang=curr)

    def resolve_call_language_turn(
        self,
        text: str,
        current_preferred_lang: Optional[str] = None,
        stt_lang: Optional[str] = None,
        confidence: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Resolves per-call language state for the current turn.
        """
        target_lang, is_switch = self.check_explicit_language_switch(text)
        if is_switch and target_lang in SUPPORTED_LANGUAGES:
            preferred = target_lang
            req_lang = target_lang
        else:
            preferred = current_preferred_lang or stt_lang or self.default_language
            req_lang = None

        detected = self.detect_spoken_language(text, current_lang=preferred)

        return {
            "detected_utterance_language": detected,
            "preferred_response_language": preferred,
            "language_switch_requested": is_switch,
            "requested_switch_language": req_lang,
            "stt_language_confidence": float(confidence or 1.0),
        }

    def get_language_switch_confirmation(
        self,
        target_lang: Optional[str] = None,
        booking_session=None,
        org_name: Optional[str] = None
    ) -> str:
        """
        Generates dynamic confirmation in the selected language.
        If booking is in progress, seamlessly prompts for the pending detail in the target language.
        """
        if not target_lang or target_lang not in SUPPORTED_LANGUAGES:
            return "I can assist you in English, Tamil, Hindi, Kannada, or Telugu. Which language would you prefer?"

        # Check if booking is in progress and prompt for the pending field
        is_booking_active = bool(booking_session and hasattr(booking_session, "is_active") and booking_session.is_active())
        pending_field = None
        if is_booking_active:
            try:
                pending_field = getattr(booking_session, "last_asked_field", None) or getattr(booking_session, "last_question_field", None)
                if not pending_field:
                    mf = getattr(booking_session, "missing_fields", None)
                    if callable(mf):
                        mf = mf()
                    if mf and isinstance(mf, list):
                        pending_field = mf[0]
            except Exception:
                pass

        if is_booking_active and pending_field:
            if pending_field == "department_or_doctor":
                if getattr(booking_session, "department", None):
                    pending_field = "doctor"
                else:
                    pending_field = "department"

            if pending_field in ("department", "service", "specialty"):
                prompts = {
                    LANG_TAMIL: "நிச்சயமாக! இனிமேல் தமிழில் பேசுகிறேன். உங்களுக்கு எந்த மருத்துவத் துறை அல்லது சிகிச்சைக்கான அப்பாயின்ட்மென்ட் வேண்டும்?",
                    LANG_ENGLISH: "Certainly! Continuing in English. Which department or service would you like to book?",
                    LANG_HINDI: "ज़रूर! अब मैं हिंदी में बात करूँगा। आप किस विभाग या सेवा के लिए अपॉइंटमेंट बुक करना चाहते हैं?",
                    LANG_KANNADA: "ಖಂಡಿತ! ಇನ್ನು ಮುಂದೆ ಕನ್ನಡದಲ್ಲಿ ಮಾತನಾಡುತ್ತೇನೆ. ನಿಮಗೆ ಯಾವ ವಿಭಾಗ ಅಥವಾ ಸೇವೆಗೆ ಅಪಾಯಿಂಟ್‌ಮೆಂಟ್ ಬೇಕು?",
                    LANG_TELUGU: "తప్పకుండా! ఇకపై తెలుగులో మాట్లాడుతాను. మీకు ఏ విభాగం లేదా సేవ కొరకు అపాయింట్‌మెంట్ కావాలి?",
                }
                return prompts.get(target_lang, prompts[LANG_ENGLISH])

            elif pending_field in ("doctor", "doctor_name"):
                prompts = {
                    LANG_TAMIL: "நிச்சயமாக! தமிழில் தொடர்கிறேன். எந்த மருத்துவரை நீங்கள் பார்க்க விரும்புகிறீர்கள்?",
                    LANG_ENGLISH: "Certainly! Continuing in English. Which doctor would you like to consult?",
                    LANG_HINDI: "ज़रूर! हिंदी में जारी रखते हैं। आप किस डॉक्टर से परामर्श करना चाहते हैं?",
                    LANG_KANNADA: "ಖಂಡಿತ! ಕನ್ನಡದಲ್ಲಿ ಮುಂದುವರಿಸುತ್ತೇನೆ. ನೀವು ಯಾವ ವೈದ್ಯರನ್ನು ಭೇಟಿಯಾಗಲು ಬಯಸುತ್ತೀರಿ?",
                    LANG_TELUGU: "తప్పకుండా! తెలుగులో కొనసాగిస్తాను. మీరు ఏ డాక్టర్‌ను సంప్రదించాలనుకుంటున్నారు?",
                }
                return prompts.get(target_lang, prompts[LANG_ENGLISH])

            elif pending_field in ("date", "time", "appointment_date", "appointment_time"):
                prompts = {
                    LANG_TAMIL: "நிச்சயமாக! தமிழில் தொடர்கிறேன். எந்த தேதி மற்றும் நேரத்தில் அப்பாயின்ட்மென்ட் வேண்டும்?",
                    LANG_ENGLISH: "Certainly! Continuing in English. What date and time would you prefer?",
                    LANG_HINDI: "ज़रूर! हिंदी में जारी रखते हैं। आप किस तारीख और समय पर अपॉइंटमेंट चाहते हैं?",
                    LANG_KANNADA: "ಖಂಡಿತ! ಕನ್ನಡದಲ್ಲಿ ಮುಂದುವರಿಸುತ್ತೇನೆ. ನಿಮಗೆ ಯಾವ ದಿನಾಂಕ ಮತ್ತು ಸಮಯ ಸೂಕ್ತ?",
                    LANG_TELUGU: "తప్పకుండా! తెలుగులో కొనసాగిస్తాను. మీకు ఏ తేదీ మరియు సమయం అనుకూలంగా ఉంటుంది?",
                }
                return prompts.get(target_lang, prompts[LANG_ENGLISH])

            elif pending_field in ("patient_name", "full_name", "name", "customer_name"):
                prompts = {
                    LANG_TAMIL: "நிச்சயமாக! தமிழில் பேசுகிறேன். அப்பாயின்ட்மென்ட் புக் செய்ய உங்கள் முழுப் பெயரை சொல்லுங்கள்.",
                    LANG_ENGLISH: "Certainly! Continuing in English. Could you please share your full name for the booking?",
                    LANG_HINDI: "ज़रूर! हिंदी में जारी रखते हैं। बुकिंग के लिए कृपया अपना पूरा नाम बताएं।",
                    LANG_KANNADA: "ಖಂಡಿತ! ಕನ್ನಡದಲ್ಲಿ ಮುಂದುವರಿಸುತ್ತೇನೆ. ಬುಕಿಂಗ್‌ಗಾಗಿ ದಯವಿಟ್ಟು ನಿಮ್ಮ ಪೂರ್ಣ ಹೆಸರನ್ನು ತಿಳಿಸಿ.",
                    LANG_TELUGU: "తప్పకుండా! తెలుగులో కొనసాగిస్తాను. బుకింగ్ కొరకు దయచేసి మీ పూర్తి పేరు తెలపండి.",
                }
                return prompts.get(target_lang, prompts[LANG_ENGLISH])

            elif pending_field in ("contact_number", "phone", "phone_number"):
                prompts = {
                    LANG_TAMIL: "நிச்சயமாக! தமிழில் தொடர்கிறேன். உங்களை தொடர்பு கொள்ள 10 இலக்க தொலைபேசி எண்ணைக் கூறுங்கள்.",
                    LANG_ENGLISH: "Certainly! Continuing in English. Could you please provide your 10-digit phone number?",
                    LANG_HINDI: "ज़रूर! हिंदी में जारी रखते हैं। कृपया अपना 10 अंकों का मोबाइल नंबर बताएं।",
                    LANG_KANNADA: "ಖಂಡಿತ! ಕನ್ನಡದಲ್ಲಿ ಮುಂದುವರಿಸುತ್ತೇನೆ. ದಯವಿಟ್ಟು ನಿಮ್ಮ 10 ಅಂಕಿಯ ಮೊಬೈಲ್ ಸಂಖ್ಯೆಯನ್ನು ತಿಳಿಸಿ.",
                    LANG_TELUGU: "తప్పకుండా! తెలుగులో కొనసాగిస్తాను. దయచేసి మీ 10 అంకెల ఫోన్ నంబర్‌ను తెలపండి.",
                }
                return prompts.get(target_lang, prompts[LANG_ENGLISH])

            elif pending_field in ("email", "mail"):
                prompts = {
                    LANG_TAMIL: "நிச்சயமாக! தமிழில் தொடர்கிறேன். அப்பாயின்ட்மென்ட் உறுதிப்படுத்த உங்கள் மின்னஞ்சல் (email) முகவரியைக் கூறுங்கள்.",
                    LANG_ENGLISH: "Certainly! Continuing in English. Could you please provide your email address for the appointment confirmation?",
                    LANG_HINDI: "ज़रूर! हिंदी में जारी रखते हैं। अपॉइंटमेंट पुष्टि के लिए कृपया अपना ईमेल पता बताएं।",
                    LANG_KANNADA: "ಖಂಡಿತ! ಕನ್ನಡದಲ್ಲಿ ಮುಂದುವರಿಸುತ್ತೇನೆ. ಅಪಾಯಿಂಟ್‌ಮೆಂಟ್ ದೃಢೀಕರಣಕ್ಕಾಗಿ ದಯವಿಟ್ಟು ನಿಮ್ಮ ಇಮೇಲ್ ವಿಳಾಸವನ್ನು ತಿಳಿಸಿ.",
                    LANG_TELUGU: "తప్పకుండా! తెలుగులో కొనసాగిస్తాను. అపాయింట్‌మెంట్ నిర్ధారణ కొరకు దయచేసి మీ ఇమెయిల్ చిరునామా తెలపండి.",
                }
                return prompts.get(target_lang, prompts[LANG_ENGLISH])

        # General acknowledgment (outside of active booking)
        general_prompts = {
            LANG_TAMIL: "நிச்சயமாக! இனிமேல் தமிழில் பேசுகிறேன். உங்களுக்கு எப்படி உதவலாம்?",
            LANG_ENGLISH: "Certainly! I will continue in English. How may I assist you today?",
            LANG_HINDI: "ज़रूर! अब मैं हिंदी में बात करूँगा। बताइए, मैं आपकी क्या सहायता कर सकता हूँ?",
            LANG_KANNADA: "ಖಂಡಿತ! ಇನ್ನು ಮುಂದೆ ಕನ್ನಡದಲ್ಲಿ ಮಾತನಾಡುತ್ತೇನೆ. ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಬಹುದು?",
            LANG_TELUGU: "తప్పకుండా! ఇకపై తెలుగులో మాట్లాడుతాను. మీకు ఎలా సహాయపడగలను?",
        }
        return general_prompts.get(target_lang, general_prompts[LANG_ENGLISH])

    def get_tts_voice_and_model(self, lang: str) -> Tuple[str, str, str]:
        """
        Returns (voice_id, model_version, provider) for the specified language.
        """
        cfg = TTS_VOICE_CONFIG.get(lang, TTS_VOICE_CONFIG[LANG_ENGLISH])
        return cfg["voice_id"], cfg["model"], cfg["provider"]

    def get_stt_language(self, lang: str) -> str:
        """Returns the Deepgram STT language code."""
        return STT_LANGUAGE_MAP.get(lang, "en")

    def get_language_name(self, lang: str) -> str:
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


language_service = LanguageDetectionService()
