"""
Multilingual Language Detection & Conversation Language Lock Service.

Supports Tamil (script & Tanglish), Hindi (script & Hinglish), Malayalam,
Telugu, Kannada, and English.
Manages conversation language locking, explicit language switches, and
entity-name immunity.

Priority pipeline:
  is_language_switch_request()  ← fast predicate (called first by voice_socket)
  check_explicit_language_switch()   ← returns (target_lang, True/False)
  determine_conversation_language()  ← full stateful resolver
"""

import logging
import re
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


LANG_TAMIL = "ta"
LANG_TANGLISH = "tanglish"
LANG_HINDI = "hi"
LANG_ENGLISH = "en"
LANG_MALAYALAM = "ml"
LANG_TELUGU = "te"
LANG_KANNADA = "kn"


# ---------------------------------------------------------------------------
# Canonical language name → code mapping
# ---------------------------------------------------------------------------
_LANG_NAME_TO_CODE = {
    "tamil": LANG_TAMIL,
    "tamizh": LANG_TAMIL,
    "தமிழ்": LANG_TAMIL,
    "hindi": LANG_HINDI,
    "हिंदी": LANG_HINDI,
    "english": LANG_ENGLISH,
    "malayalam": LANG_MALAYALAM,
    "മലയാളം": LANG_MALAYALAM,
    "telugu": LANG_TELUGU,
    "తెలుగు": LANG_TELUGU,
    "kannada": LANG_KANNADA,
    "ಕನ್ನಡ": LANG_KANNADA,
}

# Phrases that indicate a language-switch request (caller → AI direction).
# Matched case-insensitively against the full lowercased input.
_LANG_SWITCH_TRIGGER_PHRASES = [
    # Generic verbs + "in <lang>" / "<lang> please"
    r'\b(?:please\s+)?(?:talk|speak|chat|converse|respond|reply|answer|communicate)\s+(?:to\s+me\s+)?in\s+([a-zA-Z\u0b80-\u0bff\u0900-\u097f\u0d00-\u0d7f\u0c00-\u0c7f\u0c80-\u0cff]+)',
    r'\bcan\s+you\s+(?:please\s+)?(?:talk|speak|chat|converse|respond)\s+(?:to\s+me\s+)?in\s+([a-zA-Z\u0b80-\u0bff\u0900-\u097f\u0d00-\u0d7f\u0c00-\u0c7f\u0c80-\u0cff]+)',
    r'\bcould\s+you\s+(?:please\s+)?(?:talk|speak|chat)\s+(?:to\s+me\s+)?in\s+([a-zA-Z\u0b80-\u0bff\u0900-\u097f\u0d00-\u0d7f\u0c00-\u0c7f\u0c80-\u0cff]+)',
    r'\bplease\s+(?:talk|speak|chat)\s+(?:to\s+me\s+)?in\s+([a-zA-Z\u0b80-\u0bff\u0900-\u097f\u0d00-\u0d7f\u0c00-\u0c7f\u0c80-\u0cff]+)',
    # "switch to <lang>"
    r'\bswitch\s+to\s+([a-zA-Z\u0b80-\u0bff\u0900-\u097f\u0d00-\u0d7f\u0c00-\u0c7f\u0c80-\u0cff]+)',
    r'\bchange\s+(?:the\s+)?language\s+to\s+([a-zA-Z\u0b80-\u0bff\u0900-\u097f\u0d00-\u0d7f\u0c00-\u0c7f\u0c80-\u0cff]+)',
    # Repetition or standalone language command: "tamil tamil", "hindi hindi"
    r'^\s*(tamil|tamizh|hindi|english|malayalam|telugu|kannada)(?:\s+\1)+\s*$',
    # "talk <lang>" / "speak <lang>"
    r'\b(?:talk|speak)\s+([a-zA-Z]+)(?:\s+please)?\b',
    # "<lang> please" / "<lang> la pesu/pesunga"
    r'\b([a-zA-Z]+)\s+please\b',
    r'\b([a-zA-Z]+)\s+la\s+pesu(?:nga)?\b',
    r'\b([a-zA-Z]+)\s+mein\s+baat\s+karo\b',
    r'\b([a-zA-Z]+)\s+lo\s+matladandi\b',
    r'\b([a-zA-Z]+)\s+parayu(?:mo)?\b',
    r'\b([a-zA-Z]+)\s+dalli\s+mathanadi\b',
    # Native-script requests (Tamil)
    r'(தமிழில்\s+பேசுங்கள்|தமிழில்\s+பேசு|தமிழில்\s+பேச\s+முடியுமா)',
    # Hindi
    r'(हिंदी\s+में\s+बात|हिंदी\s+बोलो)',
    # Telugu
    r'(తెలుగులో\s+మాట్లాడండి|తెలుగు\s+మాట్లాడండి)',
    # Malayalam
    r'(മലയാളത്തിൽ\s+സംസാരിക്കൂ|മലയാളം\s+സംസാരിക്കൂ)',
    # Kannada
    r'(ಕನ್ನಡದಲ್ಲಿ\s+ಮಾತನಾಡಿ|ಕನ್ನಡ\s+ಮಾತನಾಡಿ)',
]


class LanguageDetectionService:
    """
    Language Detection & Lock Manager.
    Determines input language and enforces language persistence.
    """

    def __init__(self):
        # Tanglish / Mixed Tamil-English keyword indicators
        self.tanglish_keywords = {
            "tamil", "tamizh", "pannanum", "paannatum", "venum", "venom", "vendum", "iruka", "irukku", "irukkangala", "irukangala",
            "kitta", "vanakkam", "vanakam", "naalai", "nalaiku", "indru", "inru", "inniku", "innikku", "ennodaya", "peru",
            "nallairuken", "nallairukken", "seiyangga", "solloonga", "solunga", "solren", "unga", "enaku", "enakku",
            "pesunga", "pesanum", "pesalam", "mama", "nalladhu", "paapom", "paru", "parunga", "mudiyuma", "mudiyum",
            "varuma", "varum", "varuvanga", "kudunga", "kudukkavum", "eduthu", "eduthukonga", "teriyum", "theriyum", "sapadunga",
            "eppadi", "epadi", "evvalavu", "evalavu", "romba", "aama", "aamam", "illai", "illa", "seri", "sari",
            "marupadiyum", "eppo", "yeppo", "enga", "yenga", "yaar", "yaaru", "adhu", "idhu", "enna", "ethu", "edhu",
            "panna", "seiya", "seiyya", "mattum", "kaalaila", "sayandhram", "manikku", "neram", "paathutu", "sollunga"
        }

        # Hinglish / Hindi keyword indicators
        self.hinglish_keywords = {
            "hindi", "hinglish", "chahiye", "kaise", "kya", "hai", "hain", "hoon", "karo", "karna", "karenge", "aap", "aapka", "aapki", "baat",
            "batao", "batayein", "dhanyawad", "shukriya", "boliye", "kaha", "kahan", "kaun", "mera", "meri",
            "namaste", "samajh", "milna", "kal", "aaj", "kitna", "kab", "kisko", "hoga", "hogi", "theek", "accha",
            "haan", "nahi", "subah", "shaam", "baje", "samay"
        }

    # ------------------------------------------------------------------
    # Fast predicate — call this FIRST before any NLP/RAG routing
    # ------------------------------------------------------------------
    def is_language_switch_request(self, text: str) -> bool:
        """
        Returns True immediately if the text is a language-switch instruction
        (e.g. "Can you please talk in Tamil?", "Switch to Hindi").

        This MUST be called before RAG retrieval, entity extraction, and
        booking-state routing to prevent language phrases from being
        misrouted into directory/entity lookups.
        """
        _, is_switch = self.check_explicit_language_switch(text)
        return is_switch

    # ------------------------------------------------------------------
    # Core language detection
    # ------------------------------------------------------------------
    def detect_spoken_language(self, text: str, stt_lang: Optional[str] = None) -> str:
        """
        Operation A: Actual Speech-Language Detection.
        Determines the true spoken language of the customer's utterance/transcript.
        
        Crucial separation:
        "Can you please talk in Tamil?" -> Spoken utterance is English ('en').
        "வணக்கம் எனக்கு ஒரு அப்பாயின்ட்மென்ட் வேண்டும்" -> Spoken utterance is Tamil ('ta').
        "vanakkam enakku appointment book pannanum" -> Spoken utterance is Tanglish ('tanglish').
        """
        txt = (text or "").strip()
        if not txt:
            return LANG_ENGLISH

        txt_lower = txt.lower()

        # 1. Native Tamil Script (Unicode \u0b80-\u0bff)
        if re.search(r'[\u0b80-\u0bff]', txt):
            return LANG_TAMIL

        # 2. Native Hindi / Devanagari (\u0900-\u097f)
        if re.search(r'[\u0900-\u097f]', txt):
            return LANG_HINDI

        # 3. Native Malayalam (\u0d00-\u0d7f)
        if re.search(r'[\u0d00-\u0d7f]', txt):
            return LANG_MALAYALAM

        # 4. Native Telugu (\u0c00-\u0c7f)
        if re.search(r'[\u0c00-\u0c7f]', txt):
            return LANG_TELUGU

        # 5. Native Kannada (\u0c80-\u0cff)
        if re.search(r'[\u0c80-\u0cff]', txt):
            return LANG_KANNADA

        # 6. Check if this is an English language-switch request (e.g. "Can you please speak in Tamil?")
        # An English switch request spoken in English MUST be detected as English!
        target_switch, is_switch = self.check_explicit_language_switch(txt)
        words = set(re.findall(r'\b[a-z]+\b', txt_lower))
        if is_switch:
            en_switch_indicators = {"speak", "talk", "chat", "converse", "respond", "switch", "change", "in", "to", "please", "can", "could", "would"}
            if words.intersection(en_switch_indicators):
                return LANG_ENGLISH

        # 7. Tanglish / Tamil in Latin script
        # Require substantive Tamil vocabulary rather than simply mentioning the noun "tamil"
        tanglish_vocab = self.tanglish_keywords - {"tamil", "tamizh"}
        if words.intersection(tanglish_vocab):
            return LANG_TANGLISH
        if "tamil" in txt_lower or "tamizh" in txt_lower:
            if not any(w in words for w in ["speak", "talk", "switch", "change", "language", "in", "can", "could", "please"]):
                return LANG_TANGLISH

        # 8. Hinglish / Hindi in Latin script
        hinglish_vocab = self.hinglish_keywords - {"hindi"}
        if words.intersection(hinglish_vocab):
            return LANG_HINDI
        if "hindi" in txt_lower:
            if not any(w in words for w in ["speak", "talk", "switch", "change", "language", "in", "can", "could", "please"]):
                return LANG_HINDI

        # 9. STT Provider hint validation
        clean_stt = str(stt_lang or "").lower().strip()
        if clean_stt in ("ta", "tamil"):
            return LANG_TAMIL if re.search(r'[\u0b80-\u0bff]', txt) else LANG_TANGLISH
        if clean_stt in ("te", "telugu"):
            return LANG_TELUGU
        if clean_stt in ("ml", "malayalam"):
            return LANG_MALAYALAM
        if clean_stt in ("kn", "kannada"):
            return LANG_KANNADA
        if clean_stt in ("hi", "hindi"):
            if re.search(r'[\u0900-\u097f]', txt) or words.intersection(self.hinglish_keywords):
                return LANG_HINDI
            common_en = {"the", "a", "an", "i", "you", "we", "he", "she", "it", "to", "for", "in", "on", "at", "is", "are", "was", "were", "appointment", "doctor", "book", "need", "want", "hello", "hi", "tomorrow", "today", "schedule", "can", "could", "would", "available", "time", "date"}
            if words.intersection(common_en):
                return LANG_ENGLISH
            return LANG_HINDI

        return LANG_ENGLISH

    def detect_language(self, text: str, stt_lang: Optional[str] = None) -> str:
        """Alias for detect_spoken_language."""
        return self.detect_spoken_language(text, stt_lang=stt_lang)

    def check_explicit_language_switch(self, text: str) -> Tuple[Optional[str], bool]:
        """
        Operation B: Explicit Language Preference Check.
        Check if caller explicitly requests a language switch.
        Returns (target_language_code, is_explicit_switch_requested).

        Covers all natural phrasings including:
          - "Can you please talk in Tamil?"
          - "Please speak Tamil"
          - "Switch to Hindi"
          - "Tamil la pesunga"
          - "தமிழில் பேசுங்கள்"
          - "Hindi mein baat karo"
          - "Please speak in English"
        """
        txt = (text or "").strip()
        if not txt:
            return None, False

        txt_lower = txt.lower()

        # --- Native script fast checks ---
        if re.search(r'தமிழில்\s+பேசு|தமிழ்\s+பேசு|தமிழில்\s+பேச', txt):
            return LANG_TAMIL, True
        if re.search(r'हिंदी\s+में\s+बात|हिंदी\s+बोलो', txt):
            return LANG_HINDI, True
        if re.search(r'తెలుగులో\s+మాట్లాడండి|తెలుగు\s+మాట్లాడండి', txt):
            return LANG_TELUGU, True
        if re.search(r'മലയാളത്തിൽ\s+സംസാരിക്കൂ|മലയാളം\s+സംസാരിക്കൂ', txt):
            return LANG_MALAYALAM, True
        if re.search(r'ಕನ್ನಡದಲ್ಲಿ\s+ಮಾತನಾಡಿ|ಕನ್ನಡ\s+ಮಾತನಾಡಿ', txt):
            return LANG_KANNADA, True

        # --- Pattern matching ---
        for pattern in _LANG_SWITCH_TRIGGER_PHRASES:
            m = re.search(pattern, txt_lower, re.IGNORECASE)
            if m:
                # Extract the captured language name group (group 1 if present)
                lang_word = ""
                try:
                    lang_word = m.group(1).strip().lower()
                except IndexError:
                    lang_word = ""

                # Map the extracted word to a language code
                lang_code = _LANG_NAME_TO_CODE.get(lang_word)
                if lang_code:
                    logger.info(
                        f"[LANGUAGE SWITCH DETECTED] Pattern matched '{pattern}' → "
                        f"lang_word='{lang_word}' → code='{lang_code}'"
                    )
                    return lang_code, True

        return None, False

    def resolve_call_language_turn(
        self,
        text: str,
        current_preferred_lang: Optional[str] = None,
        stt_lang: Optional[str] = None,
        confidence: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Resolves per-call language state by separating:
          Operation A: Actual spoken language detection.
          Operation B: Explicit response language preference.

        Returns structured state:
          - detected_utterance_language: What language the caller actually spoke.
          - preferred_response_language: What language the AI should respond in.
          - language_switch_requested: True if an explicit switch was triggered.
          - requested_switch_language: Language code requested (or None).
          - stt_language_confidence: Confidence score from provider.
        """
        txt = (text or "").strip()
        target_switch, is_explicit_switch = self.check_explicit_language_switch(txt)
        detected_utterance = self.detect_spoken_language(txt, stt_lang=stt_lang)

        current_pref = (current_preferred_lang or LANG_ENGLISH).lower().strip()

        if is_explicit_switch and target_switch:
            new_pref = target_switch
            switch_req = True
        else:
            switch_req = False
            # If customer speaks an Indic language (Tamil, Tanglish, Hindi, Telugu, Malayalam, Kannada),
            # adapt preferred response language to that language!
            if detected_utterance in (LANG_TAMIL, LANG_TANGLISH, LANG_HINDI, LANG_MALAYALAM, LANG_TELUGU, LANG_KANNADA):
                new_pref = detected_utterance
            elif detected_utterance == LANG_ENGLISH:
                # If currently locked to an Indic language, guard against single words / affirmations / entities
                if current_pref in (LANG_TAMIL, LANG_TANGLISH, LANG_HINDI, LANG_MALAYALAM, LANG_TELUGU, LANG_KANNADA):
                    protected_words = {
                        "yes", "yeah", "yep", "no", "nah", "okay", "ok", "sure", "fine", "thanks", "thank",
                        "tomorrow", "today", "morning", "afternoon", "evening", "am", "pm", "dr", "doctor",
                        "appointment", "booking", "slot", "name", "phone", "email", "done", "correct", "right",
                        "good", "hello", "hi", "hey"
                    }
                    words = [w for w in re.findall(r'\b[a-zA-Z0-9]+\b', txt.lower()) if not w.isdigit()]
                    non_protected = [w for w in words if w not in protected_words]

                    # Only switch to English if caller provides 3+ substantive English words without Indic cues
                    if len(non_protected) >= 3 or (len(words) >= 4 and len(non_protected) >= 2):
                        logger.info(f"[LANGUAGE SWITCH] Caller switched from '{current_pref}' to English: '{txt}'")
                        new_pref = LANG_ENGLISH
                    else:
                        new_pref = current_pref
                else:
                    new_pref = LANG_ENGLISH
            else:
                new_pref = current_pref

        return {
            "detected_utterance_language": detected_utterance,
            "preferred_response_language": new_pref,
            "language_switch_requested": switch_req,
            "requested_switch_language": target_switch if is_explicit_switch else None,
            "stt_language_confidence": float(confidence or 1.0),
        }

    def determine_conversation_language(self, text: str, current_lang: Optional[str] = None, stt_lang: Optional[str] = None) -> str:
        """
        Backward-compatible wrapper returning preferred_response_language.
        """
        res = self.resolve_call_language_turn(text, current_preferred_lang=current_lang, stt_lang=stt_lang)
        return res["preferred_response_language"]

    def get_language_switch_confirmation(
        self,
        target_lang: str,
        booking_session=None
    ) -> str:
        """
        Returns a natural, bilingual language-switch acknowledgment.

        If the caller is mid-appointment, appends the current pending question
        in the new language so the conversation continues seamlessly.

        Never resets conversation or repeats already-collected information.
        """
        # Base acknowledgment per target language
        _ACK = {
            LANG_TAMIL: "நிச்சயமாக! இனிமேல் நான் தமிழில் பேசுகிறேன்.",
            LANG_HINDI: "Zaroor! Ab main Hindi mein baat karunga.",
            LANG_ENGLISH: "Sure! I'll continue in English.",
            LANG_MALAYALAM: "തീർച്ചയായും! ഇനി ഞാൻ മലയാളത്തിൽ സംസാരിക്കും.",
            LANG_TELUGU: "తప్పకుండా! ఇకపై నేను తెలుగులో మాట్లాడుతాను.",
            LANG_KANNADA: "ಖಂಡಿತ! ಇನ್ನು ನಾನು ಕನ್ನಡದಲ್ಲಿ ಮಾತನಾಡುತ್ತೇನೆ.",
        }
        ack = _ACK.get(target_lang, "Sure! I'll switch to the requested language now.")

        # If no active booking session or session is idle, just return ack + open prompt
        if not booking_session:
            _OPEN = {
                LANG_TAMIL: " எப்படி உதவலாம்?",
                LANG_HINDI: " Aapki kaise sahayata kar sakta hoon?",
                LANG_ENGLISH: " How may I assist you?",
                LANG_MALAYALAM: " എനിക്ക് നിങ്ങളെ എങ്ങനെ സഹായിക്കാൻ കഴിയും?",
                LANG_TELUGU: " మీకు ఎలా సహాయం చేయగలను?",
                LANG_KANNADA: " ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಬಹುದು?",
            }
            return ack + _OPEN.get(target_lang, " How may I assist you?")

        from src.services.booking_state_service import BookingState
        state = getattr(booking_session, "state", BookingState.IDLE)
        doctor = getattr(booking_session, "doctor", None)
        date = getattr(booking_session, "date", None)

        # Determine what was being asked before the language switch and re-ask in new language
        if state == BookingState.WAITING_FOR_DOCTOR or (state != BookingState.IDLE and not doctor):
            _CONT = {
                LANG_TAMIL: " யாரை அல்லது எதற்கு appointment வேண்டும்?",
                LANG_HINDI: " Aap kisse milna ya kya book karna chahte hain?",
                LANG_ENGLISH: " Sure. Who or what would you like to book the appointment for?",
                LANG_MALAYALAM: " ആരെ കാണണം അല്ലെങ്കിൽ എന്ത് ബുക്ക് ചെയ്യണം?",
                LANG_TELUGU: " మీరు ఎవరిని కలవాలనుకుంటున్నారు లేదా ఏమి బుక్ చేయాలనుకుంటున్నారు?",
                LANG_KANNADA: " ಯಾರನ್ನು ಭೇಟಿ ಮಾಡಬೇಕು ಅಥವಾ ಏನನ್ನು ಬುಕ್ ಮಾಡಬೇಕು?",
            }
            return ack + _CONT.get(target_lang, " How may I assist you?")

        if state == BookingState.WAITING_FOR_DATE or (doctor and not date):
            _CONT = {
                LANG_TAMIL: f" {doctor}-க்கு எந்த நாளுக்கு appointment வேண்டும்?",
                LANG_HINDI: f" {doctor} ke liye kaunsi date prefer karenge?",
                LANG_ENGLISH: f" Which date would you prefer for your {doctor} appointment?",
                LANG_MALAYALAM: f" {doctor}-ന്‌ ഏത് തീയതി appointment വേണം?",
                LANG_TELUGU: f" {doctor} కోసం ఏ తేదీ appointment కావాలి?",
                LANG_KANNADA: f" {doctor} appointment ಯಾವ ದಿನಕ್ಕೆ ಬೇಕು?",
            }
            return ack + _CONT.get(target_lang, f" Which date would you prefer for {doctor}?")

        if state == BookingState.WAITING_FOR_TIME:
            _CONT = {
                LANG_TAMIL: f" {date}-ல் எந்த நேரம் suitable-ஆ இருக்கும்?",
                LANG_HINDI: f" {date} ko kaunsa samay prefer karenge?",
                LANG_ENGLISH: f" What time would you prefer on {date}?",
                LANG_MALAYALAM: f" {date}-ൽ ഏത് സമയം ആണ് suitable?",
                LANG_TELUGU: f" {date}న ఏ సమయం suitable?",
                LANG_KANNADA: f" {date}ನಂದು ಯಾವ ಸಮಯ suitable?",
            }
            return ack + _CONT.get(target_lang, f" What time would you prefer on {date}?")

        # Default: open prompt
        _OPEN = {
            LANG_TAMIL: " எப்படி உதவலாம்?",
            LANG_HINDI: " Aapki kaise sahayata kar sakta hoon?",
            LANG_ENGLISH: " How may I assist you?",
            LANG_MALAYALAM: " എനിക்ക് നിങ്ങളെ എങ്ങനെ സഹായിക്കാൻ കഴിയും?",
            LANG_TELUGU: " మీకు ఎలా సహాయం చేయగలను?",
            LANG_KANNADA: " ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಬಹುದು?",
        }
        return ack + _OPEN.get(target_lang, " How may I assist you?")


language_service = LanguageDetectionService()
