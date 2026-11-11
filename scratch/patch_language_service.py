path = 'receptionist_backend-02-11/src/services/language_service.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

target = r'''# Name to ISO code map for parsing
_NAME_TO_CODE = {
    "tamil": LANG_TAMIL,
    "tamizh": LANG_TAMIL,
    "english": LANG_ENGLISH,
    "hindi": LANG_HINDI,
    "kannada": LANG_KANNADA,
    "telugu": LANG_TELUGU,
}

# Explicit language switch regex patterns
_SWITCH_PATTERNS = [
    # English commands with explicit language name
    (r'\b(?:change\s+(?:the\s+)?language\s+to\s+)(tamil|tamizh|english|hindi|kannada|telugu)\b', 1),
    (r'\b(?:talk|speak|chat|converse|switch|change)\s+(?:to\s+|in\s+)?(?:the\s+language\s+to\s+)?(tamil|tamizh|english|hindi|kannada|telugu)\b', 1),
    (r'\bcan\s+you\s+(?:please\s+)?(?:talk|speak|chat)\s+(?:in\s+)?(tamil|tamizh|english|hindi|kannada|telugu)\b', 1),
    (r'\bwill\s+you\s+(?:please\s+)?(?:talk|speak)\s+(?:to\s+me\s+)?in\s+(tamil|tamizh|english|hindi|kannada|telugu)\b', 1),
    (r'\bcould\s+you\s+(?:please\s+)?(?:talk|speak)\s+(?:in\s+)?(tamil|tamizh|english|hindi|kannada|telugu)\b', 1),
    (r'\bplease\s+(?:talk|speak|switch(?:\s+to)?)\s+(?:in\s+)?(tamil|tamizh|english|hindi|kannada|telugu)\b', 1),
    (r'\b(tamil|tamizh|english|hindi|kannada|telugu)\s+(?:please|language)\b', 1),
    (r'\b(?:switch|change)\s+to\s+(tamil|tamizh|english|hindi|kannada|telugu)\b', 1),

    # Transliterated Indic phrases
    (r'\b(?:tamil|tamizh)\s*(?:la|il)\s*(?:pesunga|pesalam|pesu|pesa\s*mudiyuma)\b', LANG_TAMIL),
    (r'\b(?:pesunga|pesalam)\s*(?:tamil|tamizh)\s*(?:la|il)?\b', LANG_TAMIL),
    (r'\b(?:hindi)\s*(?:me|mein)\s*(?:baat\s*karo|baat\s*kijiye|boliye|baat\s*karein)\b', LANG_HINDI),
    (r'\b(?:kannada)\s*(?:dalli|alli)?\s*(?:matadi|mathadi|matanadi|mathanadi)\b', LANG_KANNADA),
    (r'\b(?:telugu)\s*(?:lo)?\s*(?:matladandi|matladu|matladavacha)\b', LANG_TELUGU),

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
]'''

replacement = r'''# Name to ISO code map for parsing
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
}

_LANG_OR_REGEX = r'(?:tamil|tamizh|thamil|thamizh|tamilan|tamizha|english|angrezi|hindi|kannada|kanada|telugu|telgu|தமிழ்|தமிழு|தமிழில்|தமிழ்ல|ஆங்கிலம்|हिंदी|இந்தி|ஹிந்தி|ಕನ್ನಡ|ಕನ್ನಡದಲ್ಲಿ|తెలుగు|తెలుగులో)'

# Explicit language switch regex patterns
_SWITCH_PATTERNS = [
    # English commands with explicit language name
    (rf'\b(?:change\s+(?:the\s+)?language\s+to\s+)({_LANG_OR_REGEX})\b', 1),
    (rf'\b(?:talk|speak|chat|converse|switch|change)\s+(?:to\s+|in\s+)?(?:the\s+language\s+to\s+)?({_LANG_OR_REGEX})\b', 1),
    (rf'\bcan\s+you\s+(?:please\s+)?(?:talk|speak|chat)\s+(?:in\s+)?({_LANG_OR_REGEX})\b', 1),
    (rf'\bwill\s+you\s+(?:please\s+)?(?:talk|speak)\s+(?:to\s+me\s+)?(?:in\s+)?({_LANG_OR_REGEX})\b', 1),
    (rf'\bcould\s+you\s+(?:please\s+)?(?:talk|speak)\s+(?:in\s+)?({_LANG_OR_REGEX})\b', 1),
    (rf'\bplease\s+(?:talk|speak|switch(?:\s+to)?)\s+(?:in\s+)?({_LANG_OR_REGEX})\b', 1),
    (rf'\b({_LANG_OR_REGEX})\s+(?:please|language)\b', 1),
    (rf'\b(?:switch|change)\s+to\s+({_LANG_OR_REGEX})\b', 1),
    (rf'^\s*({_LANG_OR_REGEX})[\.\!\?]?\s*$', 1),
    (rf'\bin\s+({_LANG_OR_REGEX})[\.\!\?]?\s*$', 1),

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
    r'\b(?:can\s+you\s+)?(?:please\s+)?(?:talk|speak|switch)\s+(?:in|to)\b\s*$',
    r'\b(?:can\s+you\s+)?(?:please\s+)?(?:change|switch)\s+(?:the\s+)?language(?:\s+to)?\b\s*$',
    r'\b(?:will\s+you\s+)?(?:please\s+)?(?:talk|speak)\s+(?:to\s+me\s+)?in\b\s*$',
    r'\b(?:could\s+you\s+)?(?:please\s+)?(?:talk|speak)\s+in\b\s*$',
    r'\b(?:please\s+)?(?:switch|change)\s+to\b\s*$',
    r'\b(?:can\s+we\s+)?(?:talk|speak)\s+in\b\s*$',
    r'\b(?:can\s+you\s+)?(?:please\s+)?speak\s+in\b\s*$',
    r'\b(?:can\s+you\s+)?(?:please\s+)?talk\s+in\b\s*$',
]'''

assert target in content, 'target not found'
content = content.replace(target, replacement, 1)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print('SUCCESSFULLY PATCHED language_service.py part 1')

