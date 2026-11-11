path = 'receptionist_backend-02-11/src/services/language_service.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

target = r'''    def is_language_switch_request(self, text: str) -> bool:
        """Returns True if the utterance requests a language change."""
        _, is_switch = self.check_explicit_language_switch(text)
        return is_switch'''

replacement = r'''    def is_language_switch_request(self, text: str) -> bool:
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
        return prompts.get(current_lang, prompts[LANG_ENGLISH])'''

assert target in content, 'target not found'
content = content.replace(target, replacement, 1)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print('SUCCESSFULLY PATCHED language_service.py part 2')

