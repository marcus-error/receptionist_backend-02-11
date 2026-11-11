path = r'receptionist_backend-02-11\src\services\booking_state_service.py'
with open(path, 'r', encoding='utf-8') as f:
    c = f.read()

old_block = '''        # Presence check ("are you there", "can you hear me") during active booking
        presence_phrases = ["can you hear me", "are you there", "am i audible", "can you hear", "hear me", "is anyone there"]
        if any(phrase in txt_lower for phrase in presence_phrases) and not any(w in txt_lower for w in ["book", "appointment", "doctor", "schedule", "reserve"]):
            silence_prompt = self.get_silence_prompt(session)
            resp = f"Yes, I'm right here! {silence_prompt}"
            self._log_conversation_turn(session, text, resp, "presence_reassurance", state_before=state_before)
            return _ret(resp)

        session.conversation_language = "en" 

        lang = session.conversation_language'''

new_block = '''        # Presence check ("are you there", "can you hear me") during active booking
        presence_phrases = ["can you hear me", "are you there", "am i audible", "can you hear", "hear me", "is anyone there"]
        if any(phrase in txt_lower for phrase in presence_phrases) and not any(w in txt_lower for w in ["book", "appointment", "doctor", "schedule", "reserve"]):
            silence_prompt = self.get_silence_prompt(session)
            resp = f"Yes, I'm right here! {silence_prompt}"
            self._log_conversation_turn(session, text, resp, "presence_reassurance", state_before=state_before)
            return _ret(resp)

        # Dynamic Language Switch check during active booking
        from src.services.language_service import language_service, SUPPORTED_LANGUAGES
        target_lang, is_switch = language_service.check_explicit_language_switch(text)
        if is_switch and target_lang in SUPPORTED_LANGUAGES:
            session.conversation_language = target_lang
            resp = language_service.get_language_switch_confirmation(target_lang, booking_session=session)
            self._log_conversation_turn(session, text, resp, "language_switch", state_before=state_before)
            return _ret(resp)

        lang = session.conversation_language'''

if old_block.replace('\n', '\r\n') in c:
    c = c.replace(old_block.replace('\n', '\r\n'), new_block.replace('\n', '\r\n'))
else:
    c = c.replace(old_block, new_block)

with open(path, 'w', encoding='utf-8') as f:
    f.write(c)

print('booking_state_service.py updated successfully!')

