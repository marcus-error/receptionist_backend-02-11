path = r'receptionist_backend-02-11\src\services\agent_engine.py'
with open(path, 'r', encoding='utf-8') as f:
    c = f.read()

old_block = '''        # 1. ENGLISH-ONLY ENFORCEMENT & REDIRECTION GUARD
        booking_session.conversation_language = "en"
        current_lang = "en"
        if language_service.is_language_switch_request(user_text) or language_service.is_non_english_input(user_text):
            logger.info(f"[AGENT] [ENGLISH_ONLY_REDIRECT] Intercepted non-English input: '{user_text[:60]}'")
            lang_resp = language_service.get_language_switch_confirmation()
            self._log_turn(session_id, user_text, lang_resp, "LANGUAGE_REDIRECT", booking_session, channel=channel, emotion="Neutral - 100%")
            post_eval = booking_state_service.evaluate_post_response(booking_session)

            return {
                "response": lang_resp,
                "intent": "LANGUAGE_REDIRECT",
                "emotion": "Neutral - 100%",
                "language": "en",
                "stage": booking_session.current_stage,
                "next_action": post_eval.get("next_action", NextAction.USER_INPUT_REQUIRED),
                "quick_actions": self.get_quick_actions(booking_session),
                "collected_fields": booking_session.collected_fields,
                "missing_fields": booking_session.missing_fields,
                "active_rag_title": active_doc_title,
                "telemetry": tracker.log_summary()
            }'''

new_block = '''        # 1. DYNAMIC 5-LANGUAGE SWITCHING & MULTILINGUAL CONVERSATION
        target_lang, is_switch = language_service.check_explicit_language_switch(user_text)
        if is_switch:
            if target_lang in ("ta", "en", "hi", "kn", "te"):
                booking_session.conversation_language = target_lang
            effective_target = target_lang or booking_session.conversation_language
            lang_resp = language_service.get_language_switch_confirmation(
                effective_target, booking_session=booking_session
            )
            self._log_turn(session_id, user_text, lang_resp, "LANGUAGE_SWITCH", booking_session, channel=channel, emotion="Neutral - 100%")
            post_eval = booking_state_service.evaluate_post_response(booking_session)

            return {
                "response": lang_resp,
                "intent": "LANGUAGE_SWITCH",
                "emotion": "Neutral - 100%",
                "language": booking_session.conversation_language,
                "stage": booking_session.current_stage,
                "next_action": post_eval.get("next_action", NextAction.USER_INPUT_REQUIRED),
                "quick_actions": self.get_quick_actions(booking_session),
                "collected_fields": booking_session.collected_fields,
                "missing_fields": booking_session.missing_fields,
                "active_rag_title": active_doc_title,
                "telemetry": tracker.log_summary()
            }

        current_lang = language_service.determine_conversation_language(
            user_text, current_lang=booking_session.conversation_language
        )
        booking_session.conversation_language = current_lang'''

if old_block.replace('\n', '\r\n') in c:
    c = c.replace(old_block.replace('\n', '\r\n'), new_block.replace('\n', '\r\n'))
else:
    c = c.replace(old_block, new_block)

with open(path, 'w', encoding='utf-8') as f:
    f.write(c)

print('agent_engine.py updated successfully!')

