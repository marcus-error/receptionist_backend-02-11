path = r'receptionist_backend-02-11\src\services\sip_call_handler.py'
with open(path, 'r', encoding='utf-8') as f:
    c = f.read()

old_block = '''        # Enforce English-only recognition and conversation lock
        real_conf = float(metrics.get("confidence", 1.0))
        self.call_id = self.call_sid
        self.utterance_id = utt_id
        self.stt_language_confidence = real_conf
        self.original_transcript = text

        self.detected_utterance_language = "en"
        self.preferred_response_language = "en"
        self.language_switch_requested = False
        self.selected_tts_provider = "murf"
        self.selected_tts_voice = "en-IN-isha"
        self.transcript_language = "en"
        self.language_code = "en"
        self.current_lang = "en"
        self.detected_language = "English"
        self.language_confidence = real_conf
        self.ai_response_language = "en"
        if hasattr(self, "booking_session"):
            self.booking_session.conversation_language = "en"

        # Intercept foreign language input or language switch requests
        if language_service.is_language_switch_request(text) or language_service.is_non_english_input(text):
            logger.info(f"[ENGLISH_ONLY_REDIRECT] Foreign language / switch request intercepted: '{text[:60]}'")
            redirect_msg = language_service.get_language_switch_confirmation()
            self.session_history.append({"role": "user", "text": text})
            self.session_history.append({"role": "model", "text": redirect_msg})
            metrics["response_id"] = f"ai_en_redirect_{int(time.time()*1000)}"
            metrics["source"] = "ENGLISH_ONLY_REDIRECT"
            await self._speak(redirect_msg, "language_redirect", metrics)
            return

        readable_utterance = "English"
        readable_preferred = "English"
        stt_detected_lang = "en"'''

new_block = '''        # Dynamic 5-Language Switching & Multilingual Conversation
        real_conf = float(metrics.get("confidence", 1.0))
        self.call_id = self.call_sid
        self.utterance_id = utt_id
        self.stt_language_confidence = real_conf
        self.original_transcript = text

        target_lang, is_switch = language_service.check_explicit_language_switch(text)
        if is_switch:
            if target_lang in ("ta", "en", "hi", "kn", "te"):
                # Customer explicitly requested a valid supported language
                self.current_lang = target_lang
                self.language_code = target_lang
                self.detected_language = language_service.get_language_name(target_lang)
                self.detected_utterance_language = target_lang
                self.preferred_response_language = target_lang
                self.ai_response_language = target_lang
                self.transcript_language = target_lang
                self.language_switch_requested = True
                self.language_change_timestamp = datetime.now(timezone.utc).isoformat()

                v_id, _, prov = language_service.get_tts_voice_and_model(target_lang)
                self.selected_tts_voice = v_id
                self.selected_tts_provider = prov

                if hasattr(self, "booking_session") and self.booking_session:
                    self.booking_session.conversation_language = target_lang

                # Dynamically reconfigure Deepgram STT stream for the new language
                if hasattr(self, "deepgram_session") and self.deepgram_session:
                    try:
                        await self.deepgram_session.switch_language(target_lang)
                    except Exception as _stt_sw_err:
                        logger.warning(f"[STT_LANG_SWITCH_ERR] {_stt_sw_err}")

                switch_msg = language_service.get_language_switch_confirmation(
                    target_lang, booking_session=getattr(self, "booking_session", None)
                )

                self.session_history.append({"role": "user", "text": text})
                self.session_history.append({"role": "model", "text": switch_msg})
                metrics["response_id"] = f"ai_lang_switch_{int(time.time()*1000)}"
                metrics["source"] = "LANGUAGE_SWITCH"
                metrics["language"] = self.current_lang
                logger.info(f"[LANG_SWITCH_EXECUTED] call_id={self.call_sid} new_lang={target_lang} response='{switch_msg}'")
                await self._speak(switch_msg, "language_switch", metrics)
                return
            else:
                # Unsupported foreign language requested (e.g. Spanish, French)
                unsupported_msg = language_service.get_language_switch_confirmation(None)
                self.session_history.append({"role": "user", "text": text})
                self.session_history.append({"role": "model", "text": unsupported_msg})
                metrics["response_id"] = f"ai_lang_unsupported_{int(time.time()*1000)}"
                metrics["source"] = "LANGUAGE_UNSUPPORTED"
                await self._speak(unsupported_msg, "language_unsupported", metrics)
                return

        # Normal in-language conversation turn
        detected_script_lang = language_service.detect_spoken_language(text, current_lang=self.current_lang)
        self.current_lang = detected_script_lang
        self.language_code = self.current_lang
        self.detected_language = language_service.get_language_name(self.current_lang)
        self.detected_utterance_language = self.current_lang
        self.preferred_response_language = self.current_lang
        self.ai_response_language = self.current_lang
        self.transcript_language = self.current_lang
        self.language_switch_requested = False

        v_id, _, prov = language_service.get_tts_voice_and_model(self.current_lang)
        self.selected_tts_voice = v_id
        self.selected_tts_provider = prov
        if hasattr(self, "booking_session") and self.booking_session:
            self.booking_session.conversation_language = self.current_lang

        readable_utterance = self.detected_language
        readable_preferred = self.detected_language
        stt_detected_lang = self.current_lang'''

if old_block.replace('\n', '\r\n') in c:
    c = c.replace(old_block.replace('\n', '\r\n'), new_block.replace('\n', '\r\n'))
else:
    c = c.replace(old_block, new_block)

with open(path, 'w', encoding='utf-8') as f:
    f.write(c)

print('sip_call_handler.py updated successfully!')

