path = 'receptionist_backend-02-11/src/services/sip_call_handler.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. In __init__, add _pending_language_switch and RTP tracking fields
old_init_fields = r'''        # Session & Compatibility fields
        self.current_lang = "en"
        self.language_code = "en"
        self.detected_language = "English"
        self.language_confidence = 1.0
        self.original_transcript = ""'''

new_init_fields = r'''        # Session & Compatibility fields
        self.current_lang = "en"
        self.language_code = "en"
        self.detected_language = "English"
        self.language_confidence = 1.0
        self.original_transcript = ""
        self._pending_language_switch: Optional[Dict[str, Any]] = None
        self._last_rtp_seq_end: Optional[int] = None'''
assert old_init_fields in content, 'old_init_fields not found'
content = content.replace(old_init_fields, new_init_fields, 1)

# 2. In _on_deepgram_final: Split-utterance and ambiguous language switch handling
old_switch_logic = r'''        target_lang, is_switch = language_service.check_explicit_language_switch(text)
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
                return'''

new_switch_logic = r'''        # Check pending language switch context from prior split utterance
        now_t = time.monotonic()
        pending_sw = getattr(self, "_pending_language_switch", None)
        target_lang = None
        is_switch = False
        is_ambiguous = False
        prefix_used = ""

        if pending_sw and (now_t - pending_sw.get("time", 0.0)) < 8.0:
            prefix_used = pending_sw.get("text", "")
            combined_text = f"{prefix_used} {text}".strip()
            logger.info(f"[LANG_SWITCH_SPLIT_RESOLVE] Evaluating pending prefix='{prefix_used}' + text='{text}' -> combined='{combined_text}'")
            target_lang, is_switch = language_service.check_explicit_language_switch(combined_text)
            if not is_switch:
                target_lang, is_switch = language_service.check_explicit_language_switch(text)
            if is_switch and target_lang in ("ta", "en", "hi", "kn", "te"):
                self._pending_language_switch = None
            else:
                # Ambiguous word following incomplete language switch (e.g. "Can you please talk in" -> "Summer")
                # Requirement 5: Do not switch based on ambiguous words. Ask which supported language the customer wants.
                is_ambiguous = True
        else:
            self._pending_language_switch = None
            target_lang, is_switch = language_service.check_explicit_language_switch(text)

        if not is_switch and not is_ambiguous:
            if language_service.is_incomplete_language_switch(text):
                # Customer said "Can you please talk in", "Please switch to", etc.
                self._pending_language_switch = {"text": text, "time": now_t}
                clarification_msg = language_service.get_language_clarification_prompt(self.current_lang)
                self.session_history.append({"role": "user", "text": text})
                self.session_history.append({"role": "model", "text": clarification_msg})
                metrics["response_id"] = f"ai_lang_clarify_{int(time.time()*1000)}"
                metrics["source"] = "LANGUAGE_CLARIFICATION"
                logger.info(f"[LANG_SWITCH_INCOMPLETE_PROMPT] Incomplete request '{text}' -> Asking clarification in '{self.current_lang}'")
                await self._speak(clarification_msg, "language_clarification", metrics)
                return

        if is_ambiguous:
            clarification_msg = language_service.get_language_clarification_prompt(self.current_lang)
            self.session_history.append({"role": "user", "text": text})
            self.session_history.append({"role": "model", "text": clarification_msg})
            metrics["response_id"] = f"ai_lang_clarify_{int(time.time()*1000)}"
            metrics["source"] = "LANGUAGE_CLARIFICATION"
            logger.info(f"[LANG_SWITCH_AMBIGUOUS_PROMPT] Ambiguous request '{text}' after '{prefix_used}' -> Asking clarification in '{self.current_lang}'")
            await self._speak(clarification_msg, "language_clarification", metrics)
            return

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
                self._pending_language_switch = None

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
                return'''
assert old_switch_logic in content, 'old_switch_logic not found'
content = content.replace(old_switch_logic, new_switch_logic, 1)

# 3. In _stream_llm_sentences: pass pipelined_stream=True and wait for final audio after loop
old_stream_loop = r'''                # ── Speak each sentence as it arrives (sentence-streaming TTS) ──
                # Keep utt_id, response_id and speech endpoints in sent_metrics across all chunks
                sent_metrics = {
                    "is_chunk": True,
                    "response_id": response_id,
                    "utt_id": utt_id,
                    "speech_end_monotonic": metrics.get("speech_end_monotonic"),
                    "t_speech_endpoint": metrics.get("t_speech_endpoint"),
                    "t_stt_final": metrics.get("t_stt_final"),
                    "endpointing_ms": metrics.get("endpointing_ms", 300),
                    "finalization_latency_ms": metrics.get("finalization_latency_ms", 0),
                    "intent_latency_ms": metrics.get("intent_latency_ms", 0),
                    "rag_latency_ms": metrics.get("rag_latency_ms", 0),
                    "llm_latency_ms": metrics.get("llm_latency_ms", 0),
                    "audio_duration_s": metrics.get("audio_duration_s", 1.0),
                }
                if sentence_index == 0:
                    for k, v in metrics.items():
                        if k not in sent_metrics:
                            sent_metrics[k] = v
                sentence_index += 1
                await self._speak(sentence, intent_name, sent_metrics)

                if self.ended or self._barge_in_event.is_set() or stream_abort.is_set():
                    break

            full_response = " ".join(full_response_parts).strip()'''

new_stream_loop = r'''                # ── Speak each sentence as it arrives (pipelined sentence-streaming TTS) ──
                # Keep utt_id, response_id and speech endpoints in sent_metrics across all chunks
                sent_metrics = {
                    "is_chunk": True,
                    "pipelined_stream": True,
                    "chunk_idx": sentence_index,
                    "response_id": response_id,
                    "utt_id": utt_id,
                    "speech_end_monotonic": metrics.get("speech_end_monotonic"),
                    "t_speech_endpoint": metrics.get("t_speech_endpoint"),
                    "t_stt_final": metrics.get("t_stt_final"),
                    "endpointing_ms": metrics.get("endpointing_ms", 300),
                    "finalization_latency_ms": metrics.get("finalization_latency_ms", 0),
                    "intent_latency_ms": metrics.get("intent_latency_ms", 0),
                    "rag_latency_ms": metrics.get("rag_latency_ms", 0),
                    "llm_latency_ms": metrics.get("llm_latency_ms", 0),
                    "audio_duration_s": metrics.get("audio_duration_s", 1.0),
                }
                if sentence_index == 0:
                    for k, v in metrics.items():
                        if k not in sent_metrics:
                            sent_metrics[k] = v
                sentence_index += 1
                await self._speak(sentence, intent_name, sent_metrics)

                if self.ended or self._barge_in_event.is_set() or stream_abort.is_set():
                    break

            # Await remaining playback for all pipelined chunks seamlessly
            remaining_wait = max(0.0, getattr(self, "_ai_speaking_until", 0.0) - time.monotonic())
            if remaining_wait > 0.05 and not self.ended and not self._barge_in_event.is_set():
                try:
                    await asyncio.wait_for(self._barge_in_event.wait(), timeout=remaining_wait)
                    if self.ended:
                        logger.info(f"[CALL_ENDED] Call {self.call_sid} ended during playback")
                    else:
                        logger.info(f"[PLAYBACK_INTERRUPTED] Customer spoke during playback remaining_s={remaining_wait:.2f}")
                        if self.audio_clear_callback and not self.ended:
                            self.audio_clear_callback()
                except asyncio.TimeoutError:
                    logger.info(f"[PLAYBACK_COMPLETED] call_id={self.call_sid} response_id={response_id} monotonic={time.monotonic():.4f}")

            self._ai_speaking = False
            self._last_ai_speak_end_time = time.monotonic()
            self._ai_speaking_until = time.monotonic() + 0.15

            full_response = " ".join(full_response_parts).strip()'''
assert old_stream_loop in content, 'old_stream_loop not found'
content = content.replace(old_stream_loop, new_stream_loop, 1)

# 4. In _speak_locked: structured diagnostic logs, pipelined playback & lifecycle safety
old_playback_block = r'''                        tx_result = "SUCCESS"
                        cb = self.audio_send_callback
                        if cb:
                            try:
                                cb(pcmu_bytes)
                            except Exception as _cb_err:
                                tx_result = f"FAILED: {_cb_err}"
                                logger.warning(f"[RTP_TX] Callback write error: {_cb_err}")
                        else:
                            tx_result = "DROPPED: no audio_send_callback"

                        logger.info(
                            f"\n[RTP_VOICE_TRANSMISSION]"
                            f"\n  call_id={self.call_sid}"
                            f"\n  utterance_id={utt_id}"
                            f"\n  audio_source=AI_TTS"
                            f"\n  decoded_pcm_bytes={len(pcm16_audio)}"
                            f"\n  sample_rate=8000"
                            f"\n  channel_count=1"
                            f"\n  pcmu_encoded_bytes={len(pcmu_bytes)}"
                            f"\n  rtp_packet_count={pkt_count}"
                            f"\n  rtp_payload_bytes=160"
                            f"\n  rtp_destination={rtp_dest_ip}:{rtp_dest_port}"
                            f"\n  seq_start={seq_start}"
                            f"\n  seq_end={seq_end}"
                            f"\n  duration_s={audio_play_dur:.2f}"
                            f"\n  rms_energy={pcm_rms}"
                            f"\n  peak_amplitude={pcm_peak}"
                            f"\n  actual_transmission_result={tx_result}"
                        )
                        logger.info(f"[RTP_AUDIO_TRANSMISSION] call_id={self.call_sid} utterance_id={utt_id} bytes={len(pcmu_bytes)} duration_s={audio_play_dur:.2f} status={tx_result}")

                        # Feed AI speech to TwoWayCallRecorder for synchronized recording
                        if hasattr(self, "two_way_recorder") and pcm16_audio:
                            self.two_way_recorder.feed_ai_pcm16(pcm16_audio)

                        # Yield event-loop time during the audio playback window, allowing instant barge-in
                        if audio_play_dur > 0.05:
                            try:
                                await asyncio.wait_for(self._barge_in_event.wait(), timeout=audio_play_dur)
                                logger.info("[BARGE_IN] Customer spoke - interrupted playback sleep")
                                if self.audio_clear_callback:
                                    try:
                                        self.audio_clear_callback()
                                    except Exception:
                                        pass
                            except asyncio.TimeoutError:
                                pass

                        # Mark end of this audio chunk for echo guard
                        self._ai_speaking = False
                        self._last_ai_speak_end_time = time.monotonic()
                        self._ai_speaking_until = time.monotonic() + 0.15'''

new_playback_block = r'''                        tx_result = "SUCCESS"
                        cb = self.audio_send_callback
                        if cb:
                            try:
                                cb(pcmu_bytes)
                            except Exception as _cb_err:
                                tx_result = f"FAILED: {_cb_err}"
                                logger.warning(f"[RTP_TX] Callback write error: {_cb_err}")
                        else:
                            tx_result = "DROPPED: no audio_send_callback"

                        # Structured Diagnostic Logs
                        chunk_idx = metrics.get("chunk_idx", 0) if metrics else 0
                        is_pipelined = bool(metrics and metrics.get("pipelined_stream"))
                        resp_id = (metrics.get("response_id") if metrics else None) or getattr(self, "_current_response_id", "resp")

                        logger.info(f"[TTS_CHUNK_RECEIVED] call_id={self.call_sid} response_id={resp_id} chunk_idx={chunk_idx} bytes={len(audio_bytes)}")
                        logger.info(f"[MP3_DECODE_COMPLETE] call_id={self.call_sid} response_id={resp_id} raw_pcm_bytes={len(pcm16_audio)}")
                        logger.info(f"[PCM_SAMPLES_RECEIVED] call_id={self.call_sid} response_id={resp_id} samples={len(pcm16_audio)//2}")
                        logger.info(f"[RESAMPLED_SAMPLES] call_id={self.call_sid} rate=8000 channels=1 samples={len(pcm16_audio)//2}")
                        logger.info(f"[PCMU_BYTES_ENCODED] call_id={self.call_sid} response_id={resp_id} bytes={len(pcmu_bytes)} packets={pkt_count}")

                        q_depth = max(0, round((getattr(self, "_ai_speaking_until", 0.0) - time.monotonic()) * 50))
                        logger.info(f"[RTP_PACKET_ENQUEUED] call_id={self.call_sid} response_id={resp_id} packets={pkt_count} audio_queue_depth={q_depth}")
                        logger.info(f"[RTP_PACKET_SENT] call_id={self.call_sid} response_id={resp_id} packets_sent={pkt_count} seq_start={seq_start} seq_end={seq_end}")
                        logger.info(f"[PLAYBACK_STARTED] call_id={self.call_sid} response_id={resp_id} duration_s={audio_play_dur:.2f} monotonic={time.monotonic():.4f}")

                        # Sequence continuity check
                        last_end_seq = getattr(self, "_last_rtp_seq_end", None)
                        if last_end_seq is not None and seq_start != last_end_seq:
                            seq_gap = (seq_start - last_end_seq) & 0xFFFF
                            logger.warning(f"[RTP_SEQUENCE_GAP] call_id={self.call_sid} expected={last_end_seq} actual={seq_start} gap={seq_gap}")
                        self._last_rtp_seq_end = seq_end

                        # Feed AI speech to TwoWayCallRecorder for synchronized recording
                        if hasattr(self, "two_way_recorder") and pcm16_audio:
                            self.two_way_recorder.feed_ai_pcm16(pcm16_audio)

                        if is_pipelined:
                            # Advance playback end window without blocking the sentence loop
                            self._ai_speaking = True
                            self._ai_speaking_until = max(getattr(self, "_ai_speaking_until", 0.0), time.monotonic()) + audio_play_dur
                        else:
                            # Standalone utterance playback wait
                            if audio_play_dur > 0.05:
                                try:
                                    await asyncio.wait_for(self._barge_in_event.wait(), timeout=audio_play_dur)
                                    if self.ended:
                                        logger.info(f"[CALL_ENDED] Call {self.call_sid} ended during playback")
                                        return
                                    logger.info(f"[PLAYBACK_INTERRUPTED] Customer spoke - interrupted playback sleep call_id={self.call_sid}")
                                    if self.audio_clear_callback and not self.ended:
                                        try:
                                            self.audio_clear_callback()
                                        except Exception:
                                            pass
                                except asyncio.TimeoutError:
                                    logger.info(f"[PLAYBACK_COMPLETED] call_id={self.call_sid} response_id={resp_id} monotonic={time.monotonic():.4f}")

                            self._ai_speaking = False
                            self._last_ai_speak_end_time = time.monotonic()
                            self._ai_speaking_until = time.monotonic() + 0.15'''
assert old_playback_block in content, 'old_playback_block not found'
content = content.replace(old_playback_block, new_playback_block, 1)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print('SUCCESSFULLY PATCHED sip_call_handler.py')

