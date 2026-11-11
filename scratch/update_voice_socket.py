path = r'receptionist_backend-02-11\src\sockets\voice_socket.py'
with open(path, 'r', encoding='utf-8') as f:
    c = f.read()

old_block = '''                target_lang, is_lang_switch = language_service.check_explicit_language_switch(transcription)
                if is_lang_switch:
                    booking_session.conversation_language = "en"
                    current_lang = "en"
                    lang_resp = language_service.get_language_switch_confirmation()

                    norm_ans = _normalize_speech(lang_resp)
                    if norm_ans:
                        recent_ai_utterances.append(norm_ans)
                    last_ai_speech_time = time.time()

                    await websocket.send_json({"type": "analysis", "intent": "LANGUAGE_REDIRECT",
                                               "emotion": "Neutral - 100%", "language": "en"})
                    await websocket.send_json({"type": "response", "text": lang_resp})

                    audio_response = await tts_service.synthesize_stream_chunk_async(lang_resp, language="en")
                    if audio_response and len(audio_response) > 100:
                        audio_base64 = base64.b64encode(audio_response).decode('utf-8')
                        await websocket.send_json({"type": "audio", "audio": audio_base64})

                    _log_turn(session_id, transcription, lang_resp, "LANGUAGE_SWITCH", booking_session)
                    await _handle_post_response_state(websocket, booking_session, session_id, current_lang, recent_ai_utterances)
                    continue'''

new_block = '''                target_lang, is_lang_switch = language_service.check_explicit_language_switch(transcription)
                if is_lang_switch:
                    if target_lang in ("ta", "en", "hi", "kn", "te"):
                        booking_session.conversation_language = target_lang
                    current_lang = target_lang or booking_session.conversation_language
                    lang_resp = language_service.get_language_switch_confirmation(current_lang, booking_session=booking_session)

                    norm_ans = _normalize_speech(lang_resp)
                    if norm_ans:
                        recent_ai_utterances.append(norm_ans)
                    last_ai_speech_time = time.time()

                    await websocket.send_json({"type": "analysis", "intent": "LANGUAGE_SWITCH",
                                               "emotion": "Neutral - 100%", "language": current_lang})
                    await websocket.send_json({"type": "response", "text": lang_resp})

                    audio_response = await tts_service.synthesize_stream_chunk_async(lang_resp, language=current_lang)
                    if audio_response and len(audio_response) > 100:
                        audio_base64 = base64.b64encode(audio_response).decode('utf-8')
                        await websocket.send_json({"type": "audio", "audio": audio_base64})

                    _log_turn(session_id, transcription, lang_resp, "LANGUAGE_SWITCH", booking_session)
                    await _handle_post_response_state(websocket, booking_session, session_id, current_lang, recent_ai_utterances)
                    continue'''

if old_block.replace('\n', '\r\n') in c:
    c = c.replace(old_block.replace('\n', '\r\n'), new_block.replace('\n', '\r\n'))
else:
    c = c.replace(old_block, new_block)

with open(path, 'w', encoding='utf-8') as f:
    f.write(c)

print('voice_socket.py updated successfully!')

