with open('src/services/sip_call_handler.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Localized fallback in empty LLM response (Requirement 12)
old_fallback = '''            # Fallback if empty and no barge-in occurred
            if not full_response and not self._barge_in_event.is_set() and not self.ended:
                org_name = rag_service.get_organization_name(self.organization_id)
                if org_name:
                    full_response = f"Thank you for contacting {org_name}. How may I assist you today?"
                else:
                    full_response = "Thank you for contacting us. How may I assist you today?"
                fallback_metrics = dict(metrics)'''

new_fallback = '''            # Fallback if empty and no barge-in occurred (Requirement 12: Maintain active language)
            if not full_response and not self._barge_in_event.is_set() and not self.ended:
                org_name = rag_service.get_organization_name(self.organization_id)
                if self.current_lang != "en":
                    full_response = language_service.get_language_fallback_prompt(self.current_lang, org_name)
                elif org_name:
                    full_response = f"Thank you for contacting {org_name}. How may I assist you today?"
                else:
                    full_response = "Thank you for contacting us. How may I assist you today?"
                fallback_metrics = dict(metrics)'''

assert old_fallback in text, "old_fallback not found in sip_call_handler.py"
text = text.replace(old_fallback, new_fallback, 1)

# 2. Fix RTP enqueue and remove false RTP_SEQUENCE_GAP
old_rtp_block = '''                        q_depth = max(0, round((getattr(self, "_ai_speaking_until", 0.0) - time.monotonic()) * 50))
                        logger.info(f"[RTP_PACKET_ENQUEUED] call_id={self.call_sid} response_id={resp_id} packets={pkt_count} audio_queue_depth={q_depth}")
                        logger.info(f"[RTP_PACKET_SENT] call_id={self.call_sid} response_id={resp_id} packets_sent={pkt_count} seq_start={seq_start} seq_end={seq_end}")
                        logger.info(f"[PLAYBACK_STARTED] call_id={self.call_sid} response_id={resp_id} duration_s={audio_play_dur:.2f} monotonic={time.monotonic():.4f}")

                        # Sequence continuity check
                        last_end_seq = getattr(self, "_last_rtp_seq_end", None)
                        if last_end_seq is not None and seq_start != last_end_seq:
                            seq_gap = (seq_start - last_end_seq) & 0xFFFF
                            logger.warning(f"[RTP_SEQUENCE_GAP] call_id={self.call_sid} expected={last_end_seq} actual={seq_start} gap={seq_gap}")
                        self._last_rtp_seq_end = seq_end'''

new_rtp_block = '''                        # Read true hardware transmit queue depth from authoritative RTP sender
                        q_depth = max(0, round((getattr(self, "_ai_speaking_until", 0.0) - time.monotonic()) * 50))
                        if pycall and hasattr(pycall, "RTPClients") and pycall.RTPClients:
                            real_rtp = pycall.RTPClients[0]
                            real_q = getattr(getattr(real_rtp, "pmout", None), "queue_depth_packets", None)
                            if real_q is not None:
                                q_depth = real_q

                        logger.info(f"[RTP_PACKET_ENQUEUED] call_id={self.call_sid} response_id={resp_id} packets={pkt_count} audio_queue_depth={q_depth}")
                        logger.info(f"[PLAYBACK_STARTED] call_id={self.call_sid} response_id={resp_id} duration_s={audio_play_dur:.2f} monotonic={time.monotonic():.4f}")'''

assert old_rtp_block in text, "old_rtp_block not found in sip_call_handler.py"
text = text.replace(old_rtp_block, new_rtp_block, 1)

with open('src/services/sip_call_handler.py', 'w', encoding='utf-8', newline='') as f:
    f.write(text)

print("SUCCESSFULLY PATCHED sip_call_handler.py")

