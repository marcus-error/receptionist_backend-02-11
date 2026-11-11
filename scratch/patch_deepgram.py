path = 'receptionist_backend-02-11/src/services/deepgram_stt_service.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update default endpointing and utterance_end in __init__
old_init = r'''        endpointing_ms: int = 500,
        utterance_end_ms: int = 1500,
        organization_id: Optional[int] = None,
        keywords: Optional[List[str]] = None,
        language: Optional[str] = None,
    ):
        self.call_id = call_id
        self.on_partial = on_partial
        self.on_final = on_final
        self.on_error = on_error
        self.sample_rate = sample_rate
        self.organization_id = organization_id
        self.keywords = keywords or []
        self.model = model or getattr(settings, "DEEPGRAM_MODEL", "nova-3") or "nova-3"
        self.language = language or getattr(settings, "DEEPGRAM_LANGUAGE", "en") or "en"
        raw_endpointing = endpointing_ms or getattr(settings, "DEEPGRAM_ENDPOINTING_MS", 300) or 300
        self.endpointing_ms = max(250, int(raw_endpointing))
        raw_u_end = utterance_end_ms or getattr(settings, "DEEPGRAM_UTTERANCE_END_MS", 1500) or 1500
        self.utterance_end_ms = max(1000, int(raw_u_end))'''

new_init = r'''        endpointing_ms: int = 300,
        utterance_end_ms: int = 1000,
        organization_id: Optional[int] = None,
        keywords: Optional[List[str]] = None,
        language: Optional[str] = None,
    ):
        self.call_id = call_id
        self.on_partial = on_partial
        self.on_final = on_final
        self.on_error = on_error
        self.sample_rate = sample_rate
        self.organization_id = organization_id
        self.keywords = keywords or []
        self.model = model or getattr(settings, "DEEPGRAM_MODEL", "nova-3") or "nova-3"
        self.language = language or getattr(settings, "DEEPGRAM_LANGUAGE", "en") or "en"
        raw_endpointing = endpointing_ms or getattr(settings, "DEEPGRAM_ENDPOINTING_MS", 300) or 300
        self.endpointing_ms = max(250, min(1000, int(raw_endpointing)))
        raw_u_end = utterance_end_ms or getattr(settings, "DEEPGRAM_UTTERANCE_END_MS", 1000) or 1000
        self.utterance_end_ms = max(800, min(2000, int(raw_u_end)))'''
assert old_init in content, 'old_init not found'
content = content.replace(old_init, new_init, 1)

# 2. Add keyterm support to _build_ws_url
old_url = r'''        if kw_list and not is_multilingual_or_nova3:
            import urllib.parse
            kw_params = []
            for kw in kw_list[:15]:
                clean_kw = re.sub(r'[^a-zA-Z0-9\s]', '', kw).strip()
                if clean_kw and len(clean_kw) >= 3:
                    kw_params.append(f"&keywords={urllib.parse.quote(clean_kw)}:2")
            if kw_params:
                base_url += "".join(kw_params)
        return base_url'''

new_url = r'''        import urllib.parse
        if is_multilingual_or_nova3:
            language_terms = ["Tamil", "English", "Hindi", "Kannada", "Telugu"]
            kt_params = [f"&keyterm={urllib.parse.quote(term)}" for term in language_terms]
            if kw_list:
                for kw in kw_list[:10]:
                    clean_kw = re.sub(r'[^a-zA-Z0-9\s]', '', kw).strip()
                    if clean_kw and len(clean_kw) >= 3 and clean_kw not in language_terms:
                        kt_params.append(f"&keyterm={urllib.parse.quote(clean_kw)}")
            base_url += "".join(kt_params)
        elif kw_list:
            kw_params = []
            for kw in kw_list[:15]:
                clean_kw = re.sub(r'[^a-zA-Z0-9\s]', '', kw).strip()
                if clean_kw and len(clean_kw) >= 3:
                    kw_params.append(f"&keywords={urllib.parse.quote(clean_kw)}:2")
            if kw_params:
                base_url += "".join(kw_params)
        return base_url'''
assert old_url in content, 'old_url not found'
content = content.replace(old_url, new_url, 1)

# 3. Fix line 450: hardcoded en language
old_det = r'''        confidence = alt.get("confidence", 0.0)
        # Enforce English-only STT recognition
        self._last_detected_language = "en" 
        is_final = data.get("is_final", False)'''

new_det = r'''        confidence = alt.get("confidence", 0.0)
        self._last_detected_language = getattr(self, "language", "en") or "en"
        is_final = data.get("is_final", False)'''
assert old_det in content, 'old_det not found'
content = content.replace(old_det, new_det, 1)

# 4. Tune debounce fallback timer
old_deb = r'''                        if len(words) <= 3 or last_w in incomplete_words:
                            wait_s = 1.5
                        else:
                            wait_s = 1.2'''

new_deb = r'''                        if len(words) <= 3 or last_w in incomplete_words:
                            wait_s = 0.9
                        else:
                            wait_s = 0.7'''
assert old_deb in content, 'old_deb not found'
content = content.replace(old_deb, new_deb, 1)

# 5. Fix _handle_utterance_end setting _speech_endpoint_time
old_utt_end = r'''    async def _handle_utterance_end(self, data: Dict[str, Any]):
        """Deepgram indicates customer finished speaking turn."""
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
            self._debounce_task = None

        if self._turn_finalized:
            return'''

new_utt_end = r'''    async def _handle_utterance_end(self, data: Dict[str, Any]):
        """Deepgram indicates customer finished speaking turn."""
        self._speech_endpoint_time = time.monotonic()
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
            self._debounce_task = None

        if self._turn_finalized:
            return'''
assert old_utt_end in content, 'old_utt_end not found'
content = content.replace(old_utt_end, new_utt_end, 1)

# 6. Sanitize t_speech_endpoint in _finalize_utterance
old_fin_lat = r'''        t_start = self._utterance_start_time or (now - max(0.5, duration_s))
        t_first_partial = self._first_partial_time or t_start
        t_speech_endpoint = getattr(self, "_speech_endpoint_time", None) or now
        self._speech_endpoint_time = None

        first_partial_ms = max(40, round((t_first_partial - t_start) * 1000))
        final_ms = max(80, round((now - t_start) * 1000))

        actual_dur_s = duration_s if duration_s > 0.1 else round(final_ms / 1000.0, 2)
        customer_speech_duration_ms = round(actual_dur_s * 1000)
        finalization_latency_ms = max(40, round((now - t_speech_endpoint) * 1000))'''

new_fin_lat = r'''        t_start = self._utterance_start_time or (now - max(0.5, duration_s))
        t_first_partial = self._first_partial_time or t_start
        t_speech_endpoint = getattr(self, "_speech_endpoint_time", None) or now
        self._speech_endpoint_time = None
        if t_speech_endpoint < t_start or (now - t_speech_endpoint) > 4.0:
            t_speech_endpoint = now - 0.25

        first_partial_ms = max(40, round((t_first_partial - t_start) * 1000))
        final_ms = max(80, round((now - t_start) * 1000))

        actual_dur_s = duration_s if duration_s > 0.1 else round(final_ms / 1000.0, 2)
        customer_speech_duration_ms = round(actual_dur_s * 1000)
        finalization_latency_ms = max(40, round((now - t_speech_endpoint) * 1000))'''
assert old_fin_lat in content, 'old_fin_lat not found'
content = content.replace(old_fin_lat, new_fin_lat, 1)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print('SUCCESSFULLY PATCHED deepgram_stt_service.py')

