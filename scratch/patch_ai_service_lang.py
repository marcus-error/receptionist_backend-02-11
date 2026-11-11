with open('src/services/ai_service.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Update _fast_rule_response signature and body
old_def = '''    def _fast_rule_response(
        self,
        text: str,
        intent: str = "general_question",
        organization_id: Optional[int] = None,
        is_ongoing: bool = False,
        booking_session=None,
        has_context: bool = False,
        call_direction: str = "INBOUND"
    ) -> str | None:
        """
        Sub-50ms rule-based responses for common conversational pleasantries and basic identity questions.
        Strictly bound to the currently active RAG document.

        Language-switch requests are handled here as the HIGHEST-PRIORITY fast rule,
        before any RAG lookup, entity validation, or LLM call.
        """
        txt = text.lower().strip()
        txt_clean = re.sub(r'[^a-zA-Z0-9\s]', '', txt).strip()
        from src.services.cache_service import cache_service
        from src.services.rag_service import rag_service
        from src.services.language_service import language_service

        # ------------------------------------------------------------------ #
        # PRIORITY 0 — Language Switch (must fire before any RAG/LLM call)    #
        # ------------------------------------------------------------------ #
        target_lang, is_switch = language_service.check_explicit_language_switch(text)
        if is_switch:
            return language_service.get_language_switch_confirmation(
                target_lang, booking_session=booking_session, org_name=org_name
            )
        active_doc_id = rag_service._resolve_active_document_id(organization_id)
        org_name = rag_service.get_organization_name(organization_id, document_id=active_doc_id)
        welcome_greeting = rag_service.get_organization_greeting(organization_id, document_id=active_doc_id)'''

new_def = '''    def _fast_rule_response(
        self,
        text: str,
        intent: str = "general_question",
        organization_id: Optional[int] = None,
        is_ongoing: bool = False,
        booking_session=None,
        has_context: bool = False,
        call_direction: str = "INBOUND",
        conversation_language: str = "en"
    ) -> str | None:
        """
        Sub-50ms rule-based responses for common conversational pleasantries and basic identity questions.
        Strictly bound to the currently active RAG document.

        Language-switch requests are handled here as the HIGHEST-PRIORITY fast rule,
        before any RAG lookup, entity validation, or LLM call.
        """
        txt = text.lower().strip()
        txt_clean = re.sub(r'[^a-zA-Z0-9\s]', '', txt).strip()
        from src.services.cache_service import cache_service
        from src.services.rag_service import rag_service
        from src.services.language_service import language_service

        active_doc_id = rag_service._resolve_active_document_id(organization_id)
        org_name = rag_service.get_organization_name(organization_id, document_id=active_doc_id)
        welcome_greeting = rag_service.get_organization_greeting(organization_id, document_id=active_doc_id)

        # ------------------------------------------------------------------ #
        # PRIORITY 0 — Language Switch (must fire before any RAG/LLM call)    #
        # ------------------------------------------------------------------ #
        target_lang, is_switch = language_service.check_explicit_language_switch(text)
        if is_switch:
            return language_service.get_language_switch_confirmation(
                target_lang, booking_session=booking_session, org_name=org_name
            )

        # Multilingual Protection: When active conversation language is not English,
        # bypass hardcoded English pleasantries so Gemini or localized handlers respond in target language.
        curr_lang = conversation_language if conversation_language in ("ta", "hi", "kn", "te") else "en"
        if curr_lang != "en":
            greetings_native = {
                "ta": ["வணக்கம்", "வணக்கமுங்க", "வணக்கம்ங்க", "vanakkam", "vanakam", "hello", "hi"],
                "hi": ["नमस्ते", "नमस्कार", "namaste", "namaskar", "hello", "hi"],
                "kn": ["ನಮಸ್ಕಾರ", "namaskara", "namaste", "hello", "hi"],
                "te": ["నమస్కారం", "నమస్తే", "namaskaram", "namaste", "hello", "hi"],
            }
            if any(g in txt for g in greetings_native.get(curr_lang, [])):
                if curr_lang == "ta":
                    return f"வணக்கம்! நான் {org_name or 'மருத்துவமனை'} AI வரவேற்பாளர். உங்களுக்கு இன்று எவ்வாறு உதவலாம்?"
                elif curr_lang == "hi":
                    return f"नमस्ते! मैं {org_name or 'अस्पताल'} का AI स्वागतकर्ता हूँ। आज मैं आपकी क्या सहायता कर सकता हूँ?"
                elif curr_lang == "kn":
                    return f"ನಮಸ್ಕಾರ! ನಾನು {org_name or 'ಆಸ್ಪತ್ರೆ'} AI ಸ್ವಾಗತಕಾರ. ಇಂದು ನಾನು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಲಿ?"
                elif curr_lang == "te":
                    return f"నమస్కారం! నేను {org_name or 'ఆసుపత్రి'} AI రిసెప్షనిస్ట్‌ని. ఈరోజు నేను మీకు ఎలా సహాయపడగలను?"
            # For all other statements in non-English, pass to LLM to generate fluent in-language response
            return None'''

assert old_def in text, "old_def not found in ai_service.py"
text = text.replace(old_def, new_def, 1)

# 2. Update call in generate_response
old_call1 = '''        fast_resp = self._fast_rule_response(
            text, intent, organization_id=organization_id, is_ongoing=is_ongoing,
            booking_session=conversation_state, has_context=bool(clean_context),
            call_direction=call_direction
        )'''

new_call1 = '''        fast_resp = self._fast_rule_response(
            text, intent, organization_id=organization_id, is_ongoing=is_ongoing,
            booking_session=conversation_state, has_context=bool(clean_context),
            call_direction=call_direction, conversation_language=conversation_language
        )'''

assert old_call1 in text, "old_call1 not found in ai_service.py"
text = text.replace(old_call1, new_call1, 1)

# 3. Update call in generate_response_stream
assert old_call1 in text, "old_call1 for stream not found in ai_service.py"
text = text.replace(old_call1, new_call1, 1)

with open('src/services/ai_service.py', 'w', encoding='utf-8', newline='') as f:
    f.write(text)

print("SUCCESSFULLY PATCHED ai_service.py")

