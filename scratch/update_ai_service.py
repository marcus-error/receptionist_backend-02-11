import os

path = r'receptionist_backend-02-11\src\services\ai_service.py'
with open(path, 'r', encoding='utf-8') as f:
    c = f.read()

# 1. Update rule 4
old_rule4 = '"4. STRICT ENGLISH-ONLY CONVERSATION ENFORCEMENT:\\n"'
new_rule4 = '"4. DYNAMIC CONVERSATION LANGUAGE — MULTILINGUAL RECEPTIONIST:\\n"'
assert old_rule4 in c, 'old_rule4 not found'
c = c.replace(old_rule4, new_rule4)

old_r4_body = '''            "   - You MUST formulate all speech and responses strictly in English. NEVER respond in Spanish, Hindi, Tamil, Telugu, Malayalam, Kannada, or any language other than English.\\n"
            "   - If the caller speaks another language or asks to switch languages (e.g. Spanish, Hindi, Tamil, etc.), DO NOT translate and DO NOT switch languages. Politely respond in English: 'I can only assist you in English. Please could you continue in English?'\\n"
            "   - All speech recognition, entity extraction, appointment handling, and dialogue responses must remain 100% in English.\\n\\n"'''
new_r4_body = '''            "   - You MUST formulate all speech and responses strictly in the requested conversation language.\\n"
            "   - Supported languages are strictly: Tamil (ta), English (en), Hindi (hi), Kannada (kn), Telugu (te).\\n"
            "   - If the caller requests another supported language, switch immediately and naturally.\\n"
            "   - Preserve exact spelling of proper nouns, patient names, email addresses, phone numbers, and appointment dates/times without translating or corrupting them.\\n\\n"'''

if old_r4_body.replace('\n', '\r\n') in c:
    c = c.replace(old_r4_body.replace('\n', '\r\n'), new_r4_body.replace('\n', '\r\n'))
else:
    c = c.replace(old_r4_body, new_r4_body)

# 2. Update _fast_rule_response
old_fast = '''        target_lang, is_switch = language_service.check_explicit_language_switch(text)
        if is_switch and target_lang:
            return language_service.get_language_switch_confirmation(
                target_lang, booking_session=booking_session
            )'''
new_fast = '''        target_lang, is_switch = language_service.check_explicit_language_switch(text)
        if is_switch:
            return language_service.get_language_switch_confirmation(
                target_lang, booking_session=booking_session, org_name=org_name
            )'''
if old_fast.replace('\n', '\r\n') in c:
    c = c.replace(old_fast.replace('\n', '\r\n'), new_fast.replace('\n', '\r\n'))
else:
    c = c.replace(old_fast, new_fast)

# 3. Update call to _get_system_instruction in generate_response
old_sys_call = 'system_prompt = self._get_system_instruction(organization_name=org_name, is_ongoing=is_ongoing, is_rag_active=is_rag_active, emotion=emotion, call_direction=call_direction)'
new_sys_call = 'system_prompt = self._get_system_instruction(organization_name=org_name, is_ongoing=is_ongoing, is_rag_active=is_rag_active, emotion=emotion, call_direction=call_direction, conversation_language=conversation_language)'
assert old_sys_call in c, 'old_sys_call not found'
c = c.replace(old_sys_call, new_sys_call)

# 4. Replace strict english-only directive in generate_response
old_directive = '''        # Strict English-Only Directive
        system_prompt += (
            "\\n=== STRICT ENGLISH-ONLY CONVERSATION DIRECTIVE ===\\n"
            "You are an English-only AI receptionist. You MUST generate all responses strictly in English.\\n"
            "- NEVER respond in Spanish, Hindi, Tamil, Telugu, Malayalam, Kannada, or any language other than English.\\n"
            "- If the caller speaks or inquires in another language, or asks you to switch languages, DO NOT translate and DO NOT switch languages. Politely reply in English: 'I can only assist you in English. Please could you continue in English?'\\n"
            "- Maintain professional, fluent, and natural English at all times.\\n"
        )'''

new_directive = '''        # Dynamic Language Directive
        from src.services.language_service import language_service, SUPPORTED_LANGUAGES
        req_lang = conversation_language if conversation_language in SUPPORTED_LANGUAGES else "en"
        lang_name = language_service.get_language_name(req_lang)
        system_prompt += (
            f"\\n=== CONVERSATION LANGUAGE DIRECTIVE: {lang_name.upper()} ({req_lang}) ===\\n"
            f"You are speaking with the caller in {lang_name} ({req_lang}).\\n"
            f"- You MUST formulate your response strictly, entirely, and fluently in {lang_name} ({req_lang}).\\n"
            f"- Do NOT translate or corrupt caller names, doctor names, email addresses, phone numbers, or appointment dates.\\n"
            f"- Maintain professional, fluent, and natural {lang_name} dialogue at all times.\\n"
        )'''

if old_directive.replace('\n', '\r\n') in c:
    c = c.replace(old_directive.replace('\n', '\r\n'), new_directive.replace('\n', '\r\n'))
else:
    c = c.replace(old_directive, new_directive)

with open(path, 'w', encoding='utf-8') as f:
    f.write(c)

print('ai_service.py updated successfully!')

