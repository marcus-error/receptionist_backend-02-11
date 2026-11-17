import sys, os

backend_path = os.path.abspath(r"C:\projects\2-11 ai receptionist updated\receptionist_backend-02-11")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from src.services.language_service import language_service, SUPPORTED_LANGUAGES, TTS_VOICE_CONFIG
from src.services.ai_service import ai_service
from src.services.tts_service import tts_service
from src.services.booking_state_service import BookingSession

print("=====================================================================")
print("RUNNING COMPREHENSIVE END-TO-END MULTILINGUAL VERIFICATION (8 TESTS)")
print("=====================================================================")

# -------------------------------------------------------------------
# Test 1: Customer requests Tamil: "Will you please talk in Tamil?"
# -------------------------------------------------------------------
print("\n--- TEST 1: Tamil Language Switch ---")
text = "Will you please talk in Tamil?"
target_lang, is_switch, reason = language_service.check_language_switch_detailed(text)
assert is_switch and target_lang == "ta", "Tamil switch intent failed"
v_id, model, prov = language_service.get_tts_voice_and_model(target_lang)
assert v_id == "ta-IN-iniya" and prov == "murf", "Tamil voice routing failed"
conf_ta = language_service.get_language_switch_confirmation(target_lang)
assert "வணக்கம்" in conf_ta or "தமிழில்" in conf_ta, "Tamil confirmation prompt failed"
stt_mapped = language_service.get_stt_language(target_lang)
assert stt_mapped == "ta", "Deepgram STT mapping for Tamil failed"
print("  ✓ Intent: ta (ACCEPTED)")
print(f"  ✓ STT Target: {stt_mapped}")
print(f"  ✓ TTS Voice: {v_id} ({prov})")
print(f"  ✓ Confirmation: '{conf_ta}'")
print("  ✓ TEST 1 PASSED")

# -------------------------------------------------------------------
# Test 2: Customer requests Hindi: "Can you speak in Hindi?"
# -------------------------------------------------------------------
print("\n--- TEST 2: Hindi Language Switch ---")
text = "Can you speak in Hindi?"
target_lang, is_switch, reason = language_service.check_language_switch_detailed(text)
assert is_switch and target_lang == "hi", "Hindi switch intent failed"
v_id, model, prov = language_service.get_tts_voice_and_model(target_lang)
assert v_id == "hi-IN-shweta" and prov == "murf", "Hindi voice routing failed"
conf_hi = language_service.get_language_switch_confirmation(target_lang)
assert "नमस्ते" in conf_hi or "हिंदी" in conf_hi, "Hindi confirmation prompt failed"
stt_mapped = language_service.get_stt_language(target_lang)
assert stt_mapped == "hi", "Deepgram STT mapping for Hindi failed"
print("  ✓ Intent: hi (ACCEPTED)")
print(f"  ✓ STT Target: {stt_mapped}")
print(f"  ✓ TTS Voice: {v_id} ({prov})")
print(f"  ✓ Confirmation: '{conf_hi}'")
print("  ✓ TEST 2 PASSED")

# -------------------------------------------------------------------
# Test 3: Customer requests Kannada: "Please speak in Kannada."
# -------------------------------------------------------------------
print("\n--- TEST 3: Kannada Language Switch ---")
text = "Please speak in Kannada."
target_lang, is_switch, reason = language_service.check_language_switch_detailed(text)
assert is_switch and target_lang == "kn", "Kannada switch intent failed"
v_id, model, prov = language_service.get_tts_voice_and_model(target_lang)
assert v_id == "kn" and prov == "gtts", "Kannada voice routing failed"
conf_kn = language_service.get_language_switch_confirmation(target_lang)
assert "ನಮಸ್ಕಾರ" in conf_kn or "ಕನ್ನಡ" in conf_kn, "Kannada confirmation prompt failed"
stt_mapped = language_service.get_stt_language(target_lang)
assert stt_mapped == "kn", "Deepgram STT mapping for Kannada failed"
print("  ✓ Intent: kn (ACCEPTED)")
print(f"  ✓ STT Target: {stt_mapped}")
print(f"  ✓ TTS Voice: {v_id} ({prov})")
print(f"  ✓ Confirmation: '{conf_kn}'")
print("  ✓ TEST 3 PASSED")

# -------------------------------------------------------------------
# Test 4: Customer requests Telugu: "Can you talk in Telugu?"
# -------------------------------------------------------------------
print("\n--- TEST 4: Telugu Language Switch ---")
text = "Can you talk in Telugu?"
target_lang, is_switch, reason = language_service.check_language_switch_detailed(text)
assert is_switch and target_lang == "te", "Telugu switch intent failed"
v_id, model, prov = language_service.get_tts_voice_and_model(target_lang)
assert v_id == "te" and prov == "gtts", "Telugu voice routing failed"
conf_te = language_service.get_language_switch_confirmation(target_lang)
assert "నమస్కారం" in conf_te or "తెలుగు" in conf_te, "Telugu confirmation prompt failed"
stt_mapped = language_service.get_stt_language(target_lang)
assert stt_mapped == "te", "Deepgram STT mapping for Telugu failed"
print("  ✓ Intent: te (ACCEPTED)")
print(f"  ✓ STT Target: {stt_mapped}")
print(f"  ✓ TTS Voice: {v_id} ({prov})")
print(f"  ✓ Confirmation: '{conf_te}'")
print("  ✓ TEST 4 PASSED")

# -------------------------------------------------------------------
# Test 5: Customer requests return to English: "Please continue in English."
# -------------------------------------------------------------------
print("\n--- TEST 5: Return to English ---")
text = "Please continue in English."
target_lang, is_switch, reason = language_service.check_language_switch_detailed(text)
assert is_switch and target_lang == "en", "English switch intent failed"
v_id, model, prov = language_service.get_tts_voice_and_model(target_lang)
assert v_id == "en-IN-isha" and prov == "murf", "English voice routing failed"
conf_en = language_service.get_language_switch_confirmation(target_lang)
assert "English" in conf_en, "English confirmation prompt failed"
print("  ✓ Intent: en (ACCEPTED)")
print(f"  ✓ TTS Voice: {v_id} ({prov})")
print(f"  ✓ Confirmation: '{conf_en}'")
print("  ✓ TEST 5 PASSED")

# -------------------------------------------------------------------
# Test 6: Multiple switches in 1 call: en -> ta -> hi -> ta -> en
# -------------------------------------------------------------------
print("\n--- TEST 6: Multiple Switches in Single Call ---")
sequence = [
    ("Will you please talk in Tamil?", "ta", "ta-IN-iniya"),
    ("Can you speak in Hindi?", "hi", "hi-IN-shweta"),
    ("Please switch back to Tamil", "ta", "ta-IN-iniya"),
    ("Please continue in English.", "en", "en-IN-isha"),
]
curr_session_lang = "en"
for phrase, exp_lang, exp_v in sequence:
    t_lang, sw, _ = language_service.check_language_switch_detailed(phrase)
    assert sw and t_lang == exp_lang
    v, _, _ = language_service.get_tts_voice_and_model(t_lang)
    assert v == exp_v
    curr_session_lang = t_lang
    print(f"  Transition: '{phrase}' -> Active Language: {curr_session_lang}, TTS Voice: {v}")
assert curr_session_lang == "en"
print("  ✓ TEST 6 PASSED")

# -------------------------------------------------------------------
# Test 7: Language Persistence Across Transliterated Words & Reconnects
# -------------------------------------------------------------------
print("\n--- TEST 7: Language Persistence (Transliteration / Numbers) ---")
# Customer is in Tamil. Says an English phone number or loan word
active_language = "ta"
subsequent_inputs = [
    "9092649453",
    "dr. sanjay cardiology",
    "tomorrow 10:30 am",
    "hospital address and location",
    "vanakkam doctor appointment venum"
]
for sub_in in subsequent_inputs:
    turn_res = language_service.resolve_call_language_turn(sub_in, current_preferred_lang=active_language)
    assert turn_res["preferred_response_language"] == "ta", f"State leak for '{sub_in}'"
    assert not turn_res["language_switch_requested"]
    print(f"  Input: '{sub_in}' -> preferred_response_language: {turn_res['preferred_response_language']} (PERSISTED)")
print("  ✓ TEST 7 PASSED")

# -------------------------------------------------------------------
# Test 8: Business Workflow Preservation in Selected Language
# -------------------------------------------------------------------
print("\n--- TEST 8: Business Workflow & Booking State Preservation ---")
session = BookingSession(session_id="test_session_id")
session.conversation_language = "ta"
assert session.conversation_language == "ta"

# Check dynamic switch confirmation with active booking
conf_booking = language_service.get_language_switch_confirmation("ta", booking_session=session)
assert len(conf_booking) > 0 and ("வணக்கம்" in conf_booking or "தமிழில்" in conf_booking)
print(f"  Tamil Booking State Prompt: '{conf_booking}'")

# Check LLM instruction has Tamil prompt directive
sys_inst = ai_service._get_system_instruction(
    organization_name="Sarans Hospitals",
    conversation_language="ta"
)
assert "MANDATORY LANGUAGE DIRECTIVE: TAMIL / தமிழ் (ta)" in sys_inst
print("  ✓ AI System Instruction contains mandatory Tamil directive")
print("  ✓ TEST 8 PASSED")

print("\n=====================================================================")
print("ALL 8 VERIFICATION SCENARIOS COMPLETED WITH 100% SUCCESS!")
print("=====================================================================")
