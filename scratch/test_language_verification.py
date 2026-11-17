import sys, os

# Ensure backend path
backend_path = os.path.abspath(r"C:\projects\2-11 ai receptionist updated\receptionist_backend-02-11")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from src.services.language_service import language_service, SUPPORTED_LANGUAGES, TTS_VOICE_CONFIG

print("--- TESTING ALL 5 LANGUAGES DETECTION & VOICES ---")

test_phrases = [
    ("Will you please talk in Tamil?", "ta", "ta-IN-iniya", "murf"),
    ("Can you speak in Hindi?", "hi", "hi-IN-shweta", "murf"),
    ("Please speak in Kannada.", "kn", "kn", "gtts"),
    ("Can you talk in Telugu?", "te", "te", "gtts"),
    ("Please continue in English.", "en", "en-IN-isha", "murf"),
]

for phrase, exp_lang, exp_voice, exp_prov in test_phrases:
    target_lang, is_switch, reason = language_service.check_language_switch_detailed(phrase)
    v_id, model, prov = language_service.get_tts_voice_and_model(target_lang)
    conf = language_service.get_language_switch_confirmation(target_lang)
    print(f"\n[TEST] Phrase: '{phrase}'")
    print(f"  Detected: {target_lang} (is_switch={is_switch}, reason={reason})")
    print(f"  TTS: voice={v_id}, provider={prov}")
    print(f"  Confirmation: '{conf}'")
    assert is_switch, f"Expected switch for '{phrase}'"
    assert target_lang == exp_lang, f"Expected {exp_lang}, got {target_lang}"
    assert v_id == exp_voice, f"Expected voice {exp_voice}, got {v_id}"
    assert prov == exp_prov, f"Expected provider {exp_prov}, got {prov}"

print("\n--- TESTING LANGUAGE PERSISTENCE ACROSS TURNS ---")
# Customer switched to Tamil. In subsequent turns, customer speaks English numbers or transliterated Tamil
turns = [
    "9092649453",
    "hospital address",
    "dermatology department",
    "tomorrow 10:30 am",
    "sanjay kumar"
]
current_preferred = "ta"
for turn in turns:
    res = language_service.resolve_call_language_turn(turn, current_preferred_lang=current_preferred)
    print(f"[TURN PERSISTENCE] Input: '{turn}' -> preferred_response_language: {res['preferred_response_language']} (switch_requested={res['language_switch_requested']})")
    assert res['preferred_response_language'] == "ta", f"Language reverted for '{turn}'!"
    assert not res['language_switch_requested'], f"Unexpected switch for '{turn}'"

print("\n--- TESTING NON-SWITCH UTTERANCES ---")
non_switches = [
    "I have a friend in Tamil Nadu",
    "Where is the hospital located?",
    "Book an appointment for tomorrow",
    "Can you help me with a doctor consultation?"
]
for ns in non_switches:
    target_lang, is_switch, reason = language_service.check_language_switch_detailed(ns)
    print(f"[NON-SWITCH] '{ns}' -> is_switch={is_switch}")
    assert not is_switch, f"False positive switch for '{ns}'"

print("\nALL VERIFICATIONS PASSED 100% SUCCESSFULLY!")

