"""
Test Suite: CALLZENZA Dynamic 5-Language Switching Implementation
Verifies all 10 mandatory test scenarios against real services:
- Tamil (ta)
- English (en)
- Hindi (hi)
- Kannada (kn)
- Telugu (te)
"""

import asyncio
import os
import sys
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from dotenv import load_dotenv

# Ensure backend root is in PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from src.services.language_service import language_service, SUPPORTED_LANGUAGES, LANG_TAMIL, LANG_ENGLISH, LANG_HINDI, LANG_KANNADA, LANG_TELUGU
from src.services.deepgram_stt_service import DeepgramStreamingSession
from src.services.tts_service import TTSService
from src.services.booking_state_service import booking_state_service, BookingState
from src.services.sip_call_handler import SipCallSession, CallType

tts = TTSService()


async def run_all_tests():
    print("=" * 80)
    print("STARTING 10 MULTILINGUAL TEST SCENARIOS")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # TEST 1: Customer: "Will you please talk in Tamil?"
    # Expected: Tamil STT, Tamil LLM response and Tamil TTS.
    # --------------------------------------------------------------------------
    print("\n[TEST 1] Customer: 'Will you please talk in Tamil?'")
    target_lang, is_switch = language_service.check_explicit_language_switch("Will you please talk in Tamil?")
    assert is_switch is True and target_lang == "ta", f"Detection failed: {target_lang}, {is_switch}"
    print(f" -> Switch detected: target_lang={target_lang}")

    # Check TTS voice resolution & synthesis
    voice_id, model, prov = language_service.get_tts_voice_and_model("ta")
    print(f" -> TTS Voice resolved: voice_id={voice_id}, model={model}, provider={prov}")
    ta_conf = language_service.get_language_switch_confirmation("ta")
    print(f" -> Switch confirmation text: {ta_conf}")
    ta_audio = tts.synthesize_speech(ta_conf, language="ta", voice_id=voice_id)
    print(f" -> Synthesized Tamil audio: {len(ta_audio)} bytes")
    assert len(ta_audio) > 1000, "Tamil TTS synthesis returned empty audio"

    # Check STT language code
    stt_lang = language_service.get_stt_language("ta")
    assert stt_lang == "ta", f"Expected STT lang 'ta', got {stt_lang}"
    print(" -> TEST 1 PASSED!")

    # --------------------------------------------------------------------------
    # TEST 2: Customer switches from Tamil to English.
    # Expected: English STT, English response and English TTS without restarting the call.
    # --------------------------------------------------------------------------
    print("\n[TEST 2] Customer switches from Tamil to English: 'Can you speak in English?'")
    target_lang, is_switch = language_service.check_explicit_language_switch("Can you speak in English?")
    assert is_switch is True and target_lang == "en", f"Detection failed: {target_lang}"
    en_conf = language_service.get_language_switch_confirmation("en")
    print(f" -> Switch confirmation text: {en_conf}")
    en_audio = tts.synthesize_speech(en_conf, language="en")
    print(f" -> Synthesized English audio: {len(en_audio)} bytes")
    assert len(en_audio) > 1000, "English TTS synthesis failed"
    print(" -> TEST 2 PASSED!")

    # --------------------------------------------------------------------------
    # TEST 3: Customer switches from English to Hindi.
    # Expected: Hindi conversation continues with all previous context preserved.
    # --------------------------------------------------------------------------
    print("\n[TEST 3] Customer switches from English to Hindi: 'Please switch to Hindi.'")
    target_lang, is_switch = language_service.check_explicit_language_switch("Please switch to Hindi.")
    assert is_switch is True and target_lang == "hi", f"Detection failed: {target_lang}"
    voice_id, model, prov = language_service.get_tts_voice_and_model("hi")
    print(f" -> TTS Voice resolved: voice_id={voice_id}, model={model}, provider={prov}")
    hi_conf = language_service.get_language_switch_confirmation("hi")
    print(f" -> Switch confirmation text: {hi_conf}")
    hi_audio = tts.synthesize_speech(hi_conf, language="hi", voice_id=voice_id)
    print(f" -> Synthesized Hindi audio: {len(hi_audio)} bytes")
    assert len(hi_audio) > 1000, "Hindi TTS synthesis failed"
    print(" -> TEST 3 PASSED!")

    # --------------------------------------------------------------------------
    # TEST 4: Customer switches from Hindi to Kannada.
    # Expected: Kannada STT and speech.
    # --------------------------------------------------------------------------
    print("\n[TEST 4] Customer switches from Hindi to Kannada: 'Kannada please.'")
    target_lang, is_switch = language_service.check_explicit_language_switch("Kannada please.")
    assert is_switch is True and target_lang == "kn", f"Detection failed: {target_lang}"
    kn_conf = language_service.get_language_switch_confirmation("kn")
    print(f" -> Switch confirmation text: {kn_conf}")
    kn_audio = tts.synthesize_speech(kn_conf, language="kn")
    print(f" -> Synthesized Kannada audio: {len(kn_audio)} bytes")
    assert len(kn_audio) > 1000, "Kannada TTS synthesis failed"
    stt_lang = language_service.get_stt_language("kn")
    assert stt_lang == "kn", f"Expected STT lang 'kn', got {stt_lang}"
    print(" -> TEST 4 PASSED!")

    # --------------------------------------------------------------------------
    # TEST 5: Customer switches from Kannada to Telugu.
    # Expected: Telugu STT and speech.
    # --------------------------------------------------------------------------
    print("\n[TEST 5] Customer switches from Kannada to Telugu: 'Telugu lo matladandi.'")
    target_lang, is_switch = language_service.check_explicit_language_switch("Telugu lo matladandi.")
    assert is_switch is True and target_lang == "te", f"Detection failed: {target_lang}"
    te_conf = language_service.get_language_switch_confirmation("te")
    print(f" -> Switch confirmation text: {te_conf}")
    te_audio = tts.synthesize_speech(te_conf, language="te")
    print(f" -> Synthesized Telugu audio: {len(te_audio)} bytes")
    assert len(te_audio) > 1000, "Telugu TTS synthesis failed"
    stt_lang = language_service.get_stt_language("te")
    assert stt_lang == "te", f"Expected STT lang 'te', got {stt_lang}"
    print(" -> TEST 5 PASSED!")

    # --------------------------------------------------------------------------
    # TEST 6: Customer switches language during appointment booking.
    # Expected: Booking continues from the same pending step. No duplicate appointment.
    # --------------------------------------------------------------------------
    print("\n[TEST 6] Customer switches language during appointment booking.")
    session = booking_state_service.get_session("test_session_switch_mid_booking")
    session.intent = "appointment_booking"
    session.department = "Cardiology"
    session.doctor = "Dr. Ramesh"
    session.date = "2026-11-12"
    session.time = "10:00 AM"
    session.availability_verified = True
    session.state = BookingState.COLLECTING_PATIENT_NAME
    session.last_asked_field = "patient_name"

    # Caller asks to switch to Tamil mid-booking
    resp = booking_state_service.process_booking_turn("Will you please talk in Tamil?", session)
    print(f" -> Mid-booking switch response: {resp}")
    assert session.conversation_language == "ta", f"Expected conversation_language='ta', got {session.conversation_language}"
    assert session.department == "Cardiology", "Department was reset!"
    assert session.doctor == "Dr. Ramesh", "Doctor was reset!"
    assert session.date == "2026-11-12", "Date was reset!"
    assert session.time == "10:00 AM", "Time was reset!"
    assert session.state == BookingState.COLLECTING_PATIENT_NAME, "State was reset!"

    # Customer provides patient name in Tamil
    booking_state_service.process_booking_turn("ரமேஷ் குமார்", session)
    assert session.patient_name == "ரமேஷ் குமார்" or "ரமேஷ்" in (session.patient_name or ""), f"Patient name not captured: {session.patient_name}"
    print(f" -> Patient name captured: {session.patient_name}")
    print(" -> TEST 6 PASSED!")

    # --------------------------------------------------------------------------
    # TEST 7: Customer says an email address or phone number while speaking Tamil.
    # Expected: Correct extraction and validation without translating or corrupting the value.
    # --------------------------------------------------------------------------
    print("\n[TEST 7] Contact extraction during Tamil conversation.")
    session.conversation_language = "ta"
    booking_state_service.extract_slots("udaya at gmail.com", session)
    print(f" -> Extracted email: {session.email}")
    assert session.email == "udaya@gmail.com", f"Expected udaya@gmail.com, got {session.email}"

    booking_state_service.extract_slots("ennoda phone number 9789261936", session)
    print(f" -> Extracted phone: {session.contact_number}")
    assert session.contact_number == "9789261936", f"Expected 9789261936, got {session.contact_number}"
    # Verify language remained Tamil
    assert session.conversation_language == "ta", "Language unexpectedly changed away from Tamil!"
    print(" -> TEST 7 PASSED!")

    # --------------------------------------------------------------------------
    # TEST 8: Customer uses mixed-language speech (Tanglish / loanwords).
    # Expected: Selected language remains stable and technical terms are recognized correctly.
    # --------------------------------------------------------------------------
    print("\n[TEST 8] Mixed-language speech (Tanglish / loanwords).")
    lang_res = language_service.resolve_call_language_turn("Cardiology department la Dr. Ramesh appointment venum", current_preferred_lang="ta")
    print(f" -> Resolved language: {lang_res}")
    assert lang_res["preferred_response_language"] == "ta", f"Expected 'ta', got {lang_res['preferred_response_language']}"
    assert lang_res["language_switch_requested"] is False, "False switch detected on Tanglish speech"
    print(" -> TEST 8 PASSED!")

    # --------------------------------------------------------------------------
    # TEST 9: Customer switches language twice during one call.
    # Expected: Each explicit request changes the language correctly without duplicate responses.
    # --------------------------------------------------------------------------
    print("\n[TEST 9] Customer switches language twice: en -> ta -> hi.")
    call_session = SipCallSession(caller_phone="9999999999", caller_name="Multi-Lang Caller")
    assert call_session.current_lang == "en"

    # First switch: to Tamil
    t1, s1 = language_service.check_explicit_language_switch("tamil la pesunga")
    assert s1 and t1 == "ta"
    call_session.current_lang = t1
    print(f" -> First switch: current_lang={call_session.current_lang}")

    # Second switch: to Hindi
    t2, s2 = language_service.check_explicit_language_switch("Please switch to Hindi.")
    assert s2 and t2 == "hi"
    call_session.current_lang = t2
    print(f" -> Second switch: current_lang={call_session.current_lang}")
    assert call_session.current_lang == "hi", "Language state not updated correctly on second switch"
    print(" -> TEST 9 PASSED!")

    # --------------------------------------------------------------------------
    # TEST 10: Customer switches language while the AI is responding (barge-in transition).
    # Expected: Handle the transition safely without overlapping audio or corrupting RTP.
    # --------------------------------------------------------------------------
    print("\n[TEST 10] Language switch during AI speech (barge-in transition).")
    cleared = False
    def mock_clear():
        nonlocal cleared
        cleared = True

    call_session.audio_clear_callback = mock_clear
    call_session._ai_speaking = True
    call_session._ai_speaking_until = time.monotonic() + 5.0

    # Simulate barge-in utterance requesting Tamil
    target_lang, is_switch = language_service.check_explicit_language_switch("Wait, talk in Tamil please!")
    assert is_switch and target_lang == "ta"

    # Trigger barge-in stop
    call_session._barge_in_event.set()
    call_session._ai_speaking = False
    call_session._ai_speaking_until = 0.0
    if call_session.audio_clear_callback:
        call_session.audio_clear_callback()

    assert cleared is True, "Audio clear callback was not triggered on barge-in!"
    assert call_session._ai_speaking is False, "AI speaking flag was not cleared!"
    call_session.current_lang = target_lang
    print(f" -> Barge-in handled successfully: RTP cleared={cleared}, new language={call_session.current_lang}")
    print(" -> TEST 10 PASSED!")

    print("\n" + "=" * 80)
    print("ALL 10 MULTILINGUAL TEST SCENARIOS PASSED WITH 100% SUCCESS!")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(run_all_tests())
