import sys, os, time, asyncio
from unittest.mock import AsyncMock

sys.path.insert(0, os.path.abspath(r"C:\projects\2-11 ai receptionist updated\receptionist_backend-02-11"))
from src.services.language_service import language_service
from src.services.deepgram_stt_service import DeepgramStreamingSession
from src.services.booking_state_service import BookingSession

async def mock_coro():
    pass

async def test_scenarios():
    print("Testing Scenario 1 (Tamil)...", flush=True)
    phrase = "Will you please talk in Tamil?"
    target_lang, is_switch, switch_reason = language_service.check_language_switch_detailed(phrase)
    assert is_switch and target_lang == "ta", f"Failed ta switch: {target_lang}"
    v_id, model, prov = language_service.get_tts_voice_and_model("ta")
    assert v_id == "ta-IN-iniya" and prov == "murf", f"Failed ta voice: {v_id}, {prov}"
    conf = language_service.get_language_switch_confirmation("ta")
    assert "வணக்கம்" in conf, f"Failed ta conf: {conf}"
    
    stt = DeepgramStreamingSession(api_key="dummy_key", call_id="test_ta_call", language="en")
    stt._running = True
    stt._connected = True
    stt._connect_ws = AsyncMock(return_value=True)
    stt._send_loop = mock_coro
    stt._recv_loop = mock_coro
    res = await stt.switch_language("ta")
    assert res and stt.language == "ta", "Failed ta stt switch"
    print("Scenario 1 PASSED!", flush=True)

    print("Testing Scenario 2 (Hindi)...", flush=True)
    phrase = "Can you speak in Hindi?"
    target_lang, is_switch, switch_reason = language_service.check_language_switch_detailed(phrase)
    assert is_switch and target_lang == "hi", f"Failed hi switch: {target_lang}"
    v_id, model, prov = language_service.get_tts_voice_and_model("hi")
    assert v_id == "hi-IN-shweta" and prov == "murf", f"Failed hi voice: {v_id}, {prov}"
    conf = language_service.get_language_switch_confirmation("hi")
    assert "नमस्ते" in conf, f"Failed hi conf: {conf}"
    res = await stt.switch_language("hi")
    assert res and stt.language == "hi", "Failed hi stt switch"
    print("Scenario 2 PASSED!", flush=True)

    print("Testing Scenario 3 (Kannada)...", flush=True)
    phrase = "Please speak in Kannada."
    target_lang, is_switch, switch_reason = language_service.check_language_switch_detailed(phrase)
    assert is_switch and target_lang == "kn", f"Failed kn switch: {target_lang}"
    v_id, model, prov = language_service.get_tts_voice_and_model("kn")
    assert v_id == "kn" and prov == "gtts", f"Failed kn voice: {v_id}, {prov}"
    conf = language_service.get_language_switch_confirmation("kn")
    assert "ನಮಸ್ಕಾರ" in conf, f"Failed kn conf: {conf}"
    res = await stt.switch_language("kn")
    assert res and stt.language == "kn", "Failed kn stt switch"
    print("Scenario 3 PASSED!", flush=True)

    print("Testing Scenario 4 (Telugu)...", flush=True)
    phrase = "Can you talk in Telugu?"
    target_lang, is_switch, switch_reason = language_service.check_language_switch_detailed(phrase)
    assert is_switch and target_lang == "te", f"Failed te switch: {target_lang}"
    v_id, model, prov = language_service.get_tts_voice_and_model("te")
    assert v_id == "te" and prov == "gtts", f"Failed te voice: {v_id}, {prov}"
    conf = language_service.get_language_switch_confirmation("te")
    assert "నమస్కారం" in conf, f"Failed te conf: {conf}"
    res = await stt.switch_language("te")
    assert res and stt.language == "te", "Failed te stt switch"
    print("Scenario 4 PASSED!", flush=True)

    print("Testing Scenario 5 (English)...", flush=True)
    phrase = "Please continue in English."
    target_lang, is_switch, switch_reason = language_service.check_language_switch_detailed(phrase)
    assert is_switch and target_lang == "en", f"Failed en switch: {target_lang}"
    v_id, model, prov = language_service.get_tts_voice_and_model("en")
    assert v_id == "en-IN-isha" and prov == "murf", f"Failed en voice: {v_id}, {prov}"
    conf = language_service.get_language_switch_confirmation("en")
    assert "English" in conf, f"Failed en conf: {conf}"
    res = await stt.switch_language("en")
    assert res and stt.language == "en", "Failed en stt switch"
    print("Scenario 5 PASSED!", flush=True)

    print("Testing Scenario 6 (Multiple switches in 1 call: en -> ta -> hi -> ta -> en)...", flush=True)
    for phrase, expected_lang, expected_voice in [
        ("Will you please talk in Tamil?", "ta", "ta-IN-iniya"),
        ("Can you speak in Hindi?", "hi", "hi-IN-shweta"),
        ("Please speak in Tamil again", "ta", "ta-IN-iniya"),
        ("Please continue in English.", "en", "en-IN-isha"),
    ]:
        tgt_lang, is_sw, _ = language_service.check_language_switch_detailed(phrase)
        assert is_sw and tgt_lang == expected_lang
        res = await stt.switch_language(tgt_lang)
        assert res and stt.language == expected_lang
        v_id, _, _ = language_service.get_tts_voice_and_model(tgt_lang)
        assert v_id == expected_voice
    print("Scenario 6 PASSED!", flush=True)

    print("Testing Scenario 7 (Language persistence on transliterated speech)...", flush=True)
    for utt in ["9092649453", "hospital address", "appointment doctor", "yes tomorrow 10 am"]:
        res_turn = language_service.resolve_call_language_turn(utt, current_preferred_lang="ta")
        assert res_turn["preferred_response_language"] == "ta", f"Failed persistence for '{utt}': {res_turn}"
        assert not res_turn["language_switch_requested"]
    print("Scenario 7 PASSED!", flush=True)

    print("Testing Scenario 8 (Business workflow preservation in Tamil)...", flush=True)
    session = BookingSession()
    session.conversation_language = "ta"
    assert session.conversation_language == "ta"
    conf_booking = language_service.get_language_switch_confirmation("ta", booking_session=session)
    assert len(conf_booking) > 0 and "வணக்கம்" in conf_booking
    print("Scenario 8 PASSED!", flush=True)

    print("\nALL 8 SCENARIOS COMPLETED SUCCESSFULLY WITH 100% PASS RATE!", flush=True)

asyncio.run(test_scenarios())

