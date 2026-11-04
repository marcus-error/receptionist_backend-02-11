import pytest
import io
import wave
import asyncio
from unittest.mock import MagicMock
from src.services.sip_call_handler import SipCallSession, convert_tts_to_pcmu_and_pcm16
from src.services.rag_service import rag_service
from src.config.database import SessionLocal
from src.models.knowledge_doc import KnowledgeDocument, KnowledgeDocStatus


@pytest.mark.asyncio
async def test_deepgram_final_callback_no_name_error():
    """Verify that _on_deepgram_final handles final customer utterance without NameError."""
    call = SipCallSession(
        caller_phone="+919876543210",
        organization_id=1,
        audio_send_callback=lambda x: None
    )
    call._ai_speaking = False
    call._ai_speaking_until = 0.0

    # Mock conversation loop to avoid external network/LLM calls
    mock_loop = MagicMock()
    call._conversation_loop = mock_loop

    # Call _on_deepgram_final with typical caller utterance that previously crashed
    try:
        await call._on_deepgram_final("utt-1", "Hello?", {"speech_final": True, "duration": 0.5})
    except NameError as e:
        pytest.fail(f"_on_deepgram_final raised NameError: {e}")
    except Exception as e:
        # Other benign errors if mocks missing are ok, but NameError must never happen
        assert not isinstance(e, NameError)


@pytest.mark.asyncio
async def test_barge_in_logic_holds_reflex_words_and_interrupts_on_substantive():
    """Verify barge-in holds isolated reflex words ('Hello?') and triggers on substantive speech."""
    call = SipCallSession(
        caller_phone="+919876543210",
        organization_id=1,
        audio_send_callback=lambda x: None
    )
    # Simulate ongoing AI speech playback
    import time
    call._ai_speaking = True
    call._ai_speaking_until = time.time() + 5.0
    call._barge_in_event.clear()

    # 1. Reflex word "Hello?" during AI speech -> Should NOT trigger barge-in event
    await call._on_deepgram_partial("utt-p1", "Hello?", {"duration": 0.3})
    assert not call._barge_in_event.is_set(), "Reflex word 'Hello?' incorrectly triggered barge-in!"
    assert call._ai_speaking is True, "AI speaking flag was incorrectly reset by 'Hello?' reflex!"

    # 2. Reflex word "Hi" -> Should NOT trigger barge-in event
    await call._on_deepgram_partial("utt-p2", "Hi", {"duration": 0.2})
    assert not call._barge_in_event.is_set(), "Reflex word 'Hi' incorrectly triggered barge-in!"

    # 3. Substantive customer interruption -> MUST trigger barge-in event
    await call._on_deepgram_partial("utt-p3", "Excuse me, I have a question", {"duration": 1.2})
    assert call._barge_in_event.is_set(), "Substantive speech failed to trigger barge-in!"
    assert call._ai_speaking is False, "AI speaking flag should be cleared on genuine barge-in!"


def test_doc4_greeting_extraction_section_found():
    """Verify that Doc 4 (Pramerica Outbound Script) extracts the Call Opening section correctly."""
    db = SessionLocal()
    try:
        doc = db.query(KnowledgeDocument).filter(KnowledgeDocument.id == 4).first()
        if doc:
            doc.ai_usage = "Active"
            db.commit()
            rag_service.invalidate_caches(doc.organization_id)

            greeting, doc_id, doc_title = rag_service.get_active_greeting(
                organization_id=doc.organization_id,
                document_id=str(doc.id),
                scope="outbound"
            )
            assert greeting is not None
            assert "Am I speaking with" in greeting
            assert doc_id == 4
    finally:
        db.close()


def test_silence_trimming_preserves_audio_length():
    """Verify that convert_tts_to_pcmu_and_pcm16 does not excessively truncate audio."""
    import numpy as np
    # Generate 4 seconds of 24kHz audio (synthetic speech-like sine wave)
    sample_rate = 24000
    duration = 4.0
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # 440 Hz tone
    audio_data = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
    pcm_bytes = audio_data.tobytes()

    # Prepend and append 100ms silence
    silence = b"\x00\x00" * int(sample_rate * 0.1)
    full_pcm = silence + pcm_bytes + silence

    # Wrap in standard WAV container
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(full_pcm)
    wav_bytes = wav_buf.getvalue()

    pcmu, pcm16 = convert_tts_to_pcmu_and_pcm16(wav_bytes, "murf")
    # Output is 8kHz 16-bit (16000 bytes per second)
    # Duration should be close to 4.0s (>= 3.8s)
    output_duration = len(pcm16) / 16000.0
    assert output_duration >= 3.8, f"Audio was excessively truncated: {output_duration}s < 3.8s"
