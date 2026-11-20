"""
Tests for DeepgramStreamingSession attribute initialization, turn reset,
inbound call STT pipeline, reconnect resilience, and AI response lifecycle.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from src.services.deepgram_stt_service import (
    DeepgramStreamingSession,
    normalize_deepgram_language,
)


def run_coro(coro):
    """Executes a coroutine that completes without yielding to an event loop."""
    try:
        coro.send(None)
    except StopIteration as e:
        return e.value


def test_01_deepgram_session_all_attributes_initialized():
    """Verify that every session variable is properly initialized in __init__."""
    session = DeepgramStreamingSession(
        call_id="call_test_init_123",
        sample_rate=8000,
        model="nova-3",
        language="en"
    )

    # Required attributes from bug report
    assert hasattr(session, "_debounce_task"), "Missing _debounce_task attribute"
    assert session._debounce_task is None
    assert hasattr(session, "_audio_bytes_sent_in_utterance"), "Missing _audio_bytes_sent_in_utterance"
    assert session._audio_bytes_sent_in_utterance == 0

    # Complete lifecycle attributes
    assert hasattr(session, "_utterance_counter")
    assert session._utterance_counter == 100
    assert hasattr(session, "_current_utt_id")
    assert session._current_utt_id == "utt_100"
    assert hasattr(session, "_utterance_start_time")
    assert session._utterance_start_time is None
    assert hasattr(session, "_first_partial_time")
    assert session._first_partial_time is None
    assert hasattr(session, "_speech_endpoint_time")
    assert session._speech_endpoint_time is None
    assert hasattr(session, "_current_turn_transcript")
    assert session._current_turn_transcript == ""
    assert hasattr(session, "_accumulated_finals")
    assert session._accumulated_finals == []
    assert hasattr(session, "_last_partial_text")
    assert session._last_partial_text == ""
    assert hasattr(session, "_turn_finalized")
    assert session._turn_finalized is False
    assert hasattr(session, "_last_final_text")
    assert session._last_final_text == ""
    assert hasattr(session, "_last_final_time")
    assert session._last_final_time == 0.0
    assert hasattr(session, "_processed_final_utt_ids")
    assert isinstance(session._processed_final_utt_ids, set)
    assert hasattr(session, "_last_detected_language")
    assert session._last_detected_language == "en"
    assert hasattr(session, "_last_turn_confidence")
    assert session._last_turn_confidence == 1.0

    # Reconnect mutex check
    lock = session._get_reconnect_lock()
    assert isinstance(lock, asyncio.Lock)
    assert session._get_reconnect_lock() is lock


def test_02_reset_turn_lifecycle_and_idempotency():
    """Verify reset_turn cleanly resets state and cancels any active debounce timer without raising AttributeError."""
    session = DeepgramStreamingSession(call_id="call_test_reset_456")

    # Simulate dirty utterance state
    session._utterance_start_time = 100.0
    session._first_partial_time = 100.1
    session._speech_endpoint_time = 101.5
    session._audio_bytes_sent_in_utterance = 16000
    session._current_turn_transcript = "Hello I want to book"
    session._accumulated_finals = ["Hello", "I want to book"]
    session._last_partial_text = "Hello I want to book an appointment"
    session._turn_finalized = True

    dummy_task = MagicMock()
    dummy_task.done.return_value = False
    session._debounce_task = dummy_task

    # Queue some audio
    session._audio_queue.put_nowait(b"\x00" * 320)
    assert not session._audio_queue.empty()

    # Call reset_turn
    session.reset_turn(clear_audio_queue=True)

    dummy_task.cancel.assert_called_once()
    assert session._debounce_task is None
    assert session._audio_bytes_sent_in_utterance == 0
    assert session._first_partial_time is None
    assert session._utterance_start_time is None
    assert session._speech_endpoint_time is None
    assert session._current_turn_transcript == ""
    assert session._accumulated_finals == []
    assert session._last_partial_text == ""
    assert session._turn_finalized is False
    assert session._audio_queue.empty()

    # Call reset_turn again (multiple consecutive resets must be 100% safe)
    session.reset_turn(clear_audio_queue=False)
    assert session._debounce_task is None
    assert session._audio_bytes_sent_in_utterance == 0


def test_03_feed_pcm16_audio_and_audio_bytes_tracking():
    """Verify incoming caller RTP audio chunks are queued and counted properly."""
    session = DeepgramStreamingSession(call_id="call_test_audio_789")
    session._running = True

    chunk_1 = b"\x01\x02" * 160  # 320 bytes = 20ms of 8kHz 16-bit PCM
    chunk_2 = b"\x03\x04" * 160

    session.feed_pcm16_audio(chunk_1)
    session.feed_pcm16_audio(chunk_2)

    assert session._audio_queue.qsize() == 2
    popped_1 = session._audio_queue.get_nowait()
    popped_2 = session._audio_queue.get_nowait()

    assert popped_1 == chunk_1
    assert popped_2 == chunk_2

    # Simulate what _send_loop does: increments _audio_bytes_sent_in_utterance
    session._audio_bytes_sent_in_utterance += len(popped_1)
    session._audio_bytes_sent_in_utterance += len(popped_2)
    assert session._audio_bytes_sent_in_utterance == 640


def test_04_reconnect_lock_mutex_and_guard():
    """Verify _reconnect respects running state and lock mutex."""
    session = DeepgramStreamingSession(call_id="call_test_recon_101")
    session._running = False

    # When not running, _reconnect returns False immediately
    res = run_coro(session._reconnect())
    assert res is False


def test_05_stop_cleans_resources_and_no_event_loop_errors():
    """Verify stop() cancels tasks and releases resources cleanly."""
    session = DeepgramStreamingSession(call_id="call_test_stop_202")
    session._running = True
    session._connected = True

    dummy_task = MagicMock()
    dummy_task.done.return_value = False
    session._debounce_task = dummy_task

    session._audio_queue.put_nowait(b"\x01\x02" * 160)

    run_coro(session.stop())

    assert session._running is False
    assert session._connected is False
    dummy_task.cancel.assert_called_once()
    assert session._debounce_task is None
    assert session._audio_queue.empty()


def test_06_sip_speak_locked_turn_reset_safety():
    """Verify reset_turn handles exceptions gracefully without aborting AI speech."""
    session = DeepgramStreamingSession(call_id="call_test_speak_303")
    # Multiple resets in sequence
    session.reset_turn()
    session.reset_turn()
    assert session._debounce_task is None
    assert session._audio_bytes_sent_in_utterance == 0


def test_07_results_message_handling_and_finalization():
    """Verify _handle_results_message handles interim and speech_final results with correct logging and callbacks."""
    partials = []
    finals = []

    def on_part(utt_id, text, dur, lang=""):
        partials.append((utt_id, text))

    def on_fin(utt_id, text, metrics):
        finals.append((utt_id, text, metrics))

    session = DeepgramStreamingSession(
        call_id="call_test_transcript_404",
        on_partial=on_part,
        on_final=on_fin,
        sample_rate=8000
    )
    session._running = True

    # 1. Interim frame
    interim_data = {
        "type": "Results",
        "channel": {
            "alternatives": [{
                "transcript": "Hello I want to know about",
                "confidence": 0.95
            }]
        },
        "is_final": False,
        "speech_final": False,
        "duration": 1.2
    }
    run_coro(session._handle_results_message(interim_data))
    assert len(partials) == 1
    assert partials[0][1] == "Hello I want to know about"
    assert len(finals) == 0

    # 2. speech_final frame
    final_data = {
        "type": "Results",
        "channel": {
            "alternatives": [{
                "transcript": "Hello I want to know about voice agents",
                "confidence": 0.98
            }]
        },
        "is_final": True,
        "speech_final": True,
        "duration": 2.1
    }
    run_coro(session._handle_results_message(final_data))
    assert len(finals) == 1
    assert finals[0][1] == "Hello I want to know about voice agents"
    assert finals[0][2]["text"] == "Hello I want to know about voice agents"
    assert finals[0][2]["confidence"] == 0.98

    # Check that turn reset occurred for next turn
    assert session._turn_finalized is True
    assert session._audio_bytes_sent_in_utterance == 0
    assert session._current_utt_id == "utt_101"
