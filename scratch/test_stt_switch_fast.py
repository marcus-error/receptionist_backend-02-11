import sys, os, asyncio
from unittest.mock import AsyncMock

backend_path = os.path.abspath(r"C:\projects\2-11 ai receptionist updated\receptionist_backend-02-11")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from src.services.deepgram_stt_service import DeepgramStreamingSession

async def run_stt_test():
    print("Testing DeepgramStreamingSession switch_language...", flush=True)
    session = DeepgramStreamingSession(api_key="test_key", call_id="call_test_123", language="en")
    session._running = True
    session._connected = True

    async def fake_connect():
        session._connected = True
        session._ws = AsyncMock()
        session._ws.recv = AsyncMock(side_effect=asyncio.CancelledError)
        return True

    session._connect_ws = fake_connect

    # Put dummy frames in audio queue
    session._audio_queue.put_nowait(b"\x00" * 320)
    session._audio_queue.put_nowait(b"\x00" * 320)
    queue_len_before = session._audio_queue.qsize()
    assert queue_len_before == 2, f"Expected 2 frames in queue, got {queue_len_before}"

    # Perform switch to Tamil
    res = await session.switch_language("ta")
    assert res is True, "switch_language returned False"
    assert session.language == "ta", f"Expected 'ta', got {session.language}"

    # Verify audio queue was preserved across switch (not dropped)
    assert session._audio_queue.qsize() == 2, f"Audio queue dropped during switch! size={session._audio_queue.qsize()}"

    # Clean shutdown: cancel all tasks and wait for them
    session._running = False
    session._connected = False
    for t in session._tasks:
        t.cancel()
    await asyncio.gather(*session._tasks, return_exceptions=True)

    print("DEEPGRAM STT SWITCH TEST PASSED 100%!", flush=True)

asyncio.run(run_stt_test())

