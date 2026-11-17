import sys, os, time, asyncio
from unittest.mock import AsyncMock

print("Checkpoint 1", flush=True)
sys.path.insert(0, os.path.abspath(r"C:\projects\2-11 ai receptionist updated\receptionist_backend-02-11"))
print("Checkpoint 2", flush=True)
from src.services.language_service import language_service
print("Checkpoint 3", flush=True)
from src.services.deepgram_stt_service import DeepgramStreamingSession
print("Checkpoint 4", flush=True)
from src.services.booking_state_service import BookingSession
print("Checkpoint 5", flush=True)

async def main():
    print("Checkpoint in main 1", flush=True)
    phrase = "Will you please talk in Tamil?"
    target_lang, is_switch, switch_reason = language_service.check_language_switch_detailed(phrase)
    print("Checkpoint in main 2", target_lang, is_switch, flush=True)
    
    stt = DeepgramStreamingSession(api_key="dummy_key", call_id="test_ta_call", language="en")
    print("Checkpoint in main 3", flush=True)
    stt._running = True
    stt._connected = True
    stt._connect_ws = AsyncMock(return_value=True)
    async def dummy(): pass
    stt._send_loop = dummy
    stt._recv_loop = dummy
    print("Checkpoint in main 4 before switch_language", flush=True)
    res = await stt.switch_language("ta")
    print("Checkpoint in main 5 after switch_language", res, flush=True)

asyncio.run(main())
print("All finished cleanly!", flush=True)

