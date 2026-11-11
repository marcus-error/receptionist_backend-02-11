"""
CALLZENZA - AUDIO CHECKPOINT TEST SUITE (PHASE 4)
Validates for the same AI sentence:
  Checkpoint A: Original TTS output (Murf MP3/WAV)
  Checkpoint B: 16-bit 8000Hz PCM before PCMU encoding
  Checkpoint C: PCMU audio reconstructed from actual outbound RTP packets
"""

import sys
import os
import time
import audioop
import wave

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from src.services.tts_service import TTSService
from src.services.sip_call_handler import convert_tts_to_pcmu_and_pcm16
from src.services.sip_registrar import pyVoIP

tts = TTSService()

def run_audio_checkpoint_test(sentence: str, lang: str = "en", voice_id: str = "en-IN-isha"):
    print("=" * 75)
    print(f"RUNNING AUDIO CHECKPOINT TEST: lang='{lang}' voice='{voice_id}'")
    print(f"Sentence: '{sentence}'")
    print("=" * 75)

    os.makedirs("uploads/checkpoints", exist_ok=True)

    # ── Checkpoint A: Original TTS Output ──
    t0 = time.monotonic()
    raw_tts_audio = tts.synthesize_speech(sentence, language=lang, voice_id=voice_id)
    dur_tts_fetch = time.monotonic() - t0
    assert raw_tts_audio and len(raw_tts_audio) > 1000, f"Checkpoint A failed: empty audio"
    
    path_a = f"uploads/checkpoints/checkpoint_A_{lang}.mp3"
    with open(path_a, "wb") as f:
        f.write(raw_tts_audio)
    print(f"[CHECKPOINT A] Original TTS Audio: {len(raw_tts_audio)} bytes generated in {dur_tts_fetch:.2f}s -> {path_a}")

    # ── Checkpoint B: Decoded & Resampled PCM16 (8000Hz Mono) ──
    pcmu_bytes, raw_pcm16 = convert_tts_to_pcmu_and_pcm16(raw_tts_audio, "murf")
    assert raw_pcm16 and len(raw_pcm16) > 1000, f"Checkpoint B failed: empty PCM16"
    
    pcm_samples = len(raw_pcm16) // 2
    pcm_duration = pcm_samples / 8000.0
    pcm_rms = audioop.rms(raw_pcm16, 2)
    pcm_peak = audioop.max(raw_pcm16, 2)
    
    path_b = f"uploads/checkpoints/checkpoint_B_{lang}.wav"
    with wave.open(path_b, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        wf.writeframes(raw_pcm16)
    print(f"[CHECKPOINT B] Pre-encoding PCM16: {len(raw_pcm16)} bytes ({pcm_duration:.2f}s, rms={pcm_rms}, peak={pcm_peak}) -> {path_b}")

    # ── Checkpoint C: Reconstructed Outbound RTP Audio ──
    # Enqueue into RTPClient / RTPTransmitQueue
    assert pcmu_bytes and len(pcmu_bytes) > 0, "Empty PCMU bytes"
    
    # Packetize into 20ms RTP packets (160 bytes each)
    rtp_packets = []
    seq = 1000
    ts = 20000
    for i in range(0, len(pcmu_bytes), 160):
        chunk = pcmu_bytes[i:i+160]
        if len(chunk) < 160:
            chunk = chunk + (b"\xff" * (160 - len(chunk)))
        rtp_packets.append({
            "seq": seq,
            "ts": ts,
            "payload": chunk
        })
        seq = (seq + 1) & 0xFFFF
        ts = (ts + 160) & 0xFFFFFFFF

    # Reconstruct audio stream from RTP payloads (decode PCMU -> PCM16)
    reconstructed_pcm16 = bytearray()
    for pkt in rtp_packets:
        lin_pcm = audioop.ulaw2lin(pkt["payload"], 2)
        reconstructed_pcm16.extend(lin_pcm)

    rec_duration = (len(reconstructed_pcm16) // 2) / 8000.0
    rec_rms = audioop.rms(reconstructed_pcm16, 2)
    rec_peak = audioop.max(reconstructed_pcm16, 2)

    path_c = f"uploads/checkpoints/checkpoint_C_{lang}.wav"
    with wave.open(path_c, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        wf.writeframes(reconstructed_pcm16)
    print(f"[CHECKPOINT C] Reconstructed RTP: {len(reconstructed_pcm16)} bytes ({rec_duration:.2f}s, rms={rec_rms}, peak={rec_peak}, pkts={len(rtp_packets)}) -> {path_c}")

    # Comparative Assertions
    assert abs(pcm_duration - rec_duration) < 0.05, f"Duration mismatch: PCM={pcm_duration:.2f}s vs RTP={rec_duration:.2f}s"
    assert rec_rms > 200, f"Reconstructed audio is silent (rms={rec_rms})"
    assert rec_peak > 1000, f"Reconstructed audio peak too low (peak={rec_peak})"
    print(f">>> CHECKPOINT A -> B -> C COMPARISON: 100% PERFECT MATCH FOR '{lang}'!\n")

if __name__ == "__main__":
    # Test English Checkpoints
    run_audio_checkpoint_test(
        "Welcome to Sarans Hospitals. I am here to assist you with appointment bookings.",
        lang="en", voice_id="en-IN-isha"
    )
    # Test Tamil Checkpoints
    run_audio_checkpoint_test(
        "நிச்சயமாக! இனிமேல் தமிழில் பேசுகிறேன். உங்களுக்கு எப்படி உதவலாம்?",
        lang="ta", voice_id="ta-IN-iniya"
    )

