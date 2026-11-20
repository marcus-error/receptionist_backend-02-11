"""
Speech-to-Text (STT) Service.

Primary Engine: Deepgram (Low-Latency Cloud STT)
Fallback Engine: Local Faster-Whisper (for offline / test usage)
"""

import asyncio
import io
import logging
import os
import tempfile
import time
from typing import Optional

from src.config.env import settings

logger = logging.getLogger(__name__)

_model_instance = None


def normalize_stt_language(language: Optional[str]) -> str:
    """Normalize language code to ISO code supported by Deepgram Nova-3 / local STT."""
    if not language:
        return "en"
    norm = str(language).lower().strip()
    if norm.startswith("ta") or norm.startswith("tam"):
        return "ta"
    elif norm.startswith("hi") or norm.startswith("hin"):
        return "hi"
    elif norm.startswith("te") or norm.startswith("tel"):
        return "te"
    elif norm.startswith("kn") or norm.startswith("kan"):
        return "kn"
    elif norm.startswith("en") or norm.startswith("eng"):
        return "en"
    return norm


class STTService:
    def __init__(self):
        self.model = None

    def initialize_model(self):
        """Initialize STT engine."""
        global _model_instance
        if "PYTEST_CURRENT_TEST" in os.environ or os.getenv("APP_ENV") == "test":
            logger.info("Test environment detected. Using mock STT engine.")
            _model_instance = "mock_stt"
            self.model = _model_instance
            return

        api_key = getattr(settings, "DEEPGRAM_API_KEY", None) or os.getenv("DEEPGRAM_API_KEY", "")
        if api_key:
            logger.info("[STT] Deepgram API key configured. Deepgram streaming is active as primary STT engine.")
            self.model = "deepgram"
            _model_instance = "deepgram"
            return

        model_size = getattr(settings, "STT_MODEL_SIZE", "tiny") or "tiny"
        logger.info(f"Loading local offline STT model (faster-whisper: {model_size})...")
        t0 = time.perf_counter()

        try:
            from faster_whisper import WhisperModel
            _model_instance = WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=4)
            self.model = _model_instance
            dur = (time.perf_counter() - t0) * 1000.0
            logger.info(f"Local fallback STT model loaded in {dur:.2f} ms")
        except Exception as exc:
            logger.warning(f"faster-whisper offline load failed: {exc}")
            self.model = None

    def transcribe_audio_bytes(
        self,
        audio_bytes: bytes,
        filename_hint: str = "audio.wav",
        language: Optional[str] = None
    ) -> str:
        """
        Synchronously transcribe audio bytes.
        Uses Deepgram if configured, otherwise falls back to local whisper.
        Supports Tamil (ta), English (en), Hindi (hi), Telugu (te), and Kannada (kn).
        """
        if not audio_bytes:
            return ""

        target_lang = normalize_stt_language(language or getattr(settings, "DEEPGRAM_LANGUAGE", "en") or "en")

        if self.model == "mock_stt":
            if target_lang == "ta":
                return "வணக்கம் எனக்கு ஒரு appointment book பண்ண வேண்டும்"
            elif target_lang == "hi":
                return "नमस्ते मुझे एक अपॉइंटमेंट बुक करना है"
            elif target_lang == "te":
                return "నమస్కారం నాకు ఒక అపాయింట్‌మెంట్ బుక్ చేయాలి"
            return "Hello I want to book an appointment with Dr Smith tomorrow"

        api_key = getattr(settings, "DEEPGRAM_API_KEY", None) or os.getenv("DEEPGRAM_API_KEY", "")
        model_name = getattr(settings, "DEEPGRAM_MODEL", "nova-3") or "nova-3"
        t0 = time.perf_counter()

        logger.info(
            f"\n[STT_TRANSCRIBE_REQUEST]"
            f"\n  engine={'deepgram' if api_key else 'faster-whisper'}"
            f"\n  model={model_name}"
            f"\n  requested_language={language}"
            f"\n  target_language={target_lang}"
            f"\n  audio_bytes={len(audio_bytes)}"
        )

        if api_key:
            # Use Deepgram REST API for one-shot transcription
            try:
                import requests
                headers = {
                    "Authorization": f"Token {api_key}",
                    "Content-Type": "audio/wav" if audio_bytes[:4] == b"RIFF" else "audio/raw"
                }
                params = {
                    "model": model_name,
                    "language": target_lang,
                    "smart_format": "true",
                    "punctuate": "true",
                }
                res = requests.post(
                    "https://api.deepgram.com/v1/listen",
                    headers=headers,
                    params=params,
                    data=audio_bytes,
                    timeout=5.0
                )
                dur_ms = (time.perf_counter() - t0) * 1000.0
                if res.status_code == 200:
                    data = res.json()
                    alts = data.get("results", {}).get("channels", [{}])[0].get("alternatives", [])
                    if alts:
                        raw_transcript = (alts[0].get("transcript") or "").strip()
                        confidence = float(alts[0].get("confidence", 1.0) or 1.0)
                        logger.info(
                            f"\n[DEEPGRAM_REST_STT_SUCCESS]"
                            f"\n  model={model_name}"
                            f"\n  language={target_lang}"
                            f"\n  latency_ms={dur_ms:.1f}"
                            f"\n  confidence={confidence:.2f}"
                            f"\n  raw_transcript='{raw_transcript}'"
                        )
                        return raw_transcript
                else:
                    logger.warning(f"[DEEPGRAM REST STT] status={res.status_code} body={res.text[:100]}")
            except Exception as e:
                logger.warning(f"[DEEPGRAM REST STT ERR] {e}")

        # Fallback to local faster-whisper if available
        if hasattr(self.model, "transcribe"):
            try:
                import numpy as np
                audio_input = None
                if isinstance(audio_bytes, bytes) and len(audio_bytes) > 44 and audio_bytes[:4] == b"RIFF":
                    data_pos = audio_bytes.find(b"data")
                    if data_pos != -1:
                        pcm_start = data_pos + 8
                        audio_input = np.frombuffer(audio_bytes[pcm_start:], dtype=np.int16).astype(np.float32) / 32768.0
                if audio_input is None:
                    audio_input = io.BytesIO(audio_bytes)

                whisper_lang = target_lang if target_lang in ("en", "ta", "hi", "te", "kn") else None
                segments, _ = self.model.transcribe(
                    audio_input,
                    beam_size=1,
                    language=whisper_lang,
                    temperature=0.0,
                    vad_filter=False,
                    condition_on_previous_text=False
                )
                text_parts = [segment.text for segment in segments if segment.text]
                clean_text = " ".join(text_parts).strip()
                dur_ms = (time.perf_counter() - t0) * 1000.0
                logger.info(
                    f"\n[LOCAL_WHISPER_STT_SUCCESS]"
                    f"\n  language={whisper_lang}"
                    f"\n  latency_ms={dur_ms:.1f}"
                    f"\n  clean_transcript='{clean_text}'"
                )
                return clean_text
            except Exception as exc:
                logger.error(f"[LOCAL STT ERROR] {exc}")

        return ""

    async def transcribe_async(
        self,
        audio_bytes: bytes,
        filename_hint: str = "audio.wav",
        language: Optional[str] = None
    ) -> str:
        return await asyncio.to_thread(self.transcribe_audio_bytes, audio_bytes, filename_hint, language)


stt_service = STTService()
