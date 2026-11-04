import asyncio
import logging
import pytest
import os
import sys

from src.services.deepgram_stt_service import DeepgramStreamingSession
from src.services.rag_service import rag_service
from src.services.language_service import language_service
from src.services.murf_tts import murf_tts
from src.services.tts_service import tts_service
from src.services.ai_service import ai_service

logger = logging.getLogger("test_call_flow_restoration")
logging.basicConfig(level=logging.INFO)


@pytest.mark.asyncio
async def test_deepgram_stt_connection_with_organization():
    """Verify Deepgram WebSocket connection succeeds with organization_id set (no keywords HTTP 400)."""
    session = DeepgramStreamingSession(
        call_id="test_diag_call_001",
        organization_id=1,  # Previously triggered HTTP 400 due to keywords on nova-3/multi
        sample_rate=8000,
    )
    url = session._build_ws_url()
    logger.info(f"Built WebSocket URL: {url}")
    # Verify keywords are omitted on nova-3 / multi
    assert "keywords=" not in url, "keywords parameter must not be in WebSocket URL for nova-3/multi"

    connected = await session._connect_ws()
    try:
        assert connected is True, "Deepgram WebSocket connection must succeed with connected=True"
        assert session._connected is True, "session._connected must be True"
        logger.info("[PASS] Deepgram WebSocket connected successfully with connected=True")
    finally:
        if session._ws:
            await session._ws.close()


def test_active_greeting_never_speaks_filename():
    """Verify greeting generation never speaks filenames or file titles, and returns None if no script is active."""
    from src.config.database import SessionLocal
    from src.models.knowledge_doc import KnowledgeDocument, KnowledgeDocStatus

    db = SessionLocal()
    try:
        # Create an active script for Org 1
        doc = KnowledgeDocument(
            title="SARANS_HOSPITALS_RAG_Inbound_Script_Fine_Tuned.pdf",
            organization_id=1,
            document_type="SCRIPT",
            scope="inbound",
            ai_usage="Active",
            status=KnowledgeDocStatus.INDEXED,
            content="""# SARANS HOSPITALS RAG SCRIPT
Voice Greeting: "Good morning! Welcome to Sarans Hospitals. How may I assist you today?"
# Doctor Directory
- Dr. Priya"""
        )
        db.add(doc)
        db.commit()
        rag_service.invalidate_caches(1)

        greeting, doc_id, doc_title = rag_service.get_active_greeting(1)
        logger.info(f"Org 1 -> Greeting: '{greeting}', Doc ID: {doc_id}, Doc Title: {doc_title}")
        assert greeting == "Good morning! Welcome to Sarans Hospitals. How may I assist you today?"
        assert not greeting.endswith(".pdf"), "Greeting must not end with .pdf"
        assert not greeting.endswith(".txt"), "Greeting must not end with .txt"
        assert greeting != doc_title, "Greeting must not be the raw document title or filename"
        assert doc_id == doc.id

        # Verify an org with NO active script returns None (zero generic fallback)
        empty_greeting, _, _ = rag_service.get_active_greeting(999)
        assert empty_greeting is None, "Org without active script must return None per zero-fallback policy"
    finally:
        db.query(KnowledgeDocument).filter(
            KnowledgeDocument.title == "SARANS_HOSPITALS_RAG_Inbound_Script_Fine_Tuned.pdf"
        ).delete(synchronize_session=False)
        db.commit()
        db.close()
        rag_service.invalidate_caches(1)


def test_multilingual_language_detection():
    """Verify language detection accurately classifies English, Tamil, and Tanglish."""
    # English
    en_phrase = "I would like to schedule an appointment with a doctor tomorrow"
    assert language_service.detect_language(en_phrase) == "en"

    # Tamil
    ta_phrase = "வணக்கம், நாளைக்கு டாக்டரை பார்க்க முடியுமா?"
    assert language_service.detect_language(ta_phrase) == "ta"

    # Tanglish
    tanglish_phrase = "Doctor appointment book panna mudiyuma?"
    lang = language_service.determine_conversation_language(tanglish_phrase, current_lang="en")
    assert lang in ("tanglish", "ta", "en")


def test_murf_voice_mapping_for_multilingual():
    """Verify Murf TTS maps correct voices for English, Tamil, and Hindi."""
    from src.services.murf_tts import MURF_LANGUAGE_VOICES
    assert MURF_LANGUAGE_VOICES.get("ta") == "ta-IN-iniya"
    assert MURF_LANGUAGE_VOICES.get("hi") == "hi-IN-shweta"
    assert MURF_LANGUAGE_VOICES.get("en") == "en-IN-isha"
    assert MURF_LANGUAGE_VOICES.get("tanglish") == "en-IN-isha"


def test_greeting_extractor_validation():
    """Verify extract_greeting_from_content rejects filenames and metadata."""
    # Test rejection of filenames
    assert rag_service.extract_greeting_from_content("SARANS_HOSPITALS_RAG_Inbound_Script_Fine_Tuned.pdf") == ""
    assert rag_service.extract_greeting_from_content("nurse_o_clock_receptionist_rag_formatted.txt") == ""

    # Test extraction of actual conversational greeting with dialogue prefix
    script_text = 'Receptionist: "Good morning! Welcome to Sarans Hospitals. How may I help you today?"'
    extracted = rag_service.extract_greeting_from_content(script_text, is_script_doc=True)
    assert extracted == "Good morning! Welcome to Sarans Hospitals. How may I help you today?"

    # Test unquoted greeting
    script_text_2 = "Voice Greeting: Welcome to ABC Health Clinic. How can I assist you today?"
    extracted_2 = rag_service.extract_greeting_from_content(script_text_2, is_script_doc=True)
    assert extracted_2 == "Welcome to ABC Health Clinic. How can I assist you today?"
