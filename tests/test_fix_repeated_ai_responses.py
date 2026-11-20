import os
import sys
import shutil
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

SCRATCH_DB = r"C:/Users/Admin/.gemini/antigravity/brain/5b24f344-f49d-43f5-989c-2e5ba434c29c/scratch/voice_repeated_test.db"
os.environ["DATABASE_URL"] = f"sqlite:///{SCRATCH_DB}"
os.environ["APP_ENV"] = "test"
os.environ["PYTEST_CURRENT_TEST"] = "1"
os.environ["DEFAULT_LLM_PROVIDER"] = "grounded_rag"

src_db = os.path.join(BASE_DIR, "receptionist.db")
if os.path.exists(src_db):
    shutil.copy(src_db, SCRATCH_DB)
    for ext in ["-wal", "-shm"]:
        if os.path.exists(src_db + ext):
            try:
                shutil.copy(src_db + ext, SCRATCH_DB + ext)
            except Exception:
                pass

from sqlalchemy import create_engine
import src.config.database as db_mod
test_engine = create_engine(f"sqlite:///{SCRATCH_DB}", connect_args={"check_same_thread": False})
db_mod.engine = test_engine
db_mod.SessionLocal.configure(bind=test_engine)

from src.utils.latency_tracker import LatencyTracker
from src.utils.text_sanitizer import clean_markdown_for_voice
from src.sockets.voice_socket import _extract_complete_sentences


class DummyWebSocket:
    def __init__(self):
        self.sent_events = []

    def send_json(self, payload):
        self.sent_events.append(payload)


class TestFixRepeatedAiResponses(unittest.TestCase):

    def test_01_latency_tracker_get_metric(self):
        """Verify LatencyTracker has get_metric method and returns recorded or default values."""
        tracker = LatencyTracker("call_123")
        self.assertEqual(tracker.get_metric("llm_first_token_ms", 0.0), 0.0)

        tracker.record("llm_first_token_ms", 215.4)
        self.assertEqual(tracker.get_metric("llm_first_token_ms", 0.0), 215.4)

        tracker.record("tts_first_audio_ms", 450.8)
        self.assertEqual(tracker.get_metric("tts_first_audio_ms", 0.0), 450.8)
        self.assertEqual(tracker.get_metric("nonexistent_metric", 99.9), 99.9)

    def test_02_clean_markdown_for_voice_does_not_deduplicate(self):
        """Verify clean_markdown_for_voice only removes markdown syntax, not repeated sentences."""
        sentence = "Aminoid AI offers several products including AI Voice Agents."
        cleaned = clean_markdown_for_voice(sentence)
        self.assertEqual(cleaned, sentence)

        raw_markdown = "**Aminoid AI** offers *several* products: `AI Voice Agents`!"
        cleaned_md = clean_markdown_for_voice(raw_markdown)
        self.assertNotIn("**", cleaned_md)
        self.assertNotIn("*", cleaned_md)
        self.assertNotIn("`", cleaned_md)

    def test_03_extract_complete_sentences(self):
        """Verify sentence extraction properly segments stream buffer."""
        buf = "Aminoid AI offers several products including AI Voice Agents. Would you like to know more? Sure thing"
        sentences, remaining = _extract_complete_sentences(buf)
        self.assertEqual(len(sentences), 2)
        self.assertEqual(sentences[0].strip(), "Aminoid AI offers several products including AI Voice Agents.")
        self.assertEqual(sentences[1].strip(), "Would you like to know more?")
        self.assertEqual(remaining.strip(), "Sure thing")

    def test_04_streaming_pipeline_simulation_no_crash(self):
        """
        Simulate the voice WebSocket streaming loop with token stream and verify:
        1. text_chunk events are sent
        2. audio_chunk events are sent
        3. response event is sent with the full text (not cut off by get_metric crash)
        4. tracker.get_metric works without throwing AttributeError
        """
        mock_ws = DummyWebSocket()
        turn_id = "turn_test_1"
        message_id = "msg_test_1"
        session_id = "sess_test_1"

        tokens = [
            "Aminoid AI ",
            "offers several ",
            "products including ",
            "AI Voice Agents. ",
            "Would you like ",
            "to know more?"
        ]

        tracker = LatencyTracker("test_call")
        full_response_text = ""
        sentence_buffer = ""
        chunk_index = 0
        first_token_recorded = False

        for token_chunk in tokens:
            if not first_token_recorded:
                tracker.record("llm_first_token_ms", 120.0)
                first_token_recorded = True

            full_response_text += token_chunk
            sentence_buffer += token_chunk

            mock_ws.send_json({
                "type": "text_chunk",
                "text": token_chunk,
                "turn_id": turn_id,
                "message_id": message_id,
                "conversation_id": session_id
            })

            sentences, remaining = _extract_complete_sentences(sentence_buffer)
            if sentences:
                sentence_buffer = remaining
                for sentence in sentences:
                    clean_voice_sent = clean_markdown_for_voice(sentence)
                    if clean_voice_sent and len(clean_voice_sent) >= 3:
                        mock_ws.send_json({
                            "type": "audio_chunk",
                            "audio": "fake_base64_audio",
                            "text": sentence,
                            "chunk_index": chunk_index,
                            "is_first": (chunk_index == 0),
                            "turn_id": turn_id,
                            "message_id": message_id,
                            "conversation_id": session_id
                        })
                        chunk_index += 1

        remaining_clean = clean_markdown_for_voice(sentence_buffer.strip())
        if remaining_clean and len(remaining_clean) >= 3:
            mock_ws.send_json({
                "type": "audio_chunk",
                "audio": "fake_base64_audio_final",
                "text": sentence_buffer.strip(),
                "chunk_index": chunk_index,
                "is_first": (chunk_index == 0),
                "turn_id": turn_id,
                "message_id": message_id,
                "conversation_id": session_id
            })
            chunk_index += 1

        response_text = full_response_text.strip()

        # Verify tracker.get_metric works safely without crashing
        first_tok_val = tracker.get_metric('llm_first_token_ms', 0.0)
        self.assertEqual(first_tok_val, 120.0)

        # Send final complete text to UI
        mock_ws.send_json({
            "type": "response",
            "text": response_text,
            "turn_id": turn_id,
            "message_id": message_id,
            "conversation_id": session_id,
            "is_final": True
        })

        # Validate the sent events
        text_chunks = [e for e in mock_ws.sent_events if e["type"] == "text_chunk"]
        audio_chunks = [e for e in mock_ws.sent_events if e["type"] == "audio_chunk"]
        responses = [e for e in mock_ws.sent_events if e["type"] == "response"]

        self.assertEqual(len(text_chunks), len(tokens))
        self.assertEqual(len(audio_chunks), 2)
        self.assertEqual(len(responses), 1)

        expected_full = "Aminoid AI offers several products including AI Voice Agents. Would you like to know more?"
        self.assertEqual(responses[0]["text"], expected_full)
        self.assertTrue(responses[0]["is_final"])

        self.assertEqual(audio_chunks[0]["chunk_index"], 0)
        self.assertEqual(audio_chunks[1]["chunk_index"], 1)


if __name__ == "__main__":
    unittest.main()
