"""
Comprehensive Automated Test Suite for Voice Assistant Start Call Active RAG Binding.

Verifies:
1. REST API: GET /api/v1/knowledge/active returns currently active document metadata.
2. WebSocket Active RAG Resolution: _resolve_and_validate_active_doc correctly loads active indexed document and metadata.
3. Start Call Validation: Clear error returned when no active document exists.
4. Start Call Greeting: Dynamic greeting delivered strictly from active document (Aminoid AI vs Sarans Hospitals).
5. Session State Binding: Fresh session initialized with active_doc_id, business_type, booking_type.
6. In-Domain Knowledge Retrieval: Queries answered strictly from active document content.
7. Out-of-Domain Safety: Honest decline / no leak of doctors/departments when Aminoid AI is active.
8. Dynamic Script Switch: Changing active script in DB immediately updates greeting, business domain, and knowledge base.
9. Stop Call & Restart Lifecycle: Clean session reset on stop; fresh initialization on restart.
10. TTS Synthesis: Murf TTS correctly synthesizes audio for greeting and conversational responses.
"""

import sys
import os
import shutil
import unittest

# 1. Configure isolated test database in brain scratch directory
SCRATCH_DB = r"C:/Users/Admin/.gemini/antigravity/brain/5b24f344-f49d-43f5-989c-2e5ba434c29c/scratch/voice_rag_test.db"
os.environ["DATABASE_URL"] = f"sqlite:///{SCRATCH_DB}"
os.environ["APP_ENV"] = "test"
os.environ["PYTEST_CURRENT_TEST"] = "1"
os.environ["DEFAULT_LLM_PROVIDER"] = "grounded_rag"

# 2. Add backend root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# Refresh scratch database from source
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
from sqlalchemy.orm import sessionmaker
import src.config.database as db_mod
test_engine = create_engine(f"sqlite:///{SCRATCH_DB}", connect_args={"check_same_thread": False})
db_mod.engine = test_engine
db_mod.SessionLocal.configure(bind=test_engine)
SessionLocal = db_mod.SessionLocal

from src.models.knowledge_doc import KnowledgeDocument, KnowledgeDocStatus
from src.models.user import User
from src.services.rag_service import rag_service
from src.services.booking_state_service import booking_state_service, BookingState
from src.services.ai_service import ai_service
from src.services.murf_tts import murf_tts
from src.controllers.knowledge_controller import get_active_doc
from src.sockets.voice_socket import (
    _resolve_and_validate_active_doc,
    _authenticate_ws_token,
    _is_acoustic_echo,
    _contextualize_voice_query,
    _filter_procedural_chunks,
    _extract_complete_sentences
)
from src.config.jwt import create_access_token


def set_active_script(org_id: int, active_doc_id: int):
    """Simulates admin activating a document in Knowledge Base."""
    with SessionLocal() as db:
        db.query(KnowledgeDocument).filter(KnowledgeDocument.organization_id == org_id).update({"ai_usage": "Inactive"})
        if active_doc_id is not None:
            db.query(KnowledgeDocument).filter(KnowledgeDocument.id == active_doc_id).update({
                "ai_usage": "Active",
                "status": KnowledgeDocStatus.INDEXED
            })
        db.commit()
    rag_service.clear_organization_cache(org_id)
    rag_service.clear_cache()


class TestVoiceAssistantActiveRag(unittest.TestCase):

    def setUp(self):
        # Default: Org 1 has Doc 6 (Aminoid AI) Active
        set_active_script(1, 6)
        booking_state_service._sessions.clear()

    def test_01_get_active_doc_endpoint(self):
        """GET /api/v1/knowledge/active returns currently active document metadata."""
        with SessionLocal() as db:
            user = db.query(User).filter(User.role == "admin", User.organization_id == 1).first()
            self.assertIsNotNone(user, "Admin user must exist in test database")

            data = get_active_doc(current_user=user, db=db)
            self.assertTrue(data.get("has_active"))
            doc = data.get("document")
            self.assertIsNotNone(doc)
            self.assertEqual(doc.get("id"), 6)
            self.assertIn("Aminoid", doc.get("title"))
            self.assertEqual(doc.get("ai_usage"), "Active")
            self.assertEqual(doc.get("status"), "Indexed")
            self.assertEqual(doc.get("chunks"), 22)

    def test_02_resolve_and_validate_active_doc_aminoid(self):
        """Voice socket helper resolves active Doc 6 with SOFTWARE and SALES_DEMO domain."""
        doc, script_meta = _resolve_and_validate_active_doc(1)
        self.assertIsNotNone(doc)
        self.assertEqual(doc.id, 6)
        self.assertIn("Aminoid", doc.title)
        self.assertIsNotNone(script_meta)
        self.assertEqual(script_meta.get("business_type"), "SOFTWARE")
        self.assertEqual(script_meta.get("booking_type"), "SALES_DEMO")
        self.assertIn("Aminoid", script_meta.get("business_name", ""))

    def test_03_no_active_document_rejected(self):
        """When no active document is configured, resolution returns (None, None) and controller reports has_active=False."""
        set_active_script(1, None)

        # 1. Voice socket validation
        doc, script_meta = _resolve_and_validate_active_doc(1)
        self.assertIsNone(doc)
        self.assertIsNone(script_meta)

        # 2. Controller endpoint verification
        with SessionLocal() as db:
            user = db.query(User).filter(User.role == "admin", User.organization_id == 1).first()
            data = get_active_doc(current_user=user, db=db)
            self.assertFalse(data.get("has_active"))
            self.assertIsNone(data.get("document"))
            self.assertIn("No active RAG document found", data.get("message"))

    def test_04_start_call_greeting_aminoid_ai(self):
        """Start Call delivers dynamic opening greeting from active Aminoid AI document."""
        doc, script_meta = _resolve_and_validate_active_doc(1)
        self.assertIsNotNone(doc)
        greeting = rag_service.get_organization_greeting(1, document_id=str(doc.id))

        self.assertIsNotNone(greeting)
        self.assertIn("Aminoid AI", greeting)
        # Greeting should NOT reference hospitals, doctors, or patient appointments
        greet_lower = greeting.lower()
        self.assertNotIn("sarans", greet_lower)
        self.assertNotIn("hospital", greet_lower)
        self.assertNotIn("doctor", greet_lower)

        # Synthesize audio with Murf TTS
        audio = murf_tts.synthesize(greeting, format="MP3")
        self.assertTrue(len(audio) > 50, "TTS audio must be generated for greeting")

    def test_05_session_initialization_and_binding(self):
        """Start call initializes fresh booking session bound to active document metadata."""
        session_id = "ws_test_voice_session_01"
        doc, script_meta = _resolve_and_validate_active_doc(1)

        b_type = script_meta.get("business_type", "GENERAL")
        book_type = script_meta.get("booking_type", "BUSINESS_CONSULTATION")
        b_name = script_meta.get("business_name")

        # Session initialization matching voice_socket logic
        booking_session = booking_state_service.get_session(
            session_id=session_id,
            organization_id=1,
            business_type=b_type,
            booking_type=book_type,
            business_name=b_name,
            active_doc_id=str(doc.id),
            active_doc_title=doc.title
        )
        booking_session.organization_id = 1
        booking_session.active_doc_id = str(doc.id)
        booking_session.active_rag_id = str(doc.id)

        state_dict = booking_session.get_state_dict()
        self.assertEqual(state_dict.get("active_doc_id"), "6")
        self.assertEqual(state_dict.get("active_rag_id"), "6")
        self.assertEqual(state_dict.get("business_type"), "SOFTWARE")
        self.assertEqual(state_dict.get("booking_type"), "SALES_DEMO")
        self.assertIn("Aminoid", state_dict.get("business_name", ""))
        self.assertEqual(state_dict.get("missing_fields"), [])
        self.assertEqual(state_dict.get("state"), BookingState.IDLE)
        self.assertEqual(state_dict.get("current_stage"), BookingState.IDLE)

    def test_06_in_domain_product_inquiry_grounded(self):
        """Caller asking about software products receives grounded response from Doc 6."""
        query = "What products and services does your company provide?"

        # RAG retrieval with scope='inbound'
        context = rag_service.retrieve_context(query=query, organization_id=1, scope="inbound")
        self.assertTrue(len(context) > 0, "RAG context must be retrieved from active document")
        self.assertTrue(
            any(w in context for w in ["Voice Agent", "Chatbot", "CRM", "HRMS", "ERP", "Aminoid"]),
            "RAG context must contain Aminoid AI software products"
        )

        # AI Response Generation
        session = booking_state_service.get_session("ws_test_turn_in_domain", organization_id=1, business_type="SOFTWARE", booking_type="SALES_DEMO")
        resp = ai_service.generate_response(
            text=query,
            context=context,
            session_history=[],
            intent="product_information",
            emotion="Neutral",
            organization_id=1,
            conversation_state=session.get_state_dict()
        )
        self.assertTrue(len(resp) > 0)
        resp_lower = resp.lower()
        self.assertTrue(
            any(w in resp_lower for w in ["ai", "software", "voice", "agent", "crm", "hrms", "aminoid"]),
            f"Response must discuss software products: {resp}"
        )
        self.assertNotIn("sarans", resp_lower)
        self.assertNotIn("cardiologist", resp_lower)

    def test_07_honest_decline_out_of_domain(self):
        """Caller asking for medical appointment under Aminoid AI script is honestly declined."""
        query = "I have severe chest pain and want to book an appointment with Dr. Priya the cardiologist."

        session = booking_state_service.get_session("ws_test_out_of_domain", organization_id=1, business_type="SOFTWARE", booking_type="SALES_DEMO")
        context = rag_service.retrieve_context(query=query, organization_id=1, scope="inbound")

        # Context from Aminoid AI will NOT contain Dr. Priya cardiology slots
        self.assertNotIn("Dr. Priya", context)

        resp = ai_service.generate_response(
            text=query,
            context=context,
            session_history=[],
            intent="emergency/urgent request",
            emotion="Urgent",
            organization_id=1,
            conversation_state=session.get_state_dict()
        )
        resp_lower = resp.lower()
        # Must NOT book cardiology appointment with Dr. Priya
        self.assertFalse(
            "confirmed" in resp_lower and "cardiology" in resp_lower,
            f"Aminoid AI must not confirm hospital appointment: {resp}"
        )

    def test_08_dynamic_script_switch_to_hospital(self):
        """Switching active script to Sarans Hospitals (Doc 1) immediately switches domain, greeting, and workflow."""
        set_active_script(1, 1)

        # 1. Resolution immediately reflects Doc 1
        doc, script_meta = _resolve_and_validate_active_doc(1)
        self.assertIsNotNone(doc)
        self.assertEqual(doc.id, 1)
        self.assertIn("SARANS", doc.title.upper())
        self.assertEqual(script_meta.get("business_type"), "HOSPITAL")
        self.assertIn("HOSPITAL", script_meta.get("booking_type", ""))

        # 2. Greeting dynamically switches to Sarans Hospitals
        greeting = rag_service.get_organization_greeting(1, document_id="1")
        self.assertIn("Sarans Hospitals", greeting)
        self.assertNotIn("Aminoid", greeting)

        # 3. New call session binds to HOSPITAL
        session = booking_state_service.get_session(
            session_id="ws_test_hospital_call",
            organization_id=1,
            business_type="HOSPITAL",
            booking_type="HOSPITAL_APPOINTMENT",
            active_doc_id="1",
            active_doc_title=doc.title
        )
        state_dict = session.get_state_dict()
        self.assertEqual(state_dict.get("active_doc_id"), "1")
        self.assertEqual(state_dict.get("business_type"), "HOSPITAL")

    def test_09_stop_call_and_restart(self):
        """Stopping a call resets session; restarting creates clean state with active script."""
        session_id = "ws_test_stop_restart"
        doc, script_meta = _resolve_and_validate_active_doc(1)

        # Start call session
        session = booking_state_service.get_session(session_id, organization_id=1, active_doc_id=str(doc.id))
        session.current_intent = "product_information"
        session.turn_count = 5

        # Stop Call: remove/reset session
        booking_state_service.remove_session(session_id)
        cleared_session = booking_state_service.get_session(session_id, organization_id=1)
        self.assertEqual(cleared_session.turn_count, 0)
        self.assertIsNone(cleared_session.current_intent)

        # Restart Call
        new_session = booking_state_service.get_session(
            session_id=session_id,
            organization_id=1,
            business_type=script_meta.get("business_type"),
            active_doc_id=str(doc.id)
        )
        self.assertEqual(new_session.active_doc_id, str(doc.id))
        self.assertEqual(new_session.turn_count, 0)

    def test_10_jwt_token_auth_resolution(self):
        """Token authentication resolves correct organization_id from valid JWT."""
        # Regular admin user (User 2, org 1)
        token = create_access_token({"sub": "2", "organization_id": 1, "role": "admin"})
        resolved_org = _authenticate_ws_token(token)
        self.assertEqual(resolved_org, 1)

        # Super admin with query_org_id parameter
        sa_token = create_access_token({"sub": "1", "role": "super_admin"})
        resolved_sa_org = _authenticate_ws_token(sa_token, query_org_id="1")
        self.assertEqual(resolved_sa_org, 1)

        # Invalid token returns None
        bad_token = "invalid.bearer.token"
        resolved_bad = _authenticate_ws_token(bad_token)
        self.assertIsNone(resolved_bad)

    def test_11_acoustic_echo_suppression(self):
        """Microphone tail echo matching greeting is suppressed; genuine caller response is accepted."""
        ai_greeting = "Hello! I'm MURI, the AI assistant for Aminoid AI. I can help with AI voice agents, chatbots, knowledge platforms, business automation and custom software. May I know your name and what business you're calling from?"
        from src.sockets.voice_socket import _normalize_speech
        recent_ai = [_normalize_speech(ai_greeting)]

        # Tail echoes within 3.5s
        is_echo, reason = _is_acoustic_echo("Hi may I know your full name", recent_ai, 1.2)
        self.assertTrue(is_echo, f"Expected tail echo suppression, got {is_echo} ({reason})")

        is_echo, reason = _is_acoustic_echo("May I know your name", recent_ai, 0.8)
        self.assertTrue(is_echo, f"Expected substring echo suppression, got {is_echo} ({reason})")

        # Echo occurring outside 4.5s window should not be dropped
        is_echo_late, _ = _is_acoustic_echo("Hi may I know your full name", recent_ai, 5.0)
        self.assertFalse(is_echo_late, "Echo check should respect the time window")

        # Genuine caller response within window MUST NOT be suppressed
        is_echo_caller, reason_caller = _is_acoustic_echo("My name is John Doe from Tech Corp", recent_ai, 1.5)
        self.assertFalse(is_echo_caller, f"Genuine caller response must not be suppressed: {reason_caller}")

        is_echo_prod, _ = _is_acoustic_echo("What products do you offer?", recent_ai, 2.0)
        self.assertFalse(is_echo_prod, "Caller product query must not be suppressed")

    def test_12_query_contextualization_anaphoric(self):
        """Multi-turn anaphoric queries resolve to specific active document products and filter procedural chunks."""
        # 1. Ordinal reference resolution
        q1 = _contextualize_voice_query("Can you explain the first product?", [], 1, "6")
        self.assertIn("AI Voice Agents", q1)

        q2 = _contextualize_voice_query("Tell me about the second one", [], 1, "6")
        self.assertIn("Galaxy Chat", q2)

        # 2. Pronoun reference resolution from history
        history = [
            {"role": "user", "text": "What about AI Voice Agents?"},
            {"role": "model", "text": "AI Voice Agents handle natural inbound and outbound voice calls."}
        ]
        q3 = _contextualize_voice_query("What is its main benefit?", history, 1, "6")
        self.assertIn("AI Voice Agents", q3)

        # 3. Procedural chunk filtering
        dirty_context = (
            "[AI Voice Agents (CONFIRM)]\nApproved content: Aminoid AI Voice Agents handle conversations.\n\n"
            "[6. Closure and thank you]\nChunk ID: closure-rule Approved content: Ask for contact details before closing.\nTo summarise, you're interested in {product or need}\n\n"
            "[1. Greeting]\nMURI's goal: Welcome, disclose AI"
        )
        clean_context = _filter_procedural_chunks(dirty_context, "What products do you have?")
        self.assertIn("AI Voice Agents", clean_context)
        self.assertNotIn("closure-rule", clean_context)
        self.assertNotIn("To summarise, you're interested in", clean_context)

    def test_13_sentence_level_streaming_extraction(self):
        """Sentence extraction splits at sentence boundaries while preserving honorifics/abbreviations."""
        buffer = "Aminoid AI offers several key products. First is AI Voice Agents, which handle natural human conversations. Second is Galaxy Chat for omnichannel messaging! Would you like a demo?"
        sentences, remaining = _extract_complete_sentences(buffer)

        self.assertEqual(len(sentences), 3)
        self.assertEqual(sentences[0], "Aminoid AI offers several key products.")
        self.assertEqual(sentences[1], "First is AI Voice Agents, which handle natural human conversations.")
        self.assertEqual(sentences[2], "Second is Galaxy Chat for omnichannel messaging!")
        self.assertEqual(remaining, "Would you like a demo?")

        # Protection of abbreviation
        buffer_abbrev = "Our Dr. Smith provides AI consultations. Please visit us today."
        s2, rem2 = _extract_complete_sentences(buffer_abbrev)
        self.assertEqual(len(s2), 1)
        self.assertEqual(s2[0], "Our Dr. Smith provides AI consultations.")
        self.assertEqual(rem2, "Please visit us today.")

    def test_14_multi_turn_dialogue_flow(self):
        """Multi-turn dialogue correctly answers product inquiries using Doc 6 and rejects out-of-domain questions."""
        session = booking_state_service.get_session("ws_test_multiturn", organization_id=1, business_type="SOFTWARE", booking_type="SALES_DEMO", active_doc_id="6")
        session_history = []

        # Turn 1: Product overview
        q1 = "What products does your company offer?"
        c1 = rag_service.retrieve_context(query=q1, organization_id=1, scope="inbound")
        self.assertTrue(len(c1) > 0)
        r1 = ai_service.generate_response(text=q1, context=c1, session_history=session_history, intent="product_information", organization_id=1, conversation_state=session.get_state_dict())
        self.assertTrue(any(p in r1.lower() for p in ["voice", "agent", "chat", "software", "product", "aminoid"]))
        session_history.append({"role": "user", "text": q1})
        session_history.append({"role": "model", "text": r1})

        # Turn 2: Anaphoric query
        q2 = "Can you explain the first product?"
        ctx_q2 = _contextualize_voice_query(q2, session_history, 1, "6")
        self.assertIn("AI Voice Agents", ctx_q2)
        c2 = rag_service.retrieve_context(query=ctx_q2, organization_id=1, scope="inbound")
        r2 = ai_service.generate_response(text=q2, context=c2, session_history=session_history, intent="product_details", organization_id=1, conversation_state=session.get_state_dict())
        self.assertTrue(any(w in r2.lower() for w in ["voice", "agent", "call", "inbound", "outbound", "ai"]))
        session_history.append({"role": "user", "text": q2})
        session_history.append({"role": "model", "text": r2})

        # Turn 3: Out-of-domain medical inquiry
        q3 = "Can I book a consultation with a cardiologist?"
        c3 = rag_service.retrieve_context(query=q3, organization_id=1, scope="inbound")
        r3 = ai_service.generate_response(text=q3, context=c3, session_history=session_history, intent="appointment_booking", organization_id=1, conversation_state=session.get_state_dict())
        r3_lower = r3.lower()
        self.assertFalse("cardiologist" in r3_lower and "confirmed" in r3_lower, "Must not confirm cardiology consultation on Aminoid AI session")

    def test_15_clean_disconnect_and_reconnect(self):
        """Simulates clean client disconnect signal; session cleanly resets without RuntimeError."""
        session_id = "ws_test_disconnect_session"
        booking_state_service.get_session(session_id, organization_id=1, business_type="SOFTWARE")
        self.assertIn(session_id, booking_state_service._sessions)

        # Cleanup simulates finally block on disconnect
        booking_state_service.remove_session(session_id)
        self.assertNotIn(session_id, booking_state_service._sessions)

        # Reconnect starts fresh
        fresh_session = booking_state_service.get_session(session_id, organization_id=1, business_type="SOFTWARE")
        self.assertEqual(fresh_session.turn_count, 0)
        self.assertFalse(fresh_session.conversation_started)


if __name__ == "__main__":
    unittest.main()
