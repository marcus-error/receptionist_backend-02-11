import json
import base64
import logging
import asyncio
import time
import hashlib
import re
import uuid
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.config.database import SessionLocal
from src.models.knowledge_doc import KnowledgeDocument, KnowledgeDocStatus
from src.models.user import User
from src.services.ai_service import ai_service
from src.services.rag_service import rag_service
from src.services.stt_service import stt_service
from src.services.nlp_service import nlp_service
from src.services.emotion_service import emotion_service
from src.services.tts_service import tts_service
from src.services.booking_state_service import booking_state_service, NextAction, BookingState
from src.services.call_service import CallService
from src.services.language_service import language_service
from src.services.agent_engine import agent_engine
from src.utils.latency_tracker import LatencyTracker
from src.utils.text_sanitizer import clean_markdown_for_voice, deduplicate_repeated_sentences

router = APIRouter()
logger = logging.getLogger(__name__)

FAST_PATH_INTENTS = {
    "EMERGENCY", "MEDICAL_ADVICE", "GREETING", "GOODBYE", "THANK_YOU",
    "CONFIRMATION", "REJECTION", "YES", "NO", "ACKNOWLEDGEMENT",
    "LANGUAGE_SWITCH",  # language-switch requests skip ALL RAG/NLP/booking
}

# How long (ms) to wait before playing a 'please hold' TTS during slow backend ops
_SLOW_FETCH_THRESHOLD_MS: float = 700.0


async def _handle_post_response_state(websocket: WebSocket, booking_session, session_id: str, current_lang: str, recent_ai_utterances: list):
    """
    Evaluates state immediately after TTS completes, emits state to client, and triggers automatic actions if ready.
    """
    logger.info(
        f"\n[STATE]\nprevious=RESPONDING\ncurrent=TTS\n"
        f"[TTS]\nstarted\ncompleted"
    )
    post_eval = booking_state_service.evaluate_post_response(booking_session)
    logger.info(
        f"\n[POST_RESPONSE_CHECK]\n"
        f"conversation_id={booking_session.conversation_id}\n"
        f"call_sid={booking_session.call_sid}\n"
        f"intent={booking_session.current_intent}\n"
        f"stage={booking_session.current_stage}\n"
        f"missing_fields={booking_session.missing_fields}\n"
        f"[NEXT_ACTION]\n"
        f"{post_eval['next_action']}"
    )

    await websocket.send_json({
        "type": "state",
        "state": post_eval["state"],
        "next_action": post_eval["next_action"],
        "stage": booking_session.current_stage,
        "missing_fields": booking_session.missing_fields
    })

    if post_eval["next_action"] == NextAction.AUTOMATIC_ACTION_REQUIRED:
        logger.info(f"[AUTOMATIC ACTION] Triggering auto-execution for session {booking_session.session_id}")
        auto_response = booking_state_service.process_booking_turn("", booking_session)
        norm_ans = _normalize_speech(auto_response)
        if norm_ans:
            recent_ai_utterances.append(norm_ans)

        await websocket.send_json({
            "type": "response",
            "text": auto_response
        })
        audio_response = await tts_service.synthesize_stream_chunk_async(auto_response, language=current_lang)
        if audio_response and len(audio_response) > 100:
            audio_base64 = base64.b64encode(audio_response).decode('utf-8')
            await websocket.send_json({
                "type": "audio",
                "audio": audio_base64
            })
        _log_turn(session_id, "[AUTO_ACTION]", auto_response, booking_session.current_intent, booking_session)

        # Follow-up evaluation after auto execution
        next_eval = booking_state_service.evaluate_post_response(booking_session)
        await websocket.send_json({
            "type": "state",
            "state": next_eval["state"],
            "next_action": next_eval["next_action"],
            "stage": booking_session.current_stage,
            "missing_fields": booking_session.missing_fields
        })
    elif post_eval["next_action"] in (NextAction.USER_INPUT_REQUIRED, NextAction.CONFIRMATION_REQUIRED):
        logger.info(f"\n[LISTENING]\nwaiting_for_caller=true")


def _log_turn(session_id: str, user_text: str, assistant_text: str, intent: str, booking_session=None, emotion: Optional[str] = None):
    try:
        db = SessionLocal()
        try:
            svc = CallService(db)
            sentiment = "Neutral"
            if emotion:
                e_lower = emotion.lower()
                if any(w in e_lower for w in ["happy", "satisfied", "positive"]):
                    sentiment = "Positive"
                elif any(w in e_lower for w in ["frustrated", "angry", "upset", "negative", "urgent"]):
                    sentiment = "Negative"

            svc.log_conversation_message(
                session_id=session_id,
                role="user",
                message=user_text,
                intent=intent,
                channel="voice",
                emotion=emotion,
                sentiment=sentiment,
            )
            svc.log_conversation_message(
                session_id=session_id,
                role="assistant",
                message=assistant_text,
                intent=intent,
                channel="voice",
                emotion=emotion,
                sentiment=sentiment,
                doctor_name=booking_session.doctor if booking_session else None,
                appointment_date=booking_session.date if booking_session else None,
                appointment_time=booking_session.time if booking_session else None,
                availability_verified=booking_session.availability_verified if booking_session else False,
                booking_confirmed=booking_session.booking_confirmed if booking_session else False,
                appointment_id=booking_session.appointment_id if booking_session else None
            )
        finally:
            db.close()
    except Exception as e:
        logger.warning(f"Failed to log conversation turn to DB: {e}")


def _get_loading_message(lang: str = "en", booking_session=None) -> str:
    """
    Returns a natural English 'please hold' message.
    Called when a backend operation (RAG/DB) exceeds the slow-fetch threshold.
    """
    return "Just one moment, please."


def _normalize_speech(text: str) -> str:
    """Strips punctuation and extra spaces for acoustic echo comparison."""
    return re.sub(r'[^a-zA-Z0-9\s]', '', (text or "").lower()).strip()


class TurnIdempotencyTracker:
    """
    Prevents duplicate conversational response generation from repeated API requests,
    duplicate WebSocket events, rapid speech recognition triggers, retries, or fallback execution.
    """
    def __init__(self, ttl_seconds: float = 15.0):
        self._processed_turns: Dict[str, Dict[str, Any]] = {}
        self._last_transcript: Optional[str] = None
        self._last_transcript_time: float = 0.0
        self._ttl = ttl_seconds

    def check_duplicate(self, text: str, turn_id: Optional[str] = None) -> tuple[bool, Optional[Dict[str, Any]]]:
        now = time.time()
        cutoff = now - self._ttl
        self._processed_turns = {k: v for k, v in self._processed_turns.items() if v.get("timestamp", 0) > cutoff}

        norm_text = re.sub(r'[^a-zA-Z0-9\s]', '', (text or "").lower()).strip()

        # 1. Check explicit turn_id / utterance_id
        if turn_id and turn_id in self._processed_turns:
            return True, self._processed_turns[turn_id]

        # 2. Check identical normalized text arriving within 1.5s
        if self._last_transcript and norm_text and norm_text == self._last_transcript and (now - self._last_transcript_time) < 1.5:
            cached = next((v for v in self._processed_turns.values() if v.get("norm_text") == norm_text), None)
            return True, cached

        return False, None

    def record_turn(self, turn_id: str, message_id: str, text: str, response_text: str):
        now = time.time()
        norm_text = re.sub(r'[^a-zA-Z0-9\s]', '', (text or "").lower()).strip()
        self._last_transcript = norm_text
        self._last_transcript_time = now
        self._processed_turns[turn_id] = {
            "turn_id": turn_id,
            "message_id": message_id,
            "text": text,
            "norm_text": norm_text,
            "response_text": response_text,
            "timestamp": now
        }


def _authenticate_ws_token(token: Optional[str], query_org_id: Optional[str] = None) -> Optional[int]:
    """Decodes JWT access token and resolves authenticated organization ID."""
    if not token:
        return None
    try:
        clean_token = token.strip().strip('"\'')
        if clean_token.lower().startswith("bearer "):
            clean_token = clean_token[7:].strip()

        from src.config.jwt import decode_access_token
        payload = decode_access_token(clean_token)
        if payload and "sub" in payload:
            user_id = payload.get("sub")
            db_auth = SessionLocal()
            try:
                user = db_auth.query(User).filter(User.id == int(user_id), User.is_active == 1).first()
                if user:
                    if user.role == "super_admin":
                        if query_org_id:
                            try:
                                return int(query_org_id)
                            except (ValueError, TypeError):
                                return None
                        return user.organization_id
                    return user.organization_id
            finally:
                db_auth.close()
        elif payload and "organization_id" in payload:
            return payload.get("organization_id")
    except Exception as auth_err:
        logger.warning(f"[VOICE WS AUTH] Token validation failed: {auth_err}")
    return None


def _resolve_and_validate_active_doc(org_id: Optional[int]) -> tuple[Optional[KnowledgeDocument], Optional[dict]]:
    """
    Fetches the currently active, indexed RAG document for an organization from the database.
    Returns (KnowledgeDocument, script_meta) or (None, None).
    """
    if org_id is None:
        return None, None
    db = SessionLocal()
    try:
        doc = db.query(KnowledgeDocument).filter(
            KnowledgeDocument.organization_id == org_id,
            KnowledgeDocument.ai_usage == "Active",
            KnowledgeDocument.status.in_([KnowledgeDocStatus.INDEXED, "Indexed"])
        ).order_by(KnowledgeDocument.updated_at.desc(), KnowledgeDocument.id.desc()).first()
        if not doc:
            return None, None
        script_meta = rag_service.get_active_script_metadata(org_id, document_id=str(doc.id), scope="inbound")
        return doc, script_meta
    except Exception as e:
        logger.error(f"[VOICE WS] Error resolving active document for org {org_id}: {e}")
        return None, None
    finally:
        db.close()



def _is_acoustic_echo(transcription: str, recent_ai_utterances: List[str], time_since_speech: float) -> tuple[bool, str]:
    """
    Evaluates whether an incoming transcript is an acoustic echo / reverberation of recent AI speech.
    Protects against microphone feedback and acoustic reverberation while preserving genuine speech.
    Returns (is_echo, reason).
    """
    if not transcription or not recent_ai_utterances:
        return False, "empty"

    norm_user = _normalize_speech(transcription)
    if not norm_user:
        return False, "empty_norm"

    user_words = norm_user.split()
    user_word_count = len(user_words)
    user_word_set = set(user_words)

    # Echoes typically occur within 4 seconds of speech completing or while audio is playing
    if time_since_speech > 4.5:
        return False, "outside_window"

    for recent_utt in recent_ai_utterances[-4:]:
        recent_words = recent_utt.split()
        ai_word_set = set(recent_words)

        # 1. Exact match
        if norm_user == recent_utt:
            return True, "exact"

        # 2. Substring match (user transcript is a phrase contained inside recent AI speech)
        if user_word_count >= 3 and norm_user in recent_utt:
            return True, "substring"

        # 3. N-gram phrase match with word containment (e.g. "Hi may I know your full name")
        if user_word_count >= 3:
            has_ngram_match = False
            for i in range(len(user_words) - 2):
                tri = f"{user_words[i]} {user_words[i+1]} {user_words[i+2]}"
                if tri in recent_utt:
                    has_ngram_match = True
                    break

            if has_ngram_match:
                overlap = len(user_word_set & ai_word_set) / float(len(user_word_set))
                if overlap >= 0.65:
                    return True, f"ngram_overlap_{overlap:.2f}"

        # 4. Tail match (user transcript matches the trailing words of the AI utterance)
        if user_word_count >= 3 and len(recent_words) >= user_word_count:
            tail_phrase = " ".join(recent_words[-user_word_count:])
            if norm_user == tail_phrase:
                return True, "exact_tail"
            tail_word_set = set(recent_words[-max(user_word_count, 8):])
            tail_overlap = len(user_word_set & tail_word_set) / float(len(user_word_set))
            if tail_overlap >= 0.70 and user_word_count >= 3:
                return True, f"tail_overlap_{tail_overlap:.2f}"

        # 5. Whole utterance Jaccard similarity
        if user_word_count >= 4 and len(recent_words) >= 4:
            jaccard = len(user_word_set & ai_word_set) / float(len(user_word_set | ai_word_set))
            if jaccard > 0.65:
                return True, f"jaccard_{jaccard:.2f}"

    return False, "no_echo"


def _contextualize_voice_query(query: str, session_history: List[dict], org_id: Optional[int], active_doc_id: Optional[str]) -> str:
    """
    Enriches anaphoric or vague multi-turn queries with specific product or service names
    resolved from the conversation history and active document metadata.
    """
    q_clean = query.strip()
    q_low = q_clean.lower()

    # Discover known products / services for this active document
    known_products = []
    try:
        known_products = rag_service.get_organization_services(org_id, document_id=active_doc_id)
    except Exception:
        pass

    if not known_products:
        known_products = [
            "AI Voice Agents",
            "Galaxy Chat",
            "AI Knowledge Management Platform",
            "Business Automation and AI Agents",
            "Business Applications",
            "Websites and Web Applications",
            "Custom Software and Integrations"
        ]

    # 1. Ordinal references: "first product", "second one", "3rd product", "explain the first"
    ordinals = [
        (r'\b(?:first|1st|number\s*1|product\s*1)\b', 0),
        (r'\b(?:second|2nd|number\s*2|product\s*2)\b', 1),
        (r'\b(?:third|3rd|number\s*3|product\s*3)\b', 2),
        (r'\b(?:fourth|4th|number\s*4|product\s*4)\b', 3),
        (r'\b(?:fifth|5th|number\s*5|product\s*5)\b', 4),
        (r'\b(?:sixth|6th|number\s*6|product\s*6)\b', 5),
        (r'\b(?:seventh|7th|number\s*7|product\s*7)\b', 6),
    ]
    for pattern, idx in ordinals:
        if re.search(pattern, q_low):
            if idx < len(known_products):
                target = known_products[idx]
                return f"{target} {q_clean}"

    # 2. Pronoun reference: "it", "its", "that product", "that software"
    if re.search(r'\b(?:it|its|that product|that service|this product|this service|the software)\b', q_low) and session_history:
        for item in reversed(session_history):
            text_val = item.get("text", "")
            for p in known_products:
                if p.lower() in text_val.lower():
                    return f"{p} {q_clean}"

    # 3. High-level catalog queries: "what products", "what services", "offerings", "what do you do"
    if any(phrase in q_low for phrase in ["what product", "what software", "what service", "what do you offer", "what do you provide", "product catalog", "all products", "list products"]):
        return f"{q_clean} products catalog services software suite overview"

    return q_clean


def _filter_procedural_chunks(context: str, query: str) -> str:
    """
    Removes internal procedural instructions (e.g. closure rules, step instructions, lead qualification rules)
    when the caller is asking informational product or service questions.
    """
    if not context:
        return ""

    q_low = query.lower()
    is_info_query = any(w in q_low for w in ["product", "service", "feature", "explain", "benefit", "what is", "how does", "pricing", "cost", "software", "catalog"])
    if not is_info_query:
        return context

    chunks = context.split("\n\n")
    kept = []
    procedural_patterns = [
        r'\bclosure-rule\b',
        r'\boutbound-call-rule\b',
        r'To summarise, you\'re interested in',
        r'MURI\'s goal',
        r'Chunks retrieved',
        r'Lead qualification & booking flow',
        r'Ask only after the user shows interest',
        r'Ask for contact details before closing',
        r'Customer Identity Verification'
    ]

    for chk in chunks:
        if any(re.search(pat, chk, re.IGNORECASE) for pat in procedural_patterns):
            continue
        kept.append(chk)

    if not kept:
        return context
    return "\n\n".join(kept)


def _extract_complete_sentences(buffer: str) -> tuple[List[str], str]:
    """
    Extracts completed sentences from streaming text buffer based on sentence delimiters (. ? ! \n).
    Protects abbreviations (Dr., Mr., Mrs., AI., e.g., i.e., vs.) from premature splitting.
    Returns (completed_sentences, remaining_buffer).
    """
    completed = []
    remaining = buffer

    while remaining:
        match = re.search(r'([.?!]+(?:\s+|\n+)|(?:\n\n+))', remaining)
        if not match:
            break
        end_pos = match.end()
        candidate = remaining[:end_pos].strip()

        is_abbrev = re.search(r'\b(?:Dr|Mr|Mrs|Ms|Prof|vs|eg|ie|AI|Inc|Ltd|Co|No)\.$', candidate, re.IGNORECASE)
        if is_abbrev:
            next_search_start = end_pos
            match2 = re.search(r'([.?!]+(?:\s+|\n+)|(?:\n\n+))', remaining[next_search_start:])
            if not match2:
                break
            end_pos = next_search_start + match2.end()
            candidate = remaining[:end_pos].strip()

        if len(candidate) > 2:
            completed.append(candidate)
            remaining = remaining[end_pos:]
        else:
            break

    return completed, remaining


def _extract_direct_sqlite_section(doc_id: str, query: str, org_id: Optional[int]) -> str:
    """
    Directly extracts relevant sections or catalog information from SQLite KnowledgeDocument.content
    as a high-reliability fallback if vector search returns empty.
    """
    try:
        from src.config.database import SessionLocal
        from src.models.knowledge_doc import KnowledgeDocument
        db = SessionLocal()
        try:
            doc = db.query(KnowledgeDocument).filter(KnowledgeDocument.id == int(doc_id)).first()
            if not doc or not doc.content:
                return ""
            content = doc.content
            q_words = [w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', query) if w.lower() not in ("what", "your", "does", "offer", "tell", "about", "have", "with", "this", "that")]

            sections = re.split(r'\n(?=[#A-Z0-9\s/&\-\'\"]+\n)', content)
            matched_sections = []
            for sec in sections:
                sec_lower = sec.lower()
                matches = sum(1 for w in q_words if w in sec_lower)
                if matches > 0:
                    matched_sections.append((matches, sec.strip()))

            if matched_sections:
                matched_sections.sort(key=lambda x: x[0], reverse=True)
                return "\n\n".join(s for _, s in matched_sections[:3])
            return content[:3500]
        finally:
            db.close()
    except Exception as e:
        logger.warning(f"Error in _extract_direct_sqlite_section: {e}")
        return ""


@router.websocket("/ws/voice")
@router.websocket("/ws/voice-stream")
@router.websocket("/api/v1/ws/voice")
@router.websocket("/api/v1/ws/voice-stream")
@router.websocket("/api/ws/voice")
@router.websocket("/api/ws/voice-stream")
async def voice_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = f"ws_{id(websocket)}"
    session_history = []
    socket_lock = asyncio.Lock()
    last_processed_hash = None
    last_processed_time = 0.0

    # Resolve tenant organization from token query parameter or explicit organization_id
    token = websocket.query_params.get("token")
    query_org_id = websocket.query_params.get("organization_id")
    resolved_org_id: Optional[int] = _authenticate_ws_token(token, query_org_id)
    org_id: Optional[int] = resolved_org_id or 1  # Default to 1 for unauthenticated dev/test connections

    # Resolve Active RAG snapshot for this connection
    active_doc, script_meta = _resolve_and_validate_active_doc(org_id)
    active_doc_id: Optional[str] = str(active_doc.id) if active_doc else None
    active_doc_title: str = active_doc.title if active_doc else "No Active Document"
    b_type = script_meta.get("business_type", "GENERAL") if script_meta else "GENERAL"
    book_type = script_meta.get("booking_type", "BUSINESS_CONSULTATION") if script_meta else "BUSINESS_CONSULTATION"
    b_name = script_meta.get("business_name") if script_meta else None

    # Session initialization
    booking_session = booking_state_service.get_session(
        session_id=session_id,
        organization_id=org_id,
        business_type=b_type,
        booking_type=book_type,
        business_name=b_name,
        active_doc_id=active_doc_id,
        active_doc_title=active_doc_title
    )
    booking_session.organization_id = org_id
    booking_session.active_doc_id = active_doc_id
    booking_session.active_rag_id = active_doc_id

    # Inform client of session-bound active RAG on connection
    await websocket.send_json({
        "type": "rag_bound",
        "title": active_doc_title,
        "document_id": active_doc_id,
        "has_active": bool(active_doc),
        "business_type": b_type,
        "booking_type": book_type
    })

    logger.info(
        f"[TEST_VOICE_SESSION] session_id={session_id} org_id={org_id} "
        f"active_doc_id={active_doc_id} title='{active_doc_title}' "
        f"business_type={b_type} booking_type={book_type}"
    )

    # Server-Side Acoustic Echo Tracker
    recent_ai_utterances: List[str] = []
    last_ai_speech_time: float = 0.0
    is_ai_playing: bool = False
    is_on_hold = False
    last_processed_utterance_id = None
    last_processed_hash = None
    last_processed_time = 0.0
    turn_tracker = TurnIdempotencyTracker()

    try:
        while True:
            data = await websocket.receive()
            if data.get("type") in ("websocket.disconnect", "disconnect"):
                logger.info(f"[TEST_WEBSOCKET] Client disconnected: session_id={session_id}")
                break

            async with socket_lock:
                tracker = LatencyTracker(call_sid=session_id)
                tracker.mark("audio_received_at")
                transcription = ""
                utterance_id = None

                if 'bytes' in data:
                    if is_on_hold:
                        logger.info(f"[RECEPTIONIST_SESSION] session_id={session_id} on hold — ignoring audio bytes.")
                        continue

                    audio_bytes = data['bytes']
                    current_lang = getattr(booking_session, "conversation_language", "en") or "en"
                    logger.info(f"[STT] session_id={session_id} received audio packet of size {len(audio_bytes)} language={current_lang}")

                    audio_hash = hashlib.md5(audio_bytes).hexdigest()
                    now = time.time()
                    if audio_hash == last_processed_hash and (now - last_processed_time) < 0.75:
                        logger.info(f"[VOICE_UTTERANCE] session_id={session_id} skipping duplicate audio packet (<750ms).")
                        continue
                    last_processed_hash = audio_hash
                    last_processed_time = now

                    # 1. Speech-to-Text with active conversation language
                    t0 = time.perf_counter()
                    tracker.mark("stt_started_at")
                    try:
                        transcription = await stt_service.transcribe_async(audio_bytes, language=current_lang)
                    except Exception as e:
                        logger.error(f"[STT] session_id={session_id} transcription failed: {e}")
                        await websocket.send_json({"type": "error", "message": "Failed to transcribe audio"})
                        continue
                    t_stt_dur = (time.perf_counter() - t0) * 1000.0
                    tracker.record("stt", t_stt_dur)
                    tracker.mark("stt_completed_at")
                    logger.info(
                        f"\n[VOICE_WS_AUDIO_TRANSCRIPTION_RESULT]"
                        f"\n  session_id={session_id}"
                        f"\n  language={current_lang}"
                        f"\n  duration_ms={t_stt_dur:.1f}"
                        f"\n  transcript='{transcription}'"
                    )

                    if not transcription or transcription.strip() in {"Demo transcription", ""}:
                        continue

                    utterance_id = f"aud_{int(now*1000)}"

                elif 'text' in data:
                    try:
                        text_data = json.loads(data['text'])
                        msg_type = text_data.get('type')

                        if msg_type == 'set_language':
                            req_lang = text_data.get('language') or 'en'
                            norm_lang = str(req_lang).lower().strip()
                            if norm_lang.startswith('ta'):
                                target_lang = 'ta'
                                stt_code = 'ta-IN'
                            elif norm_lang.startswith('hi'):
                                target_lang = 'hi'
                                stt_code = 'hi-IN'
                            elif norm_lang.startswith('te'):
                                target_lang = 'te'
                                stt_code = 'te-IN'
                            elif norm_lang.startswith('kn'):
                                target_lang = 'kn'
                                stt_code = 'kn-IN'
                            elif norm_lang.startswith('en'):
                                target_lang = 'en'
                                stt_code = 'en-IN'
                            else:
                                target_lang = 'en'
                                stt_code = 'en-IN'

                            booking_session.conversation_language = target_lang
                            logger.info(
                                f"\n[VOICE_WS_SET_LANGUAGE]"
                                f"\n  session_id={session_id}"
                                f"\n  requested_language='{req_lang}'"
                                f"\n  target_language='{target_lang}'"
                                f"\n  stt_code='{stt_code}'"
                            )
                            await websocket.send_json({
                                "type": "language_changed",
                                "language": target_lang,
                                "stt_lang": stt_code
                            })
                            continue

                        if msg_type in ('auth', 'init'):
                            auth_tok = text_data.get('token')
                            if auth_tok:
                                tok_org = _authenticate_ws_token(auth_tok, text_data.get('organization_id'))
                                if tok_org is not None:
                                    org_id = tok_org
                                    resolved_org_id = tok_org

                            active_doc, script_meta = _resolve_and_validate_active_doc(org_id)
                            active_doc_id = str(active_doc.id) if active_doc else None
                            active_doc_title = active_doc.title if active_doc else "No Active Document"
                            b_type = script_meta.get("business_type", "GENERAL") if script_meta else "GENERAL"
                            book_type = script_meta.get("booking_type", "BUSINESS_CONSULTATION") if script_meta else "BUSINESS_CONSULTATION"
                            b_name = script_meta.get("business_name") if script_meta else None

                            booking_session.organization_id = org_id
                            booking_session.active_doc_id = active_doc_id
                            booking_session.active_rag_id = active_doc_id
                            booking_session.business_type = b_type
                            booking_session.booking_type = book_type
                            booking_session.business_name = b_name

                            await websocket.send_json({
                                "type": "rag_bound",
                                "title": active_doc_title,
                                "document_id": active_doc_id,
                                "has_active": bool(active_doc),
                                "business_type": b_type,
                                "booking_type": book_type
                            })
                            continue

                        if msg_type == 'start_call':
                            logger.info(f"[VOICE WS] Received start_call command for session {session_id}")
                            start_tok = text_data.get('token')
                            if start_tok:
                                tok_org = _authenticate_ws_token(start_tok, text_data.get('organization_id'))
                                if tok_org is not None:
                                    org_id = tok_org
                                    resolved_org_id = tok_org

                            # Fetch currently active RAG document from DB to guarantee fresh state
                            active_doc, script_meta = _resolve_and_validate_active_doc(org_id)
                            if not active_doc:
                                logger.warning(f"[VOICE WS] start_call rejected: No active RAG document for org {org_id}")
                                await websocket.send_json({
                                    "type": "call_error",
                                    "message": "Cannot start call: No active RAG document found for your organization. Please upload and activate a document in Knowledge Base first."
                                })
                                continue

                            active_doc_id = str(active_doc.id)
                            active_doc_title = active_doc.title
                            b_type = script_meta.get("business_type", "GENERAL") if script_meta else "GENERAL"
                            book_type = script_meta.get("booking_type", "BUSINESS_CONSULTATION") if script_meta else "BUSINESS_CONSULTATION"
                            b_name = script_meta.get("business_name") if script_meta else None

                            # Initialize completely new voice conversation session
                            booking_state_service.remove_session(session_id)
                            booking_session = booking_state_service.get_session(
                                session_id=session_id,
                                organization_id=org_id,
                                business_type=b_type,
                                booking_type=book_type,
                                business_name=b_name,
                                active_doc_id=active_doc_id,
                                active_doc_title=active_doc_title
                            )
                            req_lang = text_data.get('language')
                            if req_lang:
                                norm_l = str(req_lang).lower().strip()
                                if norm_l.startswith('ta'):
                                    booking_session.conversation_language = 'ta'
                                elif norm_l.startswith('hi'):
                                    booking_session.conversation_language = 'hi'
                                elif norm_l.startswith('te'):
                                    booking_session.conversation_language = 'te'
                                elif norm_l.startswith('kn'):
                                    booking_session.conversation_language = 'kn'
                                elif norm_l.startswith('en'):
                                    booking_session.conversation_language = 'en'
                            else:
                                booking_session.conversation_language = 'en'

                            booking_session.greeting_sent = True
                            booking_session.conversation_started = True
                            booking_session.turn_count = 0
                            session_history.clear()
                            recent_ai_utterances.clear()
                            is_on_hold = False

                            logger.info(
                                f"\n[VOICE_CALL_INITIALIZED]"
                                f"\n  session_id={session_id}"
                                f"\n  organization_id={org_id}"
                                f"\n  active_doc_id={active_doc_id}"
                                f"\n  active_doc_title='{active_doc_title}'"
                                f"\n  business_type={b_type}"
                                f"\n  booking_type={book_type}"
                                f"\n  language={booking_session.conversation_language}"
                            )

                            await websocket.send_json({
                                "type": "call_started",
                                "session_id": session_id,
                                "document_id": active_doc.id,
                                "title": active_doc_title,
                                "business_type": b_type,
                                "booking_type": book_type
                            })

                            await websocket.send_json({
                                "type": "rag_bound",
                                "title": active_doc_title,
                                "document_id": active_doc_id,
                                "has_active": True
                            })

                            # Start opening greeting from the active document
                            try:
                                opening_greeting = rag_service.get_organization_greeting(org_id, document_id=active_doc_id)
                                if not opening_greeting:
                                    opening_greeting = f"Good morning! Thank you for calling {b_name or 'our office'}. How may I assist you?"

                                clean_norm = _normalize_speech(opening_greeting)
                                if clean_norm:
                                    recent_ai_utterances.append(clean_norm)
                                last_ai_speech_time = time.time()

                                greeting_turn_id = "turn_greeting"
                                greeting_msg_id = "msg_greeting"
                                await websocket.send_json({
                                    "type": "state",
                                    "state": "AI_SPEAKING"
                                })
                                await websocket.send_json({
                                    "type": "response",
                                    "text": opening_greeting,
                                    "turn_id": greeting_turn_id,
                                    "message_id": greeting_msg_id,
                                    "conversation_id": session_id,
                                    "is_final": True
                                })
                                audio_response = await tts_service.synthesize_speech_async(clean_markdown_for_voice(opening_greeting), language=booking_session.conversation_language or "en")
                                if audio_response and len(audio_response) > 100:
                                    audio_base64 = base64.b64encode(audio_response).decode('utf-8')
                                    await websocket.send_json({
                                        "type": "audio",
                                        "audio": audio_base64,
                                        "text": opening_greeting,
                                        "turn_id": greeting_turn_id,
                                        "message_id": greeting_msg_id,
                                        "conversation_id": session_id
                                    })
                                turn_tracker.record_turn(greeting_turn_id, greeting_msg_id, "[CALL_START]", opening_greeting)
                                _log_turn(session_id, "[CALL_START]", opening_greeting, "greeting", booking_session)
                                await _handle_post_response_state(websocket, booking_session, session_id, booking_session.conversation_language or "en", recent_ai_utterances)
                            except Exception as exc:
                                logger.error(f"[VOICE WS] Opening greeting error: {exc}")
                            continue

                        if msg_type == 'stop_call':
                            logger.info(f"[VOICE WS] Received stop_call command for session {session_id}")
                            try:
                                if org_id is not None:
                                    asyncio.create_task(asyncio.to_thread(agent_engine.generate_session_summary, session_id, org_id))
                            except Exception as _e:
                                logger.warning(f"Could not trigger post-call summary: {_e}")
                            booking_state_service.remove_session(session_id)
                            session_history.clear()
                            recent_ai_utterances.clear()
                            await websocket.send_json({
                                "type": "call_ended",
                                "status": "idle"
                            })
                            await websocket.send_json({
                                "type": "state",
                                "state": "IDLE"
                            })
                            continue

                        if msg_type == 'interrupt':
                            logger.info(f"[VOICE_UTTERANCE] [BARGE-IN] session_id={session_id} received client voice interrupt signal.")
                            continue

                        if msg_type == 'hold':
                            is_on_hold = True
                            booking_session.state = "ON_HOLD"
                            logger.info(f"[RECEPTIONIST_SESSION] session_id={session_id} put ON_HOLD.")
                            await websocket.send_json({"type": "state", "state": "ON_HOLD"})
                            continue

                        if msg_type == 'resume':
                            is_on_hold = False
                            booking_session.state = "LISTENING"
                            logger.info(f"[RECEPTIONIST_SESSION] session_id={session_id} RESUMED.")
                            await websocket.send_json({"type": "state", "state": "LISTENING"})
                            continue

                        if msg_type == 'playback_started':
                            is_ai_playing = True
                            last_ai_speech_time = time.time()
                            logger.info(f"[TEST_TTS_PLAYBACK] session_id={session_id} browser playback started")
                            continue

                        if msg_type == 'playback_completed':
                            is_ai_playing = False
                            last_ai_speech_time = time.time()
                            logger.info(f"[TEST_TTS_PLAYBACK] session_id={session_id} browser playback completed")
                            continue

                        if msg_type in ('reschedule', 'cancellation', 'callback', 'handoff', 'human_handoff'):
                            target_intent = {
                                'reschedule': 'reschedule',
                                'cancellation': 'cancellation',
                                'callback': 'callback',
                                'handoff': 'human_handoff',
                                'human_handoff': 'human_handoff'
                            }.get(msg_type, msg_type)
                            booking_session.current_intent = target_intent
                            logger.info(f"[RECEPTIONIST_SESSION] session_id={session_id} explicit signal received for action '{msg_type}'. Setting intent={target_intent}.")
                            if not text_data.get('text'):
                                text_data['text'] = f"I would like to request {msg_type}."

                        if is_on_hold:
                            logger.info(f"[RECEPTIONIST_SESSION] session_id={session_id} on hold — ignoring text message.")
                            continue

                        transcription = text_data.get('text', '').strip()
                        utterance_id = text_data.get('utterance_id')
                        confidence = text_data.get('confidence', 1.0)
                    except Exception:
                        transcription = ""
                        confidence = 1.0

                    if not transcription:
                        continue

                    now = time.time()
                    if utterance_id and utterance_id == last_processed_utterance_id and (now - last_processed_time) < 0.75:
                        logger.info(f"[VOICE_UTTERANCE] session_id={session_id} utterance_id={utterance_id} dropped duplicate (<750ms).")
                        continue

                    msg_hash = hashlib.md5(f"{transcription}_{utterance_id or ''}".encode('utf-8')).hexdigest()
                    if msg_hash == last_processed_hash and (now - last_processed_time) < 0.75:
                        logger.info(f"[VOICE_UTTERANCE] session_id={session_id} dropped duplicate text packet (<750ms).")
                        continue

                    last_processed_hash = msg_hash
                    last_processed_utterance_id = utterance_id
                    last_processed_time = now

                if not transcription:
                    continue

                # Turn Identification & Turn-Level Idempotency
                turn_id = utterance_id or f"turn_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"
                message_id = f"msg_{turn_id}"

                is_dup_turn, cached_turn = turn_tracker.check_duplicate(transcription, turn_id)
                if is_dup_turn:
                    logger.info(f"[TURN IDEMPOTENCY] session_id={session_id} turn_id={turn_id} duplicate turn rejected.")
                    if cached_turn and cached_turn.get("response_text"):
                        await websocket.send_json({
                            "type": "response",
                            "text": cached_turn["response_text"],
                            "turn_id": cached_turn.get("turn_id", turn_id),
                            "message_id": cached_turn.get("message_id", message_id),
                            "conversation_id": session_id,
                            "is_final": True
                        })
                    continue

                # PRECISE SERVER-SIDE ACOUSTIC ECHO REJECTION
                time_since_speech = time.time() - last_ai_speech_time
                if is_ai_playing and time_since_speech < 15.0:
                    logger.info(
                        f"[TEST_WEBSOCKET] [SPEECH DURING AI PLAYBACK DROPPED] session_id={session_id} "
                        f"Ignored transcript captured while browser audio was playing ({time_since_speech:.2f}s): '{transcription}'"
                    )
                    continue

                is_echo, echo_reason = _is_acoustic_echo(transcription, recent_ai_utterances, time_since_speech)
                if is_echo:
                    logger.warning(
                        f"[TEST_WEBSOCKET] [ACOUSTIC ECHO SUPPRESSED] session_id={session_id} "
                        f"Dropped incoming transcript matching recent AI speech ({echo_reason}, {time_since_speech:.2f}s ago): '{transcription}'"
                    )
                    continue

                logger.info(
                    f"\n[VOICE_TURN_START]"
                    f"\n  session_id={session_id}"
                    f"\n  call_sid={booking_session.call_sid}"
                    f"\n  utterance_id={utterance_id}"
                    f"\n  turn_id={turn_id}"
                    f"\n  message_id={message_id}"
                    f"\n  transcript='{transcription}'"
                    f"\n  confidence={confidence}"
                    f"\n  language={booking_session.conversation_language}"
                    f"\n  state_before={booking_session.state}"
                    f"\n  extracted_fields={json.dumps({k: v for k, v in booking_session.booking_data.items() if v is not None and k != 'idempotency_key'})}"
                )
                await websocket.send_json({
                    "type": "transcription",
                    "text": transcription,
                    "utterance_id": utterance_id,
                    "turn_id": turn_id,
                    "message_id": message_id,
                    "conversation_id": session_id
                })

                # ============================================================
                # LANGUAGE-SWITCH FAST PATH — must fire BEFORE NLP, RAG,
                # booking state, and entity extraction.
                # ============================================================
                target_lang, is_lang_switch = language_service.check_explicit_language_switch(transcription)
                if is_lang_switch:
                    if target_lang in ("ta", "en", "hi", "kn", "te"):
                        booking_session.conversation_language = target_lang
                    current_lang = target_lang or booking_session.conversation_language
                    lang_resp = language_service.get_language_switch_confirmation(current_lang, booking_session=booking_session)

                    stt_code_map = {"ta": "ta-IN", "hi": "hi-IN", "te": "te-IN", "kn": "kn-IN", "en": "en-IN"}
                    stt_code = stt_code_map.get(current_lang, "en-IN")
                    await websocket.send_json({
                        "type": "language_changed",
                        "language": current_lang,
                        "stt_lang": stt_code,
                        "turn_id": turn_id,
                        "conversation_id": session_id
                    })

                    norm_ans = _normalize_speech(lang_resp)
                    if norm_ans:
                        recent_ai_utterances.append(norm_ans)
                    last_ai_speech_time = time.time()

                    await websocket.send_json({"type": "analysis", "intent": "LANGUAGE_SWITCH",
                                               "emotion": "Neutral - 100%", "language": current_lang,
                                               "turn_id": turn_id, "conversation_id": session_id})
                    await websocket.send_json({
                        "type": "response",
                        "text": lang_resp,
                        "turn_id": turn_id,
                        "message_id": message_id,
                        "conversation_id": session_id,
                        "is_final": True
                    })

                    audio_response = await tts_service.synthesize_stream_chunk_async(lang_resp, language=current_lang)
                    if audio_response and len(audio_response) > 100:
                        audio_base64 = base64.b64encode(audio_response).decode('utf-8')
                        await websocket.send_json({
                            "type": "audio",
                            "audio": audio_base64,
                            "turn_id": turn_id,
                            "message_id": message_id,
                            "conversation_id": session_id
                        })

                    turn_tracker.record_turn(turn_id, message_id, transcription, lang_resp)
                    _log_turn(session_id, transcription, lang_resp, "LANGUAGE_SWITCH", booking_session)
                    await _handle_post_response_state(websocket, booking_session, session_id, current_lang, recent_ai_utterances)
                    continue
                # ============================================================

                # 2. Language Lock & Single-Pass Intent / Emotion Understanding
                t_n_start = time.perf_counter()
                prev_lang = booking_session.conversation_language
                booking_session.conversation_language = language_service.determine_conversation_language(
                    transcription, current_lang=booking_session.conversation_language
                )
                current_lang = booking_session.conversation_language
                tracker.mark("language_detected_at")

                if current_lang != prev_lang:
                    stt_code_map = {"ta": "ta-IN", "hi": "hi-IN", "te": "te-IN", "kn": "kn-IN", "en": "en-IN"}
                    stt_code = stt_code_map.get(current_lang, "en-IN")
                    await websocket.send_json({
                        "type": "language_changed",
                        "language": current_lang,
                        "stt_lang": stt_code,
                        "turn_id": turn_id,
                        "conversation_id": session_id
                    })

                try:
                    booking_state_service.extract_slots(transcription, booking_session)
                    booking_state_service.save_session_state(booking_session)
                except Exception as e:
                    logger.warning(f"[VOICE_SOCKET] Slot extraction error: {e}")

                nlp_task = nlp_service.detect_intent_async(transcription)
                emotion_task = emotion_service.detect_emotion_async(transcription)

                nlp_res, emotion_res = await asyncio.gather(nlp_task, emotion_task)
                intent_name = nlp_res.intent
                emotion = f"{emotion_res.emotion.title()} - {int(emotion_res.confidence * 100)}%"
                t_nlp_dur = (time.perf_counter() - t_n_start) * 1000.0
                tracker.record("intent_nlp", t_nlp_dur)
                tracker.mark("intent_detected_at")

                logger.info(
                    f"[AGENT] session_id={session_id} utterance_id={utterance_id} intent={intent_name} emotion={emotion} lang={current_lang}"
                )

                await websocket.send_json({
                    "type": "analysis",
                    "intent": intent_name,
                    "emotion": emotion,
                    "language": current_lang
                })

                org_id = getattr(booking_session, "organization_id", None)

                # -------------------------------------------------------------
                # HUMAN HANDOFF HANDLING
                # -------------------------------------------------------------
                if intent_name == "transfer request":
                    handoff_msg = "I am transferring you to our human representative right away. Please stay on the line."
                    if current_lang in ("ta", "tanglish", "ta_mix"):
                        handoff_msg = "உங்களை எங்கள் human representative-க்கு transfer செய்கிறேன். தயவுசெய்து line-ல் இருங்கள்."
                    elif current_lang == "hi":
                        handoff_msg = "Main aapko hamare human representative ke paas transfer kar raha hoon. Kripya line par bane rahein."

                    logger.info(f"[HANDOFF] [RECEPTIONIST_SESSION] session_id={session_id} initiated human handoff.")
                    await websocket.send_json({
                        "type": "handoff",
                        "status": "initiated",
                        "customer": booking_session.full_name,
                        "phone": booking_session.phone_number,
                        "reason": "Caller requested human agent"
                    })
                    await websocket.send_json({
                        "type": "response",
                        "text": handoff_msg,
                        "turn_id": turn_id,
                        "message_id": message_id,
                        "conversation_id": session_id,
                        "is_final": True
                    })
                    handoff_audio = await tts_service.synthesize_speech_async(clean_markdown_for_voice(handoff_msg), language=current_lang)
                    if handoff_audio and len(handoff_audio) > 100:
                        await websocket.send_json({
                            "type": "audio",
                            "audio": base64.b64encode(handoff_audio).decode('utf-8'),
                            "text": handoff_msg,
                            "turn_id": turn_id,
                            "message_id": message_id,
                            "conversation_id": session_id
                        })
                    turn_tracker.record_turn(turn_id, message_id, transcription, handoff_msg)
                    _log_turn(session_id, transcription, handoff_msg, "transfer request", booking_session)
                    await _handle_post_response_state(websocket, booking_session, session_id, current_lang, recent_ai_utterances)
                    continue

                # -------------------------------------------------------------
                # CALLBACK HANDLING
                # -------------------------------------------------------------
                if intent_name == "callback":
                    cb_phone = booking_session.phone_number or "your number"
                    cb_msg = f"I have registered your callback request for {cb_phone}. Our team will contact you shortly."
                    if current_lang in ("ta", "tanglish", "ta_mix"):
                        cb_msg = f"உங்க callback request register பண்ணியாச்சு. எங்கள் team உங்களை விரைவில் தொடர்புகொள்வார்கள்."
                    elif current_lang == "hi":
                        cb_msg = f"Aapka callback request note kar liya gaya hai. Hamari team aapse jaldi sampark karegi."

                    logger.info(f"[WORKFLOW] [CALLBACK] session_id={session_id} registered callback.")
                    await websocket.send_json({
                        "type": "response",
                        "text": cb_msg,
                        "turn_id": turn_id,
                        "message_id": message_id,
                        "conversation_id": session_id,
                        "is_final": True
                    })
                    cb_audio = await tts_service.synthesize_speech_async(clean_markdown_for_voice(cb_msg), language=current_lang)
                    if cb_audio and len(cb_audio) > 100:
                        await websocket.send_json({
                            "type": "audio",
                            "audio": base64.b64encode(cb_audio).decode('utf-8'),
                            "text": cb_msg,
                            "turn_id": turn_id,
                            "message_id": message_id,
                            "conversation_id": session_id
                        })
                    turn_tracker.record_turn(turn_id, message_id, transcription, cb_msg)
                    _log_turn(session_id, transcription, cb_msg, "callback", booking_session)
                    await _handle_post_response_state(websocket, booking_session, session_id, current_lang, recent_ai_utterances)
                    continue

                # -------------------------------------------------------------
                # APPOINTMENT CANCELLATION HANDLING
                # -------------------------------------------------------------
                if intent_name == "appointment_cancellation":
                    from src.services.receptionist_tools import cancel_appointment
                    appt_id_match = re.search(r'\b(?:id|appointment\s+id|appointment\s*#|#)\s*(\d+)\b', transcription, re.IGNORECASE)
                    target_id = int(appt_id_match.group(1)) if appt_id_match else booking_session.appointment_id
                    if target_id:
                        res = cancel_appointment(target_id)
                        cancel_msg = res.get("message", f"Appointment #{target_id} has been cancelled.")
                    else:
                        if current_lang == "ta":
                            cancel_msg = "உங்கள் பதிவை ரத்து செய்ய, தயவுசெய்து உங்கள் முன்பதிவு எண் அல்லது பதிவு செய்யப்பட்ட தொலைபேசி எண்ணைக் கூறவும்."
                        else:
                            cancel_msg = "Please provide your appointment ID or registered phone number so I can cancel your booking."

                    logger.info(f"[WORKFLOW] [APPOINTMENT_CANCEL] session_id={session_id} target_id={target_id}")
                    await websocket.send_json({
                        "type": "response",
                        "text": cancel_msg,
                        "turn_id": turn_id,
                        "message_id": message_id,
                        "conversation_id": session_id,
                        "is_final": True
                    })
                    cancel_audio = await tts_service.synthesize_speech_async(clean_markdown_for_voice(cancel_msg), language=current_lang)
                    if cancel_audio and len(cancel_audio) > 100:
                        await websocket.send_json({
                            "type": "audio",
                            "audio": base64.b64encode(cancel_audio).decode('utf-8'),
                            "text": cancel_msg,
                            "turn_id": turn_id,
                            "message_id": message_id,
                            "conversation_id": session_id
                        })
                    turn_tracker.record_turn(turn_id, message_id, transcription, cancel_msg)
                    _log_turn(session_id, transcription, cancel_msg, "appointment_cancellation", booking_session)
                    await _handle_post_response_state(websocket, booking_session, session_id, current_lang, recent_ai_utterances)
                    continue

                # -------------------------------------------------------------
                # APPOINTMENT RESCHEDULING HANDLING
                # -------------------------------------------------------------
                if intent_name == "appointment_rescheduling":
                    from src.services.receptionist_tools import reschedule_appointment
                    appt_id_match = re.search(r'\b(?:id|appointment\s+id|appointment\s*#|#)\s*(\d+)\b', transcription, re.IGNORECASE)
                    target_id = int(appt_id_match.group(1)) if appt_id_match else booking_session.appointment_id

                    # Check if new date and time are provided in the message
                    booking_state_service.extract_slots(transcription, booking_session)
                    if target_id and booking_session.date and booking_session.time:
                        res = reschedule_appointment(target_id, booking_session.date, booking_session.time, org_id)
                        reschedule_msg = res.get("message", f"Appointment #{target_id} rescheduled.")
                    else:
                        if current_lang == "ta":
                            reschedule_msg = "நிச்சயமாக, உங்கள் முன்பதிவை மாற்றியமைக்க உதவுகிறேன். எந்த புதிய தேதி மற்றும் நேரத்தை விரும்புகிறீர்கள்?"
                        else:
                            reschedule_msg = "Sure, I can help you reschedule. What new date and time would you prefer?"

                    logger.info(f"[WORKFLOW] [APPOINTMENT_RESCHEDULE] session_id={session_id} target_id={target_id}")
                    await websocket.send_json({
                        "type": "response",
                        "text": reschedule_msg,
                        "turn_id": turn_id,
                        "message_id": message_id,
                        "conversation_id": session_id,
                        "is_final": True
                    })
                    resched_audio = await tts_service.synthesize_speech_async(clean_markdown_for_voice(reschedule_msg), language=current_lang)
                    if resched_audio and len(resched_audio) > 100:
                        await websocket.send_json({
                            "type": "audio",
                            "audio": base64.b64encode(resched_audio).decode('utf-8'),
                            "text": reschedule_msg,
                            "turn_id": turn_id,
                            "message_id": message_id,
                            "conversation_id": session_id
                        })
                    turn_tracker.record_turn(turn_id, message_id, transcription, reschedule_msg)
                    _log_turn(session_id, transcription, reschedule_msg, "appointment_rescheduling", booking_session)
                    await _handle_post_response_state(websocket, booking_session, session_id, current_lang, recent_ai_utterances)
                    continue

                is_ongoing_turn = bool(booking_session.conversation_started or booking_session.greeting_sent or booking_session.turn_count > 0)
                fast_rule_answer = ai_service._fast_rule_response(
                    transcription, intent_name,
                    organization_id=org_id,
                    is_ongoing=is_ongoing_turn,
                    booking_session=booking_session,
                    conversation_language=current_lang
                )

                if fast_rule_answer:
                    logger.info(f"[AGENT] [FAST PATH ROUTE] Satisfied '{transcription}' in < 50ms without LLM/RAG.")
                    t_tts_start = time.perf_counter()
                    tracker.mark("tts_started_at")

                    # Track for echo suppression
                    norm_ans = _normalize_speech(fast_rule_answer)
                    if norm_ans:
                        recent_ai_utterances.append(norm_ans)
                    last_ai_speech_time = time.time()

                    await websocket.send_json({
                        "type": "response",
                        "text": fast_rule_answer,
                        "turn_id": turn_id,
                        "message_id": message_id,
                        "conversation_id": session_id,
                        "is_final": True
                    })

                    audio_response = await tts_service.synthesize_speech_async(clean_markdown_for_voice(fast_rule_answer), language=current_lang)
                    if audio_response and len(audio_response) > 100:
                        tracker.mark("first_audio_at")
                        audio_base64 = base64.b64encode(audio_response).decode('utf-8')
                        await websocket.send_json({
                            "type": "audio",
                            "audio": audio_base64,
                            "text": fast_rule_answer,
                            "turn_id": turn_id,
                            "message_id": message_id,
                            "conversation_id": session_id
                        })
                    tracker.record("tts_total", (time.perf_counter() - t_tts_start) * 1000.0)
                    tracker.mark("tts_completed_at")

                    turn_tracker.record_turn(turn_id, message_id, transcription, fast_rule_answer)
                    telemetry = tracker.log_summary()
                    await websocket.send_json({"type": "telemetry", "data": telemetry})
                    _log_turn(session_id, transcription, fast_rule_answer, intent_name, booking_session)
                    await _handle_post_response_state(websocket, booking_session, session_id, current_lang, recent_ai_utterances)
                    continue

                # Check if intent switched to informational query
                is_intent_switched = booking_state_service.check_and_handle_intent_switch(booking_session, intent_name, transcription)

                # 3. BOOKING / LEAD COLLECTION STATE MACHINE ROUTING
                is_callback_intent = (
                    intent_name == "callback"
                    or any(cb in transcription.lower() for cb in [
                        "arrange a call", "arrange call", "make an arrangement", "make arrangement",
                        "speak with your team", "speak with the team", "contact me", "call me back",
                        "schedule a call", "schedule a callback", "yes, contact me", "yes contact me"
                    ])
                )
                is_booking_intent = (
                    intent_name in ("appointment_booking", "appointment_availability", "sales_demo_booking", "demo_booking", "schedule_demo")
                    or any(w in transcription.lower() for w in ["book an appointment", "book appointment", "schedule an appointment", "reserve a slot"])
                )
                if not is_intent_switched and (booking_session.is_active() or is_booking_intent or is_callback_intent):
                    if is_callback_intent:
                        booking_session.workflow_mode = "LEAD_COLLECTION"
                    elif is_booking_intent:
                        booking_session.workflow_mode = "APPOINTMENT_BOOKING"
                        if booking_session.is_completed:
                            booking_session.reset()

                    response_text = booking_state_service.process_booking_turn(transcription, booking_session)

                    norm_ans = _normalize_speech(response_text)
                    if norm_ans:
                        recent_ai_utterances.append(norm_ans)
                    last_ai_speech_time = time.time()

                    await websocket.send_json({
                        "type": "response",
                        "text": response_text,
                        "turn_id": turn_id,
                        "message_id": message_id,
                        "conversation_id": session_id,
                        "is_final": True
                    })

                    t_tts = time.perf_counter()
                    tracker.mark("tts_started_at")
                    try:
                        audio_response = await tts_service.synthesize_speech_async(clean_markdown_for_voice(response_text), language=current_lang)
                        if audio_response and len(audio_response) > 100:
                            tracker.mark("first_audio_at")
                            audio_base64 = base64.b64encode(audio_response).decode('utf-8')
                            await websocket.send_json({
                                "type": "audio",
                                "audio": audio_base64,
                                "text": response_text,
                                "turn_id": turn_id,
                                "message_id": message_id,
                                "conversation_id": session_id
                            })
                    except Exception as e:
                        logger.error(f"Booking state TTS generation failed: {e}")

                    tts_dur_ms = (time.perf_counter() - t_tts) * 1000.0
                    tracker.record("tts_total", tts_dur_ms)
                    tracker.mark("tts_completed_at")
                    turn_tracker.record_turn(turn_id, message_id, transcription, response_text)
                    telemetry = tracker.log_summary()
                    logger.info(
                        f"\n[VOICE_TURN_COMPLETE]"
                        f"\n  session_id={session_id}"
                        f"\n  utterance_id={utterance_id}"
                        f"\n  intent={intent_name}"
                        f"\n  state_after={booking_session.state}"
                        f"\n  extracted_fields={json.dumps({k: v for k, v in booking_session.booking_data.items() if v is not None and k != 'idempotency_key'})}"
                        f"\n  tts_latency_ms={tts_dur_ms:.1f}"
                    )
                    await websocket.send_json({"type": "telemetry", "data": telemetry})
                    _log_turn(session_id, transcription, response_text, intent_name, booking_session)
                    session_history.append({"role": "user", "text": transcription})
                    session_history.append({"role": "model", "text": response_text})
                    await _handle_post_response_state(websocket, booking_session, session_id, current_lang, recent_ai_utterances)
                    continue

                # 4. INFORMATIONAL / STREAMING RAG ROUTING (STRICT ACTIVE DOCUMENT ISOLATION)
                booking_session.conversation_started = True
                booking_session.turn_count += 1
                context = ""
                response_text = ""

                if intent_name not in FAST_PATH_INTENTS:
                    t_rag = time.perf_counter()
                    tracker.mark("retrieval_started_at")
                    try:
                        current_doc_id = str(booking_session.active_doc_id) if getattr(booking_session, "active_doc_id", None) else rag_service._resolve_active_document_id(org_id)
                        contextualized_query = _contextualize_voice_query(transcription, session_history, org_id, current_doc_id)

                        rag_task = asyncio.create_task(
                            asyncio.to_thread(
                                rag_service.retrieve_context,
                                query=contextualized_query,
                                n_results=5,
                                organization_id=org_id,
                                intent=intent_name,
                                document_id=current_doc_id,
                                scope="inbound"
                            )
                        )
                        raw_context = await rag_task
                        context = _filter_procedural_chunks(raw_context, transcription)

                        # SQLite section direct extraction fallback
                        if not context and current_doc_id:
                            context = _extract_direct_sqlite_section(current_doc_id, contextualized_query, org_id)

                        doc_title_to_send = active_doc_title
                        if current_doc_id and org_id is not None:
                            try:
                                _d = SessionLocal()
                                doc_obj = _d.query(KnowledgeDocument).filter(
                                    KnowledgeDocument.id == int(current_doc_id),
                                    KnowledgeDocument.organization_id == org_id
                                ).first()
                                if doc_obj:
                                    doc_title_to_send = doc_obj.title
                                _d.close()
                            except Exception:
                                pass

                        logger.info(
                            f"[TEST_RAG_RETRIEVAL] session_id={session_id} doc_id={current_doc_id} "
                            f"query='{transcription}' contextualized='{contextualized_query}' "
                            f"context_length={len(context)} latency_ms={round((time.perf_counter() - t_rag) * 1000)}"
                        )

                        await websocket.send_json({
                            "type": "rag",
                            "source": f"{doc_title_to_send} (ID: {current_doc_id})" if (current_doc_id and doc_title_to_send not in ("None", "No Active Document")) else "Active Knowledge Base"
                        })
                    except Exception as e:
                        logger.error(f"[TEST_RAG_RETRIEVAL] RAG retrieval error: {e}")
                        context = ""
                    tracker.record("retrieval_ms", (time.perf_counter() - t_rag) * 1000.0)
                    tracker.mark("retrieval_completed_at")

                # Live Calendar Verification if availability query
                if intent_name in ("appointment_availability", "appointment_booking"):
                    try:
                        from src.services.calendar_verification_service import calendar_verification_service
                        cal_v = calendar_verification_service.verify_availability(transcription, session_history)
                        if cal_v and cal_v.is_checked:
                            await websocket.send_json({
                                "type": "calendar_verification",
                                "verified": cal_v.is_available,
                                "doctor": cal_v.doctor_name,
                                "date": cal_v.requested_date,
                                "time": cal_v.requested_time,
                                "reason": cal_v.reason,
                                "alternative_slots": cal_v.alternative_slots
                            })
                    except Exception as e:
                        logger.warning(f"Voice WS calendar verification error: {e}")

                # 5. STREAMING LLM & STREAMING TTS PIPELINE (< 1.5s time to first audio chunk)
                t_llm_start = time.perf_counter()
                tracker.mark("llm_started_at")
                session_history.append({"role": "user", "text": transcription})
                full_response_text = ""
                sentence_buffer = ""
                chunk_index = 0

                first_token_recorded = False
                first_audio_recorded = False

                try:
                    token_stream = ai_service.generate_response_stream(
                        text=transcription,
                        context=context,
                        session_history=session_history,
                        intent=intent_name,
                        emotion=emotion,
                        conversation_language=current_lang,
                        organization_id=org_id,
                        conversation_state=booking_session.get_state_dict()
                    )

                    for token_chunk in token_stream:
                        if not token_chunk:
                            continue

                        if not first_token_recorded:
                            t_first_tok = (time.perf_counter() - t_llm_start) * 1000.0
                            tracker.record("llm_first_token_ms", t_first_tok)
                            tracker.mark("first_token_at")
                            first_token_recorded = True

                        full_response_text += token_chunk
                        sentence_buffer += token_chunk

                        logger.info(
                            f"[DIAG_STAGE_B_ACCUMULATED] turn_id={turn_id} message_id={message_id} "
                            f"token_chunk_len={len(token_chunk)} accumulated_len={len(full_response_text)} "
                            f"chunk={token_chunk!r}"
                        )

                        # Stream text chunk to UI so user sees real-time visual transcription/typing
                        await websocket.send_json({
                            "type": "text_chunk",
                            "text": token_chunk,
                            "turn_id": turn_id,
                            "message_id": message_id,
                            "conversation_id": session_id
                        })

                        # Extract completed sentences to synthesize immediate TTS chunk
                        sentences, remaining = _extract_complete_sentences(sentence_buffer)
                        if sentences:
                            sentence_buffer = remaining
                            for sentence in sentences:
                                clean_voice_sent = clean_markdown_for_voice(sentence)
                                if clean_voice_sent and len(clean_voice_sent) >= 3:
                                    logger.info(
                                        f"[DIAG_STAGE_C_FINAL_SERVICE_CHUNK] turn_id={turn_id} chunk_index={chunk_index} "
                                        f"sentence={clean_voice_sent!r}"
                                    )
                                    t_tts_c_start = time.perf_counter()
                                    try:
                                        audio_chunk = await tts_service.synthesize_speech_async(clean_voice_sent, language=current_lang)
                                        if not audio_chunk:
                                            # Retry once reusing existing sentence text without generating new conversational response
                                            audio_chunk = await tts_service.synthesize_speech_async(clean_voice_sent, language=current_lang)

                                        if audio_chunk and len(audio_chunk) > 100:
                                            chunk_b64 = base64.b64encode(audio_chunk).decode('utf-8')
                                            first_audio_ms = (time.perf_counter() - t_llm_start) * 1000.0
                                            if not first_audio_recorded:
                                                tracker.record("tts_first_audio_ms", first_audio_ms)
                                                tracker.mark("first_audio_at")
                                                first_audio_recorded = True

                                            logger.info(
                                                f"[TEST_TTS_PLAYBACK] session_id={session_id} chunk_index={chunk_index} "
                                                f"first_audio_ms={first_audio_ms:.1f} tts_ms={(time.perf_counter() - t_tts_c_start)*1000:.1f} "
                                                f"audio_bytes={len(audio_chunk)} text='{clean_voice_sent[:50]}...'"
                                            )

                                            await websocket.send_json({"type": "state", "state": "AI_SPEAKING"})
                                            await websocket.send_json({
                                                "type": "audio_chunk",
                                                "audio": chunk_b64,
                                                "text": sentence,
                                                "chunk_index": chunk_index,
                                                "is_first": (chunk_index == 0),
                                                "turn_id": turn_id,
                                                "message_id": message_id,
                                                "conversation_id": session_id
                                            })
                                            chunk_index += 1
                                    except Exception as tts_err:
                                        logger.warning(f"[VOICE STREAMING TTS] Chunk {chunk_index} synthesis failed: {tts_err}")

                    # Flush any remaining text in sentence buffer
                    remaining_clean = clean_markdown_for_voice(sentence_buffer.strip())
                    if remaining_clean and len(remaining_clean) >= 3:
                        logger.info(
                            f"[DIAG_STAGE_C_FINAL_SERVICE_CHUNK] turn_id={turn_id} chunk_index={chunk_index} (final_buffer) "
                            f"sentence={remaining_clean!r}"
                        )
                        t_tts_c_start = time.perf_counter()
                        try:
                            audio_chunk = await tts_service.synthesize_speech_async(remaining_clean, language=current_lang)
                            if not audio_chunk:
                                audio_chunk = await tts_service.synthesize_speech_async(remaining_clean, language=current_lang)

                            if audio_chunk and len(audio_chunk) > 100:
                                chunk_b64 = base64.b64encode(audio_chunk).decode('utf-8')
                                first_audio_ms = (time.perf_counter() - t_llm_start) * 1000.0
                                if not first_audio_recorded:
                                    tracker.record("tts_first_audio_ms", first_audio_ms)
                                    tracker.mark("first_audio_at")
                                    first_audio_recorded = True

                                logger.info(
                                    f"[TEST_TTS_PLAYBACK] session_id={session_id} chunk_index={chunk_index} (final) "
                                    f"first_audio_ms={first_audio_ms:.1f} tts_ms={(time.perf_counter() - t_tts_c_start)*1000:.1f} "
                                    f"audio_bytes={len(audio_chunk)} text='{remaining_clean[:50]}...'"
                                )

                                await websocket.send_json({"type": "state", "state": "AI_SPEAKING"})
                                await websocket.send_json({
                                    "type": "audio_chunk",
                                    "audio": chunk_b64,
                                    "text": sentence_buffer.strip(),
                                    "chunk_index": chunk_index,
                                    "is_first": (chunk_index == 0),
                                    "turn_id": turn_id,
                                    "message_id": message_id,
                                    "conversation_id": session_id
                                })
                                chunk_index += 1
                        except Exception as tts_err:
                            logger.warning(f"[VOICE STREAMING TTS] Final chunk synthesis failed: {tts_err}")

                    response_text = full_response_text.strip()
                    response_text = deduplicate_repeated_sentences(clean_markdown_for_voice(response_text))
                    session_history.append({"role": "model", "text": response_text})

                except Exception as e:
                    logger.error(f"[TEST_LLM_RESPONSE] LLM streaming pipeline failed: {e}")
                    if current_lang == "ta":
                        response_text = "எங்கள் வரவேற்பாளர் பிரிவை தொடர்பு கொண்டதற்கு நன்றி. உங்களுக்கு எவ்வாறு உதவலாம்?"
                    else:
                        response_text = "Thank you for contacting our reception desk. How may I assist you?"

                total_llm_ms = (time.perf_counter() - t_llm_start) * 1000.0
                tracker.record("llm_total_ms", total_llm_ms)
                tracker.mark("llm_completed_at")

                first_tok_val = tracker.get_metric('llm_first_token_ms', 0.0) if hasattr(tracker, 'get_metric') else getattr(tracker, 'metrics', {}).get('llm_first_token_ms', 0.0)
                logger.info(
                    f"[TEST_LLM_RESPONSE] session_id={session_id} "
                    f"first_token_ms={first_tok_val or 0:.1f} "
                    f"total_ms={total_llm_ms:.1f} chunks_synthesized={chunk_index} "
                    f"text='{response_text[:80].replace(chr(10), ' ')}...'"
                )

                norm_full = _normalize_speech(response_text)
                if norm_full:
                    recent_ai_utterances.append(norm_full)
                    if len(recent_ai_utterances) > 15:
                        recent_ai_utterances = recent_ai_utterances[-15:]
                last_ai_speech_time = time.time()

                logger.info(
                    f"[DIAG_STAGE_D_BACKEND_TO_WS] turn_id={turn_id} message_id={message_id} "
                    f"event_type=response text_len={len(response_text)} text={response_text!r}"
                )
                # Send final complete text to UI
                await websocket.send_json({
                    "type": "response",
                    "text": response_text,
                    "turn_id": turn_id,
                    "message_id": message_id,
                    "conversation_id": session_id,
                    "is_final": True
                })

                # If no chunks were successfully synthesized (e.g. streaming fallback), synthesize whole response
                if chunk_index == 0:
                    clean_voice_text = clean_markdown_for_voice(response_text)
                    if clean_voice_text:
                        try:
                            audio_resp = await tts_service.synthesize_speech_async(clean_voice_text, language=current_lang)
                            if not audio_resp:
                                audio_resp = await tts_service.synthesize_speech_async(clean_voice_text, language=current_lang)

                            if audio_resp and len(audio_resp) > 100:
                                audio_b64 = base64.b64encode(audio_resp).decode('utf-8')
                                await websocket.send_json({"type": "state", "state": "AI_SPEAKING"})
                                await websocket.send_json({
                                    "type": "audio",
                                    "audio": audio_b64,
                                    "text": response_text,
                                    "turn_id": turn_id,
                                    "message_id": message_id,
                                    "conversation_id": session_id
                                })
                        except Exception as e:
                            logger.error(f"Voice fallback TTS synthesis failed: {e}")

                turn_tracker.record_turn(turn_id, message_id, transcription, response_text)

                telemetry = tracker.log_summary()
                await websocket.send_json({"type": "telemetry", "data": telemetry})
                logger.info(
                    f"\n[VOICE_TURN_COMPLETE]"
                    f"\n  session_id={session_id}"
                    f"\n  utterance_id={utterance_id}"
                    f"\n  intent={intent_name}"
                    f"\n  state_after={booking_session.state}"
                    f"\n  llm_ms={total_llm_ms:.1f}"
                )
                _log_turn(session_id, transcription, response_text, intent_name, booking_session, emotion=emotion)
                await _handle_post_response_state(websocket, booking_session, session_id, current_lang, recent_ai_utterances)

    except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError) as e:
        logger.info(f"[TEST_WEBSOCKET] Voice WebSocket disconnected ({e}): session_id={session_id}")
    except Exception as e:
        logger.error(f"[TEST_WEBSOCKET] WebSocket error: {e}")
    finally:
        try:
            org_id = getattr(booking_session, "organization_id", None) if 'booking_session' in locals() and booking_session else None
            if org_id is not None:
                asyncio.create_task(asyncio.to_thread(agent_engine.generate_session_summary, session_id, org_id))
        except Exception as _e:
            logger.warning(f"Could not trigger post-call summary: {_e}")
        booking_state_service.remove_session(session_id)
