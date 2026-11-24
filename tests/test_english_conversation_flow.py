"""
Test Suite: English Conversation Flow, Duplicate Greeting & Response Repeating Fixes.

Verifies:
1. Initial opening greeting plays only once and is never repeated in subsequent turns.
2. Greeting return to "Good afternoon", "Hello", etc. responds with a natural, standalone greeting acknowledgement.
3. Connection check "Hello? Are you there?" returns a single reassurance without prefix compounding or replaying greetings.
4. Genuine inquiries ("What are the products and services you have?", "Can you explain your services?") route to RAG and are answered using knowledge base context without greeting replays.
5. "Can I arrange a team call?" is recognized as callback/sales lead collection intent.
6. Multiple "Hello?" statements vary naturally without loops or compounding prefixes.
7. Strict preservation of Tamil conversation intelligence and language switching.
"""

import pytest
import os
import sys

# Ensure backend root is in sys.path
backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_root not in sys.path:
    sys.path.insert(0, backend_root)

from src.services.nlp_service import (
    nlp_service,
    INTENT_GREETING,
    INTENT_CONNECTION_REASSURANCE,
    INTENT_PRODUCT_INFORMATION,
    INTENT_SERVICE_INFORMATION,
    INTENT_CALLBACK,
    INTENT_CHANGE_LANGUAGE
)
from src.services.ai_service import ai_service
from src.services.rag_service import rag_service
from src.services.language_service import language_service


@pytest.fixture(scope="module", autouse=True)
def setup_nlp():
    nlp_service.initialize_model()


def test_english_1_initial_greeting_followed_by_good_afternoon():
    """English Test 1: Caller says 'Good afternoon.' after initial greeting."""
    history = [
        {"role": "assistant", "text": "Good afternoon! Thank you for calling Aminoid AI. How may I assist you?"}
    ]
    session_state = {"greeting_sent": True, "state": "IDLE"}

    intent = nlp_service.detect_intent("Good afternoon.").intent
    assert intent == INTENT_GREETING

    resp = ai_service._fast_rule_response(
        "Good afternoon.",
        intent=intent,
        organization_id=1,
        is_ongoing=True,
        booking_session=session_state,
        session_history=history,
        conversation_language="en"
    )

    assert resp is not None
    # 1. Natural greeting acknowledgment
    assert "good afternoon" in resp.lower()
    # 2. Company intro must NOT be repeated
    assert "thank you for calling" not in resp.lower()
    assert "aminoid ai" not in resp.lower()
    # 3. Must NOT concatenate old response
    assert "how may i assist you?" not in resp.lower() or resp.count("?") <= 1
    assert "yes, i'm right here!" not in resp.lower()


def test_english_2_connection_reassurance():
    """English Test 2: Caller says 'Hello? Are you there?' -> Single clean reassurance."""
    history = [
        {"role": "assistant", "text": "Good afternoon! Thank you for calling Aminoid AI. How may I assist you?"},
        {"role": "user", "text": "Good afternoon."},
        {"role": "assistant", "text": "Good afternoon! How can I help you today?"}
    ]
    session_state = {"greeting_sent": True, "state": "IDLE"}

    intent = nlp_service.detect_intent("Hello? Are you there?").intent
    assert intent == INTENT_CONNECTION_REASSURANCE

    resp = ai_service._fast_rule_response(
        "Hello? Are you there?",
        intent=intent,
        organization_id=1,
        is_ongoing=True,
        booking_session=session_state,
        session_history=history,
        conversation_language="en"
    )

    assert resp is not None
    # Single clean reassurance
    assert "right here" in resp.lower() or "can hear you" in resp.lower() or "here with you" in resp.lower()
    # Must NOT compound "Yes, I'm right here!" multiple times
    assert resp.lower().count("right here") <= 1
    assert resp.lower().count("here with you") <= 1
    # Must NOT replay the company greeting
    assert "thank you for calling" not in resp.lower()
    assert "aminoid ai" not in resp.lower()


def test_english_3_product_service_inquiry():
    """English Test 3: Caller asks 'What are the products and services you have?'"""
    history = [
        {"role": "assistant", "text": "Good afternoon! Thank you for calling Aminoid AI. How may I assist you?"},
        {"role": "user", "text": "Good afternoon."},
        {"role": "assistant", "text": "Good afternoon! How can I help you today?"}
    ]
    session_state = {"greeting_sent": True, "state": "IDLE"}

    intent = nlp_service.detect_intent("What are the products and services you have?").intent
    assert intent in (INTENT_PRODUCT_INFORMATION, INTENT_SERVICE_INFORMATION)

    # Fast path must be bypassed (return None) so RAG/LLM handles it
    fast_resp = ai_service._fast_rule_response(
        "What are the products and services you have?",
        intent=intent,
        organization_id=1,
        is_ongoing=True,
        booking_session=session_state,
        session_history=history,
        conversation_language="en"
    )
    assert fast_resp is None

    # Full LLM response generation with RAG context
    rag_context = rag_service.retrieve_context("What are the products and services you have?", organization_id=1)
    llm_resp = ai_service.generate_response(
        text="What are the products and services you have?",
        context=rag_context,
        session_history=history,
        intent=intent,
        conversation_language="en",
        organization_id=1,
        conversation_state=session_state
    )

    assert llm_resp is not None and len(llm_resp) > 0
    # Must NOT replay opening greeting
    assert "Good afternoon! Thank you for calling Aminoid AI. How may I assist you?" not in llm_resp
    # Must contain active knowledge base products/services
    llm_lower = llm_resp.lower()
    print("DEBUG LLM RESP:", repr(llm_resp))
    assert any(term in llm_lower for term in ["crm", "hrms", "erp", "website", "software", "project management", "survey", "ai", "solution", "voice agent", "application", "business"])


def test_english_4_detailed_product_explanation():
    """English Test 4: Caller asks 'Can you explain your services?'"""
    intent = nlp_service.detect_intent("Can you explain your services?").intent
    assert intent in (INTENT_SERVICE_INFORMATION, INTENT_PRODUCT_INFORMATION)

    session_state = {"greeting_sent": True, "state": "IDLE"}
    history = [
        {"role": "assistant", "text": "Good afternoon! How can I help you today?"}
    ]

    fast_resp = ai_service._fast_rule_response(
        "Can you explain your services?",
        intent=intent,
        organization_id=1,
        is_ongoing=True,
        booking_session=session_state,
        session_history=history,
        conversation_language="en"
    )
    assert fast_resp is None


def test_english_5_sales_callback_request():
    """English Test 5: Caller asks 'Can I arrange a team call?'"""
    intent = nlp_service.detect_intent("Can I arrange a team call?").intent
    assert intent == INTENT_CALLBACK

    # Also test variants
    assert nlp_service.detect_intent("I want to speak with your team").intent == INTENT_CALLBACK
    assert nlp_service.detect_intent("Arrange a team call please").intent == INTENT_CALLBACK


def test_english_6_multiple_hello_repetition_prevention():
    """English Test 6: Multiple 'Hello?' statements vary naturally without prefix compounding."""
    history = [
        {"role": "assistant", "text": "Good afternoon! Thank you for calling Aminoid AI. How may I assist you?"}
    ]
    session_state = {"greeting_sent": True, "state": "IDLE"}

    # Turn 1
    r1 = ai_service._fast_rule_response("Hello?", intent="greeting", is_ongoing=True, booking_session=session_state, session_history=history)
    assert r1 is not None
    assert "thank you for calling" not in r1.lower()
    history.append({"role": "user", "text": "Hello?"})
    history.append({"role": "assistant", "text": r1})

    # Turn 2
    r2 = ai_service._fast_rule_response("Hello, are you there?", intent="connection_reassurance", is_ongoing=True, booking_session=session_state, session_history=history)
    assert r2 is not None
    assert "thank you for calling" not in r2.lower()
    # Must NOT compound prefixes
    assert r2.lower().count("right here") <= 1
    assert r2.lower().count("here with you") <= 1
    history.append({"role": "user", "text": "Hello, are you there?"})
    history.append({"role": "assistant", "text": r2})

    # Turn 3
    r3 = ai_service._fast_rule_response("Hello?", intent="greeting", is_ongoing=True, booking_session=session_state, session_history=history)
    assert r3 is not None
    assert "thank you for calling" not in r3.lower()
    assert r3.lower().count("right here") <= 1


def test_tamil_1_native_tamil_greeting():
    """Tamil Regression Test 1: Caller says 'வணக்கம்.' -> Natural spoken Tamil response."""
    history = [
        {"role": "assistant", "text": "வணக்கம்! Aminoid AI-க்கு அழைத்ததற்கு நன்றி. நான் உங்களுக்கு எப்படி உதவ முடியும்?"}
    ]
    session_state = {"greeting_sent": True, "state": "IDLE", "conversation_language": "ta"}

    resp = ai_service._fast_rule_response(
        "வணக்கம்.",
        intent="greeting",
        organization_id=1,
        is_ongoing=True,
        booking_session=session_state,
        session_history=history,
        conversation_language="ta"
    )

    assert resp is not None
    assert any(c in resp for c in ["வணக்கம்", "சொல்லுங்க", "help", "உதவி"])
    # Zero English intrusion in primary flow
    assert "good afternoon" not in resp.lower()


def test_tamil_2_explicit_language_switch():
    """Tamil Regression Test 2: Explicit language switch request."""
    lang, is_switch = language_service.check_explicit_language_switch("தமிழ்ல பேசுங்க.")
    assert is_switch is True
    assert lang == "ta"

    conf = language_service.get_language_switch_confirmation("ta")
    assert conf is not None
    assert "தமிழ்" in conf or "சொல்லுங்க" in conf or "வணக்கம்" in conf


def test_tamil_3_tamil_conversation_continuity():
    """Tamil Regression Test 3: Audio check in Tamil returns clear Tamil response."""
    session_state = {"greeting_sent": True, "state": "IDLE", "conversation_language": "ta"}
    resp = ai_service._fast_rule_response(
        "உங்களுக்கு கேட்கிறதா?",
        intent="connection_reassurance",
        organization_id=1,
        is_ongoing=True,
        booking_session=session_state,
        session_history=[],
        conversation_language="ta"
    )
    assert resp is not None
    assert any(w in resp for w in ["voice", "தெளிவா", "கேக்குது", "help", "சொல்லுங்க"])


def test_compound_audio_check_with_product_query():
    """Verify compound queries like 'Can you hear me? What are your products?' prioritize product query."""
    intent_res = nlp_service.detect_intent("Can you hear me? What are your products?")
    assert intent_res.intent == INTENT_PRODUCT_INFORMATION

    session_state = {"greeting_sent": True, "state": "IDLE"}
    fast_resp = ai_service._fast_rule_response(
        "Can you hear me? What are your products?",
        intent=intent_res.intent,
        organization_id=1,
        is_ongoing=True,
        booking_session=session_state
    )
    assert fast_resp is None, "Compound business question must bypass fast-path to RAG/LLM"
