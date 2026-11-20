import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.services.rag_service import rag_service
from src.services.ai_service import ai_service
import json

query = "Okay, what are the products are available in Aminadi A I?"
doc_id = "6"
org_id = 1

context = rag_service.retrieve_context(
    query=query,
    n_results=5,
    organization_id=org_id,
    document_id=doc_id,
    scope="inbound"
)

print(f"=== STAGE 1: RAG RETRIEVED CONTEXT (len={len(context)}) ===")
print(context[:400])

conv_state = {
    "active_doc_id": "6",
    "business_name": "Aminoid AI",
    "business_type": "SOFTWARE",
    "booking_type": "SALES_DEMO",
    "current_stage": "IDLE",
    "current_intent": "product_information",
    "collected_fields": {},
    "missing_fields": []
}

session_history = [
    {"role": "model", "text": "Hello! I'm MURI, the AI assistant for Aminoid AI. I can help with AI voice agents, chatbots, automation, business applications, websites and custom software. May I know your name?"}
]

print("\n=== STAGE 2: CALLING AI SERVICE GENERATE_RESPONSE_STREAM ===")
chunks = []
for idx, chunk in enumerate(ai_service.generate_response_stream(
    text=query,
    context=context,
    session_history=session_history,
    intent="product_information",
    emotion="neutral",
    conversation_language="en",
    organization_id=org_id,
    conversation_state=conv_state
)):
    chunks.append(chunk)
    print(f"Chunk {idx} (len={len(chunk)}): {repr(chunk)}")

full_accumulated = "".join(chunks)
print("\n=== STAGE 3: FULL ACCUMULATED STREAM ===")
print("Length:", len(full_accumulated))
print("Text:\n", full_accumulated)
