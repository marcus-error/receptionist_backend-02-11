import sys, os
sys.path.insert(0, 'receptionist_backend-02-11')

from src.services.rag_service import rag_service
from src.services.ai_service import ai_service
from src.sockets.voice_socket import _contextualize_voice_query, _filter_procedural_chunks, _extract_complete_sentences

query = "Okay, what are the products are available in Aminadi A I?"
doc_id = "6"
org_id = 1

# What was session history at turn 2?
# Turn 1 was greeting:
greeting_text = rag_service.get_organization_greeting(org_id, document_id=doc_id)
print("Turn 1 Greeting:", greeting_text)

session_history = [
    {"role": "model", "text": greeting_text or "Good morning! Thank you for calling Aminoid AI. How may I assist you?"}
]

contextualized_query = _contextualize_voice_query(query, session_history, org_id, doc_id)
raw_context = rag_service.retrieve_context(
    query=contextualized_query,
    n_results=5,
    organization_id=org_id,
    intent="product_information",
    document_id=doc_id,
    scope="inbound"
)
filtered_context = _filter_procedural_chunks(raw_context, query)

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

print("\n--- INITIATING STREAM ---")
stream = ai_service.generate_response_stream(
    text=query,
    context=filtered_context,
    session_history=session_history,
    intent="product_information",
    emotion="neutral",
    conversation_language="en",
    organization_id=org_id,
    conversation_state=conv_state
)

full_text = ""
sentence_buffer = ""
tts_chunks = []

for idx, chunk in enumerate(stream):
    print(f"\n[STREAM CHUNK {idx}] len={len(chunk)}: {repr(chunk)}")
    full_text += chunk
    sentence_buffer += chunk
    
    sentences, remaining = _extract_complete_sentences(sentence_buffer)
    if sentences:
        sentence_buffer = remaining
        for s in sentences:
            tts_chunks.append(s)
            print(f"  -> Extracted completed TTS sentence {len(tts_chunks)-1}: {repr(s)}")

if sentence_buffer.strip():
    tts_chunks.append(sentence_buffer.strip())
    print(f"  -> Extracted final TTS sentence {len(tts_chunks)-1}: {repr(sentence_buffer.strip())}")

print("\n=== TOTAL STREAM RESULT ===")
print("Full Text Length:", len(full_text))
print("Full Text:\n", full_text)
print("\nTTS Chunks count:", len(tts_chunks))
for i, c in enumerate(tts_chunks):
    print(f"TTS Chunk {i}: {repr(c)}")
