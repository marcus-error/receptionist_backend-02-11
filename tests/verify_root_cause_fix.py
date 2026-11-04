import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config.database import SessionLocal
from src.models.knowledge_doc import KnowledgeDocument, KnowledgeDocStatus
from src.services.rag_service import rag_service

def verify():
    print("=== 1. Test Inbound Document Resolution for Org 1 ===")
    rag_service.clear_cache()
    inbound_doc_id = rag_service._resolve_active_document_id(organization_id=1, scope="inbound")
    print(f"Inbound doc_id: {inbound_doc_id}")
    assert inbound_doc_id == "2", f"Expected '2', got {inbound_doc_id}"

    print("\n=== 2. Test Outbound Document Resolution for Org 1 ===")
    rag_service.clear_cache()
    outbound_doc_id = rag_service._resolve_active_document_id(organization_id=1, scope="outbound")
    print(f"Outbound doc_id: {outbound_doc_id}")
    assert outbound_doc_id == "2", f"Expected '2', got {outbound_doc_id}"

    print("\n=== 3. Test Greeting Resolution for Org 1 (Inbound) ===")
    rag_service.clear_cache()
    greeting_in, doc_id_in, title_in = rag_service.get_active_greeting(organization_id=1, scope="inbound")
    print(f"Inbound Greeting: '{greeting_in}'")
    print(f"Doc ID: {doc_id_in}, Title: '{title_in}'")
    assert "Nurse O' Clock" in greeting_in, f"Greeting does not mention Nurse O' Clock: {greeting_in}"
    assert doc_id_in == 2, f"Expected doc_id 2, got {doc_id_in}"

    print("\n=== 4. Test Greeting Resolution for Org 1 (Outbound) ===")
    rag_service.clear_cache()
    greeting_out, doc_id_out, title_out = rag_service.get_active_greeting(organization_id=1, scope="outbound")
    print(f"Outbound Greeting: '{greeting_out}'")
    print(f"Doc ID: {doc_id_out}, Title: '{title_out}'")
    assert "Nurse O' Clock" in greeting_out, f"Greeting does not mention Nurse O' Clock: {greeting_out}"
    assert doc_id_out == 2, f"Expected doc_id 2, got {doc_id_out}"

    print("\n=== 5. Test Master Context Retrieval for Org 1 ===")
    rag_service.clear_cache()
    ctx = rag_service.get_organization_master_context(organization_id=1, document_id="2")
    print(f"Master Context length: {len(ctx)}")
    assert "Nurse O' Clock" in ctx, "Context missing Nurse O' Clock"

    print("\n=== 6. Test RAG Retrieval for Org 1 Service Question ===")
    rag_service.clear_cache()
    rag_res = rag_service.retrieve_context(
        query="What services do you provide?",
        organization_id=1,
        document_id="2",
        scope="inbound"
    )
    print(f"RAG result length: {len(rag_res)}")
    assert len(rag_res) > 0, "RAG result was empty!"
    assert any(term in rag_res.lower() for term in ["nursing", "physiotherapy", "assisted living", "elderly"]), "RAG result missing services!"

    print("\n=== ALL ROOT CAUSE RESOLUTION CHECKS PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    verify()
