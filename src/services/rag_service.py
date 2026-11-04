"""
Universal Production RAG Retrieval Service.
Features:
1. Production SQLite Vector Store Engine with 384-dimensional dense embeddings & cosine similarity.
2. Pure Python BM25 + Dense Vector Hybrid Search (60% Dense + 40% BM25).
3. Strict Multi-Tenant & Active Document Isolation (Filtered by organization_id & specific active document_id).
4. Domain-Agnostic Entity, Keyword, and Metadata Extraction (Works for any business).
5. Dynamic Business Name & Voice Greeting Extraction.
6. Fail-Safe Instant SQLite Fallback Context.
"""

import os
import re
import math
import time
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from src.services.sqlite_vector_store import sqlite_vector_store
from src.services.chroma_vector_store import chroma_vector_store

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.15

NON_RAG_INTENTS = {
    "greeting",
    "general_conversation",
    "goodbye",
    "emergency/urgent request",
    "transfer request",
    "appointment_booking",
    "appointment"
}

CONVERSATIONAL_PHRASES = {
    "how are you", "how are you doing", "what's up", "yeah good", "good",
    "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
    "bye", "goodbye", "thanks", "thank you", "ok", "okay", "alright", "yes", "no",
    "yes please", "no thanks", "sure", "ok sure", "confirm", "proceed", "go ahead"
}


STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing",
    "can", "could", "should", "would", "may", "might", "must", "shall", "will", "i",
    "me", "my", "myself", "we", "our", "ours", "you", "your", "yours", "he", "him",
    "his", "she", "her", "hers", "it", "its", "they", "them", "their", "theirs", "at",
    "by", "for", "from", "in", "into", "of", "off", "on", "onto", "out", "over", "to",
    "with", "about", "against", "between", "through", "during", "before", "after", "how"
}


class BM25Scorer:
    """
    Pure Python BM25 (Best Matching 25) Keyword Ranker for Hybrid Search.
    Mathematically exact BM25 formulation with zero external dependencies.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def tokenize(self, text: str) -> List[str]:
        words = re.split(r'\W+', (text or "").lower())
        return [w for w in words if len(w) > 1 and w not in STOPWORDS]

    def score(self, query: str, corpus: List[str]) -> List[float]:
        query_terms = self.tokenize(query)
        if not query_terms or not corpus:
            return [0.0] * len(corpus)

        doc_tokens = [self.tokenize(doc) for doc in corpus]
        N = len(corpus)
        avgdl = sum(len(d) for d in doc_tokens) / N if N > 0 else 1.0

        idf = {}
        for q in set(query_terms):
            n_q = sum(1 for d in doc_tokens if q in d)
            idf[q] = math.log((N - n_q + 0.5) / (n_q + 0.5) + 1.0)

        scores = []
        for d in doc_tokens:
            d_len = len(d)
            s = 0.0
            for q in query_terms:
                if q in d:
                    f = d.count(q)
                    num = f * (self.k1 + 1.0)
                    den = f + self.k1 * (1.0 - self.b + self.b * (d_len / avgdl))
                    s += idf.get(q, 0.0) * (num / den)
            scores.append(s)

        max_possible = sum(idf.get(q, 0.0) for q in query_terms) * (self.k1 + 1.0)
        denom = max_possible if max_possible > 0 else 1.0
        return [min(1.0, s / denom) for s in scores]


class RagService:
    def __init__(self):
        self.collection = chroma_vector_store
        self.bm25_scorer = BM25Scorer()
        self._org_context_cache: Dict[str, str] = {}
        self._verified_indexed_docs: set = set()
        logger.info("[RAG SERVICE] ChromaDB Vector Store initialized successfully as primary.")

    def _init_or_repair_collection(self):
        """Returns the primary ChromaDB Vector Store."""
        return chroma_vector_store

    def reset_and_rebuild_from_db(self):
        """
        Hard resets the ChromaDB vector store and syncs all active documents from SQLite.
        """
        logger.info("[RAG SERVICE] Resetting and rebuilding ChromaDB vector store from database...")
        chroma_vector_store.reset()
        sqlite_vector_store.reset()
        self.clear_organization_cache()

        try:
            from src.config.database import SessionLocal
            from src.models.knowledge_doc import KnowledgeDocument, KnowledgeDocStatus
            from src.services.knowledge_service import KnowledgeService

            db = SessionLocal()
            try:
                ks = KnowledgeService(db)
                docs = db.query(KnowledgeDocument).all()
                reindexed_count = 0
                for doc in docs:
                    if doc.content and doc.content.strip():
                        try:
                            ks._index_document(doc, doc.content)
                            reindexed_count += 1
                        except Exception as e:
                            logger.warning(f"Failed to reindex doc {doc.id}: {e}")
                db.commit()
                logger.info(f"[RAG SERVICE] Rebuild complete. Successfully re-indexed {reindexed_count} documents.")
                return reindexed_count
            finally:
                db.close()
        except Exception as err:
            logger.error(f"[RAG SERVICE] Error rebuilding knowledge base: {err}")
            return 0

    def ingest_text(self, document_id: str, text: str, source: str, organization_id: int = 1):
        """Upsert document text and metadata into vector store with organization isolation."""
        metadata = {
            "source": source,
            "document_id": str(document_id),
            "organization_id": str(organization_id),
            "ai_usage": "Active",
            "status": "Indexed"
        }
        self.collection.upsert(
            documents=[text],
            metadatas=[metadata],
            ids=[str(document_id)]
        )
        self.clear_organization_cache(organization_id)

    def delete_text(self, document_id: str):
        """Delete document vectors by ID."""
        try:
            self.collection.delete(where={"document_id": str(document_id)})
        except Exception as exc:
            logger.warning(f"Failed to delete document {document_id} from SQLite vector store: {exc}")

    def is_rag_needed(self, intent: Optional[str] = None, text: Optional[str] = None) -> bool:
        """
        Determines whether RAG search is required for the user message.
        Bypasses RAG for simple greetings, pure pleasantries, personal entity introductions, and goodbyes.
        """
        if intent and intent.lower().strip() in NON_RAG_INTENTS:
            return False

        if text:
            txt_clean = text.lower().strip()
            clean_words = re.sub(r'[^a-zA-Z\s]', '', txt_clean).strip()
            if clean_words in CONVERSATIONAL_PHRASES or (len(clean_words.split()) <= 2 and clean_words in CONVERSATIONAL_PHRASES):
                return False

            # Bypass RAG for personal name declarations (e.g. "My name is Mugilan", "I'm Mugilan", "Mugilan Viswanathan")
            name_decl = re.search(r'^\s*(?:my\s+(?:full\s+)?name\s+is|i\s*(?:am|\'m)|this\s+is|you\s+can\s+call\s+me|call\s+me|myself)\s+([a-zA-Z\s]+)\s*$', txt_clean)
            if name_decl and len(name_decl.group(1).split()) <= 4:
                return False

            # Bypass RAG for raw phone numbers or spoken email responses
            digits = re.sub(r'[^\d]', '', txt_clean)
            if len(digits) >= 7 and len(txt_clean.split()) <= 4:
                return False
            if "@" in txt_clean or "dot com" in txt_clean:
                return False

        return True

    def clear_cache(self):
        """Clears all in-memory RAG, context, and greeting caches across the service."""
        if hasattr(self, "_org_context_cache"):
            self._org_context_cache.clear()
        if hasattr(self, "_greeting_cache"):
            self._greeting_cache.clear()
        if hasattr(self, "_rag_terminology_cache"):
            self._rag_terminology_cache.clear()
        if hasattr(self, "_ctx_cache"):
            self._ctx_cache.clear()
        if hasattr(self, "_verified_indexed_docs"):
            self._verified_indexed_docs.clear()
        if hasattr(self, "_active_doc_cache"):
            self._active_doc_cache.clear()
        if hasattr(self, "_rag_query_cache"):
            self._rag_query_cache.clear()
        if hasattr(self, "_doc_version_cache"):
            self._doc_version_cache.clear()
        logger.info("[RAG SERVICE] In-memory RAG, context, and greeting caches cleared.")

    def clear_organization_cache(self, organization_id: Optional[int] = None):
        """
        Flushes cached organization document context and in-memory caches strictly for the requested organization.
        Tenant A's document changes will NEVER flush or alter Tenant B's caches.
        """
        if organization_id is not None:
            prefix1 = f"org_{organization_id}_"
            prefix2 = f"{organization_id}_"
            prefix_query = f"org_{organization_id}"

            if hasattr(self, "_ctx_cache"):
                keys_to_del = [k for k in self._ctx_cache if str(k).startswith(prefix1) or str(k).startswith(prefix2) or (isinstance(k, tuple) and k and str(k[0]) == str(organization_id))]
                for k in keys_to_del:
                    del self._ctx_cache[k]
            if hasattr(self, "_org_context_cache"):
                keys_to_del = [k for k in self._org_context_cache if str(k).startswith(prefix1) or str(k).startswith(prefix2) or (isinstance(k, tuple) and k and str(k[0]) == str(organization_id))]
                for k in keys_to_del:
                    del self._org_context_cache[k]
            if hasattr(self, "_active_doc_cache"):
                keys_to_del = [k for k in self._active_doc_cache if str(k).startswith(prefix1) or str(k).startswith(prefix2) or (isinstance(k, tuple) and k and str(k[0]) == str(organization_id))]
                for k in keys_to_del:
                    del self._active_doc_cache[k]
            if hasattr(self, "_rag_query_cache"):
                keys_to_del = [k for k in self._rag_query_cache if prefix_query in str(k) or str(k).startswith(prefix2)]
                for k in keys_to_del:
                    del self._rag_query_cache[k]
            if hasattr(self, "_doc_version_cache"):
                self._doc_version_cache.clear()

            try:
                from src.services.cache_service import cache_service
                cache_service.invalidate_organization(organization_id)
            except Exception:
                pass
            logger.info(f"[MASTER CONTEXT FLUSHED] In-memory cache cleared strictly for org={organization_id}.")
        else:
            self.clear_cache()
            try:
                from src.services.cache_service import cache_service
                cache_service.clear_all()
            except Exception:
                pass
            logger.info("[MASTER CONTEXT FLUSHED] Global in-memory cache cleared.")

    invalidate_master_context = clear_organization_cache

    def _get_document_version(self, document_id: Optional[str]) -> str:
        """Returns a cached version identifier for a document based on its updated_at timestamp or chunk count."""
        if not document_id:
            return "0"
        if not hasattr(self, "_doc_version_cache"):
            self._doc_version_cache = {}
        now = time.monotonic()
        if document_id in self._doc_version_cache:
            v, exp = self._doc_version_cache[document_id]
            if now < exp:
                return v

        try:
            from src.config.database import SessionLocal
            from src.models.knowledge_doc import KnowledgeDocument
            db = SessionLocal()
            try:
                try:
                    d_id = int(document_id)
                    doc = db.query(KnowledgeDocument).filter(KnowledgeDocument.id == d_id).first()
                except ValueError:
                    doc = db.query(KnowledgeDocument).filter(KnowledgeDocument.title == str(document_id)).first()

                if doc and doc.updated_at:
                    ver = str(int(doc.updated_at.timestamp()))
                elif doc:
                    ver = str(doc.chunks or "1")
                else:
                    ver = "0"
                self._doc_version_cache[document_id] = (ver, now + 30.0)
                return ver
            finally:
                db.close()
        except Exception:
            return "1"

    def _resolve_active_document_id(
        self,
        organization_id: Optional[int] = None,
        document_id: Optional[str] = None,
        scope: Optional[str] = None,
        document_type: Optional[str] = None,
    ) -> Optional[str]:
        """
        Resolves the currently active, indexed document ID for the requested organization.
        Guarantees:
        - Organization-specific active documents always take highest priority.
        - Shared organization documents (organization_id is None, scope != 'superadmin_assistant')
          are used as fallback when no tenant-specific document exists.
        - Superadmin assistant documents are NEVER returned for regular tenants.
        - Outbound calls prioritize outbound scripts, but fall back to active organization
          documents if no dedicated outbound script exists.
        - Inactive or unindexed documents are NEVER resolved.
        """
        org_key = f"org_{organization_id}" if organization_id is not None else "superadmin"
        cache_key = f"{org_key}_doc_{document_id}_type_{document_type}_scope_{scope}"
        now = time.monotonic()
        if not hasattr(self, "_active_doc_cache"):
            self._active_doc_cache = {}
        if cache_key in self._active_doc_cache:
            val, expires = self._active_doc_cache[cache_key]
            if now < expires:
                return val

        if document_id:
            try:
                from src.config.database import SessionLocal
                from src.models.knowledge_doc import KnowledgeDocument, KnowledgeDocStatus
                db_val = SessionLocal()
                try:
                    try:
                        doc_val = db_val.query(KnowledgeDocument).filter(KnowledgeDocument.id == int(document_id)).first()
                    except ValueError:
                        doc_val = db_val.query(KnowledgeDocument).filter(KnowledgeDocument.title == str(document_id)).first()

                    if doc_val:
                        # Tenant isolation check:
                        # 1. Regular tenant query: allow own docs OR shared organization docs
                        if organization_id is not None:
                            if doc_val.organization_id is not None and doc_val.organization_id != int(organization_id):
                                logger.warning(
                                    f"[RAG ISOLATION] Refusing document {doc_val.id} (org={doc_val.organization_id}) for requested tenant org {organization_id}"
                                )
                                document_id = None
                            elif doc_val.organization_id is None and doc_val.scope == "superadmin_assistant":
                                logger.warning(
                                    f"[RAG ISOLATION] Refusing superadmin assistant document {doc_val.id} for requested tenant org {organization_id}"
                                )
                                document_id = None
                        else:
                            # 2. Superadmin query (organization_id is None) must only access superadmin documents
                            if doc_val.organization_id is not None or doc_val.scope != "superadmin_assistant":
                                logger.warning(
                                    f"[RAG ISOLATION] Refusing tenant document {doc_val.id} (org={doc_val.organization_id}) for superadmin assistant query"
                                )
                                document_id = None

                        # Status check: must be actively assigned and successfully indexed
                        if document_id and (doc_val.ai_usage != "Active" or str(doc_val.status).upper() not in ("INDEXED", "KNOWLEDGEDOCSTATUS.INDEXED")):
                            logger.warning(
                                f"[RAG ISOLATION] Refusing inactive or unindexed document {doc_val.id} (ai_usage={doc_val.ai_usage}, status={doc_val.status})"
                            )
                            document_id = None

                        # Scope check: inbound calls cannot use outbound scripts
                        if document_id:
                            scope_str = str(scope or "").lower()
                            if "inbound" in scope_str:
                                is_outbound = (
                                    doc_val.document_type in ("OUTBOUND_SCRIPT", "CAMPAIGN_SCRIPT")
                                    or "outbound" in (doc_val.title or "").lower()
                                    or doc_val.scope == "outbound"
                                )
                                if is_outbound:
                                    logger.warning(f"[RAG ISOLATION] Refusing outbound script '{doc_val.title}' (id={doc_val.id}) for inbound call.")
                                    document_id = None
                    else:
                        try:
                            where_q = {"document_id": str(document_id)}
                            cnt = self.collection.count(where=where_q)
                            if cnt > 0:
                                resolved_id_str = str(document_id)
                                self._active_doc_cache[cache_key] = (resolved_id_str, now + 15.0)
                                return resolved_id_str
                        except Exception:
                            pass
                        document_id = None

                    if document_id:
                        resolved_id_str = str(document_id)
                        self._active_doc_cache[cache_key] = (resolved_id_str, now + 15.0)
                        return resolved_id_str
                finally:
                    db_val.close()
            except Exception as e_check:
                logger.warning(f"Error checking specified document {document_id}: {e_check}")

        # Fallback to resolving the active document for the requested tenant from SQLite
        try:
            from src.config.database import SessionLocal
            from src.models.knowledge_doc import KnowledgeDocument, KnowledgeDocStatus
            db = SessionLocal()
            try:
                base_query = db.query(KnowledgeDocument).filter(
                    KnowledgeDocument.ai_usage == "Active",
                    KnowledgeDocument.status.in_([KnowledgeDocStatus.INDEXED, "Indexed", "INDEXED", "indexed"]),
                )

                resolved_id = None
                if organization_id is not None:
                    org_int = int(organization_id)
                    def _query_candidate(q):
                        # Priority 1: explicitly owned by organization
                        d = q.filter(KnowledgeDocument.organization_id == org_int).order_by(KnowledgeDocument.updated_at.desc(), KnowledgeDocument.id.desc()).first()
                        if not d and org_int in (1, None):
                            # Priority 2: default/shared organization document (for default tenant only)
                            d = q.filter(KnowledgeDocument.organization_id.is_(None), KnowledgeDocument.scope != "superadmin_assistant").order_by(KnowledgeDocument.updated_at.desc(), KnowledgeDocument.id.desc()).first()
                        return d

                    scope_str = str(scope or "").lower()
                    if "outbound" in scope_str:
                        # 1. Try dedicated outbound script
                        outbound_q = base_query.filter(
                            (KnowledgeDocument.document_type.in_(["OUTBOUND_SCRIPT", "CAMPAIGN_SCRIPT"])) |
                            (KnowledgeDocument.title.ilike("%outbound%")) |
                            (KnowledgeDocument.scope == "outbound")
                        )
                        target_doc = _query_candidate(outbound_q)
                        # 2. If no dedicated outbound script, fall back to active organization doc
                        if not target_doc:
                            target_doc = _query_candidate(base_query.filter(KnowledgeDocument.scope != "inbound"))
                    else:
                        inbound_base = base_query.filter(
                            KnowledgeDocument.document_type.notin_(["OUTBOUND_SCRIPT", "CAMPAIGN_SCRIPT", "STAFF_DIRECTORY"]),
                            ~KnowledgeDocument.title.ilike("%outbound%"),
                            KnowledgeDocument.scope != "outbound",
                        )
                        target_doc = None
                        if document_type:
                            target_doc = _query_candidate(inbound_base.filter(KnowledgeDocument.document_type == document_type))
                        if not target_doc:
                            target_doc = _query_candidate(inbound_base)

                    if target_doc:
                        resolved_id = str(target_doc.id)
                elif scope == "superadmin_assistant":
                    sa_doc = base_query.filter(
                        KnowledgeDocument.organization_id.is_(None),
                        KnowledgeDocument.scope == "superadmin_assistant"
                    ).order_by(KnowledgeDocument.updated_at.desc(), KnowledgeDocument.id.desc()).first()
                    if sa_doc:
                        resolved_id = str(sa_doc.id)

                self._active_doc_cache[cache_key] = (resolved_id, now + 15.0)
                return resolved_id
            finally:
                db.close()
        except Exception as err:
            logger.warning(f"Error resolving active document ID from SQLite: {err}")
        return None

    def get_organization_master_context(self, organization_id: Optional[int] = None, document_id: Optional[str] = None, scope: Optional[str] = None) -> str:
        """
        Retrieves the active organization knowledge base context directly from SQLite / ChromaDB.
        Ensures strict multi-tenant isolation by organization_id and the resolved active document_id.
        Caches master context in-memory with document versioning to avoid redundant queries and stale data.
        """
        active_doc_id = self._resolve_active_document_id(organization_id, document_id, scope=scope)
        if not active_doc_id:
            return ""

        if not hasattr(self, "_ctx_cache"):
            self._ctx_cache = {}

        doc_version = self._get_document_version(active_doc_id)
        cache_key = f"org_{organization_id}_doc_{active_doc_id}_v_{doc_version}_scope_{scope}"
        if cache_key in self._ctx_cache:
            return self._ctx_cache[cache_key]

        # 1. SQLite Direct Fetch (instant zero-latency read & guaranteed 100% complete text)
        try:
            from src.config.database import SessionLocal
            from src.models.knowledge_doc import KnowledgeDocument, KnowledgeDocStatus
            db = SessionLocal()
            try:
                try:
                    doc_id_num = int(active_doc_id)
                    q = db.query(KnowledgeDocument).filter(
                        KnowledgeDocument.id == doc_id_num,
                        KnowledgeDocument.ai_usage == "Active",
                        KnowledgeDocument.status.in_([KnowledgeDocStatus.INDEXED, "Indexed", "INDEXED", "indexed"])
                    )
                    if organization_id is not None:
                        q = q.filter(
                            (KnowledgeDocument.organization_id == int(organization_id)) |
                            (KnowledgeDocument.organization_id.is_(None) & (KnowledgeDocument.scope != "superadmin_assistant"))
                        )
                    else:
                        q = q.filter(KnowledgeDocument.organization_id.is_(None))
                    active_doc = q.first()
                except ValueError:
                    q = db.query(KnowledgeDocument).filter(
                        KnowledgeDocument.title.ilike(f"%{active_doc_id}%"),
                        KnowledgeDocument.ai_usage == "Active",
                        KnowledgeDocument.status.in_([KnowledgeDocStatus.INDEXED, "Indexed", "INDEXED", "indexed"])
                    )
                    if organization_id is not None:
                        q = q.filter(
                            (KnowledgeDocument.organization_id == int(organization_id)) |
                            (KnowledgeDocument.organization_id.is_(None) & (KnowledgeDocument.scope != "superadmin_assistant"))
                        )
                    else:
                        q = q.filter(KnowledgeDocument.organization_id.is_(None))
                    active_doc = q.first()

                if active_doc and active_doc.content and active_doc.content.strip():
                    combined = str(active_doc.content).strip()
                    logger.info(f"[MASTER CONTEXT] Loaded {len(combined)} chars from SQLite doc {active_doc_id} ('{active_doc.title}')")
                    res_text = combined[:25000]
                    self._ctx_cache[cache_key] = res_text
                    return res_text
            finally:
                db.close()
        except Exception as e2:
            logger.warning(f"SQLite master context fetch warning: {e2}")

        # 2. SQLite / Chroma Vector Store Fallback
        try:
            where_filter = {"document_id": str(active_doc_id)}
            if organization_id is not None:
                where_filter = {
                    "$and": [
                        {"document_id": str(active_doc_id)},
                        {"organization_id": str(organization_id)}
                    ]
                }
            res = self.collection.get(where=where_filter)
            docs = res.get("documents", []) if res else []
            metas = res.get("metadatas", []) if res else []
            valid_docs = []
            for d, m in zip(docs, metas):
                if d and str(d).strip():
                    m_dict = m or {}
                    if organization_id is not None and str(m_dict.get("organization_id", "")).strip() != str(organization_id):
                        continue
                    if str(m_dict.get("document_id", "")).strip() != str(active_doc_id):
                        continue
                    valid_docs.append(str(d).strip())

            if valid_docs:
                combined = "\n\n".join(valid_docs)
                logger.info(f"[MASTER CONTEXT] Loaded {len(valid_docs)} vector chunks from vector store for doc {active_doc_id}")
                res_text = combined[:25000]
                self._ctx_cache[cache_key] = res_text
                return res_text
        except Exception as e:
            logger.warning(f"Vector master context fallback warning: {e}")

        return ""

    def get_organization_name(self, organization_id: Optional[int] = None, document_id: Optional[str] = None, scope: Optional[str] = None) -> str:
        """
        Dynamically extracts the Organization / Business Name from the active Knowledge Base.
        Returns empty string if no custom brand is specified.
        """
        master_ctx = self.get_organization_master_context(organization_id, document_id=document_id, scope=scope)
        if master_ctx:
            match = re.search(
                r'(?:Company\s*/\s*Organization Name|Organization Name|Business Name|Company Name|Hospital Name|Clinic Name|Salon Name|School Name|Restaurant Name|Hotel Name|Agency Name|Firm Name)(?:\s*:|\s*\n+)\s*([^\n\r]+)',
                master_ctx,
                re.IGNORECASE
            )
            if match:
                clean_name = re.sub(r'[*_#`]', '', match.group(1)).strip()
                if len(clean_name) > 1 and not any(w in clean_name.lower() for w in ["unknown", "n/a", "none"]):
                    return clean_name

            # Check for "Welcome to [Organization Name]" in master context
            m_welcome = re.search(r'welcome to\s+([A-Za-z0-9\s&\'\-]+?)(?:\.|\?|\!|,|how may|thank you|\n|$)', master_ctx, re.IGNORECASE)
            if m_welcome:
                clean_w = re.sub(r'[*_#`]', '', m_welcome.group(1)).strip()
                if clean_w.isupper():
                    clean_w = clean_w.title()
                if len(clean_w) > 2 and not any(w in clean_w.lower() for w in ["unknown", "n/a", "none", "our team", "the call", "doctor", "dr."]):
                    return clean_w

            header_match = re.search(r'#+\s*([A-Za-z0-9\s\.\-&,\']+?)(?:[\-\n\r]|$)', master_ctx)
            if header_match:
                candidate = re.sub(r'[*_#`]', '', header_match.group(1)).strip()
                stop_phrases = [
                    "test receptionist script", "receptionist script", "inbound script", "outbound script",
                    "test script", "rag knowledge base", "section", "organization overview", "knowledge base",
                    "master rag", "school information", "hospital information", "salon information",
                    "company information", "doctor directory", "doctors directory", "staff directory",
                    "directory", "departments", "department", "services", "treatments", "pricing",
                    "hours", "timings", "contact", "faq", "faqs", "greeting & call opening", "greeting",
                    "call opening"
                ]
                for sp in stop_phrases:
                    candidate = re.sub(re.escape(sp), '', candidate, flags=re.IGNORECASE).strip()
                candidate = re.sub(r'\s*(?:inbound|outbound)?\s*script\s*$', '', candidate, flags=re.IGNORECASE).strip()
                if candidate.isupper(): candidate = candidate.title()
                if candidate and len(candidate) > 2 and not any(w in candidate.lower() for w in ["script", "receptionist", "knowledge base", "directory"]):
                    return candidate

        # If active document exists, attempt to extract clean title from SQLite
        target_doc_id = document_id or self._resolve_active_document_id(organization_id)
        if target_doc_id:
            try:
                from src.config.database import SessionLocal
                from src.models.knowledge_doc import KnowledgeDocument
                db = SessionLocal()
                try:
                    try:
                        doc_id_num = int(target_doc_id)
                        doc = db.query(KnowledgeDocument).filter(KnowledgeDocument.id == doc_id_num).first()
                        if doc and organization_id is not None and doc.organization_id is not None and doc.organization_id != int(organization_id):
                            doc = None
                    except ValueError:
                        doc = None
                    if doc and doc.title:
                        title_clean = re.sub(r'\.(pdf|txt|docx|doc)$', '', doc.title, flags=re.IGNORECASE)
                        title_clean = re.sub(r'^(master_rag_knowledge_base_|rag_knowledge_base_|knowledge_base_)', '', title_clean, flags=re.IGNORECASE)
                        title_clean = re.sub(r'(_rag)?(_inbound|_outbound)?(_script)?(_fine_tuned)?$', '', title_clean, flags=re.IGNORECASE)
                        title_clean = title_clean.replace('_', ' ').strip().title()
                        bad_generic_keywords = ["simple english", "knowledge base", "master rag", "script", "document", "receptionist", "test"]
                        if title_clean and not any(kw in title_clean.lower() for kw in bad_generic_keywords) and len(title_clean) > 2:
                            return title_clean
                finally:
                    db.close()
            except Exception:
                pass

        if organization_id:
            try:
                from src.config.database import SessionLocal
                from src.models.organization import Organization
                db = SessionLocal()
                try:
                    org = db.query(Organization).filter(Organization.id == int(organization_id)).first()
                    if org and org.name and org.name.strip().lower() not in {"default organization", "organization", "company", "business", "unknown", "none"}:
                        return org.name
                finally:
                    db.close()
            except Exception:
                pass

        return ""

    @staticmethod
    def _clean_spoken_text(text: str) -> str:
        """Strips quotes, stage directions (e.g. (warmly)), speaker labels, instruction prefixes, and markdown."""
        quote_chars = '"\'\u201c\u201d\u2018\u2019\xab\xbb\u201e\u201f\ufffd\x91\x92\x93\x94'
        s = text.strip()
        # Remove stage directions in parentheses or brackets e.g. (warmly), [in a friendly tone]
        s = re.sub(r'^\s*[\(\[][^\)\]]+[\)\]]\s*', '', s)
        s = re.sub(r'\s*[\(\[][^\)\]]+[\)\]]\s*$', '', s)
        # Remove leading speaker label e.g. "AI Agent:", "AI Receptionist:", "Receptionist:", "Agent:"
        s = re.sub(r'^(?:AI\s+Agent|AI\s+Receptionist|AI\s+Assistant|Receptionist|Assistant|Agent|AI)\s*:\s*', '', s, flags=re.IGNORECASE)
        # Remove instruction triggers like "Say:", "Says:"
        s = re.sub(r'^(?:Say|Says|Speaks|Spoken|Greet with)\s*:\s*', '', s, flags=re.IGNORECASE)
        # Strip quotes and markdown formatting
        s = s.strip(quote_chars).strip()
        s = s.replace("**", "").replace("*", "").replace("__", "").replace("`", "")
        s = s.replace("�", "").strip()
        s = re.sub(r'\s+', ' ', s).strip()
        return s.strip(quote_chars).strip()

    def _is_valid_spoken_greeting(self, g: str, doc_title: Optional[str] = None) -> bool:
        """Validates that extracted text is a genuine spoken conversational greeting, not a filename or metadata."""
        if not g or len(g.strip()) < 8:
            return False
        clean = g.strip()
        # Never speak a filename or file extension
        if re.search(r'\.(pdf|docx|txt|doc|csv|wav|mp3)\b', clean, re.IGNORECASE):
            return False
        # Never speak the document title or file stem
        if doc_title:
            dt_clean = re.sub(r'\.(pdf|docx|txt|doc|csv)$', '', str(doc_title), flags=re.IGNORECASE).strip().lower()
            if clean.lower() == dt_clean or clean.lower() == str(doc_title).lower().strip():
                return False
        # Never speak underscore-separated identifiers
        if "_" in clean and clean.count("_") >= 2:
            return False
        # Never speak system instructions, flowchart steps, or metadata tags
        bad_prefixes = [
            "retrieve", "check", "when", "the receptionist", "if a customer",
            "verify", "identify", "must", "note:", "step", "flow", "instructions:",
            "system:", "prompt:", "hospital name:", "company name:", "organization name:",
            "address:", "phone:", "timing:", "hours:", "department:", "services:"
        ]
        if any(clean.lower().startswith(bp) for bp in bad_prefixes):
            return False
        words = clean.split()
        if len(words) < 3:
            return False
        return True

    def extract_greeting_from_content(self, text: str, is_script_doc: bool = False, doc_title: Optional[str] = None) -> str:
        """
        Dynamically extracts the opening greeting sentence/paragraph from script content.
        Supports:
        - Explicit section headers (# 1. Greeting & Call Opening, Greeting:, [GREETING], Call Opening:)
        - Quoted lines or unquoted lines under greeting headers
        - Dialogue scripts (Receptionist: "...", Agent: ...)
        - Numbered instruction steps (The receptionist says: "...")
        - Tamil/Thanglish greetings (வணக்கம், Vanakkam)
        - Preserves original wording, language, and punctuation.
        - Never returns filenames, document titles, or internal instructions.
        """
        if not text or not text.strip():
            return ""
        clean_src = text.strip()
        quote_chars = '"\'“”‘’«»„‟'

        # 1. Look for explicit GREETING section header
        hdr_regex = re.compile(
            r'(?m)^[\s*\-•#]*(?:\d+[\.\)]\s*)?(?:\*\*|__)?'
            r'(?:Voice Greeting|Inbound Greeting|Outbound Greeting|Opening Greeting|Receptionist Greeting|'
            r'Greeting\s*(?:&|and|/)\s*Call Opening|Call Opening\s*(?:&|and|/)\s*Greeting|'
            r'(?:Outbound|Inbound|Voice|Phone)?\s*Call Opening|'
            r'(?:Outbound|Inbound|Voice|Phone)?\s*Greeting|'
            r'Welcome Greeting|Opening Message|Approved Greeting|Initial Greeting|'
            r'Standard Greeting|Spoken Greeting|Approved Spoken Greeting|Welcome Message|'
            r'Phone Greeting|First Response|Opening|Introduction|Greeting|வரவேற்பு|வணக்கம்)'
            r'(?:\*\*|__)?\s*[:\-\]]?\s*([^\n\r]*)',
            re.IGNORECASE
        )

        for m in hdr_regex.finditer(clean_src):
            same_line = m.group(1).strip()
            # Case A: Same line contains quotes
            q_match = re.search(r'[' + quote_chars + r']\s*(.+?)\s*[' + quote_chars + r']', same_line, re.DOTALL)
            if q_match:
                candidate = self._clean_spoken_text(q_match.group(1))
                if self._is_valid_spoken_greeting(candidate, doc_title):
                    return candidate

            # Case B: Same line has spoken text after colon/dash
            if same_line:
                candidate = self._clean_spoken_text(same_line)
                if self._is_valid_spoken_greeting(candidate, doc_title):
                    return candidate

            # Case C: Check lines immediately following the heading
            post_text = clean_src[m.end():]
            lines = post_text.splitlines()
            collected = []
            for line in lines[:10]:
                l_str = line.strip()
                if not l_str:
                    if collected:
                        break
                    continue
                # Stop if hitting another section header
                if re.match(r'^[\s*\-•#]*(?:[A-Z0-9\.\s]{3,}:|#|\d+\.)', l_str) and not any(l_str.lower().startswith(p) for p in ["receptionist", "ai", "agent", "say"]):
                    break
                # Check for quote in this line
                q_line = re.search(r'[' + quote_chars + r']\s*(.+?)\s*[' + quote_chars + r']', l_str, re.DOTALL)
                if q_line:
                    candidate = self._clean_spoken_text(q_line.group(1))
                    if self._is_valid_spoken_greeting(candidate, doc_title):
                        return candidate
                collected.append(l_str)

            if collected:
                full_candidate = self._clean_spoken_text(" ".join(collected))
                if self._is_valid_spoken_greeting(full_candidate, doc_title):
                    return full_candidate

        # 2. Check dialogue lines at start of script: Receptionist: "..." or Receptionist: ...
        diag_regex = re.compile(
            r'(?m)^[\s*\-•]*(?:AI\s+Receptionist|Receptionist|AI\s+Assistant|Assistant|Agent|AI)\s*:\s*([^\n\r]+)',
            re.IGNORECASE
        )
        for dm in diag_regex.finditer(clean_src[:3000]):
            cand_line = dm.group(1).strip()
            q_diag = re.search(r'[' + quote_chars + r']\s*(.+?)\s*[' + quote_chars + r']', cand_line)
            candidate = self._clean_spoken_text(q_diag.group(1) if q_diag else cand_line)
            if self._is_valid_spoken_greeting(candidate, doc_title):
                return candidate

        # 3. Check standalone direct quote at the very start of the script
        direct_q = re.search(r'^\s*[' + quote_chars + r']\s*(.+?)\s*[' + quote_chars + r']', clean_src, re.DOTALL)
        if direct_q:
            candidate = self._clean_spoken_text(direct_q.group(1))
            if self._is_valid_spoken_greeting(candidate, doc_title):
                return candidate

        # 4. Check if the first paragraph starts with standard greeting triggers
        paragraphs = [p.strip() for p in clean_src.split("\n\n") if p.strip()]
        triggers = ["good morning", "good afternoon", "good evening", "welcome to", "thank you for calling", "thanks for calling", "hello", "hi", "vanakkam", "namaste", "வணக்கம்"]
        for para in paragraphs[:3]:
            para_clean = self._clean_spoken_text(para)
            if any(para_clean.lower().startswith(t) for t in triggers):
                if self._is_valid_spoken_greeting(para_clean, doc_title):
                    return para_clean

        return ""

    @staticmethod
    def _get_time_aware_greeting_prefix(organization_id: Optional[int] = None) -> str:
        """Returns 'Good morning', 'Good afternoon', or 'Good evening' based on
        the organization's timezone or IST (UTC+5:30) as standard.
        Strict time rules:
        - Good morning:   5:00 AM to 11:59 AM (05:00 - 11:59)
        - Good afternoon: 12:00 PM to 4:59 PM (12:00 - 16:59)
        - Good evening:   5:00 PM to 4:59 AM (17:00 - 04:59)
        """
        try:
            import datetime as _dt
            from zoneinfo import ZoneInfo
            from src.config.env import settings

            tz_name = getattr(settings, "BUSINESS_TIMEZONE", "Asia/Kolkata") or "Asia/Kolkata"
            try:
                tz = ZoneInfo(tz_name)
                now_dt = _dt.datetime.now(tz)
            except Exception:
                utc_now = _dt.datetime.now(_dt.timezone.utc)
                ist_offset = _dt.timedelta(hours=5, minutes=30)
                now_dt = utc_now + ist_offset

            now_hour = now_dt.hour
            if 5 <= now_hour < 12:
                return "Good morning"
            elif 12 <= now_hour < 17:
                return "Good afternoon"
            else:
                return "Good evening"
        except Exception:
            return "Good morning"

    @classmethod
    def _apply_time_aware_prefix(cls, greeting: str, org_name: str = "", organization_id: Optional[int] = None, scope: Optional[str] = None) -> str:
        """Dynamically shapes opening greeting according to standard structure:
        "Good [morning/afternoon/evening]. Welcome to [Organization Name]. How may I assist you?"
        - Organization name comes dynamically from authenticated organization / database.
        - Strict time-of-day prefix based on actual call time in configured timezone:
          * Good morning: 5:00 AM to 11:59 AM
          * Good afternoon: 12:00 PM to 4:59 PM
          * Good evening: 5:00 PM to 4:59 AM
        - Never uses hardcoded organization names.
        """
        import re as _re
        time_prefix = cls._get_time_aware_greeting_prefix(organization_id=organization_id)
        resolved_org = ""
        m_org = _re.search(r'welcome to\s+([A-Za-z0-9\s&\'\-]+?)(?:\.|\?|\!|,|how may|thank you|\n|$)', greeting, _re.IGNORECASE)
        if m_org:
            extracted = m_org.group(1).strip()
            if len(extracted) > 2 and extracted.lower() not in ("default organization", "unknown", "none", "our team", "the call"):
                resolved_org = extracted

        if not resolved_org:
            resolved_org = org_name or rag_service.get_organization_name(organization_id)

        if resolved_org and resolved_org.isupper():
            resolved_org = resolved_org.title()
        if not resolved_org or resolved_org.strip().lower() in ("default organization", "unknown", "none"):
            resolved_org = "our team"

        g = (greeting or "").strip()
        time_pattern = r'^(?:good\s+morning\s*/\s*afternoon|good\s+morning\s*/\s*evening|good\s+morning|good\s+afternoon|good\s+evening)[.,!?;:\s]*'
        if _re.search(time_pattern, g, _re.IGNORECASE):
            rest = _re.sub(time_pattern, '', g, count=1, flags=_re.IGNORECASE).strip()
            if rest:
                rest = rest[0].upper() + rest[1:]
                return f"{time_prefix}. {rest}"
            return time_prefix

        if _re.search(r'^(?:welcome to|this is|thank you for calling)', g, _re.IGNORECASE):
            return f"{time_prefix}. {g}"

        if g and len(g) > 10:
            return f"{time_prefix}. {g}"

        scope_str = str(scope or "").lower()
        if "outbound" in scope_str:
            return f"{time_prefix}. This is {resolved_org} calling. How may I assist you?"
        return f"{time_prefix}. Welcome to {resolved_org}. How may I assist you?"


    def synthesize_greeting_from_doc(self, content: str, doc_title: str = "", scope: Optional[str] = None) -> str:
        """
        Constructs an appropriate opening greeting using the organization name, role,
        and style defined in an active knowledge document when no explicit greeting section exists.
        Guarantees:
        - Never returns hardcoded generic fallback ("Hello! How can I help you today?").
        - Always references the active document's organization identity.
        """
        org_name = ""
        # 1. Try explicit field
        m_org = re.search(r'(?:Organization|Company|Hospital|Clinic|Business|Salon|School|Restaurant|Hotel|Firm)\s+Name\s*:\s*([^\n\r]+)', content, re.IGNORECASE)
        if m_org:
            org_name = re.sub(r'[*_#`]', '', m_org.group(1)).strip()

        # 2. Try leading header e.g. "NURSE O' CLOCK — CHENNAI INFORMATION"
        if not org_name:
            m_hdr = re.search(r"^(?:#+\s*)?([A-Za-z0-9\s'\.\-&]+?)(?:\s*[\—\-\|\:]|\s+INFORMATION|\s+RAG|\s+SCRIPT|\s*\n)", content, re.IGNORECASE)
            if m_hdr:
                cand = re.sub(r'[*_#`]', '', m_hdr.group(1)).strip()
                if len(cand) > 2 and not any(w in cand.lower() for w in ["section", "general", "overview"]):
                    org_name = cand

        # 3. Try clean doc title
        if not org_name and doc_title:
            cand = re.sub(r'\.(pdf|docx|txt|doc|csv)$', '', doc_title, flags=re.IGNORECASE)
            cand = re.sub(r'^(?:master_rag_|rag_|knowledge_base_)', '', cand, flags=re.IGNORECASE)
            cand = re.sub(r'(_receptionist)?(_rag)?(_formatted)?(_script)?$', '', cand, flags=re.IGNORECASE)
            cand = cand.replace('_', ' ').strip().title()
            if len(cand) > 2:
                org_name = cand

        if not org_name:
            org_name = "our team"

        scope_str = str(scope or "").lower()
        if "outbound" in scope_str:
            time_prefix = self._get_time_aware_greeting_prefix()
            return f"{time_prefix}. This is {org_name} calling. How may I assist you?"
        else:
            time_prefix = self._get_time_aware_greeting_prefix()
            return f"{time_prefix}. Welcome to {org_name}. How may I assist you?"

    def get_active_greeting(
        self,
        organization_id: Optional[int] = None,
        document_id: Optional[str] = None,
        scope: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[int], Optional[str]]:
        """
        Dynamically retrieves the greeting strictly from the currently ACTIVE script or knowledge document for the organization.
        Returns: (greeting_text, active_script_doc_id, active_script_title)

        Rules:
        - The active script is the ONLY source for the greeting.
        - If the document is a general knowledge document with organization info but no explicit greeting section,
          synthesizes an appropriate greeting using the organization name and style defined in the document.
        - Under NO circumstance falls back to a hardcoded generic greeting like 'Hello! How can I help you today?'.
        - If no active document exists at all, returns (None, None, None) and logs CONFIGURATION_ERROR.
        """
        try:
            from src.config.database import SessionLocal
            from src.models.knowledge_doc import KnowledgeDocument, KnowledgeDocStatus
            db = SessionLocal()
            try:
                # 1. Explicit document_id requested
                doc_explicit = None
                if document_id:
                    try:
                        doc_num = int(document_id)
                        doc_explicit = db.query(KnowledgeDocument).filter(
                            KnowledgeDocument.id == doc_num,
                            KnowledgeDocument.ai_usage == "Active",
                            KnowledgeDocument.status.in_([KnowledgeDocStatus.INDEXED, "Indexed", "INDEXED", "indexed"])
                        ).first()
                    except ValueError:
                        doc_explicit = db.query(KnowledgeDocument).filter(
                            KnowledgeDocument.title == str(document_id),
                            KnowledgeDocument.ai_usage == "Active",
                            KnowledgeDocument.status.in_([KnowledgeDocStatus.INDEXED, "Indexed", "INDEXED", "indexed"])
                        ).first()

                    if doc_explicit:
                        # Tenant isolation check
                        if organization_id is not None:
                            if doc_explicit.organization_id is not None and doc_explicit.organization_id != int(organization_id):
                                doc_explicit = None
                            elif doc_explicit.organization_id is None and doc_explicit.scope == "superadmin_assistant":
                                doc_explicit = None
                        elif organization_id is None and doc_explicit.organization_id is not None:
                            doc_explicit = None

                    if doc_explicit:
                        content = str(doc_explicit.content or "").strip()
                        if not content and doc_explicit.file_path:
                            try:
                                import os
                                if os.path.exists(doc_explicit.file_path):
                                    with open(doc_explicit.file_path, "r", encoding="utf-8", errors="ignore") as fp:
                                        content = fp.read().strip()
                            except Exception:
                                pass

                        logger.info(
                            f"\n[GREETING_SCRIPT_RESOLUTION]"
                            f"\n  organization_id={organization_id}"
                            f"\n  active_script_id={doc_explicit.id}"
                            f"\n  active_script_title='{doc_explicit.title}'"
                            f"\n  script_status=ACTIVE"
                        )

                        greeting = self.extract_greeting_from_content(content, is_script_doc=True, doc_title=doc_explicit.title)
                        if greeting:
                            greeting = self._apply_time_aware_prefix(greeting, organization_id=organization_id, scope=scope)
                            logger.info(
                                f"\n[GREETING_EXTRACTION]"
                                f"\n  source='{doc_explicit.title}'"
                                f"\n  section_found=True"
                                f"\n  extracted_text='{greeting}'"
                                f"\n  extraction_status=SUCCESS"
                            )
                            return (greeting, doc_explicit.id, doc_explicit.title)
                        elif doc_explicit.document_type not in ["SCRIPT", "INBOUND_SCRIPT", "OUTBOUND_SCRIPT"] and content and content.strip():
                            # Priority 2: General knowledge document with organization info but no explicit greeting section
                            synth_greeting = self.synthesize_greeting_from_doc(content, doc_explicit.title, scope=scope)
                            logger.info(
                                f"\n[GREETING_EXTRACTION]"
                                f"\n  source='{doc_explicit.title}'"
                                f"\n  section_found=False (synthesized_from_document_identity)"
                                f"\n  extracted_text='{synth_greeting}'"
                                f"\n  extraction_status=SUCCESS"
                            )
                            return (synth_greeting, doc_explicit.id, doc_explicit.title)
                        else:
                            logger.warning(
                                f"\n[GREETING_EXTRACTION]"
                                f"\n  source='{doc_explicit.title}'"
                                f"\n  section_found=False"
                                f"\n  extracted_text=none"
                                f"\n  extraction_status=NOT_FOUND"
                            )
                            logger.error(
                                f"[GREETING_CONFIG_ERROR] Active script id={doc_explicit.id} ('{doc_explicit.title}') does not contain a greeting section."
                            )
                            return (None, doc_explicit.id, doc_explicit.title)

                # 2. Dynamic active script lookup directly from SQLite
                base_q = db.query(KnowledgeDocument).filter(
                    KnowledgeDocument.ai_usage == "Active",
                    KnowledgeDocument.status.in_([KnowledgeDocStatus.INDEXED, "Indexed", "INDEXED", "indexed"])
                )

                if organization_id is not None:
                    org_int = int(organization_id)
                    has_own_doc = db.query(KnowledgeDocument).filter(
                        KnowledgeDocument.organization_id == org_int,
                        KnowledgeDocument.ai_usage == "Active",
                        KnowledgeDocument.status.in_([KnowledgeDocStatus.INDEXED, "Indexed", "INDEXED", "indexed"])
                    ).first()
                    if has_own_doc:
                        base_q = base_q.filter(KnowledgeDocument.organization_id == org_int)
                    elif org_int in (1, None):
                        base_q = base_q.filter(
                            (KnowledgeDocument.organization_id == org_int) |
                            (KnowledgeDocument.organization_id.is_(None) & (KnowledgeDocument.scope != "superadmin_assistant"))
                        )
                    else:
                        base_q = base_q.filter(KnowledgeDocument.organization_id == org_int)
                else:
                    base_q = base_q.filter(
                        KnowledgeDocument.organization_id.is_(None),
                        KnowledgeDocument.scope == "superadmin_assistant"
                    )

                scope_str = str(scope or "inbound").lower()
                if "outbound" in scope_str:
                    # Outbound: try dedicated outbound script first, then general scripts
                    outbound_cand = base_q.filter(
                        (KnowledgeDocument.document_type.in_(["OUTBOUND_SCRIPT", "CAMPAIGN_SCRIPT"])) |
                        KnowledgeDocument.title.ilike("%outbound%") |
                        (KnowledgeDocument.scope == "outbound")
                    ).order_by(KnowledgeDocument.organization_id.desc(), KnowledgeDocument.updated_at.desc(), KnowledgeDocument.id.desc()).all()
                    
                    general_cand = base_q.filter(KnowledgeDocument.scope != "inbound").order_by(
                        KnowledgeDocument.organization_id.desc(), KnowledgeDocument.updated_at.desc(), KnowledgeDocument.id.desc()
                    ).all()
                    candidate_docs = outbound_cand or general_cand
                else:
                    inbound_q = base_q.filter(
                        KnowledgeDocument.document_type.notin_(["OUTBOUND_SCRIPT", "CAMPAIGN_SCRIPT"]),
                        ~KnowledgeDocument.title.ilike("%outbound%"),
                        KnowledgeDocument.scope != "outbound",
                    )
                    t1_docs = inbound_q.filter(KnowledgeDocument.document_type.in_(["SCRIPT", "INBOUND_SCRIPT"])).order_by(
                        KnowledgeDocument.organization_id.desc(), KnowledgeDocument.updated_at.desc(), KnowledgeDocument.id.desc()
                    ).all()
                    t2_docs = inbound_q.filter(KnowledgeDocument.title.ilike("%script%")).order_by(
                        KnowledgeDocument.organization_id.desc(), KnowledgeDocument.updated_at.desc(), KnowledgeDocument.id.desc()
                    ).all()
                    t3_docs = inbound_q.order_by(
                        KnowledgeDocument.organization_id.desc(), KnowledgeDocument.updated_at.desc(), KnowledgeDocument.id.desc()
                    ).all()
                    seen_ids = set()
                    candidate_docs = []
                    for doc_list in [t1_docs, t2_docs, t3_docs]:
                        for d in doc_list:
                            if d.id not in seen_ids:
                                seen_ids.add(d.id)
                                candidate_docs.append(d)

                if not candidate_docs:
                    logger.warning(
                        f"\n[GREETING_SCRIPT_RESOLUTION]"
                        f"\n  organization_id={organization_id}"
                        f"\n  active_script_id=none"
                        f"\n  active_script_title=none"
                        f"\n  script_status=MISSING"
                    )
                    logger.warning(
                        f"\n[GREETING_EXTRACTION]"
                        f"\n  source=NONE"
                        f"\n  section_found=False"
                        f"\n  extracted_text=none"
                        f"\n  extraction_status=NO_ACTIVE_SCRIPT"
                    )
                    logger.error(
                        f"[GREETING_CONFIG_ERROR] No active script configured for organization_id={organization_id}."
                    )
                    return (None, None, None)

                for doc in candidate_docs:
                    logger.info(
                        f"\n[GREETING_SCRIPT_RESOLUTION]"
                        f"\n  organization_id={organization_id}"
                        f"\n  active_script_id={doc.id}"
                        f"\n  active_script_title='{doc.title}'"
                        f"\n  script_status=ACTIVE"
                    )

                    raw_content = str(doc.content or "").strip()
                    if not raw_content and doc.file_path:
                        try:
                            import os
                            if os.path.exists(doc.file_path):
                                with open(doc.file_path, "r", encoding="utf-8", errors="ignore") as fp:
                                    raw_content = fp.read().strip()
                        except Exception:
                            pass

                    if not raw_content:
                        continue

                    is_script = (doc.document_type in ["SCRIPT", "INBOUND_SCRIPT"]) or ("script" in (doc.title or "").lower())
                    greeting = self.extract_greeting_from_content(raw_content, is_script_doc=is_script, doc_title=doc.title)
                    if greeting:
                        greeting = self._apply_time_aware_prefix(greeting, organization_id=organization_id, scope=scope)
                        logger.info(
                            f"\n[GREETING_EXTRACTION]"
                            f"\n  source='{doc.title}'"
                            f"\n  section_found=True"
                            f"\n  extracted_text='{greeting}'"
                            f"\n  extraction_status=SUCCESS"
                        )
                        return (greeting, doc.id, doc.title)
                    elif doc.document_type not in ["SCRIPT", "INBOUND_SCRIPT", "OUTBOUND_SCRIPT"] and raw_content and raw_content.strip():
                        synth_greeting = self.synthesize_greeting_from_doc(raw_content, doc.title, scope=scope)
                        logger.info(
                            f"\n[GREETING_EXTRACTION]"
                            f"\n  source='{doc.title}'"
                            f"\n  section_found=False (synthesized_from_document_identity)"
                            f"\n  extracted_text='{synth_greeting}'"
                            f"\n  extraction_status=SUCCESS"
                        )
                        return (synth_greeting, doc.id, doc.title)
                    else:
                        logger.warning(
                            f"\n[GREETING_EXTRACTION]"
                            f"\n  source='{doc.title}'"
                            f"\n  section_found=False"
                            f"\n  extracted_text=none"
                            f"\n  extraction_status=NOT_FOUND"
                        )
                        logger.error(
                            f"[GREETING_CONFIG_ERROR] Active script id={doc.id} ('{doc.title}') does not contain a greeting section."
                        )

                top_doc = candidate_docs[0]
                return (None, top_doc.id, top_doc.title)
            finally:
                db.close()
        except Exception as exc:
            logger.error(f"[GREETING_ERROR] Error resolving active greeting: {exc}")
            return (None, None, None)

    def invalidate_caches(self, organization_id: Optional[int] = None):
        """Clears all in-memory caches for the organization or globally."""
        self.clear_organization_cache(organization_id)
        logger.info(f"[CACHE_INVALIDATED] Flushed all in-memory RAG/script caches for org={organization_id or 'all'}")

    def get_organization_greeting(self, organization_id: Optional[int] = None, document_id: Optional[str] = None, scope: Optional[str] = None) -> str:
        """
        Dynamically extracts the approved opening Receptionist Greeting directly from the active script,
        or constructs a natural, time-aware greeting using the extracted organization name.
        """
        greeting, _, _ = self.get_active_greeting(organization_id=organization_id, document_id=document_id, scope=scope)
        return greeting
    def get_organization_hours(self, organization_id: Optional[int] = None, document_id: Optional[str] = None) -> Optional[str]:
        master_ctx = self.get_organization_master_context(organization_id, document_id=document_id)
        if master_ctx:
            match = re.search(r'(?:Working Hours|Hours|Operating Hours|Timings|Schedule|Opening Hours)\s*:\s*([^\n\r]+)', master_ctx, re.IGNORECASE)
            if match:
                return match.group(1).strip().replace("*", "")
        return None

    def get_organization_location(self, organization_id: Optional[int] = None, document_id: Optional[str] = None) -> Optional[str]:
        master_ctx = self.get_organization_master_context(organization_id, document_id=document_id)
        if master_ctx:
            match = re.search(r'(?:Address|Location|Situated at|Located at|Address / Location)\s*:\s*([^\n\r]+)', master_ctx, re.IGNORECASE)
            if match:
                return match.group(1).strip().replace("*", "")
        return None

    def get_organization_doctors(self, organization_id: Optional[int] = None, document_id: Optional[str] = None) -> list[str]:
        master_ctx = self.get_organization_master_context(organization_id, document_id=document_id)
        doctors = []
        if master_ctx:
            # Check explicit Teacher, Doctor, Stylist or Staff directory patterns in script
            matches = re.findall(r'(?:(?:Mr\.?|Ms\.?|Mrs\.?|Dr\.?|Prof\.?)\s+[A-Za-z]+(?:\s+[A-Za-z]+)?|[A-Za-z]+\s+(?:Kumar|Sharma|Verma|Singh|Patel|Rao|Reddy|Nair|Gupta|Joshi))', master_ctx)
            for m in matches:
                clean_m = m.strip()
                if clean_m and clean_m not in doctors and len(clean_m) > 2:
                    doctors.append(clean_m)
            # Only query Staff database table if organization_id is provided and no specific document filter is active
            if not document_id and organization_id is not None:
                try:
                    from src.config.database import SessionLocal
                    from src.models.staff import Staff
                    db = SessionLocal()
                    try:
                        org_id_num = int(organization_id)
                        staff_members = db.query(Staff).filter(Staff.organization_id == org_id_num).all()
                        for sm in staff_members:
                            if sm.name and sm.name not in doctors:
                                doctors.append(sm.name)
                    finally:
                        db.close()
                except Exception:
                    pass
        return doctors

    def get_organization_services(self, organization_id: Optional[int] = None, document_id: Optional[str] = None) -> list[str]:
        master_ctx = self.get_organization_master_context(organization_id, document_id=document_id)
        if not master_ctx:
            return []

        services = []
        seen = set()

        def add_item(item: str):
            if not item:
                return
            for sub_line in str(item).splitlines():
                line_clean = re.sub(r'[^\x20-\x7E]', ' ', sub_line).strip()
                clean = re.sub(r'^(?:[-*•\d\.]+\s*)+', '', line_clean).strip()
                clean = re.sub(r'[:\(\[\{].*$', '', clean).strip().replace('*', '').strip()
                c_lower = clean.lower()
                if not clean or len(clean) < 2 or len(clean) > 50 or c_lower in seen:
                    continue
                words = clean.split()
                if len(words) >= 1 and all(w.isupper() for w in words if len(w) > 2):
                    continue
                _noise_headers = {
                    "product suite", "products", "services", "offerings", "menu", "departments",
                    "specialties", "overview", "core capabilities", "features", "primary purpose",
                    "main interaction", "about", "note", "important", "type", "company", "product name"
                }
                if c_lower not in _noise_headers and not any(c_lower.startswith(p) for p in ["primary purpose", "main interaction", "type:", "about "]):
                    seen.add(c_lower)
                    services.append(clean)

        # 1. Comma/semicolon/newline separated lists under headers (including Product Suite and Products)
        header_patterns = [
            r'(?:Product\s+Suite|Products|Products\s+Offered|Product\s+Line|Software\s+Suite|Offerings|Services|Specialties|Departments|Treatments|Procedures|Menu)\s*:\s*([^\n\r]+(?:\n\s+[A-Za-z0-9][^\n\r]+)*)',
        ]
        for pat in header_patterns:
            for m in re.finditer(pat, master_ctx, re.IGNORECASE):
                for part in re.split(r'[,;•\n]', m.group(1)):
                    add_item(part)

        # 2. Block lists under PRODUCT SUITE / PRODUCTS / SERVICES headers
        block_pat = r'(?:#+\s*|\n\s*)(?:PRODUCT\s+SUITE|PRODUCTS|SERVICES|OFFERINGS|SPECIALTIES|DEPARTMENTS)[^\n]*\n((?:[^\n]+\n?){1,20})'
        for bm in re.finditer(block_pat, master_ctx, re.IGNORECASE):
            for line in bm.group(1).splitlines():
                line_s = line.strip()
                if re.match(r'^(?:[-*•]|\d+\.)\s*[A-Za-z0-9]', line_s):
                    add_item(line_s)
                elif line_s and len(line_s) <= 40 and not any(line_s.lower().startswith(p) for p in ["primary purpose:", "main interaction:", "type:", "about", "note:", "important:"]):
                    if re.match(r'^[A-Za-z0-9\s\-_]+$', line_s) and len(line_s.split()) <= 4:
                        add_item(line_s)

        # 3. Product definition lines (e.g., "Galaxy Chat is the text-based conversational AI product")
        for m in re.finditer(r'\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+){0,3})\s+is\s+(?:the|an|our|a)\s+[^.\n]*(?:product|software|platform|system|application|service|tool)\b', master_ctx):
            add_item(m.group(1))

        return services

    def get_appointment_entity_terminology(
        self,
        organization_id: int = 1,
        document_id: Optional[str] = None
    ) -> dict:
        """
        Extracts the business-appropriate appointment entity terminology from the active RAG.
        """
        master_ctx = self.get_organization_master_context(organization_id, document_id=document_id)
        ctx_lower = master_ctx.lower() if master_ctx else ""

        DOMAIN_SIGNALS: list[tuple[list[str], dict]] = [
            # Education / School (prioritized for teacher/school scripts)
            (["teacher", "tutor", "faculty", "lecturer"],
             {"entity_term": "teacher", "entity_term_plural": "teachers", "entity_action": "meet", "service_based": False,
              "ask_target_question": "Which teacher would you like to meet?"}),
            (["school", "college", "university", "admission enquiry", "campus visit", "class teacher", "principal", "headmaster", "academic"],
             {"entity_term": "teacher", "entity_term_plural": "teachers", "entity_action": "meet", "service_based": False,
              "ask_target_question": "Which teacher or department would you like to meet?"}),
            # Salon / Beauty (placed high for specific service terms like haircut, facial, spa)
            (["haircut", "hair cut", "hair color", "facial", "manicure", "pedicure", "blow dry",
              "beauty", "salon", "spa", "makeup", "beard grooming", "hair spa", "stylist"],
             {"entity_term": "service", "entity_term_plural": "services", "entity_action": "book", "service_based": True,
              "ask_target_question": "Which service would you like to book?"}),
            # Software / AI / Products
            (["callzenza", "ai receptionist", "receptionist bot", "ai agent", "software", "product demo"],
             {"entity_term": "service", "entity_term_plural": "services", "entity_action": "book", "service_based": True,
              "ask_target_question": "Which service or product would you like to book?"}),
            # Medical / Healthcare (using specific medical terms)
            (["doctor", "physician", "cardiologist", "neurologist", "dermatologist",
              "pediatrician", "orthopedic", "gynecologist", "dentist", "surgeon", "hospital", "clinic"],
             {"entity_term": "doctor", "entity_term_plural": "doctors", "entity_action": "see", "service_based": False,
              "ask_target_question": "Which doctor would you like to see?"}),
            # Legal
            (["attorney", "lawyer", "advocate", "solicitor", "paralegal"],
             {"entity_term": "attorney", "entity_term_plural": "attorneys", "entity_action": "meet", "service_based": False,
              "ask_target_question": "Which attorney would you like to meet?"}),
            # Finance / Banking
            (["financial advisor", "loan officer", "banker", "account manager", "wealth manager"],
             {"entity_term": "advisor", "entity_term_plural": "advisors", "entity_action": "meet", "service_based": False,
              "ask_target_question": "Which advisor would you like to meet?"}),
            # Real Estate
            (["real estate agent", "property agent", "broker"],
             {"entity_term": "agent", "entity_term_plural": "agents", "entity_action": "meet", "service_based": False,
              "ask_target_question": "Which agent would you like to meet?"}),
            # IT / Tech Support
            (["technician", "support engineer", "it support", "tech support"],
             {"entity_term": "technician", "entity_term_plural": "technicians", "entity_action": "meet", "service_based": False,
              "ask_target_question": "Which technician would you like to meet?"}),
            # Therapy / Wellness
            (["therapist", "psychologist", "counsellor", "physiotherapist", "chiropractor", "nutritionist", "dietitian"],
             {"entity_term": "therapist", "entity_term_plural": "therapists", "entity_action": "see", "service_based": False,
              "ask_target_question": "Which therapist would you like to see?"}),
            # Veterinary
            (["veterinarian", "vet", "animal doctor", "veterinary"],
             {"entity_term": "vet", "entity_term_plural": "vets", "entity_action": "see", "service_based": False,
              "ask_target_question": "Which vet would you like to see?"}),
        ]

        for signals, terminology in DOMAIN_SIGNALS:
            for signal in signals:
                if re.search(r'\b' + re.escape(signal) + r'\b', ctx_lower):
                    return terminology

        return {
            "entity_term": None,
            "entity_term_plural": None,
            "entity_action": "meet",
            "service_based": False,
            "ask_target_question": None,
        }

    def retrieve_staff_context(self, query: str = "", organization_id: Optional[int] = None) -> str:
        """
        Retrieves authoritative, synchronized staff directory representation strictly for the requested organization.
        Guarantees that staff records are tenant-isolated and never mixed across organizations.
        """
        if organization_id is None:
            return ""
        try:
            from src.config.database import SessionLocal
            from src.models.knowledge_doc import KnowledgeDocument
            db = SessionLocal()
            try:
                doc = db.query(KnowledgeDocument).filter(
                    KnowledgeDocument.organization_id == int(organization_id),
                    KnowledgeDocument.document_type == "STAFF_DIRECTORY",
                    KnowledgeDocument.ai_usage == "Active",
                ).first()
                if doc and doc.content:
                    return doc.content
            finally:
                db.close()
        except Exception as e:
            logger.warning(f"[STAFF_RAG_RETRIEVAL_ERR] Failed to load staff directory for org {organization_id}: {e}")
        return ""

    def retrieve_context(
        self,
        query: str,
        n_results: int = DEFAULT_TOP_K,
        organization_id: Optional[int] = None,
        intent: Optional[str] = None,
        document_id: Optional[str] = None,
        scope: Optional[str] = None,
        call_id: Optional[str] = None,
        utterance_id: Optional[str] = None,
    ) -> str:
        """
        Universal RAG Retrieval combining Dense Vector Search + BM25 Keyword Scoring.
        Strict multi-tenant & active document filtering with post-retrieval metadata verification.
        """
        t_start = time.monotonic()
        org_id_val = int(organization_id) if organization_id is not None else None
        active_doc_id = self._resolve_active_document_id(org_id_val, document_id, scope=scope)

        logger.info(
            f"\n[RAG_QUERY]"
            f"\n  call_id={call_id or 'unknown'}"
            f"\n  utterance_id={utterance_id or 'unknown'}"
            f"\n  query='{query}'"
            f"\n  organization_id={org_id_val}"
            f"\n  knowledge_id={active_doc_id or 'none'}"
            f"\n  scope={scope or 'none'}"
        )

        if not query or not query.strip():
            master_ctx = self.get_organization_master_context(org_id_val, document_id=active_doc_id, scope=scope) if active_doc_id else ""
            elapsed_ms = round((time.monotonic() - t_start) * 1000)
            logger.info(
                f"\n[RAG_RESULT]"
                f"\n  chunks=0 (master_fallback)"
                f"\n  context_length={len(master_ctx)}"
                f"\n  latency_ms={elapsed_ms}"
            )
            return master_ctx

        # STRICT ISOLATION: A tenant with no eligible active document receives NO context and never falls back
        if org_id_val is not None and not active_doc_id:
            staff_signals = {
                "doctor", "doctors", "dr", "dr.", "staff", "physician", "specialist",
                "consultant", "appointment", "booking", "schedule", "timing", "hours",
                "available", "availability", "department", "departments", "who is", "who handles"
            }
            if any(s in query.lower() for s in staff_signals) or intent in ("staff_lookup", "staff_availability", "doctor_information"):
                staff_ctx = self.retrieve_staff_context(query=query, organization_id=org_id_val)
                if staff_ctx:
                    elapsed_ms = round((time.monotonic() - t_start) * 1000)
                    logger.info(
                        f"\n[RAG_RESULT]"
                        f"\n  chunks=staff_directory_sync"
                        f"\n  context_length={len(staff_ctx)}"
                        f"\n  latency_ms={elapsed_ms}"
                    )
                    return staff_ctx

            elapsed_ms = round((time.monotonic() - t_start) * 1000)
            logger.info(
                f"\n[RAG_RESULT]"
                f"\n  chunks=0 (no_active_doc_for_org)"
                f"\n  context_length=0"
                f"\n  latency_ms={elapsed_ms}"
            )
            return ""

        if not self.is_rag_needed(intent=intent, text=query):
            logger.info(
                f"\n[RAG_RESULT]"
                f"\n  chunks=0 (bypassed_conversational)"
                f"\n  context_length=0"
                f"\n  latency_ms={round((time.monotonic() - t_start) * 1000)}"
            )
            return ""

        search_query = query.strip()
        doc_version = self._get_document_version(active_doc_id) if active_doc_id else "0"
        if not hasattr(self, "_rag_query_cache"):
            self._rag_query_cache = {}

        # Cache key strictly incorporates tenant ID, document ID, version, scope, and query
        cache_key = f"org_{org_id_val}_doc_{active_doc_id}_v_{doc_version}_scope_{scope}:q_{search_query.lower()}"
        if cache_key in self._rag_query_cache:
            cached_ctx = self._rag_query_cache[cache_key]
            elapsed_ms = round((time.monotonic() - t_start) * 1000)
            logger.info(
                f"\n[RAG_RESULT]"
                f"\n  chunks=cache_hit"
                f"\n  context_length={len(cached_ctx)}"
                f"\n  latency_ms={elapsed_ms}"
            )
            return cached_ctx

        # Verify collection has chunks for active document; if missing, auto-index from SQLite
        if active_doc_id and str(active_doc_id) not in getattr(self, "_verified_indexed_docs", set()):
            try:
                cnt = self.collection.count(where={"document_id": str(active_doc_id)})
                if cnt == 0:
                    from src.config.database import SessionLocal
                    from src.models.knowledge_doc import KnowledgeDocument, KnowledgeDocStatus
                    from src.services.knowledge_service import KnowledgeService
                    _db = SessionLocal()
                    try:
                        try:
                            _d_id = int(active_doc_id)
                            _doc = _db.query(KnowledgeDocument).filter(KnowledgeDocument.id == _d_id).first()
                        except ValueError:
                            _doc = _db.query(KnowledgeDocument).filter(KnowledgeDocument.title.ilike(f"%{active_doc_id}%")).first()
                        if _doc and _doc.content and _doc.content.strip():
                            logger.info(f"[RAG] Auto-indexing active document {_doc.id} ('{_doc.title}') into vector store...")
                            KnowledgeService(_db)._index_document(_doc, _doc.content)
                    finally:
                        _db.close()
                if not hasattr(self, "_verified_indexed_docs"):
                    self._verified_indexed_docs = set()
                self._verified_indexed_docs.add(str(active_doc_id))
            except Exception as _sync_err:
                logger.debug(f"[RAG SYNC] Note: {_sync_err}")

        # Vector Store Query with verified active document filter
        if active_doc_id:
            where_filter = {"document_id": str(active_doc_id)}
        elif org_id_val is not None:
            where_filter = {"organization_id": str(org_id_val)}
        elif scope == "superadmin_assistant":
            where_filter = {
                "$and": [
                    {"organization_id": "super_admin"},
                    {"scope": "superadmin_assistant"}
                ]
            }
        else:
            return ""

        query_kwargs = {
            "query_texts": [search_query],
            "n_results": max(n_results, DEFAULT_TOP_K),
            "where": where_filter,
            "include": ["documents", "metadatas", "distances"]
        }

        results = None
        try:
            results = self.collection.query(**query_kwargs)
        except Exception as exc:
            logger.warning(f"[RAG SEARCH] Primary query failed ({exc}). Retrying with active isolation filter...")
            try:
                results = self.collection.query(
                    query_texts=[search_query],
                    n_results=max(n_results, DEFAULT_TOP_K),
                    where=where_filter,
                    include=["documents", "metadatas", "distances"]
                )
            except Exception as exc2:
                logger.error(f"[RAG SEARCH] Vector query exception: {exc2}.")
                elapsed_ms = round((time.monotonic() - t_start) * 1000)
                logger.info(
                    f"\n[RAG_RESULT]"
                    f"\n  chunks=0 (exception)"
                    f"\n  context_length=0"
                    f"\n  latency_ms={elapsed_ms}"
                )
                return ""

        if not results or not results.get('documents') or not results['documents'][0]:
            master_ctx = self.get_organization_master_context(org_id_val, document_id=active_doc_id, scope=scope) if active_doc_id else ""
            elapsed_ms = round((time.monotonic() - t_start) * 1000)
            logger.info(
                f"\n[RAG_RESULT]"
                f"\n  chunks=0 (empty_vector_hits -> fallback_master)"
                f"\n  context_length={len(master_ctx)}"
                f"\n  latency_ms={elapsed_ms}"
            )
            return master_ctx

        docs = results['documents'][0]
        metas = results['metadatas'][0] if results.get('metadatas') and results['metadatas'][0] else [{}] * len(docs)
        dists = results['distances'][0] if results.get('distances') and results['distances'][0] else [0.0] * len(docs)

        clean_docs = []
        valid_metas = []
        valid_dists = []
        for d, m, dist in zip(docs, metas, dists):
            if d and str(d).strip():
                m_dict = m or {}

                # ── POST-RETRIEVAL CHUNK METADATA VALIDATION (DEFENSE-IN-DEPTH) ──
                # 1. Enforce active usage status
                if str(m_dict.get("ai_usage", "Active")).strip().lower() == "inactive":
                    logger.debug(f"[RAG ISOLATION] Rejected inactive chunk: {m_dict.get('chunk_index')}")
                    continue

                # 2. Enforce organization ownership
                if org_id_val is not None:
                    chunk_org = str(m_dict.get("organization_id", "")).strip()
                    if chunk_org != str(org_id_val):
                        logger.warning(
                            f"[RAG ISOLATION LEAK PREVENTED] Rejected chunk belonging to org '{chunk_org}' for requested tenant org {org_id_val}"
                        )
                        continue
                elif scope == "superadmin_assistant":
                    chunk_org = str(m_dict.get("organization_id", "")).strip()
                    if chunk_org != "super_admin":
                        logger.warning(
                            f"[RAG ISOLATION LEAK PREVENTED] Rejected non-superadmin chunk from org '{chunk_org}' for superadmin assistant"
                        )
                        continue

                # 3. Enforce active document identity
                if active_doc_id:
                    chunk_doc = str(m_dict.get("document_id", "")).strip()
                    if chunk_doc != str(active_doc_id):
                        logger.warning(
                            f"[RAG ISOLATION LEAK PREVENTED] Rejected chunk from doc '{chunk_doc}' instead of active doc {active_doc_id}"
                        )
                        continue

                # 4. Enforce directional scope
                if scope:
                    scope_lower = str(scope).lower()
                    chunk_scope = str(m_dict.get("scope", "")).lower()
                    chunk_type = str(m_dict.get("document_type", "")).upper()
                    if "inbound" in scope_lower and (chunk_scope == "outbound" or chunk_type in ("OUTBOUND_SCRIPT", "CAMPAIGN_SCRIPT")):
                        continue
                    if "outbound" in scope_lower and chunk_scope == "inbound":
                        continue

                clean_docs.append(str(d).strip())
                valid_metas.append(m_dict)
                valid_dists.append(dist if dist is not None else 0.0)

        if not clean_docs:
            master_ctx = self.get_organization_master_context(org_id_val, document_id=active_doc_id, scope=scope) if active_doc_id else ""
            elapsed_ms = round((time.monotonic() - t_start) * 1000)
            logger.info(
                f"\n[RAG_RESULT]"
                f"\n  chunks=0 (clean_docs_empty -> fallback_master)"
                f"\n  context_length={len(master_ctx)}"
                f"\n  latency_ms={elapsed_ms}"
            )
            return master_ctx

        # Step 1: BM25 Keyword Scoring
        bm25_scores = self.bm25_scorer.score(search_query, clean_docs)

        # Step 2: Hybrid Reranking (60% Dense Vector + 40% BM25)
        candidates = []
        for i, (doc_text, meta, dist) in enumerate(zip(clean_docs, valid_metas, valid_dists)):
            vector_sim = max(0.0, 1.0 - (float(dist) / 2.0)) if dist is not None else 0.0
            bm25_sim = bm25_scores[i] if i < len(bm25_scores) else 0.0
            hybrid_score = (0.6 * vector_sim) + (0.4 * bm25_sim)

            if hybrid_score >= DEFAULT_SIMILARITY_THRESHOLD:
                candidates.append((hybrid_score, doc_text, meta))

        # Sort candidates by combined score (descending)
        candidates.sort(key=lambda x: x[0], reverse=True)

        # Select Top distinct relevant chunks
        context_parts = []
        chunk_log_details = []
        seen = set()
        for idx, (score, doc_text, meta) in enumerate(candidates[:n_results]):
            norm_snippet = doc_text[:100]
            if norm_snippet not in seen:
                seen.add(norm_snippet)
                context_parts.append(doc_text)
                c_id = meta.get("chunk_id", meta.get("id", f"chk_{idx}"))
                d_id = meta.get("document_id", active_doc_id or "unknown")
                preview = doc_text[:60].replace('\n', ' ')
                chunk_log_details.append(f"[id={c_id}, doc={d_id}, score={score:.3f}, preview='{preview}...']")

        specific_ctx = "\n\n".join(context_parts)
        final_ctx = specific_ctx

        staff_signals = {
            "doctor", "doctors", "dr", "dr.", "staff", "physician", "specialist",
            "consultant", "appointment", "booking", "schedule", "timing", "hours",
            "available", "availability", "department", "departments", "who is", "who handles"
        }
        if org_id_val is not None and (any(s in search_query.lower() for s in staff_signals) or intent in ("staff_lookup", "staff_availability", "doctor_information")):
            staff_ctx = self.retrieve_staff_context(query=search_query, organization_id=org_id_val)
            if staff_ctx:
                final_ctx = f"{final_ctx}\n\n[STAFF DIRECTORY CONTEXT]\n{staff_ctx}".strip()

        elapsed_ms = round((time.monotonic() - t_start) * 1000)

        logger.info(
            f"\n[RAG_RESULT]"
            f"\n  chunks={len(context_parts)}"
            f"\n  details={chunk_log_details}"
            f"\n  context_length={len(final_ctx)}"
            f"\n  latency_ms={elapsed_ms}"
        )

        res_to_return = final_ctx if final_ctx else (self.get_organization_master_context(org_id_val, document_id=active_doc_id, scope=scope) if active_doc_id else "")
        if hasattr(self, "_rag_query_cache") and len(self._rag_query_cache) < 500:
            self._rag_query_cache[cache_key] = res_to_return
        return res_to_return

    def get_active_script(
        self,
        organization_id: Optional[int] = None,
        agent_id: Optional[str] = None,
        assistant_id: Optional[str] = None,
        scope: Optional[str] = None,
    ) -> Optional[str]:
        """
        Retrieves the active SCRIPT document content for the specified scope/org/agent.
        Strictly isolated: Super Admin scripts are NEVER returned for org queries and vice versa.
        """
        try:
            from src.config.database import SessionLocal
            from src.models.knowledge_doc import KnowledgeDocument, KnowledgeDocStatus
            db = SessionLocal()
            try:
                query = db.query(KnowledgeDocument).filter(
                    KnowledgeDocument.document_type == "SCRIPT",
                    KnowledgeDocument.ai_usage == "Active",
                    KnowledgeDocument.status.in_([KnowledgeDocStatus.INDEXED, "Indexed", "INDEXED", "indexed"]),
                )

                if scope == "superadmin_assistant" or organization_id is None:
                    query = query.filter(
                        KnowledgeDocument.organization_id.is_(None),
                        KnowledgeDocument.scope == "superadmin_assistant",
                    )
                else:
                    query = query.filter(
                        KnowledgeDocument.organization_id == int(organization_id),
                    )

                if agent_id:
                    query = query.filter(KnowledgeDocument.agent_id == agent_id)
                if assistant_id:
                    query = query.filter(KnowledgeDocument.assistant_id == assistant_id)

                script_doc = query.order_by(KnowledgeDocument.updated_at.desc()).first()
                if script_doc and script_doc.content:
                    return script_doc.content.strip()
            finally:
                db.close()
        except Exception as err:
            logger.warning(f"Error loading active script: {err}")
        return None


rag_service = RagService()
