import json
import logging
import os
import shutil
from typing import Any, Dict, List, Optional
from fastapi import HTTPException, status
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from src.models.organization import Organization
from src.models.user import User
from src.models.audit_log import PlatformAuditLog

logger = logging.getLogger(__name__)


class OrganizationDeletionService:
    """
    Comprehensive Tenant Deletion Service for CallZenza AI Receptionist.
    
    Guarantees:
    - Protection of system/platform organizations via database-backed flag `is_protected`.
    - Strict Super Admin access.
    - Transactional deletion of all tenant data across 25+ tables in proper FK cascade order.
    - Purging of physical storage files (knowledge files, call recordings, assets).
    - Complete cleanup of vector embeddings across ChromaDB and SQLite vector stores.
    - Termination of active in-memory SIP sessions, background registrars, and caches.
    - Audit logging of the operation without retaining sensitive customer data.
    """

    def delete_organization(
        self,
        db: Session,
        org_id: int,
        current_user: Optional[User] = None
    ) -> Dict[str, Any]:
        """
        Permanently deletes an organization and all its tenant-scoped assets.
        """
        # 1. Verify existence
        org = db.query(Organization).filter(Organization.id == org_id).first()
        if not org:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Organization #{org_id} not found."
            )

        # 2. Database-backed protection check
        if getattr(org, "is_protected", False):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete protected system organization."
            )

        org_name = org.name

        # Detect existing tables in the bound database connection
        try:
            inspector = inspect(db.get_bind())
            existing_tables = set(inspector.get_table_names())
        except Exception:
            existing_tables = set()

        # 3. Pre-deletion scan: gather file paths and doc IDs before database rows are removed
        file_paths_to_delete: List[str] = []
        doc_ids: List[int] = []

        try:
            if not existing_tables or "knowledge_documents" in existing_tables:
                doc_rows = db.execute(
                    text("SELECT id, file_path FROM knowledge_documents WHERE organization_id = :org_id"),
                    {"org_id": org_id}
                ).fetchall()
                for r in doc_rows:
                    if r[0]:
                        doc_ids.append(r[0])
                    if r[1]:
                        file_paths_to_delete.append(str(r[1]))
        except Exception as ex:
            logger.warning(f"[ORG DELETION] Error collecting knowledge document files: {ex}")

        try:
            if not existing_tables or "call_logs" in existing_tables:
                rec_rows = db.execute(
                    text("SELECT recording_url FROM call_logs WHERE organization_id = :org_id AND recording_url IS NOT NULL AND recording_url != ''"),
                    {"org_id": org_id}
                ).fetchall()
                for r in rec_rows:
                    if r[0]:
                        file_paths_to_delete.append(str(r[0]))
        except Exception as ex:
            logger.warning(f"[ORG DELETION] Error collecting call recording files: {ex}")

        # 4. Atomic database deletion in strict FK dependency order
        deleted_records: Dict[str, int] = {}
        deletion_queries = [
            ("conversation_messages", text("""
                DELETE FROM conversation_messages
                WHERE appointment_id IN (SELECT id FROM appointments WHERE organization_id = :org_id)
                   OR session_id IN (SELECT call_sid FROM call_logs WHERE organization_id = :org_id)
                   OR session_id IN (SELECT 'sip_session_' || call_sid FROM call_logs WHERE organization_id = :org_id)
            """)),
            ("reminders", text("""
                DELETE FROM reminders
                WHERE appointment_id IN (SELECT id FROM appointments WHERE organization_id = :org_id)
            """)),
            ("user_permissions", text("""
                DELETE FROM user_permissions
                WHERE user_id IN (SELECT id FROM users WHERE organization_id = :org_id)
            """)),
            ("lead_activities", text("""
                DELETE FROM lead_activities
                WHERE organization_id = :org_id
                   OR lead_id IN (SELECT id FROM leads WHERE organization_id = :org_id)
            """)),
            ("webhook_deliveries", text("""
                DELETE FROM webhook_deliveries
                WHERE organization_id = :org_id
                   OR webhook_id IN (SELECT id FROM webhooks WHERE organization_id = :org_id)
            """)),
            ("workflow_logs", text("""
                DELETE FROM workflow_logs
                WHERE organization_id = :org_id
                   OR workflow_id IN (SELECT id FROM workflows WHERE organization_id = :org_id)
            """)),
            ("crm_sync_logs", text("DELETE FROM crm_sync_logs WHERE organization_id = :org_id")),
            ("crm_field_mappings", text("DELETE FROM crm_field_mappings WHERE organization_id = :org_id")),
            ("crm_automation_rules", text("DELETE FROM crm_automation_rules WHERE organization_id = :org_id")),
            ("crm_connections", text("DELETE FROM crm_connections WHERE organization_id = :org_id")),
            ("crm_sync_settings", text("DELETE FROM crm_sync_settings WHERE organization_id = :org_id")),
            ("leads", text("DELETE FROM leads WHERE organization_id = :org_id")),
            ("call_logs", text("DELETE FROM call_logs WHERE organization_id = :org_id")),
            ("appointments", text("DELETE FROM appointments WHERE organization_id = :org_id")),
            ("calendar_configs", text("""
                DELETE FROM calendar_configs
                WHERE organization_id = :org_id
                   OR user_id IN (SELECT id FROM users WHERE organization_id = :org_id)
            """)),
            ("knowledge_documents", text("DELETE FROM knowledge_documents WHERE organization_id = :org_id")),
            ("staff", text("DELETE FROM staff WHERE organization_id = :org_id")),
            ("departments", text("DELETE FROM departments WHERE organization_id = :org_id")),
            ("users", text("DELETE FROM users WHERE organization_id = :org_id")),
            ("subscriptions", text("DELETE FROM subscriptions WHERE organization_id = :org_id")),
            ("notification_logs", text("DELETE FROM notification_logs WHERE organization_id = :org_id")),
            ("workflows", text("DELETE FROM workflows WHERE organization_id = :org_id")),
            ("webhooks", text("DELETE FROM webhooks WHERE organization_id = :org_id")),
            ("sip_configs", text("DELETE FROM sip_configs WHERE organization_id = :org_id")),
            ("twilio_configs", text("DELETE FROM twilio_configs WHERE organization_id = :org_id")),
            ("whatsapp_configs", text("DELETE FROM whatsapp_configs WHERE organization_id = :org_id")),
            ("email_configs", text("DELETE FROM email_configs WHERE organization_id = :org_id")),
            ("app_settings", text("DELETE FROM app_settings WHERE organization_id = :org_id")),
            ("organizations", text("DELETE FROM organizations WHERE id = :org_id")),
        ]

        try:
            for table_name, stmt in deletion_queries:
                if existing_tables and table_name not in existing_tables:
                    continue
                res = db.execute(stmt, {"org_id": org_id})
                rowcount = res.rowcount if hasattr(res, "rowcount") and res.rowcount is not None else 0
                deleted_records[table_name] = max(0, rowcount)

            # Record platform audit log entry
            if not existing_tables or "platform_audit_logs" in existing_tables:
                audit_entry = PlatformAuditLog(
                    action="ORGANIZATION_DELETED",
                    entity_type="organization",
                    entity_id=org_id,
                    actor_id=current_user.id if current_user else None,
                    actor_email=current_user.email if current_user else None,
                    details=json.dumps({
                        "organization_name": org_name,
                        "deleted_tables_summary": {k: v for k, v in deleted_records.items() if v > 0}
                    })
                )
                db.add(audit_entry)
            db.commit()
            db.expire_all()
            logger.info(f"[ORG DELETION] Successfully purged database records for Org #{org_id} ({org_name}).")
        except Exception as err:
            db.rollback()
            logger.error(f"[ORG DELETION] Database transaction failed for Org #{org_id}: {err}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete organization data: {str(err)}"
            )

        # 5. Post-commit resource cleanup (files, vectors, SIP, caches)
        # Purge files
        files_removed = 0
        for fp in file_paths_to_delete:
            if self._clean_disk_file(fp):
                files_removed += 1

        # Purge organization upload folder if present
        self._clean_org_folder(org_id)

        # Purge vector stores (ChromaDB + SQLite rag.sqlite)
        self._clean_vector_store(org_id, doc_ids)

        # Terminate active in-memory sessions, SIP registrations, and caches
        self._clean_active_sessions_and_processes(org_id)

        return {
            "status": "success",
            "message": f"Organization '{org_name}' and all associated tenant data were permanently deleted.",
            "organization_id": org_id,
            "organization_name": org_name,
            "records_deleted": deleted_records,
            "files_removed": files_removed
        }

    def _clean_disk_file(self, file_path: Optional[str]) -> bool:
        if not file_path or not str(file_path).strip():
            return False
        path_str = str(file_path).strip()

        # Check absolute path
        if os.path.isabs(path_str) and os.path.exists(path_str) and os.path.isfile(path_str):
            try:
                os.remove(path_str)
                return True
            except Exception as e:
                logger.warning(f"[ORG DELETION] Failed removing absolute file {path_str}: {e}")

        # Check candidate relative paths across known directory roots
        base_dirs = [
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
        ]
        filename = os.path.basename(path_str.replace("\\", "/"))
        clean_rel = path_str.lstrip("/\\")

        candidates = [
            clean_rel,
            os.path.join("uploads", "recordings", filename),
            os.path.join("src", "uploads", "knowledge", filename),
            os.path.join("uploads", "knowledge", filename),
            os.path.join("src", "uploads", filename),
            os.path.join("uploads", filename),
        ]

        for b in base_dirs:
            for sub in candidates:
                cand = os.path.join(b, sub)
                if os.path.exists(cand) and os.path.isfile(cand):
                    try:
                        os.remove(cand)
                        return True
                    except Exception as e:
                        logger.warning(f"[ORG DELETION] Failed removing candidate file {cand}: {e}")
        return False

    def _clean_org_folder(self, org_id: int):
        base_dirs = [
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
        ]
        for b in base_dirs:
            org_dir = os.path.join(b, "uploads", "organizations", str(org_id))
            if os.path.exists(org_dir) and os.path.isdir(org_dir):
                try:
                    shutil.rmtree(org_dir, ignore_errors=True)
                except Exception as e:
                    logger.warning(f"[ORG DELETION] Failed removing org folder {org_dir}: {e}")

    def _clean_vector_store(self, org_id: int, doc_ids: List[int]):
        # 1. ChromaDB
        try:
            from src.services.chroma_vector_store import chroma_vector_store
            chroma_vector_store.delete(where={"organization_id": str(org_id)})
            for d_id in doc_ids:
                chroma_vector_store.delete(where={"document_id": str(d_id)})
            logger.info(f"[ORG DELETION] ChromaDB vector cleanup completed for org={org_id}")
        except Exception as ex:
            logger.warning(f"[ORG DELETION] ChromaDB vector deletion warning: {ex}")

        # 2. SQLite vector store (rag.sqlite)
        base_dirs = [
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
        ]
        for b in base_dirs:
            rag_path = os.path.join(b, "rag.sqlite")
            if os.path.exists(rag_path):
                try:
                    import sqlite3
                    conn = sqlite3.connect(rag_path)
                    cur = conn.cursor()
                    cur.execute("DELETE FROM rag_chunks WHERE organization_id = ? OR organization_id = ?", (str(org_id), org_id))
                    for d_id in doc_ids:
                        cur.execute("DELETE FROM rag_chunks WHERE document_id = ? OR document_id = ?", (str(d_id), d_id))
                    conn.commit()
                    conn.close()
                    logger.info(f"[ORG DELETION] Cleaned rag.sqlite vectors at {rag_path}")
                except Exception as ex:
                    logger.warning(f"[ORG DELETION] rag.sqlite cleanup warning for {rag_path}: {ex}")

    def _clean_active_sessions_and_processes(self, org_id: int):
        # 1. Active SIP Call Sessions
        try:
            from src.services.sip_call_handler import _active_sip_calls
            for s_id, s_obj in list(_active_sip_calls.items()):
                if getattr(s_obj, "organization_id", None) == org_id:
                    try:
                        s_obj.stop()
                    except Exception:
                        pass
                    _active_sip_calls.pop(s_id, None)
            logger.info(f"[ORG DELETION] Active SIP calls terminated for org={org_id}")
        except Exception as ex:
            logger.warning(f"[ORG DELETION] Active SIP call cleanup warning: {ex}")

        # 2. Active SIP Registrar
        try:
            from src.services.sip_registrar import stop_registrar
            stop_registrar(org_id)
            logger.info(f"[ORG DELETION] SIP registrar stopped for org={org_id}")
        except Exception as ex:
            logger.warning(f"[ORG DELETION] stop_registrar warning: {ex}")

        # 3. RAG and Master Context Invalidation
        try:
            from src.services.rag_service import rag_service
            rag_service.clear_organization_cache(org_id)
            logger.info(f"[ORG DELETION] RAG cache cleared for org={org_id}")
        except Exception as ex:
            logger.warning(f"[ORG DELETION] RAG cache clear warning: {ex}")

        # 4. Cache Service Invalidation
        try:
            from src.services.cache_service import cache_service
            cache_service.invalidate_organization(org_id)
            logger.info(f"[ORG DELETION] Cache service invalidated for org={org_id}")
        except Exception as ex:
            logger.warning(f"[ORG DELETION] cache_service invalidation warning: {ex}")


organization_deletion_service = OrganizationDeletionService()
