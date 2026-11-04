"""
Staff Directory to RAG Synchronization Service.
Ensures real-time synchronization between the Staff Directory and the RAG Knowledge Base
with strict multi-tenant isolation, structured audit logging, and cache invalidation.
"""
import json
import logging
import time
from typing import List, Optional
from sqlalchemy.orm import Session

from src.models.staff import Staff, StaffStatus
from src.models.knowledge_doc import KnowledgeDocument, KnowledgeDocStatus

logger = logging.getLogger(__name__)


class StaffRAGSyncService:
    @staticmethod
    def _generate_staff_chunk(staff: Staff) -> str:
        """
        Generates a structured, privacy-safe text chunk for a single staff member.
        Strictly excludes personal phone numbers and email addresses.
        """
        status_val = getattr(staff.status, "value", str(staff.status)) if staff.status else "Available"
        is_available = status_val.lower() == "available"

        # Parse working days
        days_str = staff.working_days or "Monday to Friday"
        try:
            if days_str.startswith("["):
                days_list = json.loads(days_str)
                if isinstance(days_list, list):
                    days_str = ", ".join(days_list)
        except Exception:
            pass

        # Parse services
        services_str = ""
        if staff.services:
            try:
                s_list = json.loads(staff.services) if isinstance(staff.services, str) else staff.services
                if isinstance(s_list, list) and s_list:
                    services_str = ", ".join(str(s) for s in s_list)
            except Exception:
                services_str = str(staff.services)

        chunk = (
            f"Staff Member: {staff.name}\n"
            f"Department: {staff.department}\n"
            f"Working Hours: {staff.working_hours or '09:00-18:00'}\n"
            f"Available Days: {days_str}\n"
            f"Appointment Duration: {staff.appointment_duration or '30 min'}\n"
            f"Status: {status_val}\n"
            f"Accepting Appointments: {'Yes' if is_available else 'No'}\n"
        )
        if services_str:
            chunk += f"Services/Treatments Handled: {services_str}\n"

        if is_available:
            chunk += (
                f"AI Instruction: {staff.name} is available for appointments in the {staff.department} department "
                f"during {staff.working_hours or '09:00-18:00'} on {days_str}. "
                f"Slot duration is {staff.appointment_duration or '30 min'}.\n"
            )
        else:
            chunk += (
                f"AI Instruction: {staff.name} is currently {status_val.lower()} and CANNOT accept new appointments. "
                f"Inform callers they are currently unavailable and suggest another provider in {staff.department} if possible.\n"
            )

        return chunk

    @classmethod
    def generate_directory_text(cls, staff_members: List[Staff], organization_id: int) -> str:
        """
        Generates the complete Staff Directory knowledge base text for an organization.
        """
        if not staff_members:
            return f"Staff Directory for Organization {organization_id}: No staff members currently registered."

        lines = [
            f"# STAFF DIRECTORY AND SCHEDULING RULES (ORGANIZATION ID: {organization_id})",
            "This document is the authoritative staff directory for appointment booking and routing.",
            "Rules for AI Receptionist:",
            "1. Only schedule appointments with staff members whose Status is 'Available'.",
            "2. Never disclose staff personal telephone numbers or personal email addresses.",
            "3. Use specified working hours and available days to confirm appointment slots.",
            "4. Do NOT confuse staff records with company products or services.",
            "\n--- ACTIVE STAFF DIRECTORY ---\n"
        ]

        for s in staff_members:
            lines.append(cls._generate_staff_chunk(s))
            lines.append("---")

        return "\n".join(lines)

    @classmethod
    def sync_staff(
        cls,
        staff_id: int,
        organization_id: int,
        db: Session,
        event_type: str = "UPDATE",
        changed_fields: Optional[List[str]] = None,
    ) -> bool:
        """
        Synchronizes a single staff member event (CREATE or UPDATE) into the organization's RAG knowledge base.
        Emits structured logs: [STAFF_CREATED], [STAFF_UPDATED], [RAG_STAFF_SYNC_STARTED], [RAG_STAFF_SYNC_COMPLETED], [RAG_STAFF_SYNC_FAILED].
        """
        version = int(time.time())
        if event_type == "CREATE":
            logger.info(f"[STAFF_CREATED] organization_id={organization_id} staff_id={staff_id}")
        else:
            logger.info(f"[STAFF_UPDATED] organization_id={organization_id} staff_id={staff_id} changed_fields={changed_fields or []}")

        logger.info(f"[RAG_STAFF_SYNC_STARTED] organization_id={organization_id} staff_id={staff_id} source_version={version}")

        try:
            return cls.sync_organization_staff(organization_id, db, staff_id=staff_id)
        except Exception as e:
            logger.error(f"[RAG_STAFF_SYNC_FAILED] organization_id={organization_id} staff_id={staff_id} error={e}", exc_info=True)
            return False

    @classmethod
    def delete_staff(
        cls,
        staff_id: int,
        organization_id: int,
        db: Session,
    ) -> bool:
        """
        Handles staff deletion event and immediately resynchronizes the organization's RAG directory.
        Emits [STAFF_DELETED], [RAG_STAFF_SYNC_STARTED], [RAG_STAFF_SYNC_COMPLETED].
        """
        version = int(time.time())
        logger.info(f"[STAFF_DELETED] organization_id={organization_id} staff_id={staff_id}")
        logger.info(f"[RAG_STAFF_SYNC_STARTED] organization_id={organization_id} staff_id={staff_id} source_version={version}")

        try:
            return cls.sync_organization_staff(organization_id, db, staff_id=staff_id)
        except Exception as e:
            logger.error(f"[RAG_STAFF_SYNC_FAILED] organization_id={organization_id} staff_id={staff_id} error={e}", exc_info=True)
            return False

    @classmethod
    def sync_organization_staff(
        cls,
        organization_id: int,
        db: Session,
        staff_id: Optional[int] = None,
    ) -> bool:
        """
        Re-indexes the entire organization's staff directory into a dedicated organization-scoped KnowledgeDocument,
        ingests into vector store, and clears RAG caches.
        """
        from src.services.rag_service import rag_service

        staff_list = db.query(Staff).filter(Staff.organization_id == organization_id).order_by(Staff.id.asc()).all()
        doc_text = cls.generate_directory_text(staff_list, organization_id)
        doc_title = f"Staff Directory - Organization {organization_id}"

        # Locate or create the dedicated STAFF_DIRECTORY KnowledgeDocument
        doc = db.query(KnowledgeDocument).filter(
            KnowledgeDocument.organization_id == organization_id,
            KnowledgeDocument.document_type == "STAFF_DIRECTORY",
        ).first()

        chunk_count = len(staff_list)
        if not doc:
            doc = KnowledgeDocument(
                organization_id=organization_id,
                title=doc_title,
                document_type="STAFF_DIRECTORY",
                source_type="system_sync",
                visibility="organization",
                scope="organization",
                file_type="TEXT",
                content=doc_text,
                chunks=chunk_count,
                ai_usage="Active",
                status=KnowledgeDocStatus.INDEXED,
            )
            db.add(doc)
        else:
            doc.title = doc_title
            doc.content = doc_text
            doc.chunks = chunk_count
            doc.ai_usage = "Active"
            doc.status = KnowledgeDocStatus.INDEXED

        db.commit()
        db.refresh(doc)

        # Ingest into vector store under dedicated document ID
        vector_doc_id = f"staff_dir_{organization_id}"
        rag_service.ingest_text(
            document_id=vector_doc_id,
            text=doc_text,
            source="STAFF_DIRECTORY",
            organization_id=organization_id,
        )

        # Invalidate organization caches
        rag_service.clear_organization_cache(organization_id)
        rag_service.invalidate_caches(organization_id)

        target_sid = staff_id if staff_id is not None else (staff_list[0].id if staff_list else 0)
        logger.info(f"[RAG_STAFF_SYNC_COMPLETED] organization_id={organization_id} staff_id={target_sid} indexed_chunks={chunk_count} sync_status=SUCCESS")
        return True

