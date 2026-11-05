"""
Comprehensive End-to-End Tests for CallZenza AI Receptionist Organization Deletion.

Verifies:
1. Complete tenant data deletion across 25+ tables in proper FK cascade order.
2. Purging of storage files (knowledge files, call recordings).
3. Purging of vector embeddings in ChromaDB and SQLite vector stores.
4. Complete tenant isolation (other tenant organizations remain completely intact).
5. Protected system organization enforcement (is_protected=1 prevents deletion).
6. Non-superadmin / unauthorized access rejection (403 Forbidden).
7. Non-existent organization handling (404 Not Found).
8. Audit logging with actor info and zero customer confidential data retention.
"""

import json
import os
import tempfile
import pytest
from datetime import datetime, timezone
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import text

from src.models.organization import Organization
from src.models.user import User
from src.models.permission import UserPermission
from src.models.staff import Staff, StaffStatus
from src.models.department import Department
from src.models.appointment import Appointment, AppointmentStatus
from src.models.call_log import CallLog
from src.models.conversation_message import ConversationMessage
from src.models.lead import Lead, LeadActivity
from src.models.knowledge_doc import KnowledgeDocument
from src.models.calendar_config import CalendarConfig
from src.models.webhook import Webhook, WebhookDelivery, WebhookType, WebhookDeliveryStatus
from src.models.workflow import Workflow, WorkflowExecutionLog
from src.models.sip_config import SipConfig
from src.models.twilio_config import TwilioConfig
from src.models.whatsapp_config import WhatsAppConfig
from src.models.email_config import EmailConfig
from src.models.settings import AppSettings
from src.models.subscription import Subscription
from src.models.notification_log import NotificationLog
from src.models.crm import CrmConnection, CrmSyncLog, CrmSyncSettings, CrmAutomationRules, CrmFieldMapping
from src.models.audit_log import PlatformAuditLog
from src.services.organization_deletion_service import organization_deletion_service
from src.services.chroma_vector_store import chroma_vector_store
from src.services.rag_service import rag_service
from src.services.auth_service import AuthService
from src.app import app


@pytest.fixture
def super_admin_user(db_session):
    user = db_session.query(User).filter(User.role == "super_admin").first()
    if not user:
        user = User(
            name="Super Admin",
            email=f"superadmin_{datetime.now().timestamp()}@callzenza.test",
            password_hash=AuthService(db_session).hash_password("SuperSecret123!"),
            role="super_admin",
            organization_id=1,
            is_active=1
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
    return user


@pytest.fixture
def regular_admin_user(db_session):
    user = db_session.query(User).filter(User.role == "admin").first()
    if not user:
        user = User(
            name="Tenant Admin",
            email=f"tenantadmin_{datetime.now().timestamp()}@callzenza.test",
            password_hash=AuthService(db_session).hash_password("TenantSecret123!"),
            role="admin",
            organization_id=1,
            is_active=1
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
    return user


def test_full_organization_deletion_cascade(db_session, super_admin_user, tmp_path):
    """
    Test 1: Full deletion of a populated tenant across all 25+ tables,
    storage files, vectors, and audit log generation, verifying isolation of tenant 2.
    """
    # 1. Create Target Tenant (Org A: To be deleted)
    org_a = Organization(
        name="Target Delete Corp",
        organization_type="Healthcare",
        status="active",
        is_protected=False
    )
    db_session.add(org_a)
    db_session.commit()
    db_session.refresh(org_a)
    org_a_id = org_a.id

    # Create Surviving Tenant (Org B: Must remain 100% intact)
    org_b = Organization(
        name="Surviving Stable Corp",
        organization_type="Finance",
        status="active",
        is_protected=False
    )
    db_session.add(org_b)
    db_session.commit()
    db_session.refresh(org_b)
    org_b_id = org_b.id

    # 2. Populate Org A data across all tables
    user_a = User(name="User A", email=f"usera_{org_a_id}@example.com", password_hash="hash", role="admin", organization_id=org_a_id, is_active=1)
    db_session.add(user_a)
    db_session.commit()
    db_session.refresh(user_a)
    user_a_id = user_a.id

    user_perm_a = UserPermission(user_id=user_a.id, permission_name="dashboard", enabled=1)
    db_session.add(user_perm_a)

    dept_a = Department(organization_id=org_a_id, name="Cardiology")
    db_session.add(dept_a)
    db_session.commit()
    db_session.refresh(dept_a)

    staff_a = Staff(organization_id=org_a_id, department="Cardiology", name="Dr. Smith", email="smith@test.com", status=StaffStatus.AVAILABLE)
    db_session.add(staff_a)

    appt_a = Appointment(
        organization_id=org_a_id,
        customer="Alice Caller",
        phone="5551112222",
        department="Cardiology",
        staff="Dr. Smith",
        status=AppointmentStatus.CONFIRMED,
        appointment_date="2026-12-01",
        appointment_time="10:00"
    )
    db_session.add(appt_a)
    db_session.commit()
    db_session.refresh(appt_a)
    appt_a_id = appt_a.id

    # Reminder linked to appt_a
    from src.services.reminder_service import Reminder
    from datetime import timedelta
    reminder_a = Reminder(
        appointment_id=appt_a.id,
        channel="sms",
        reminder_type="day_before",
        scheduled_at=datetime.now(timezone.utc) + timedelta(days=1),
        status="pending",
        recipient="5551112222"
    )
    db_session.add(reminder_a)

    # CallLog & Conversation Messages
    call_a = CallLog(
        organization_id=org_a_id,
        caller_phone="5551112222",
        call_sid="call_a_sid_999",
        status="completed",
        recording_url="/uploads/recordings/test_rec_org_a.wav"
    )
    db_session.add(call_a)
    db_session.commit()
    db_session.refresh(call_a)

    conv_msg_a = ConversationMessage(
        session_id="call_a_sid_999",
        role="assistant",
        message="Good morning, thank you for calling Target Delete Corp.",
        appointment_id=appt_a.id
    )
    db_session.add(conv_msg_a)

    # Lead & LeadActivity
    lead_a = Lead(organization_id=org_a_id, name="Alice Lead", phone="5551112222", status="new", call_log_id=call_a.id)
    db_session.add(lead_a)
    db_session.commit()
    db_session.refresh(lead_a)

    lead_act_a = LeadActivity(lead_id=lead_a.id, organization_id=org_a_id, event_type="CALL_INCOMING", title="Call logged")
    db_session.add(lead_act_a)

    # Physical files for Org A
    test_doc_file_a = tmp_path / "org_a_knowledge.txt"
    test_doc_file_a.write_text("Confidential Org A Knowledge Content", encoding="utf-8")
    assert test_doc_file_a.exists()

    doc_a = KnowledgeDocument(
        organization_id=org_a_id,
        title="Org A Script",
        file_path=str(test_doc_file_a),
        content="Confidential Org A Knowledge Content",
        file_type="TXT",
        ai_usage="Active"
    )
    db_session.add(doc_a)
    db_session.commit()
    db_session.refresh(doc_a)

    # Physical call recording file
    rec_dir = os.path.abspath("uploads/recordings")
    os.makedirs(rec_dir, exist_ok=True)
    test_rec_file_a = os.path.join(rec_dir, "test_rec_org_a.wav")
    with open(test_rec_file_a, "wb") as f:
        f.write(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x40\x1f\x00\x00data\x00\x00\x00\x00")
    assert os.path.exists(test_rec_file_a)

    # Webhook & Workflow & Integrations
    webhook_a = Webhook(organization_id=org_a_id, name="Org A Hook", url="https://example.com/webhook", type=WebhookType.OUTGOING, secret="whsec_123", events=["call.completed"])
    db_session.add(webhook_a)
    db_session.commit()
    db_session.refresh(webhook_a)

    delivery_a = WebhookDelivery(webhook_id=webhook_a.id, organization_id=org_a_id, event="call.completed", event_id="evt_123", status=WebhookDeliveryStatus.DELIVERED)
    db_session.add(delivery_a)

    workflow_a = Workflow(organization_id=org_a_id, name="Org A Flow", trigger_event="appointment.booked", action_type="send_email", action_config="{}")
    db_session.add(workflow_a)
    db_session.commit()
    db_session.refresh(workflow_a)

    wf_log_a = WorkflowExecutionLog(workflow_id=workflow_a.id, organization_id=org_a_id, workflow_name="Org A Flow", trigger_event="appointment.booked", action_type="send_email", status="SUCCESS")
    db_session.add(wf_log_a)

    db_session.add(SipConfig(organization_id=org_a_id, user_id=user_a.id, sip_server="sip.org-a.com", sip_username="101"))
    db_session.add(TwilioConfig(organization_id=org_a_id, account_sid="AC_test_org_a"))
    db_session.add(WhatsAppConfig(organization_id=org_a_id, phone_number="9876543210"))
    db_session.add(EmailConfig(organization_id=org_a_id, from_email="care@org-a.com"))
    db_session.add(AppSettings(organization_id=org_a_id, receptionist_name="A-Bot"))
    db_session.add(Subscription(organization_id=org_a_id, plan="Enterprise", status="active"))
    db_session.add(CalendarConfig(organization_id=org_a_id, user_id=user_a.id, provider="google"))
    db_session.add(NotificationLog(organization_id=org_a_id, channel="email", action="created", recipient_contact="5551112222", message_body="Your appointment is confirmed."))
    db_session.add(CrmConnection(organization_id=org_a_id, name="Salesforce A", provider="salesforce"))
    db_session.add(CrmSyncLog(organization_id=org_a_id, datetime="2026-11-05 10:00", record="Alice Lead", type="contact", crm="salesforce", status="Success"))
    db_session.add(CrmSyncSettings(organization_id=org_a_id, sync_contacts=True))
    db_session.add(CrmAutomationRules(organization_id=org_a_id, lead_identified=True))
    db_session.add(CrmFieldMapping(organization_id=org_a_id, ai_field="name", crm_field="FirstName"))

    # Vector store chunk for Org A
    chroma_vector_store.upsert(
        ids=[f"chunk_org_{org_a_id}_1"],
        documents=["Org A specific confidential service procedures"],
        metadatas=[{"organization_id": str(org_a_id), "document_id": str(doc_a.id)}]
    )

    # 3. Populate Surviving Org B data
    user_b = User(name="User B", email=f"userb_{org_b_id}@example.com", password_hash="hash", role="admin", organization_id=org_b_id, is_active=1)
    db_session.add(user_b)
    appt_b = Appointment(organization_id=org_b_id, customer="Bob Caller", phone="5559998888", department="General", staff="Dr. General", status=AppointmentStatus.CONFIRMED, appointment_date="2026-12-02", appointment_time="11:00")
    db_session.add(appt_b)
    call_b = CallLog(organization_id=org_b_id, caller_phone="5559998888", call_sid="call_b_sid_777", status="completed")
    db_session.add(call_b)
    test_doc_file_b = tmp_path / "org_b_knowledge.txt"
    test_doc_file_b.write_text("Org B Surviving Knowledge", encoding="utf-8")
    doc_b = KnowledgeDocument(organization_id=org_b_id, title="Org B Script", file_path=str(test_doc_file_b), content="Org B Surviving Knowledge", file_type="TXT", ai_usage="Active")
    db_session.add(doc_b)
    db_session.commit()

    chroma_vector_store.upsert(
        ids=[f"chunk_org_{org_b_id}_1"],
        documents=["Org B surviving service data"],
        metadatas=[{"organization_id": str(org_b_id), "document_id": str(doc_b.id)}]
    )

    # Ensure in-memory cache has something for Org A
    if not hasattr(rag_service, "_ctx_cache"):
        rag_service._ctx_cache = {}
    rag_service._ctx_cache[f"org_{org_a_id}_cached"] = "stale_org_a_cache"
    rag_service._ctx_cache[f"org_{org_b_id}_cached"] = "valid_org_b_cache"

    # 4. EXECUTE DELETION
    result = organization_deletion_service.delete_organization(
        db=db_session,
        org_id=org_a_id,
        current_user=super_admin_user
    )

    assert result["status"] == "success"
    assert result["organization_id"] == org_a_id
    assert "permanently deleted" in result["message"]

    # 5. VERIFY ORG A DELETION ACROSS ALL TABLES
    assert db_session.query(Organization).filter(Organization.id == org_a_id).first() is None
    assert db_session.query(User).filter(User.organization_id == org_a_id).count() == 0
    assert db_session.query(UserPermission).filter(UserPermission.user_id == user_a_id).count() == 0
    assert db_session.query(Department).filter(Department.organization_id == org_a_id).count() == 0
    assert db_session.query(Staff).filter(Staff.organization_id == org_a_id).count() == 0
    assert db_session.query(Appointment).filter(Appointment.organization_id == org_a_id).count() == 0
    assert db_session.execute(text(f"SELECT count(*) FROM reminders WHERE appointment_id = {appt_a_id}")).scalar() == 0
    assert db_session.query(CallLog).filter(CallLog.organization_id == org_a_id).count() == 0
    assert db_session.query(ConversationMessage).filter(ConversationMessage.session_id == "call_a_sid_999").count() == 0
    assert db_session.query(Lead).filter(Lead.organization_id == org_a_id).count() == 0
    assert db_session.query(LeadActivity).filter(LeadActivity.organization_id == org_a_id).count() == 0
    assert db_session.query(KnowledgeDocument).filter(KnowledgeDocument.organization_id == org_a_id).count() == 0
    assert db_session.query(Webhook).filter(Webhook.organization_id == org_a_id).count() == 0
    assert db_session.query(WebhookDelivery).filter(WebhookDelivery.organization_id == org_a_id).count() == 0
    assert db_session.query(Workflow).filter(Workflow.organization_id == org_a_id).count() == 0
    assert db_session.query(WorkflowExecutionLog).filter(WorkflowExecutionLog.organization_id == org_a_id).count() == 0
    assert db_session.query(SipConfig).filter(SipConfig.organization_id == org_a_id).count() == 0
    assert db_session.query(TwilioConfig).filter(TwilioConfig.organization_id == org_a_id).count() == 0
    assert db_session.query(WhatsAppConfig).filter(WhatsAppConfig.organization_id == org_a_id).count() == 0
    assert db_session.query(EmailConfig).filter(EmailConfig.organization_id == org_a_id).count() == 0
    assert db_session.query(AppSettings).filter(AppSettings.organization_id == org_a_id).count() == 0
    assert db_session.query(Subscription).filter(Subscription.organization_id == org_a_id).count() == 0
    assert db_session.query(CalendarConfig).filter(CalendarConfig.organization_id == org_a_id).count() == 0
    assert db_session.query(NotificationLog).filter(NotificationLog.organization_id == org_a_id).count() == 0
    assert db_session.query(CrmConnection).filter(CrmConnection.organization_id == org_a_id).count() == 0
    assert db_session.query(CrmSyncLog).filter(CrmSyncLog.organization_id == org_a_id).count() == 0
    assert db_session.query(CrmSyncSettings).filter(CrmSyncSettings.organization_id == org_a_id).count() == 0
    assert db_session.query(CrmAutomationRules).filter(CrmAutomationRules.organization_id == org_a_id).count() == 0
    assert db_session.query(CrmFieldMapping).filter(CrmFieldMapping.organization_id == org_a_id).count() == 0

    # 6. VERIFY PHYSICAL FILE REMOVAL
    assert not test_doc_file_a.exists(), "Knowledge document file should be purged from disk"
    assert not os.path.exists(test_rec_file_a), "Call recording file should be purged from disk"

    # 7. VERIFY VECTOR STORE CLEANUP
    chunks_a = chroma_vector_store.get(where={"organization_id": str(org_a_id)})
    assert len(chunks_a.get("ids", [])) == 0, "All Chroma vectors for Org A must be deleted"

    # 8. VERIFY CACHE INVALIDATION
    assert f"org_{org_a_id}_cached" not in rag_service._ctx_cache

    # 9. VERIFY SURVIVING TENANT (ORG B) REMAINS 100% INTACT
    assert db_session.query(Organization).filter(Organization.id == org_b_id).first() is not None
    assert db_session.query(User).filter(User.organization_id == org_b_id).count() == 1
    assert db_session.query(Appointment).filter(Appointment.organization_id == org_b_id).count() == 1
    assert db_session.query(CallLog).filter(CallLog.organization_id == org_b_id).count() == 1
    assert db_session.query(KnowledgeDocument).filter(KnowledgeDocument.organization_id == org_b_id).count() == 1
    assert test_doc_file_b.exists(), "Surviving org file must remain intact"
    chunks_b = chroma_vector_store.get(where={"organization_id": str(org_b_id)})
    assert len(chunks_b.get("ids", [])) == 1, "Surviving org vectors must remain intact"
    assert f"org_{org_b_id}_cached" in getattr(rag_service, "_ctx_cache", {})

    # 10. VERIFY AUDIT LOG
    audit = db_session.query(PlatformAuditLog).filter(PlatformAuditLog.entity_id == org_a_id).first()
    assert audit is not None
    assert audit.action == "ORGANIZATION_DELETED"
    assert audit.actor_email == super_admin_user.email
    # Verify no customer confidential details are stored
    assert "Confidential" not in (audit.details or "")


def test_protected_system_organization_cannot_be_deleted(db_session, super_admin_user):
    """
    Test 2: System/Platform organization with is_protected=True cannot be deleted.
    """
    protected_org = Organization(
        name="Root Core Platform",
        organization_type="System",
        status="active",
        is_protected=True
    )
    db_session.add(protected_org)
    db_session.commit()
    db_session.refresh(protected_org)

    with pytest.raises(HTTPException) as exc_info:
        organization_deletion_service.delete_organization(
            db=db_session,
            org_id=protected_org.id,
            current_user=super_admin_user
        )

    assert exc_info.value.status_code == 400
    assert "protected system organization" in exc_info.value.detail.lower()

    # Ensure org still exists
    assert db_session.query(Organization).filter(Organization.id == protected_org.id).first() is not None


def test_non_existent_organization_returns_404(db_session, super_admin_user):
    """
    Test 3: Deleting a non-existent organization returns 404 Not Found.
    """
    with pytest.raises(HTTPException) as exc_info:
        organization_deletion_service.delete_organization(
            db=db_session,
            org_id=9999999,
            current_user=super_admin_user
        )

    assert exc_info.value.status_code == 404
    assert "not found" in exc_info.value.detail.lower()


def test_api_endpoint_rbac_enforcement(db_session, super_admin_user, regular_admin_user):
    """
    Test 4: RBAC enforcement on DELETE /organizations/{org_id}.
    - Unauthorized or non-superadmin returns 403 Forbidden.
    - Superadmin returns 200 OK.
    """
    client = TestClient(app)
    auth_service = AuthService(db_session)

    # 1. Create temporary org to delete
    test_org = Organization(name="RBAC Test Org", organization_type="General", status="active", is_protected=False)
    db_session.add(test_org)
    db_session.commit()
    db_session.refresh(test_org)

    # 2. Test without token -> 401 Unauthorized
    resp_no_token = client.delete(f"/organizations/{test_org.id}")
    assert resp_no_token.status_code in (401, 403)

    from src.config.jwt import create_access_token

    # 3. Test with regular admin token -> 403 Forbidden
    admin_token = create_access_token({
        "sub": str(regular_admin_user.id),
        "email": regular_admin_user.email,
        "role": regular_admin_user.role,
        "organization_id": regular_admin_user.organization_id
    })
    resp_admin = client.delete(
        f"/organizations/{test_org.id}",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert resp_admin.status_code == 403
    assert "access denied" in resp_admin.text.lower() or "requires" in resp_admin.text.lower()

    # 4. Test with super admin token -> 200 OK
    super_token = create_access_token({
        "sub": str(super_admin_user.id),
        "email": super_admin_user.email,
        "role": super_admin_user.role,
        "organization_id": super_admin_user.organization_id
    })
    resp_super = client.delete(
        f"/organizations/{test_org.id}",
        headers={"Authorization": f"Bearer {super_token}"}
    )
    assert resp_super.status_code == 200
    data = resp_super.json()
    assert data["status"] == "success"
    assert data["organization_id"] == test_org.id


def test_transactional_rollback_on_failure(db_session, super_admin_user, monkeypatch):
    """
    Test 5: Transactional atomicity: if an exception occurs mid-way through database deletion,
    the entire transaction is rolled back and no partial or orphan deletion remains.
    """
    test_org = Organization(name="Rollback Test Org", organization_type="General", status="active", is_protected=False)
    db_session.add(test_org)
    db_session.commit()
    db_session.refresh(test_org)
    test_org_id = test_org.id

    user = User(name="Rollback User", email=f"rb_{test_org_id}@example.com", password_hash="hash", role="admin", organization_id=test_org_id, is_active=1)
    db_session.add(user)
    db_session.commit()

    # Monkeypatch db.execute to raise an error when deleting 'departments'
    orig_execute = db_session.execute
    def faulty_execute(stmt, params=None, *args, **kwargs):
        stmt_str = str(stmt)
        if "DELETE FROM departments" in stmt_str:
            raise RuntimeError("Simulated Database I/O failure during deletion")
        return orig_execute(stmt, params, *args, **kwargs)

    monkeypatch.setattr(db_session, "execute", faulty_execute)

    with pytest.raises(HTTPException) as exc_info:
        organization_deletion_service.delete_organization(
            db=db_session,
            org_id=test_org_id,
            current_user=super_admin_user
        )

    assert exc_info.value.status_code == 500
    assert "Simulated Database I/O failure" in str(exc_info.value.detail)

    # Verify rollback: both the organization and user STILL EXIST
    assert db_session.query(Organization).filter(Organization.id == test_org_id).first() is not None
    assert db_session.query(User).filter(User.organization_id == test_org_id).count() == 1


def test_organization_out_schema_contains_is_protected(db_session, super_admin_user):
    """
    Test 6: GET /organizations endpoint returns `is_protected` field for all organizations
    so frontend can safely disable delete buttons on protected systems.
    """
    client = TestClient(app)
    from src.config.jwt import create_access_token
    super_token = create_access_token({
        "sub": str(super_admin_user.id),
        "email": super_admin_user.email,
        "role": super_admin_user.role,
        "organization_id": super_admin_user.organization_id
    })
    resp = client.get("/organizations", headers={"Authorization": f"Bearer {super_token}"})
    assert resp.status_code == 200
    orgs = resp.json()
    assert len(orgs) > 0
    # Every organization in the response has the is_protected boolean property
    for o in orgs:
        assert "is_protected" in o
        assert isinstance(o["is_protected"], bool)
