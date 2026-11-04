"""
Comprehensive test suite for Staff Directory Multi-Tenant Organization Data Isolation,
RAG Synchronization, and Voice Assistant Retrieval.
Covers all 14 mandatory verification steps:
1. Create Staff A in Org A -> Org A owns Staff A.
2. Org A lists staff -> sees Staff A.
3. Org B lists staff -> does NOT see Staff A.
4. Create Staff B in Org B -> Org B owns Staff B.
5. Org B lists staff -> sees Staff B, does NOT see Staff A.
6. Org A lists staff -> sees Staff A, does NOT see Staff B.
7. Cross-tenant GET /staff/{staff_a_id} by Org B -> 404 / 403.
8. Cross-tenant PUT /staff/{staff_a_id} by Org B -> 404 / 403.
9. Cross-tenant DELETE /staff/{staff_a_id} by Org B -> 404 / 403.
10. Availability route isolation: GET & PUT /{staff_id}/availability.
11. RAG synchronization: Update Staff A -> Org A RAG updated; Org B RAG unaffected.
12. Voice assistant / RAG retrieval isolation: Org A vs Org B inquiries.
13. Deactivation & deletion: Deactivate / Delete Staff A -> RAG reflects unavailable / removed.
14. Super Admin verification: Global visibility vs scoped filtering.
"""
import os
import sys

# Ensure backend root is on sys.path and development mode is enabled
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["APP_ENV"] = "development"

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from main import app
from src.config.database import SessionLocal
from src.config.jwt import create_access_token
from src.models.organization import Organization
from src.models.user import User
from src.models.staff import Staff, StaffStatus
from src.models.department import Department
from src.models.knowledge_doc import KnowledgeDocument
from src.services.rag_service import rag_service
from src.services.staff_rag_sync_service import StaffRAGSyncService

client = TestClient(app)


@pytest.fixture(scope="module")
def setup_tenants():
    """Sets up two test organizations (Org A=101, Org B=102) and their admin users."""
    db: Session = SessionLocal()
    try:
        # Create or ensure Org A
        org_a = db.query(Organization).filter(Organization.id == 101).first()
        if not org_a:
            org_a = Organization(id=101, name="Apex Cardiology Center", status="active")
            db.add(org_a)

        # Create or ensure Org B
        org_b = db.query(Organization).filter(Organization.id == 102).first()
        if not org_b:
            org_b = Organization(id=102, name="Beacon Dental Clinic", status="active")
            db.add(org_b)

        # Create Users
        user_a = db.query(User).filter(User.id == 1001).first()
        if not user_a:
            user_a = User(id=1001, organization_id=101, name="Admin Org A", email="admin.a@test.com", password_hash="dummy", role="admin", is_active=True)
            db.add(user_a)

        user_b = db.query(User).filter(User.id == 1002).first()
        if not user_b:
            user_b = User(id=1002, organization_id=102, name="Admin Org B", email="admin.b@test.com", password_hash="dummy", role="admin", is_active=True)
            db.add(user_b)

        user_sa = db.query(User).filter(User.id == 1000).first()
        if not user_sa:
            user_sa = User(id=1000, organization_id=None, name="Super Admin User", email="superadmin@test.com", password_hash="dummy", role="super_admin", is_active=True)
            db.add(user_sa)

        # Clean existing test staff for orgs 101 and 102
        db.query(Staff).filter(Staff.organization_id.in_([101, 102])).delete()
        db.query(KnowledgeDocument).filter(KnowledgeDocument.organization_id.in_([101, 102])).delete()
        db.commit()

        token_a = create_access_token({"sub": "1001", "email": "admin.a@test.com", "role": "admin", "organization_id": 101})
        token_b = create_access_token({"sub": "1002", "email": "admin.b@test.com", "role": "admin", "organization_id": 102})
        token_sa = create_access_token({"sub": "1000", "email": "superadmin@test.com", "role": "super_admin", "organization_id": None})

        yield {
            "token_a": token_a,
            "token_b": token_b,
            "token_sa": token_sa,
            "org_a_id": 101,
            "org_b_id": 102,
        }

        # Cleanup
        db.query(Staff).filter(Staff.organization_id.in_([101, 102])).delete()
        db.query(KnowledgeDocument).filter(KnowledgeDocument.organization_id.in_([101, 102])).delete()
        db.commit()
    finally:
        db.close()


def test_steps_1_to_6_creation_and_listing_isolation(setup_tenants):
    """
    Steps 1-6:
    Create Staff A in Org A, Create Staff B in Org B.
    Verify each tenant sees strictly its own staff and never the other's.
    """
    token_a = setup_tenants["token_a"]
    token_b = setup_tenants["token_b"]

    headers_a = {"Authorization": f"Bearer {token_a}"}
    headers_b = {"Authorization": f"Bearer {token_b}"}

    # Step 1: Create Staff A in Organization A
    res_a = client.post(
        "/api/v1/staff",
        json={
            "name": "Dr. Sarah Adams",
            "department": "Cardiology",
            "working_hours": "09:00-17:00",
            "appointment_duration": "30 min",
            "status": "Available",
            # Attempt cross-tenant spoofing: even if payload claims organization_id=102, backend MUST override to 101!
            "organization_id": 102
        },
        headers=headers_a,
    )
    assert res_a.status_code == 201, res_a.text
    staff_a = res_a.json()
    assert staff_a["name"] == "Dr. Sarah Adams"
    assert staff_a["organization_id"] == 101, "Spoofed organization_id was not overridden!"

    # Step 2: Org A GET /staff -> sees Staff A
    res_list_a = client.get("/api/v1/staff", headers=headers_a)
    assert res_list_a.status_code == 200
    staff_ids_a = [s["id"] for s in res_list_a.json()]
    assert staff_a["id"] in staff_ids_a

    # Step 3: Org B GET /staff -> Staff A is NOT in response
    res_list_b = client.get("/api/v1/staff", headers=headers_b)
    assert res_list_b.status_code == 200
    staff_ids_b = [s["id"] for s in res_list_b.json()]
    assert staff_a["id"] not in staff_ids_b

    # Step 4: Create Staff B in Organization B
    res_b = client.post(
        "/api/v1/staff",
        json={
            "name": "Dr. Brian Baker",
            "department": "Orthodontics",
            "working_hours": "10:00-18:00",
            "appointment_duration": "45 min",
            "status": "Available",
        },
        headers=headers_b,
    )
    assert res_b.status_code == 201, res_b.text
    staff_b = res_b.json()
    assert staff_b["name"] == "Dr. Brian Baker"
    assert staff_b["organization_id"] == 102

    # Step 5: Org B GET /staff -> sees Staff B, does NOT see Staff A
    res_list_b2 = client.get("/api/v1/staff", headers=headers_b)
    b_ids = [s["id"] for s in res_list_b2.json()]
    assert staff_b["id"] in b_ids
    assert staff_a["id"] not in b_ids

    # Step 6: Org A GET /staff -> sees Staff A, does NOT see Staff B
    res_list_a2 = client.get("/api/v1/staff", headers=headers_a)
    a_ids = [s["id"] for s in res_list_a2.json()]
    assert staff_a["id"] in a_ids
    assert staff_b["id"] not in a_ids


def test_steps_7_to_9_cross_tenant_crud_rejection(setup_tenants):
    """
    Steps 7-9:
    Attempt cross-tenant GET, PUT, DELETE by ID.
    Verify requests are strictly rejected with 404/403.
    """
    token_a = setup_tenants["token_a"]
    token_b = setup_tenants["token_b"]

    headers_a = {"Authorization": f"Bearer {token_a}"}
    headers_b = {"Authorization": f"Bearer {token_b}"}

    # Fetch Staff A id
    list_a = client.get("/api/v1/staff", headers=headers_a).json()
    staff_a_id = list_a[0]["id"]

    # Step 7: Cross-tenant GET /staff/{staff_a_id} as Org B -> 404
    res_cross_get = client.get(f"/api/v1/staff/{staff_a_id}", headers=headers_b)
    assert res_cross_get.status_code in (403, 404)

    # Step 8: Cross-tenant PUT /staff/{staff_a_id} as Org B -> 404
    res_cross_put = client.put(
        f"/api/v1/staff/{staff_a_id}",
        json={"name": "Hacked Name"},
        headers=headers_b,
    )
    assert res_cross_put.status_code in (403, 404)

    # Step 9: Cross-tenant DELETE /staff/{staff_a_id} as Org B -> 404
    res_cross_del = client.delete(f"/api/v1/staff/{staff_a_id}", headers=headers_b)
    assert res_cross_del.status_code in (403, 404)

    # Verify Staff A was NOT modified or deleted
    verify_a = client.get(f"/api/v1/staff/{staff_a_id}", headers=headers_a).json()
    assert verify_a["name"] == "Dr. Sarah Adams"


def test_step_10_availability_routes_isolation(setup_tenants):
    """
    Step 10:
    Verify /{staff_id}/availability adheres to tenant isolation for read and update.
    """
    token_a = setup_tenants["token_a"]
    token_b = setup_tenants["token_b"]
    headers_a = {"Authorization": f"Bearer {token_a}"}
    headers_b = {"Authorization": f"Bearer {token_b}"}

    list_a = client.get("/api/v1/staff", headers=headers_a).json()
    staff_a_id = list_a[0]["id"]

    # Org A can read Staff A availability
    res_avail_a = client.get(f"/api/v1/staff/{staff_a_id}/availability", headers=headers_a)
    assert res_avail_a.status_code == 200

    # Org B CANNOT read Staff A availability
    res_avail_b = client.get(f"/api/v1/staff/{staff_a_id}/availability", headers=headers_b)
    assert res_avail_b.status_code in (403, 404)

    # Org B CANNOT update Staff A availability
    res_upd_b = client.put(
        f"/api/v1/staff/{staff_a_id}/availability",
        json={"start_time": "08:00", "end_time": "16:00", "working_days": ["Monday", "Tuesday"]},
        headers=headers_b,
    )
    assert res_upd_b.status_code in (403, 404)

    # Org A can update Staff A availability
    res_upd_a = client.put(
        f"/api/v1/staff/{staff_a_id}/availability",
        json={"start_time": "08:00", "end_time": "16:00", "working_days": ["Monday", "Tuesday", "Wednesday"]},
        headers=headers_a,
    )
    assert res_upd_a.status_code == 200
    assert res_upd_a.json()["start_time"] == "08:00"


def test_steps_11_to_13_rag_synchronization_and_voice_retrieval(setup_tenants):
    """
    Steps 11-13:
    11. Verify RAG synchronization on staff update (Org A RAG updated, Org B unaffected).
    12. Voice assistant / RAG retrieval returns only Org's active staff with updated hours.
    13. Deactivation & deletion removes or flags staff in RAG.
    """
    token_a = setup_tenants["token_a"]
    token_b = setup_tenants["token_b"]
    headers_a = {"Authorization": f"Bearer {token_a}"}
    headers_b = {"Authorization": f"Bearer {token_b}"}

    list_a = client.get("/api/v1/staff", headers=headers_a).json()
    staff_a_id = list_a[0]["id"]

    # Step 11: Update Staff A's working hours
    upd_res = client.put(
        f"/api/v1/staff/{staff_a_id}",
        json={"working_hours": "07:30-15:30", "appointment_duration": "40 min"},
        headers=headers_a,
    )
    assert upd_res.status_code == 200

    # Query RAG staff context for Org A
    ctx_a = rag_service.retrieve_staff_context("Who is available in Cardiology?", organization_id=101)
    assert "Dr. Sarah Adams" in ctx_a
    assert "07:30-15:30" in ctx_a
    assert "40 min" in ctx_a

    # Query RAG staff context for Org B -> Must NOT contain Sarah Adams
    ctx_b = rag_service.retrieve_staff_context("Who is available in Cardiology?", organization_id=102)
    assert "Dr. Sarah Adams" not in ctx_b
    assert "Dr. Brian Baker" in ctx_b

    # Step 12: Voice Assistant Retrieval test via rag_service.retrieve_context
    rag_ret_a = rag_service.retrieve_context("Tell me about Dr Sarah Adams hours", organization_id=101)
    assert "Dr. Sarah Adams" in rag_ret_a
    assert "07:30-15:30" in rag_ret_a

    rag_ret_b = rag_service.retrieve_context("Tell me about Dr Sarah Adams hours", organization_id=102)
    assert "Dr. Sarah Adams" not in rag_ret_b

    # Step 13: Staff deactivation and deletion test
    # Deactivate Staff A
    client.put(f"/api/v1/staff/{staff_a_id}", json={"status": "Offline"}, headers=headers_a)
    ctx_a_deactivated = rag_service.retrieve_staff_context("Is Sarah Adams available?", organization_id=101)
    assert "Accepting Appointments: No" in ctx_a_deactivated or "CANNOT accept new appointments" in ctx_a_deactivated

    # Delete Staff A
    del_res = client.delete(f"/api/v1/staff/{staff_a_id}", headers=headers_a)
    assert del_res.status_code == 200

    ctx_a_deleted = rag_service.retrieve_staff_context("Is Sarah Adams available?", organization_id=101)
    assert "Dr. Sarah Adams" not in ctx_a_deleted


def test_step_14_super_admin_access(setup_tenants):
    """
    Step 14:
    Verify Super Admin can view all organizations or filter by organization_id.
    """
    token_sa = setup_tenants["token_sa"]
    headers_sa = {"Authorization": f"Bearer {token_sa}"}

    # Global listing without organization_id filter
    all_staff = client.get("/api/v1/staff", headers=headers_sa).json()
    assert isinstance(all_staff, list)

    # Scoped listing for Org B (102)
    scoped_b = client.get("/api/v1/staff?organization_id=102", headers=headers_sa).json()
    assert all(s["organization_id"] == 102 for s in scoped_b)
    assert any(s["name"] == "Dr. Brian Baker" for s in scoped_b)
