"""
Unit & Integration Test Suite for Department-wise Multiple Staff Management.

Tests and verifies:
1. Create a department using the existing API/service.
2. Add three existing employees to that department.
3. Confirm all three employees appear under the correct department with actual IDs, names, roles, and contacts.
4. Refresh DB session and confirm assignments persist.
5. Add another employee to the same department.
6. Prevent assigning the same employee twice (duplicate assignment rejection with 400 Bad Request).
7. Remove one employee from the department without deleting the employee record.
8. Verify another organization cannot access or modify these assignments (strict multi-tenant isolation).
9. Confirm existing department creation and staff directory workflows still work seamlessly.
"""

import pytest
from fastapi import HTTPException
from sqlalchemy.orm import Session

from src.config.database import SessionLocal
from src.models.organization import Organization
from src.models.user import User
from src.models.department import Department, DepartmentStaff
from src.models.staff import Staff, StaffStatus
from src.services.department_service import DepartmentService
from src.services.staff_service import StaffService
from src.validators.department_validator import DepartmentCreate, DepartmentUpdate, DepartmentStaffAssign


@pytest.fixture
def db_session():
    db = SessionLocal()
    yield db
    db.close()


@pytest.fixture
def setup_tenants(db_session: Session):
    """Sets up two isolated tenant organizations with test departments and staff."""
    # Ensure Organization 101 exists
    org1 = db_session.query(Organization).filter(Organization.id == 101).first()
    if not org1:
        org1 = Organization(id=101, name="Apex Cardiology Center", organization_type="Hospital", status="active")
        db_session.add(org1)

    # Ensure Organization 102 exists
    org2 = db_session.query(Organization).filter(Organization.id == 102).first()
    if not org2:
        org2 = Organization(id=102, name="Zenith Dental Care", organization_type="Clinic", status="active")
        db_session.add(org2)

    db_session.commit()
    return {"org1": org1, "org2": org2}


def test_01_create_department_existing_form_and_api(db_session: Session, setup_tenants):
    """Scenario 1: Create a department using the existing form and existing API."""
    svc = DepartmentService(db_session)
    payload = DepartmentCreate(
        name="Cardiology",
        code="CARD",
        description="Comprehensive cardiac care, interventions, and diagnostics.",
        head_of_department="Dr. Priya Sharma",
        phone="+919876543210",
        email="cardiology@apex.com",
        status="Active"
    )

    dept = svc.create_department(payload, organization_id=101)
    assert dept.id is not None
    assert dept.name == "Cardiology"
    assert dept.code == "CARD"
    assert dept.head_of_department == "Dr. Priya Sharma"
    assert dept.status == "Active"
    assert dept.staff_count == 0


def test_02_and_03_add_three_employees_and_verify_details(db_session: Session, setup_tenants):
    """
    Scenario 2 & 3:
    - Add three existing employees to that department.
    - Confirm all three employees appear under the correct department with actual IDs, names, roles, and contacts.
    """
    dept_svc = DepartmentService(db_session)
    staff_svc = StaffService(db_session)

    # Fetch Cardiology department in Org 101
    dept = db_session.query(Department).filter(Department.name == "Cardiology", Department.organization_id == 101).first()
    assert dept is not None

    # Create 3 existing staff members in Org 101
    s1 = Staff(
        name="Dr. Alistair Finch",
        organization_id=101,
        department="General",
        email="afinch@apex.com",
        phone="+919876500001",
        working_hours="08:00-16:00",
        appointment_duration="30 min",
        status=StaffStatus.AVAILABLE
    )
    s2 = Staff(
        name="Dr. Beatrice Vance",
        organization_id=101,
        department="General",
        email="bvance@apex.com",
        phone="+919876500002",
        working_hours="09:00-17:00",
        appointment_duration="45 min",
        status=StaffStatus.AVAILABLE
    )
    s3 = Staff(
        name="Nurse Carol Danvers",
        organization_id=101,
        department="General",
        email="cdanvers@apex.com",
        phone="+919876500003",
        working_hours="07:00-15:00",
        appointment_duration="15 min",
        status=StaffStatus.AVAILABLE
    )
    db_session.add_all([s1, s2, s3])
    db_session.commit()
    db_session.refresh(s1)
    db_session.refresh(s2)
    db_session.refresh(s3)

    # Assign all 3 existing employees to Cardiology
    assigned = dept_svc.assign_staff_to_department(
        dept_id=dept.id,
        staff_ids=[s1.id, s2.id, s3.id],
        organization_id=101
    )

    assert len(assigned) == 3
    assigned_names = [a.name for a in assigned]
    assigned_ids = [a.staff_id for a in assigned]

    assert "Dr. Alistair Finch" in assigned_names
    assert "Dr. Beatrice Vance" in assigned_names
    assert "Nurse Carol Danvers" in assigned_names
    assert s1.id in assigned_ids
    assert s2.id in assigned_ids
    assert s3.id in assigned_ids

    # Verify contact info and working hours match database
    finch_entry = next(a for a in assigned if a.staff_id == s1.id)
    assert finch_entry.email == "afinch@apex.com"
    assert finch_entry.phone == "+919876500001"
    assert finch_entry.working_hours == "08:00-16:00"
    assert finch_entry.department == "Cardiology"

    # Verify department staff_count reflects 3
    dept_out = dept_svc.get_department_out(dept.id, organization_id=101)
    assert dept_out.staff_count == 3


def test_04_persistence_across_fresh_session(setup_tenants):
    """Scenario 4: Refresh the session/page and confirm assignments persist in database."""
    fresh_db = SessionLocal()
    try:
        dept_svc = DepartmentService(fresh_db)
        dept = fresh_db.query(Department).filter(Department.name == "Cardiology", Department.organization_id == 101).first()
        assert dept is not None

        staff_list = dept_svc.get_department_staff(dept.id, organization_id=101)
        assert len(staff_list) == 3

        # Confirm mapping table records exist
        mappings = fresh_db.query(DepartmentStaff).filter(
            DepartmentStaff.department_id == dept.id,
            DepartmentStaff.organization_id == 101
        ).all()
        assert len(mappings) == 3
    finally:
        fresh_db.close()


def test_05_add_fourth_employee(db_session: Session, setup_tenants):
    """Scenario 5: Add another employee to the same department."""
    dept_svc = DepartmentService(db_session)
    dept = db_session.query(Department).filter(Department.name == "Cardiology", Department.organization_id == 101).first()
    assert dept is not None

    s4 = Staff(
        name="Receptionist Derek Cole",
        organization_id=101,
        department="General",
        email="dcole@apex.com",
        phone="+919876500004",
        working_hours="08:00-18:00",
        appointment_duration="15 min",
        status=StaffStatus.AVAILABLE
    )
    db_session.add(s4)
    db_session.commit()
    db_session.refresh(s4)

    # Assign 4th employee individually
    assigned = dept_svc.assign_staff_to_department(
        dept_id=dept.id,
        staff_ids=[s4.id],
        organization_id=101
    )

    assert len(assigned) == 4
    assert any(a.staff_id == s4.id for a in assigned)

    dept_out = dept_svc.get_department_out(dept.id, organization_id=101)
    assert dept_out.staff_count == 4


def test_06_prevent_duplicate_assignment(db_session: Session, setup_tenants):
    """Scenario 6: Prevent assigning the same employee twice (raises 400 Bad Request)."""
    dept_svc = DepartmentService(db_session)
    dept = db_session.query(Department).filter(Department.name == "Cardiology", Department.organization_id == 101).first()
    assert dept is not None

    # Get one of the already assigned staff
    staff_member = db_session.query(Staff).filter(Staff.name == "Dr. Alistair Finch", Staff.organization_id == 101).first()
    assert staff_member is not None

    # Attempt to assign the same employee again
    with pytest.raises(HTTPException) as exc_info:
        dept_svc.assign_staff_to_department(
            dept_id=dept.id,
            staff_ids=[staff_member.id],
            organization_id=101
        )

    assert exc_info.value.status_code == 400
    assert "already assigned" in exc_info.value.detail.lower()

    # Confirm count did not increase
    dept_out = dept_svc.get_department_out(dept.id, organization_id=101)
    assert dept_out.staff_count == 4


def test_07_remove_staff_without_deleting_employee_record(db_session: Session, setup_tenants):
    """
    Scenario 7:
    - Remove one employee from the department.
    - Confirm staff count decreases.
    - Confirm the employee record in staff table is NOT deleted.
    """
    dept_svc = DepartmentService(db_session)
    dept = db_session.query(Department).filter(Department.name == "Cardiology", Department.organization_id == 101).first()
    assert dept is not None

    staff_to_remove = db_session.query(Staff).filter(Staff.name == "Receptionist Derek Cole", Staff.organization_id == 101).first()
    assert staff_to_remove is not None
    removed_staff_id = staff_to_remove.id

    # Remove from department
    dept_svc.remove_staff_from_department(dept_id=dept.id, staff_id=removed_staff_id, organization_id=101)

    # Verify staff count is now 3
    remaining_staff = dept_svc.get_department_staff(dept.id, organization_id=101)
    assert len(remaining_staff) == 3
    assert not any(a.staff_id == removed_staff_id for a in remaining_staff)

    # CRUCIAL: Verify Derek Cole still exists in staff table!
    retained_staff = db_session.query(Staff).filter(Staff.id == removed_staff_id).first()
    assert retained_staff is not None
    assert retained_staff.name == "Receptionist Derek Cole"
    assert retained_staff.email == "dcole@apex.com"


def test_08_multi_tenant_isolation(db_session: Session, setup_tenants):
    """
    Scenario 8:
    - Organization 102 cannot view Organization 101's department staff (returns 404).
    - Organization 102 cannot assign Organization 101's staff or department.
    - Organization 102 cannot remove Organization 101's staff.
    """
    dept_svc = DepartmentService(db_session)
    dept_org1 = db_session.query(Department).filter(Department.name == "Cardiology", Department.organization_id == 101).first()
    assert dept_org1 is not None

    # Org 102 creates their own staff member
    s_org2 = Staff(
        name="Dr. Ethan Hunt",
        organization_id=102,
        department="Dental",
        email="ehunt@zenith.com",
        status=StaffStatus.AVAILABLE
    )
    db_session.add(s_org2)
    db_session.commit()
    db_session.refresh(s_org2)

    # 1. Org 102 queries Org 101's department -> 404 Not Found
    with pytest.raises(HTTPException) as exc_info:
        dept_svc.get_department_staff(dept_org1.id, organization_id=102)
    assert exc_info.value.status_code == 404

    # 2. Org 102 tries to assign their staff to Org 101's department -> 404 Not Found
    with pytest.raises(HTTPException) as exc_info:
        dept_svc.assign_staff_to_department(dept_org1.id, staff_ids=[s_org2.id], organization_id=102)
    assert exc_info.value.status_code == 404

    # 3. Org 101 tries to assign Org 102's staff -> 404 Not Found (cross-tenant staff rejected)
    with pytest.raises(HTTPException) as exc_info:
        dept_svc.assign_staff_to_department(dept_org1.id, staff_ids=[s_org2.id], organization_id=101)
    assert exc_info.value.status_code == 404

    # 4. Org 102 tries to remove Org 101's staff -> 404 Not Found
    finch = db_session.query(Staff).filter(Staff.name == "Dr. Alistair Finch", Staff.organization_id == 101).first()
    with pytest.raises(HTTPException) as exc_info:
        dept_svc.remove_staff_from_department(dept_org1.id, finch.id, organization_id=102)
    assert exc_info.value.status_code == 404


def test_09_existing_department_crud_and_staff_workflows_preserved(db_session: Session, setup_tenants):
    """
    Scenario 9: Confirm existing department creation, update, listing, and staff directory continue to work.
    """
    dept_svc = DepartmentService(db_session)
    staff_svc = StaffService(db_session)

    # 1. Department listing returns staff_count
    all_depts = dept_svc.list_departments(organization_id=101)
    assert len(all_depts) >= 1
    cardio = next(d for d in all_depts if d.name == "Cardiology")
    assert cardio.staff_count == 3

    # 2. Update department metadata preserves staff mappings
    updated = dept_svc.update_department(
        cardio.id,
        DepartmentUpdate(phone="+919876599999", description="Updated cardiac department description"),
        organization_id=101
    )
    assert updated.phone == "+919876599999"
    assert updated.staff_count == 3

    # 3. Existing StaffService methods work seamlessly
    staff_in_cardio = staff_svc.get_staff_for_department("Cardiology", organization_id=101)
    assert len(staff_in_cardio) >= 1

    # 4. Staff can be queried by name
    found_staff = staff_svc.find_staff_by_name("Alistair Finch", organization_id=101)
    assert found_staff is not None
    assert found_staff.email == "afinch@apex.com"


def test_10_api_http_endpoints_full_lifecycle(db_session: Session, setup_tenants):
    """
    Scenario 10: Verify FastAPI HTTP endpoints with authentication headers.
    - POST /departments/{dept_id}/staff
    - GET /departments/{dept_id}/staff
    - DELETE /departments/{dept_id}/staff/{staff_id}
    - GET /departments
    """
    from fastapi.testclient import TestClient
    from src.app import app
    from src.config.jwt import create_access_token

    client = TestClient(app)

    # Org 101 and 102 test users
    u101 = db_session.query(User).filter(User.email == "admin101@apex.com").first()
    if not u101:
        u101 = User(id=9101, email="admin101@apex.com", name="Apex Admin", password_hash="dummy_hash_101", role="admin", organization_id=101, is_active=True)
        db_session.add(u101)

    u102 = db_session.query(User).filter(User.email == "admin102@zenith.com").first()
    if not u102:
        u102 = User(id=9102, email="admin102@zenith.com", name="Zenith Admin", password_hash="dummy_hash_102", role="admin", organization_id=102, is_active=True)
        db_session.add(u102)

    db_session.commit()

    # Org 101 user token
    token_101 = create_access_token({
        "sub": str(u101.id),
        "email": u101.email,
        "role": "admin",
        "organization_id": 101
    })
    headers_101 = {"Authorization": f"Bearer {token_101}"}

    # Org 102 user token
    token_102 = create_access_token({
        "sub": str(u102.id),
        "email": u102.email,
        "role": "admin",
        "organization_id": 102
    })
    headers_102 = {"Authorization": f"Bearer {token_102}"}

    # Create a new department in Org 101 via HTTP
    resp_create = client.post(
        "/departments",
        json={
            "name": "Neurology Center",
            "code": "NEURO",
            "description": "Brain and nerve care",
            "head_of_department": "Dr. Xavier",
            "status": "Active"
        },
        headers=headers_101
    )
    assert resp_create.status_code == 201
    neuro_id = resp_create.json()["id"]

    # Create two staff in Org 101
    s_n1 = Staff(name="Dr. Stephen Strange", organization_id=101, department="General", status=StaffStatus.AVAILABLE)
    s_n2 = Staff(name="Nurse Christine Palmer", organization_id=101, department="General", status=StaffStatus.AVAILABLE)
    db_session.add_all([s_n1, s_n2])
    db_session.commit()
    db_session.refresh(s_n1)
    db_session.refresh(s_n2)

    # 1. Assign via HTTP POST
    resp_assign = client.post(
        f"/departments/{neuro_id}/staff",
        json={"staff_ids": [s_n1.id, s_n2.id]},
        headers=headers_101
    )
    print("RESP_ASSIGN:", resp_assign.status_code, resp_assign.text)
    assert resp_assign.status_code == 200
    assigned_json = resp_assign.json()
    assert len(assigned_json) == 2
    assert any(a["name"] == "Dr. Stephen Strange" for a in assigned_json)

    # 2. Get department staff via HTTP GET
    resp_get = client.get(f"/departments/{neuro_id}/staff", headers=headers_101)
    assert resp_get.status_code == 200
    assert len(resp_get.json()) == 2

    # 3. Org 102 attempt to get Org 101 department staff -> 404 Forbidden/Not Found
    resp_cross = client.get(f"/departments/{neuro_id}/staff", headers=headers_102)
    assert resp_cross.status_code == 404

    # 4. Remove one staff member via HTTP DELETE
    resp_del = client.delete(f"/departments/{neuro_id}/staff/{s_n1.id}", headers=headers_101)
    assert resp_del.status_code == 200

    # Verify count is now 1
    resp_get_after = client.get(f"/departments/{neuro_id}/staff", headers=headers_101)
    assert resp_get_after.status_code == 200
    assert len(resp_get_after.json()) == 1
    assert resp_get_after.json()[0]["staff_id"] == s_n2.id

    # Verify employee Stephen Strange still exists in DB
    assert db_session.query(Staff).filter(Staff.id == s_n1.id).first() is not None

    # 5. List departments via HTTP GET returns staff_count == 1
    resp_list = client.get("/departments", headers=headers_101)
    assert resp_list.status_code == 200
    neuro_dept = next(d for d in resp_list.json() if d["id"] == neuro_id)
    assert neuro_dept["staff_count"] == 1

