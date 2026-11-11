from typing import List, Optional
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from src.models.department import Department
from src.models.staff import Staff
from src.repositories.department_repository import DepartmentRepository
from src.validators.department_validator import (
    DepartmentCreate,
    DepartmentOut,
    DepartmentStaffOut,
    DepartmentUpdate,
)


class DepartmentService:
    def __init__(self, db: Session):
        self.db = db
        self.repo = DepartmentRepository(db)

    def _to_out(self, dept: Department, staff_count: int = 0) -> DepartmentOut:
        return DepartmentOut(
            id=dept.id,
            organization_id=dept.organization_id,
            name=dept.name,
            code=dept.code,
            description=dept.description,
            head_of_department=dept.head_of_department,
            phone=dept.phone,
            email=dept.email,
            status=dept.status,
            staff_count=staff_count,
            created_at=dept.created_at,
            updated_at=dept.updated_at,
        )

    def list_departments(self, organization_id: int) -> List[DepartmentOut]:
        departments = self.repo.get_all(organization_id=organization_id)
        counts = self.repo.get_staff_counts_for_org(organization_id=organization_id)
        return [self._to_out(d, counts.get(d.id, 0)) for d in departments]

    def get_department(self, dept_id: int, organization_id: int) -> Department:
        dept = self.repo.get_by_id(dept_id, organization_id=organization_id)
        if not dept:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Department with ID {dept_id} not found in your organization"
            )
        return dept

    def get_department_out(self, dept_id: int, organization_id: int) -> DepartmentOut:
        dept = self.get_department(dept_id, organization_id=organization_id)
        staff_list = self.repo.get_department_staff(dept.id, organization_id=organization_id)
        return self._to_out(dept, len(staff_list))

    def create_department(self, payload: DepartmentCreate, organization_id: int) -> DepartmentOut:
        data = payload.model_dump()
        data["organization_id"] = organization_id
        
        # Check duplicate name within the same organization
        existing = self.repo.get_by_name(payload.name, organization_id=organization_id)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"A department named '{payload.name}' already exists in your organization."
            )
        dept = self.repo.create(data)
        return self._to_out(dept, 0)

    def update_department(self, dept_id: int, payload: DepartmentUpdate, organization_id: int) -> DepartmentOut:
        dept = self.get_department(dept_id, organization_id=organization_id)
        data = payload.model_dump(exclude_unset=True)
        
        if "name" in data and data["name"] != dept.name:
            existing = self.repo.get_by_name(data["name"], organization_id=organization_id)
            if existing and existing.id != dept_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"A department named '{data['name']}' already exists in your organization."
                )
        updated = self.repo.update(dept, data)
        try:
            from src.services.staff_rag_sync_service import StaffRAGSyncService
            StaffRAGSyncService.sync_organization_staff(organization_id, self.db)
        except Exception:
            pass
        staff_list = self.repo.get_department_staff(updated.id, organization_id=organization_id)
        return self._to_out(updated, len(staff_list))

    def delete_department(self, dept_id: int, organization_id: int) -> None:
        dept = self.get_department(dept_id, organization_id=organization_id)
        self.repo.delete(dept)
        try:
            from src.services.staff_rag_sync_service import StaffRAGSyncService
            StaffRAGSyncService.sync_organization_staff(organization_id, self.db)
        except Exception:
            pass

    def get_department_staff(self, dept_id: int, organization_id: int) -> List[DepartmentStaffOut]:
        """Lists all staff members assigned to a department."""
        dept = self.get_department(dept_id, organization_id=organization_id)
        raw_staff = self.repo.get_department_staff(dept_id=dept.id, organization_id=organization_id)
        
        out_list: List[DepartmentStaffOut] = []
        for staff, mapping in raw_staff:
            status_val = staff.status.value if hasattr(staff.status, "value") else str(staff.status or "Available")
            out_list.append(
                DepartmentStaffOut(
                    id=mapping.id if mapping else staff.id,
                    staff_id=staff.id,
                    name=staff.name,
                    department=dept.name,
                    organization_id=staff.organization_id,
                    email=staff.email,
                    phone=staff.phone,
                    status=status_val,
                    working_hours=staff.working_hours or "09:00-18:00",
                    appointment_duration=staff.appointment_duration or "30 min",
                    role=getattr(staff, "role", None) or "Staff Member",
                    assigned_at=mapping.created_at if mapping else staff.created_at,
                )
            )
        return out_list

    def assign_staff_to_department(
        self, dept_id: int, staff_ids: List[int], organization_id: int
    ) -> List[DepartmentStaffOut]:
        """
        Assigns multiple existing staff members to a department.
        Validates tenant isolation and prevents duplicate assignments.
        """
        dept = self.get_department(dept_id, organization_id=organization_id)

        if not staff_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one staff member ID must be provided for assignment."
            )

        # Validate all staff exist in the same organization
        staff_records = (
            self.db.query(Staff)
            .filter(Staff.id.in_(staff_ids), Staff.organization_id == organization_id)
            .all()
        )
        found_ids = {s.id for s in staff_records}
        missing_ids = set(staff_ids) - found_ids
        if missing_ids:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Staff member(s) with ID(s) {list(missing_ids)} not found in your organization."
            )

        # Check for duplicates before making changes
        for staff in staff_records:
            if self.repo.is_staff_assigned(dept_id=dept.id, staff_id=staff.id, organization_id=organization_id):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Employee '{staff.name}' (ID: {staff.id}) is already assigned to department '{dept.name}'."
                )

        # Assign each staff member
        for staff in staff_records:
            self.repo.assign_staff(dept_id=dept.id, staff_id=staff.id, organization_id=organization_id)

        # Sync RAG so AI assistant has updated department staff knowledge
        try:
            from src.services.staff_rag_sync_service import StaffRAGSyncService
            StaffRAGSyncService.sync_organization_staff(organization_id, self.db)
        except Exception:
            pass

        return self.get_department_staff(dept_id=dept.id, organization_id=organization_id)

    def remove_staff_from_department(
        self, dept_id: int, staff_id: int, organization_id: int
    ) -> None:
        """
        Removes a staff member from a department without deleting the employee record.
        """
        dept = self.get_department(dept_id, organization_id=organization_id)
        staff = self.db.query(Staff).filter(Staff.id == staff_id, Staff.organization_id == organization_id).first()
        if not staff:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Staff member with ID {staff_id} not found in your organization."
            )

        if not self.repo.is_staff_assigned(dept_id=dept.id, staff_id=staff.id, organization_id=organization_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Employee '{staff.name}' (ID: {staff.id}) is not assigned to department '{dept.name}'."
            )

        self.repo.remove_staff(dept_id=dept.id, staff_id=staff.id, organization_id=organization_id)

        # Sync RAG
        try:
            from src.services.staff_rag_sync_service import StaffRAGSyncService
            StaffRAGSyncService.sync_organization_staff(organization_id, self.db)
        except Exception:
            pass
