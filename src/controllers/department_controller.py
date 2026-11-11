from typing import List, Optional
from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.config.database import get_db
from src.middleware.auth_middleware import get_current_user
from src.models.user import User
from src.services.department_service import DepartmentService
from src.validators.department_validator import (
    DepartmentCreate,
    DepartmentOut,
    DepartmentStaffAssign,
    DepartmentStaffOut,
    DepartmentUpdate,
)


def _get_org_id(user: Optional[User]) -> int:
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication token required")
    if user.organization_id is not None:
        return user.organization_id
    if user.role == "super_admin":
        return 1
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User is not assigned to an organization")


def list_departments(
    current_user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> List[DepartmentOut]:
    org_id = _get_org_id(current_user)
    return DepartmentService(db).list_departments(organization_id=org_id)


def get_department(
    dept_id: int,
    current_user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> DepartmentOut:
    org_id = _get_org_id(current_user)
    return DepartmentService(db).get_department_out(dept_id, organization_id=org_id)


def create_department(
    payload: DepartmentCreate,
    current_user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> DepartmentOut:
    org_id = _get_org_id(current_user)
    return DepartmentService(db).create_department(payload, organization_id=org_id)


def update_department(
    dept_id: int,
    payload: DepartmentUpdate,
    current_user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> DepartmentOut:
    org_id = _get_org_id(current_user)
    return DepartmentService(db).update_department(dept_id, payload, organization_id=org_id)


def delete_department(
    dept_id: int,
    current_user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> dict:
    org_id = _get_org_id(current_user)
    DepartmentService(db).delete_department(dept_id, organization_id=org_id)
    return {"message": "Department deleted successfully"}


def get_department_staff(
    dept_id: int,
    current_user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> List[DepartmentStaffOut]:
    org_id = _get_org_id(current_user)
    return DepartmentService(db).get_department_staff(dept_id, organization_id=org_id)


def assign_staff_to_department(
    dept_id: int,
    payload: DepartmentStaffAssign,
    current_user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> List[DepartmentStaffOut]:
    org_id = _get_org_id(current_user)
    staff_ids = payload.staff_ids or []
    if payload.staff_id and payload.staff_id not in staff_ids:
        staff_ids.append(payload.staff_id)
    return DepartmentService(db).assign_staff_to_department(dept_id, staff_ids=staff_ids, organization_id=org_id)


def remove_staff_from_department(
    dept_id: int,
    staff_id: int,
    current_user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> dict:
    org_id = _get_org_id(current_user)
    DepartmentService(db).remove_staff_from_department(dept_id, staff_id=staff_id, organization_id=org_id)
    return {"message": "Staff member removed from department successfully"}
