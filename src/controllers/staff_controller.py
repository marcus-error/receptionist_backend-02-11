from typing import List, Optional

from fastapi import Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from src.config.database import get_db
from src.middleware.auth_middleware import get_current_user
from src.models.department import Department
from src.models.user import User
from src.services.staff_service import StaffService
from src.validators.staff_validator import StaffCreate, StaffOut, StaffUpdate


def _get_org_id(user: Optional[User], requested_org_id: Optional[int] = None) -> Optional[int]:
    """
    Enforces strict organization scoping based on the authenticated user's credentials.
    - Non-super admins are strictly bound to their token's organization_id.
    - Super admins can optionally specify an organization_id query parameter, or None for global view.
    """
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token required"
        )
    if user.role == "super_admin":
        return requested_org_id if requested_org_id is not None else user.organization_id

    if user.organization_id is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User is not assigned to an organization"
        )

    if requested_org_id is not None and requested_org_id != user.organization_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Cannot access staff from another organization"
        )

    return user.organization_id


def _validate_department_ownership(dept_name: Optional[str], org_id: Optional[int], db: Session) -> None:
    """
    Validates that if a department exists in the database, it belongs to the target organization.
    Prevents cross-tenant department contamination.
    """
    if not dept_name or not org_id:
        return
    clean_name = dept_name.strip()
    other_org_dept = db.query(Department).filter(
        Department.name.ilike(clean_name),
        Department.organization_id != org_id
    ).first()
    same_org_dept = db.query(Department).filter(
        Department.name.ilike(clean_name),
        Department.organization_id == org_id
    ).first()

    if other_org_dept and not same_org_dept:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Department '{dept_name}' belongs to another organization and cannot be linked."
        )


def list_staff(
    organization_id: Optional[int] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> List[StaffOut]:
    org_id = _get_org_id(current_user, requested_org_id=organization_id)
    return StaffService(db).list_staff(organization_id=org_id)


def get_staff(
    staff_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> StaffOut:
    target_org = None if current_user.role == "super_admin" else _get_org_id(current_user)
    return StaffService(db).get_staff(staff_id, organization_id=target_org)


def create_staff(
    payload: StaffCreate,
    organization_id: Optional[int] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> StaffOut:
    if current_user.role == "super_admin":
        target_org = organization_id or payload.organization_id or current_user.organization_id or 1
    else:
        target_org = _get_org_id(current_user)

    _validate_department_ownership(payload.department, target_org, db)
    return StaffService(db).create_staff(payload, organization_id=target_org)


def update_staff(
    staff_id: int,
    payload: StaffUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> StaffOut:
    target_org = None if current_user.role == "super_admin" else _get_org_id(current_user)
    staff = StaffService(db).get_staff(staff_id, organization_id=target_org)

    if payload.department:
        _validate_department_ownership(payload.department, staff.organization_id, db)

    return StaffService(db).update_staff(staff_id, payload, organization_id=target_org)


def delete_staff(
    staff_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> dict:
    target_org = None if current_user.role == "super_admin" else _get_org_id(current_user)
    StaffService(db).delete_staff(staff_id, organization_id=target_org)
    return {"message": "Staff member deleted"}
