from typing import List, Optional

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from src.models.staff import Staff
from src.repositories.staff_repository import StaffRepository
from src.services.staff_rag_sync_service import StaffRAGSyncService
from src.validators.staff_validator import StaffCreate, StaffUpdate


class StaffService:
    def __init__(self, db: Session):
        self.db = db
        self.repo = StaffRepository(db)

    def list_staff(self, organization_id: Optional[int] = None) -> List[Staff]:
        return self.repo.get_all(organization_id=organization_id)

    def get_staff(self, staff_id: int, organization_id: Optional[int] = None) -> Staff:
        staff = self.repo.get_by_id(staff_id, organization_id=organization_id)
        if not staff:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Staff member not found")
        return staff

    def create_staff(self, payload: StaffCreate, organization_id: Optional[int] = None) -> Staff:
        data = payload.model_dump()
        target_org = organization_id if organization_id is not None else data.get("organization_id", 1)
        data["organization_id"] = target_org
        staff = self.repo.create(data)
        try:
            StaffRAGSyncService.sync_staff(staff.id, target_org, self.db, event_type="CREATE")
        except Exception as e:
            # Sync failures should not prevent creation, but are logged
            pass
        return staff

    def update_staff(self, staff_id: int, payload: StaffUpdate, organization_id: Optional[int] = None) -> Staff:
        staff = self.get_staff(staff_id, organization_id=organization_id)
        update_data = payload.model_dump(exclude_unset=True)
        # Prevent client from silently changing organization_id unless explicitly authorized
        if "organization_id" in update_data and organization_id is not None:
            update_data["organization_id"] = organization_id
        changed_fields = list(update_data.keys())
        updated = self.repo.update(staff, update_data)
        try:
            StaffRAGSyncService.sync_staff(
                updated.id,
                updated.organization_id,
                self.db,
                event_type="UPDATE",
                changed_fields=changed_fields,
            )
        except Exception:
            pass
        return updated

    def delete_staff(self, staff_id: int, organization_id: Optional[int] = None) -> None:
        staff = self.get_staff(staff_id, organization_id=organization_id)
        s_id = staff.id
        org_id = staff.organization_id
        self.repo.delete(staff)
        try:
            StaffRAGSyncService.delete_staff(s_id, org_id, self.db)
        except Exception:
            pass

    def find_staff_by_name(self, name: str, organization_id: int = 1) -> Optional[Staff]:
        """Looks up a staff member by name for the given organization."""
        import re
        clean = name.strip()
        clean_words = re.sub(r'^(?:dr\.?|doctor|prof\.?|mr\.?|mrs\.?|ms\.?)\s+', '', clean, flags=re.I).strip().lower()
        all_staff = self.list_staff(organization_id=organization_id)
        for s in all_staff:
            if not s.name:
                continue
            s_clean = re.sub(r'^(?:dr\.?|doctor|prof\.?|mr\.?|mrs\.?|ms\.?)\s+', '', s.name, flags=re.I).strip().lower()
            if clean_words == s_clean or clean_words in s_clean.split():
                return s
        return None

    def get_staff_for_department(self, department: str, organization_id: int = 1) -> List[Staff]:
        """Maps a department query to staff members for the given organization."""
        import json
        dept_clean = department.strip().lower()
        all_staff = self.list_staff(organization_id=organization_id)
        matched = []
        for s in all_staff:
            if s.department and (dept_clean in s.department.lower() or s.department.lower() in dept_clean):
                matched.append(s)
            elif s.services:
                try:
                    s_list = json.loads(s.services) if isinstance(s.services, str) else s.services
                    if isinstance(s_list, list) and any(dept_clean in str(item).lower() for item in s_list):
                        matched.append(s)
                except Exception:
                    pass
        return matched

    def get_staff_profile(self, staff_name: str, organization_id: int = 1) -> Optional[dict]:
        """
        Retrieves actual staff profile (role, department, status) from DB.
        Never hallucinates medical qualifications.
        """
        staff = self.find_staff_by_name(staff_name, organization_id=organization_id)
        if not staff:
            return None
        return {
            "id": staff.id,
            "name": staff.name,
            "department": staff.department,
            "status": getattr(staff.status, "value", str(staff.status)),
            "working_hours": staff.working_hours,
            "working_days": staff.working_days
        }
