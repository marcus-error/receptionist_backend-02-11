from typing import Dict, List, Optional, Tuple
from sqlalchemy.orm import Session
from src.models.department import Department, DepartmentStaff
from src.models.staff import Staff


class DepartmentRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_all(self, organization_id: int) -> List[Department]:
        return (
            self.db.query(Department)
            .filter(Department.organization_id == organization_id)
            .order_by(Department.id.asc())
            .all()
        )

    def get_by_id(self, dept_id: int, organization_id: int) -> Optional[Department]:
        return (
            self.db.query(Department)
            .filter(Department.id == dept_id, Department.organization_id == organization_id)
            .first()
        )

    def get_by_name(self, name: str, organization_id: int) -> Optional[Department]:
        return (
            self.db.query(Department)
            .filter(Department.name.ilike(name.strip()), Department.organization_id == organization_id)
            .first()
        )

    def create(self, data: dict) -> Department:
        dept = Department(**data)
        self.db.add(dept)
        self.db.commit()
        self.db.refresh(dept)
        return dept

    def update(self, dept: Department, data: dict) -> Department:
        for key, value in data.items():
            if value is not None:
                setattr(dept, key, value)
        self.db.commit()
        self.db.refresh(dept)
        return dept

    def delete(self, dept: Department) -> None:
        self.db.delete(dept)
        self.db.commit()

    def get_department_staff(self, dept_id: int, organization_id: int) -> List[Tuple[Staff, Optional[DepartmentStaff]]]:
        """
        Retrieves all staff assigned to a department, ensuring full multi-tenant isolation.
        Automatically bridges legacy staff records whose string department matches the department name.
        """
        dept = self.get_by_id(dept_id, organization_id)
        if not dept:
            return []

        # 1. Fetch all explicit mappings from department_staff
        mappings = (
            self.db.query(Staff, DepartmentStaff)
            .join(DepartmentStaff, DepartmentStaff.staff_id == Staff.id)
            .filter(
                DepartmentStaff.department_id == dept_id,
                DepartmentStaff.organization_id == organization_id,
                Staff.organization_id == organization_id
            )
            .all()
        )

        mapped_staff_ids = {s.id for s, _ in mappings}

        # 2. Check for legacy staff records matching department name without mapping row yet
        legacy_staff = (
            self.db.query(Staff)
            .filter(
                Staff.organization_id == organization_id,
                Staff.department.ilike(dept.name.strip())
            )
            .all()
        )

        results = list(mappings)
        for s in legacy_staff:
            if s.id not in mapped_staff_ids:
                try:
                    new_mapping = DepartmentStaff(
                        department_id=dept.id,
                        staff_id=s.id,
                        organization_id=organization_id
                    )
                    self.db.add(new_mapping)
                    self.db.commit()
                    self.db.refresh(new_mapping)
                    results.append((s, new_mapping))
                    mapped_staff_ids.add(s.id)
                except Exception:
                    self.db.rollback()
                    results.append((s, None))

        return results

    def get_staff_counts_for_org(self, organization_id: int) -> Dict[int, int]:
        """
        Returns a dictionary of {dept_id: staff_count} for all departments in the organization.
        """
        departments = self.get_all(organization_id)
        counts: Dict[int, int] = {}
        for d in departments:
            staff_list = self.get_department_staff(d.id, organization_id)
            counts[d.id] = len(staff_list)
        return counts

    def is_staff_assigned(self, dept_id: int, staff_id: int, organization_id: int) -> bool:
        """Checks if a staff member is already assigned to this department."""
        dept = self.get_by_id(dept_id, organization_id)
        if not dept:
            return False

        mapping = (
            self.db.query(DepartmentStaff)
            .filter(
                DepartmentStaff.department_id == dept_id,
                DepartmentStaff.staff_id == staff_id,
                DepartmentStaff.organization_id == organization_id
            )
            .first()
        )
        if mapping:
            return True

        # Check legacy staff match
        staff = self.db.query(Staff).filter(Staff.id == staff_id, Staff.organization_id == organization_id).first()
        if staff and staff.department and staff.department.strip().lower() == dept.name.strip().lower():
            return True

        return False

    def assign_staff(self, dept_id: int, staff_id: int, organization_id: int) -> DepartmentStaff:
        """Assigns a staff member to a department."""
        mapping = DepartmentStaff(
            department_id=dept_id,
            staff_id=staff_id,
            organization_id=organization_id
        )
        self.db.add(mapping)
        self.db.commit()
        self.db.refresh(mapping)
        return mapping

    def remove_staff(self, dept_id: int, staff_id: int, organization_id: int) -> bool:
        """
        Removes a staff member from a department without deleting the employee/staff record.
        """
        dept = self.get_by_id(dept_id, organization_id)
        staff = self.db.query(Staff).filter(Staff.id == staff_id, Staff.organization_id == organization_id).first()
        if not dept or not staff:
            return False

        # Remove mapping row
        self.db.query(DepartmentStaff).filter(
            DepartmentStaff.department_id == dept_id,
            DepartmentStaff.staff_id == staff_id,
            DepartmentStaff.organization_id == organization_id
        ).delete()

        # If staff's primary department string matches this department, update to another department if available
        if staff.department and staff.department.strip().lower() == dept.name.strip().lower():
            other_mapping = (
                self.db.query(Department)
                .join(DepartmentStaff, DepartmentStaff.department_id == Department.id)
                .filter(
                    DepartmentStaff.staff_id == staff_id,
                    DepartmentStaff.organization_id == organization_id,
                    Department.id != dept_id
                )
                .first()
            )
            if other_mapping:
                staff.department = other_mapping.name
            else:
                staff.department = "General"

        self.db.commit()
        return True
