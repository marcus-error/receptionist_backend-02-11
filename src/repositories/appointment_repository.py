"""
Data-access layer for appointments. Nothing here knows about HTTP -
it only talks to the DB session. Services call into this layer instead
of touching SQLAlchemy models directly.
"""
from datetime import datetime
from typing import List, Optional, Tuple, Union

from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from src.models.appointment import Appointment


def _parse_appointment_date(date_str: Optional[str]):
    if not date_str:
        return None
    cleaned = str(date_str).strip()
    for fmt in ("%d %b %Y", "%d %B %Y", "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            pass
    return None


class AppointmentRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_all(
        self,
        organization_id: Optional[int] = None,
        page: Optional[int] = None,
        limit: Optional[int] = None,
        search: Optional[str] = None,
        status: Optional[str] = None,
        time_filter: Optional[str] = None,
    ) -> Union[List[Appointment], Tuple[List[Appointment], int]]:
        query = self.db.query(Appointment)
        if organization_id is not None:
            query = query.filter(Appointment.organization_id == organization_id)

        # 1. Search filter across text columns
        if search and search.strip():
            term = f"%{search.strip()}%"
            query = query.filter(
                or_(
                    Appointment.customer.ilike(term),
                    Appointment.customer_id.ilike(term),
                    Appointment.phone.ilike(term),
                    Appointment.email.ilike(term),
                    Appointment.staff.ilike(term),
                    Appointment.department.ilike(term),
                    Appointment.service.ilike(term),
                )
            )

        # 2. Status filter
        if status and status.strip().lower() != "all":
            query = query.filter(func.lower(Appointment.status) == status.strip().lower())

        # 3. Time filter (all, today, weekly, monthly)
        if time_filter and time_filter.strip().lower() != "all":
            all_records = query.order_by(Appointment.id.desc()).all()
            today_date = datetime.now().date()
            filtered_records = []
            tf = time_filter.strip().lower()

            for appt in all_records:
                parsed_d = _parse_appointment_date(appt.appointment_date)
                if not parsed_d:
                    filtered_records.append(appt)
                    continue

                if tf == "today":
                    if parsed_d == today_date:
                        filtered_records.append(appt)
                elif tf == "weekly":
                    diff_days = abs((parsed_d - today_date).days)
                    if diff_days <= 7 or (parsed_d.year == today_date.year and parsed_d.month == today_date.month and abs(parsed_d.day - today_date.day) <= 7):
                        filtered_records.append(appt)
                elif tf == "monthly":
                    diff_days = abs((parsed_d - today_date).days)
                    if diff_days <= 30 or (parsed_d.year == today_date.year and parsed_d.month == today_date.month):
                        filtered_records.append(appt)
                else:
                    filtered_records.append(appt)

            total = len(filtered_records)
            if page is not None or limit is not None:
                eff_limit = limit or 10
                eff_page = page or 1
                offset = (eff_page - 1) * eff_limit
                paged = filtered_records[offset : offset + eff_limit]
                return paged, total
            return filtered_records

        # Pagination without time_filter
        if page is not None or limit is not None:
            total = query.count()
            eff_limit = limit or 10
            eff_page = page or 1
            offset = (eff_page - 1) * eff_limit
            records = query.order_by(Appointment.id.desc()).offset(offset).limit(eff_limit).all()
            return records, total

        return query.order_by(Appointment.id.desc()).all()

    def get_by_id(self, appointment_id: int, organization_id: Optional[int] = None) -> Optional[Appointment]:
        query = self.db.query(Appointment).filter(Appointment.id == appointment_id)
        if organization_id is not None:
            query = query.filter(Appointment.organization_id == organization_id)
        return query.first()

    def create(self, data: dict) -> Appointment:
        from src.config.database import execute_with_retry
        valid_cols = {c.name for c in Appointment.__table__.columns}
        filtered_data = {k: v for k, v in data.items() if k in valid_cols}
        appointment = Appointment(**filtered_data)
        def _commit_tx():
            self.db.add(appointment)
            self.db.commit()
            self.db.refresh(appointment)
            return appointment
        return execute_with_retry(_commit_tx)

    def update(self, appointment: Appointment, data: dict) -> Appointment:
        for key, value in data.items():
            if value is not None:
                setattr(appointment, key, value)
        self.db.commit()
        self.db.refresh(appointment)
        return appointment

    def delete(self, appointment: Appointment) -> None:
        self.db.delete(appointment)
        self.db.commit()
