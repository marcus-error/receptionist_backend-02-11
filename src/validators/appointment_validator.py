import re
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.models.appointment import AppointmentStatus


class AppointmentCreate(BaseModel):
    customer: str = Field(..., min_length=1, max_length=150)
    customer_id: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    location: Optional[str] = None
    department: str = Field(..., min_length=1, max_length=100)
    staff: str = Field(..., min_length=1, max_length=150)
    service: Optional[str] = Field(default=None, max_length=200)
    appointment_date: str = Field(..., min_length=1, max_length=50)
    appointment_time: str = Field(..., min_length=1, max_length=50)
    duration: Optional[str] = "30 Minutes"
    notes: Optional[str] = None
    reminder: Optional[str] = "WhatsApp + Email"
    status: Optional[AppointmentStatus] = AppointmentStatus.PENDING

    @field_validator("location")
    @classmethod
    def validate_location(cls, v: Optional[str]) -> Optional[str]:
        if not v:
            return None
        loc = v.strip()
        if "@" in loc:
            raise ValueError("Location cannot be an email address.")
        if re.search(r'\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b', loc):
            raise ValueError("Location cannot contain an email address.")
        clean_num = re.sub(r'[\s\-\(\)\+]', '', loc)
        if clean_num.isdigit() and len(clean_num) >= 6:
            raise ValueError("Location cannot be a phone number.")
        return loc

    @model_validator(mode="after")
    def validate_location_not_customer_or_email(self) -> "AppointmentCreate":
        loc = getattr(self, "location", None)
        if not loc:
            return self
        loc_clean = loc.strip().lower()
        cust = getattr(self, "customer", "") or ""
        cust_clean = cust.strip().lower()
        if cust_clean:
            if loc_clean == cust_clean:
                raise ValueError("Location cannot be the customer's name.")
            name_parts = [w for w in cust_clean.split() if len(w) > 2]
            if loc_clean in name_parts:
                raise ValueError("Location cannot be the customer's name.")
        em = getattr(self, "email", "") or ""
        if em and "@" in em:
            em_user = em.split("@")[0].strip().lower()
            if loc_clean == em_user:
                raise ValueError("Location cannot be the customer's email username.")
            em_domain = em.split("@")[1].strip().lower()
            if loc_clean == em_domain:
                raise ValueError("Location cannot be an email domain.")
        return self


class AppointmentUpdate(BaseModel):
    customer: Optional[str] = None
    customer_id: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    location: Optional[str] = None
    department: Optional[str] = None
    staff: Optional[str] = None
    service: Optional[str] = Field(default=None, max_length=200)
    appointment_date: Optional[str] = None
    appointment_time: Optional[str] = None
    duration: Optional[str] = None
    notes: Optional[str] = None
    reminder: Optional[str] = None
    status: Optional[AppointmentStatus] = None


class AppointmentOut(BaseModel):
    id: int
    customer: str
    customer_id: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    location: Optional[str] = None
    department: str
    staff: str
    service: Optional[str] = None
    appointment_date: str
    appointment_time: str
    duration: Optional[str] = None
    notes: Optional[str] = None
    reminder: Optional[str] = "WhatsApp + Email"
    status: AppointmentStatus
    calendar_event_id: Optional[str] = None
    calendar_sync_status: Optional[str] = "PENDING"
    email_status: Optional[str] = "PENDING"
    whatsapp_status: Optional[str] = "PENDING"

    model_config = ConfigDict(from_attributes=True)

