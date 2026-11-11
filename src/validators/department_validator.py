from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class DepartmentCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    code: Optional[str] = Field(None, max_length=50)
    description: Optional[str] = None
    head_of_department: Optional[str] = Field(None, max_length=150)
    phone: Optional[str] = Field(None, max_length=50)
    email: Optional[str] = Field(None, max_length=200)
    status: str = Field("Active", max_length=20)


class DepartmentUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    code: Optional[str] = Field(None, max_length=50)
    description: Optional[str] = None
    head_of_department: Optional[str] = Field(None, max_length=150)
    phone: Optional[str] = Field(None, max_length=50)
    email: Optional[str] = Field(None, max_length=200)
    status: Optional[str] = Field(None, max_length=20)


class DepartmentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    organization_id: Optional[int] = None
    name: str
    code: Optional[str] = None
    description: Optional[str] = None
    head_of_department: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    status: str = "Active"
    staff_count: Optional[int] = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class DepartmentStaffAssign(BaseModel):
    staff_ids: Optional[List[int]] = None
    staff_id: Optional[int] = None


class DepartmentStaffOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    staff_id: int
    name: str
    department: str
    organization_id: Optional[int] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    status: Optional[str] = "Available"
    working_hours: Optional[str] = "09:00-18:00"
    appointment_duration: Optional[str] = "30 min"
    role: Optional[str] = None
    assigned_at: Optional[datetime] = None
