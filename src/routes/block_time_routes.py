import logging
from datetime import date, datetime, timedelta
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.config.database import get_db
from src.middleware.auth_middleware import get_current_user
from src.models.block_time import BlockTime
from src.models.staff import Staff
from src.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Block Time"])


def _get_org_id(user: Optional[User]) -> int:
    if not user:
        return 1
    if user.role == "super_admin" and user.organization_id is None:
        return 1
    return user.organization_id or 1


class BlockTimeCreate(BaseModel):
    staff: Optional[str] = None
    staff_id: Optional[int] = None
    date: Optional[str] = None
    startTime: Optional[str] = None
    endTime: Optional[str] = None
    start_datetime: Optional[str] = None
    end_datetime: Optional[str] = None
    reason: Optional[str] = "Blocked"


class BlockTimeUpdate(BaseModel):
    staff: Optional[str] = None
    staff_id: Optional[int] = None
    start_datetime: Optional[str] = None
    end_datetime: Optional[str] = None
    reason: Optional[str] = None


def _parse_datetimes(payload: BlockTimeCreate) -> tuple[datetime, datetime]:
    """Parse start and end datetimes flexibly from either ISO datetime or date+time fields."""
    start_dt = None
    end_dt = None

    if payload.start_datetime and payload.end_datetime:
        try:
            start_dt = datetime.fromisoformat(payload.start_datetime.replace("Z", "+00:00")).replace(tzinfo=None)
            end_dt = datetime.fromisoformat(payload.end_datetime.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            pass

    if not start_dt or not end_dt:
        if not payload.date:
            raise HTTPException(400, "Date is required when start_datetime/end_datetime are not provided")

        from src.services.calendar_verification_service import calendar_verification_service
        d = calendar_verification_service.parse_date(payload.date)

        t_start_str = payload.startTime or "09:00"
        t_end_str = payload.endTime or "10:00"

        s_24 = calendar_verification_service.parse_time_to_24h(t_start_str) or "09:00"
        e_24 = calendar_verification_service.parse_time_to_24h(t_end_str) or "10:00"

        t_start = datetime.strptime(s_24, "%H:%M").time()
        t_end = datetime.strptime(e_24, "%H:%M").time()

        start_dt = datetime.combine(d, t_start)
        end_dt = datetime.combine(d, t_end)

    if end_dt <= start_dt:
        raise HTTPException(400, "End time must be after start time")

    return start_dt, end_dt


@router.get("/calendar/blocked-time")
@router.get("/calendar/block-time")
def list_blocked_times(
    staff_id: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    org_id = _get_org_id(current_user)
    query = db.query(BlockTime).filter(
        (BlockTime.organization_id == org_id) | (BlockTime.organization_id.is_(None))
    )

    if staff_id:
        query = query.filter(BlockTime.staff_id == staff_id)

    if start_date:
        from src.services.calendar_verification_service import calendar_verification_service
        s_date = calendar_verification_service.parse_date(start_date)
        query = query.filter(BlockTime.end_datetime >= datetime.combine(s_date, datetime.min.time()))

    if end_date:
        from src.services.calendar_verification_service import calendar_verification_service
        e_date = calendar_verification_service.parse_date(end_date)
        query = query.filter(BlockTime.start_datetime <= datetime.combine(e_date, datetime.max.time()))

    records = query.order_by(BlockTime.start_datetime.asc()).all()
    return [r.to_dict() for r in records]


@router.post("/calendar/blocked-time", status_code=201)
@router.post("/calendar/block-time", status_code=201)
def create_blocked_time(
    payload: BlockTimeCreate,
    current_user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    org_id = _get_org_id(current_user)
    start_dt, end_dt = _parse_datetimes(payload)

    staff_name = payload.staff
    staff_id = payload.staff_id

    if staff_name and not staff_id:
        staff_rec = db.query(Staff).filter(
            Staff.name.ilike(f"%{staff_name.strip()}%"),
            (Staff.organization_id == org_id) | (Staff.organization_id.is_(None))
        ).first()
        if staff_rec:
            staff_id = staff_rec.id
            staff_name = staff_rec.name
    elif staff_id and not staff_name:
        staff_rec = db.query(Staff).filter(Staff.id == staff_id).first()
        if staff_rec:
            staff_name = staff_rec.name

    block = BlockTime(
        organization_id=org_id,
        staff_id=staff_id,
        staff_name=staff_name,
        start_datetime=start_dt,
        end_datetime=end_dt,
        reason=payload.reason or "Blocked",
    )
    db.add(block)
    db.commit()
    db.refresh(block)
    logger.info(f"[BLOCK_TIME] Created BlockTime #{block.id} for staff={staff_name} ({start_dt} to {end_dt})")

    # Broadcast real-time update
    try:
        from src.sockets.socket_manager import sio
        import asyncio
        data = block.to_dict()
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(sio.emit("block_time_created", data))
        except Exception:
            pass
    except Exception:
        pass

    return block.to_dict()


@router.put("/calendar/blocked-time/{block_id}")
@router.put("/calendar/block-time/{block_id}")
def update_blocked_time(
    block_id: int,
    payload: BlockTimeUpdate,
    current_user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    org_id = _get_org_id(current_user)
    block = db.query(BlockTime).filter(
        BlockTime.id == block_id,
        (BlockTime.organization_id == org_id) | (BlockTime.organization_id.is_(None))
    ).first()

    if not block:
        raise HTTPException(404, "Blocked time record not found")

    if payload.staff_id is not None:
        block.staff_id = payload.staff_id
    if payload.staff is not None:
        block.staff_name = payload.staff
    if payload.reason is not None:
        block.reason = payload.reason

    if payload.start_datetime:
        block.start_datetime = datetime.fromisoformat(payload.start_datetime.replace("Z", "+00:00")).replace(tzinfo=None)
    if payload.end_datetime:
        block.end_datetime = datetime.fromisoformat(payload.end_datetime.replace("Z", "+00:00")).replace(tzinfo=None)

    if block.end_datetime <= block.start_datetime:
        raise HTTPException(400, "End time must be after start time")

    db.commit()
    db.refresh(block)
    logger.info(f"[BLOCK_TIME] Updated BlockTime #{block.id}")

    try:
        from src.sockets.socket_manager import sio
        import asyncio
        data = block.to_dict()
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(sio.emit("block_time_updated", data))
        except Exception:
            pass
    except Exception:
        pass

    return block.to_dict()


@router.delete("/calendar/blocked-time/{block_id}")
@router.delete("/calendar/block-time/{block_id}")
def delete_blocked_time(
    block_id: int,
    current_user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    org_id = _get_org_id(current_user)
    block = db.query(BlockTime).filter(
        BlockTime.id == block_id,
        (BlockTime.organization_id == org_id) | (BlockTime.organization_id.is_(None))
    ).first()

    if not block:
        raise HTTPException(404, "Blocked time record not found")

    db.delete(block)
    db.commit()
    logger.info(f"[BLOCK_TIME] Deleted BlockTime #{block_id}")

    try:
        from src.sockets.socket_manager import sio
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(sio.emit("block_time_deleted", {"id": block_id}))
        except Exception:
            pass
    except Exception:
        pass

    return {"message": "Blocked time removed successfully", "id": block_id}
