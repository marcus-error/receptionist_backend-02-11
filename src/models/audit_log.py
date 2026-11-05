from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.sql import func

from src.config.database import Base


class PlatformAuditLog(Base):
    __tablename__ = "platform_audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    action = Column(String(100), nullable=False, index=True)  # e.g., "ORGANIZATION_DELETED"
    entity_type = Column(String(50), nullable=False)          # e.g., "organization"
    entity_id = Column(Integer, nullable=False)
    actor_id = Column(Integer, nullable=True)
    actor_email = Column(String(150), nullable=True)
    details = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

