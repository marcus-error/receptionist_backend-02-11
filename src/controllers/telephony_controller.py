from typing import Optional, List
from fastapi import Depends, Query, HTTPException, status
from sqlalchemy.orm import Session

from src.config.database import get_db
from src.middleware.auth_middleware import get_current_user, get_optional_current_user
from src.models.user import User
from src.services.telephony_service import TelephonyService
from src.validators.telephony_validator import (
    TelephonySettingsOut,
    TelephonySettingsUpdate,
    TestConnectionRequest,
    TestConnectionResponse,
    ValidateSipRequest,
    ValidateSipResponse,
    SipConfigIn,
    SipConfigOut,
    SipTestRequest,
    SipTestResponse,
    SipE2ETestRequest,
    SipE2ETestResponse,
    SipRegisterRequest,
    SipRegistrationStatusResponse,
    SipUnregisterResponse,
    SipOutboundCallRequest,
    SipOutboundCallResponse,
)


def _get_ids(current_user: Optional[User], organization_id: Optional[int] = None):
    user_id = current_user.id if current_user else 1
    if not current_user:
        return user_id, (organization_id or 1)
    if current_user.role == "super_admin":
        return user_id, (organization_id or current_user.organization_id or 1)
    # Strict multi-tenant isolation: tenant admins can ONLY manage their own organization
    if current_user.organization_id:
        return user_id, current_user.organization_id
    return user_id, (organization_id or 1)


def get_settings(
    organization_id: Optional[int] = Query(None),
    current_user: Optional[User] = Depends(get_optional_current_user),
    db: Session = Depends(get_db)
) -> TelephonySettingsOut:
    user_id, org_id = _get_ids(current_user, organization_id)
    return TelephonyService(db, user_id=user_id, org_id=org_id).get_settings()


def update_settings(
    payload: TelephonySettingsUpdate,
    organization_id: Optional[int] = Query(None),
    current_user: Optional[User] = Depends(get_optional_current_user),
    db: Session = Depends(get_db)
) -> TelephonySettingsOut:
    user_id, org_id = _get_ids(current_user, organization_id)
    return TelephonyService(db, user_id=user_id, org_id=org_id).update_settings(payload)


async def test_connection(
    payload: TestConnectionRequest,
    organization_id: Optional[int] = Query(None),
    current_user: Optional[User] = Depends(get_optional_current_user),
    db: Session = Depends(get_db)
) -> TestConnectionResponse:
    user_id, org_id = _get_ids(current_user, organization_id)
    result = await TelephonyService(db, user_id=user_id, org_id=org_id).test_connection(payload)
    return TestConnectionResponse(**result)


async def validate_sip(
    payload: ValidateSipRequest,
    organization_id: Optional[int] = Query(None),
    current_user: Optional[User] = Depends(get_optional_current_user),
    db: Session = Depends(get_db)
) -> ValidateSipResponse:
    user_id, org_id = _get_ids(current_user, organization_id)
    result = await TelephonyService(db, user_id=user_id, org_id=org_id).validate_sip(payload)
    return ValidateSipResponse(**result)


def list_sip_configs(
    organization_id: Optional[int] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> List[SipConfigOut]:
    user_id, org_id = _get_ids(current_user, organization_id)
    return TelephonyService(db, user_id=user_id, org_id=org_id).list_sip_configs()


def get_sip_config(
    probe: bool = Query(True),
    config_id: Optional[int] = None,
    organization_id: Optional[int] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> SipConfigOut:
    user_id, org_id = _get_ids(current_user, organization_id)
    return TelephonyService(db, user_id=user_id, org_id=org_id).get_sip_config(probe=probe, config_id=config_id)


def create_sip_config(
    payload: SipConfigIn,
    organization_id: Optional[int] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> SipConfigOut:
    user_id, org_id = _get_ids(current_user, organization_id)
    return TelephonyService(db, user_id=user_id, org_id=org_id).create_sip_config(payload)


def update_sip_config(
    payload: SipConfigIn,
    config_id: Optional[int] = None,
    organization_id: Optional[int] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> SipConfigOut:
    user_id, org_id = _get_ids(current_user, organization_id)
    return TelephonyService(db, user_id=user_id, org_id=org_id).update_sip_config(payload, config_id=config_id)


def delete_sip_config(
    config_id: int,
    organization_id: Optional[int] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> dict:
    user_id, org_id = _get_ids(current_user, organization_id)
    return TelephonyService(db, user_id=user_id, org_id=org_id).delete_sip_config(config_id)


def activate_sip_config(
    config_id: int,
    organization_id: Optional[int] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> SipConfigOut:
    user_id, org_id = _get_ids(current_user, organization_id)
    return TelephonyService(db, user_id=user_id, org_id=org_id).activate_sip_config(config_id)


def deactivate_sip_config(
    config_id: Optional[int] = None,
    organization_id: Optional[int] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> SipConfigOut:
    user_id, org_id = _get_ids(current_user, organization_id)
    return TelephonyService(db, user_id=user_id, org_id=org_id).deactivate_sip_config(config_id)


async def test_sip_connection(
    payload: SipTestRequest,
    organization_id: Optional[int] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> SipTestResponse:
    user_id, org_id = _get_ids(current_user, organization_id)
    result = await TelephonyService(db, user_id=user_id, org_id=org_id).test_sip_connection(payload)
    return SipTestResponse(**result)


async def run_e2e_test(
    payload: Optional[SipE2ETestRequest] = None,
    organization_id: Optional[int] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> SipE2ETestResponse:
    user_id, org_id = _get_ids(current_user, organization_id)
    if payload is None:
        payload = SipE2ETestRequest()
    result = await TelephonyService(db, user_id=user_id, org_id=org_id).run_e2e_test(payload)
    return SipE2ETestResponse(**result)


def register_sip_trunk(
    payload: Optional[SipRegisterRequest] = None,
    organization_id: Optional[int] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> dict:
    """Trigger SIP REGISTER with FreeSWITCH / Asterisk for the authenticated organization."""
    user_id, org_id = _get_ids(current_user, organization_id)
    if payload is None:
        payload = SipRegisterRequest()
    return TelephonyService(db, user_id=user_id, org_id=org_id).register_sip_trunk(payload)


def get_registration_status(
    organization_id: Optional[int] = Query(None),
    current_user: Optional[User] = Depends(get_optional_current_user),
    db: Session = Depends(get_db)
) -> SipRegistrationStatusResponse:
    """Return live SIP UA registration state for the authenticated organization."""
    user_id, org_id = _get_ids(current_user, organization_id)
    result = TelephonyService(db, user_id=user_id, org_id=org_id).get_registration_status()
    return SipRegistrationStatusResponse(
        status=result.get("status", "unregistered"),
        detail=result.get("detail", ""),
        sip_host=result.get("sip_host"),
        sip_username=result.get("sip_username"),
        sip_port=result.get("sip_port", 5060),
        did_number=result.get("did_number"),
        registered_at=result.get("registered_at"),
        last_attempt_at=result.get("last_attempt_at"),
        error_message=result.get("error_message"),
        active_calls=result.get("active_calls", 0),
    )


def unregister_sip_trunk(
    organization_id: Optional[int] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> SipUnregisterResponse:
    """Stop the SIP UA and unregister for the authenticated organization."""
    user_id, org_id = _get_ids(current_user, organization_id)
    result = TelephonyService(db, user_id=user_id, org_id=org_id).unregister_sip_trunk()
    return SipUnregisterResponse(**result)


def make_outbound_call(
    payload: SipOutboundCallRequest,
    organization_id: Optional[int] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> SipOutboundCallResponse:
    user_id, org_id = _get_ids(current_user, organization_id)
    result = TelephonyService(db, user_id=user_id, org_id=org_id).make_outbound_call(payload)
    return SipOutboundCallResponse(**result)
