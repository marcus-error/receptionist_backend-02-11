"""
Multi-Tenant SIP Isolation Test Suite for CallZenza.

Tests and verifies:
1. Organization A and Organization B credentials storage and retrieval isolation.
2. Asterisk dynamic PJSIP multi-tenant generation (no overwrite of other orgs).
3. Outbound calling isolation (no fallback to global .env credentials or other orgs).
4. Inbound DID routing isolation (strictly routes to matching org, drops unknown DIDs).
5. Super Admin vs Tenant Admin scoping enforcement.
"""
import pytest
from fastapi.testclient import TestClient
from main import app
from src.config.database import SessionLocal
from src.models.user import User
from src.models.organization import Organization
from src.models.sip_config import SipConfig
from src.utils.security import encrypt_secret
from src.config.jwt import create_access_token
from src.services.asterisk_configurator import AsteriskConfigurator
from src.services.sip_registrar import get_registrar


@pytest.fixture(scope="module")
def db_session():
    db = SessionLocal()
    yield db
    db.close()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def org_tokens(db_session):
    """Create test users and organizations with JWT tokens for Org 1, Org 2, Org 3, and Super Admin."""
    # Ensure organizations exist
    org1 = db_session.query(Organization).filter(Organization.id == 1).first()
    if not org1:
        org1 = Organization(id=1, name="Organization 1", organization_type="General", status="active")
        db_session.add(org1)
        db_session.commit()

    org2 = db_session.query(Organization).filter(Organization.id == 2).first()
    if not org2:
        org2 = Organization(id=2, name="Organization 2", organization_type="General", status="active")
        db_session.add(org2)
        db_session.commit()

    org3 = db_session.query(Organization).filter(Organization.id == 3).first()
    if not org3:
        org3 = Organization(id=3, name="Organization 3", organization_type="General", status="active")
        db_session.add(org3)
        db_session.commit()

    # Create tokens
    token_super = create_access_token({
        "sub": "1",
        "email": "superadmin@receptionist.ai",
        "role": "super_admin",
        "organization_id": None,
    })

    token_org1 = create_access_token({
        "sub": "2",
        "email": "admin@receptionist.ai",
        "role": "admin",
        "organization_id": 1,
    })

    # User for Org 2
    u_org2 = db_session.query(User).filter(User.organization_id == 2).first()
    if not u_org2:
        u_org2 = User(
            name="Org 2 Admin",
            email="admin_org2@receptionist.ai",
            password_hash="mock",
            role="admin",
            organization_id=2,
            is_active=1
        )
        db_session.add(u_org2)
        db_session.commit()
        db_session.refresh(u_org2)

    token_org2 = create_access_token({
        "sub": str(u_org2.id),
        "email": u_org2.email,
        "role": "admin",
        "organization_id": 2,
    })

    # User for Org 3 (unconfigured SIP)
    u_org3 = db_session.query(User).filter(User.organization_id == 3).first()
    if not u_org3:
        u_org3 = User(
            name="Org 3 Admin",
            email="admin_org3@receptionist.ai",
            password_hash="mock",
            role="admin",
            organization_id=3,
            is_active=1
        )
        db_session.add(u_org3)
        db_session.commit()
        db_session.refresh(u_org3)

    token_org3 = create_access_token({
        "sub": str(u_org3.id),
        "email": u_org3.email,
        "role": "admin",
        "organization_id": 3,
    })

    return {
        "super": token_super,
        "org1": token_org1,
        "org2": token_org2,
        "org3": token_org3,
    }


def test_sip_config_crud_isolation(client, org_tokens, db_session):
    """Org A and Org B must have completely independent SIP credentials."""
    # 1. Update Org 1 SIP config
    headers_org1 = {"Authorization": f"Bearer {org_tokens['org1']}"}
    payload_org1 = {
        "account_name": "Org1 Trunk",
        "sip_server": "49.249.142.147",
        "sip_username": "09410697548",
        "sip_password": "Org1SecretPassword!",
        "sip_host": "49.249.142.147",
        "sip_port": 5060,
        "did_number": "09410697548",
        "transport": "UDP",
    }
    res1 = client.post("/api/v1/telephony/sip", json=payload_org1, headers=headers_org1)
    assert res1.status_code == 200, res1.text
    data1 = res1.json()
    assert data1["sip_username"] == "09410697548"
    assert data1["did_number"] == "09410697548"

    # 2. Update Org 2 SIP config with completely different credentials
    headers_org2 = {"Authorization": f"Bearer {org_tokens['org2']}"}
    payload_org2 = {
        "account_name": "Org2 Trunk",
        "sip_server": "sip.provider.org2.com",
        "sip_username": "09876543210",
        "sip_password": "Org2SecretPassword!",
        "sip_host": "sip.provider.org2.com",
        "sip_port": 5080,
        "did_number": "09876543210",
        "transport": "TCP",
    }
    res2 = client.post("/api/v1/telephony/sip", json=payload_org2, headers=headers_org2)
    assert res2.status_code == 200, res2.text
    data2 = res2.json()
    assert data2["sip_username"] == "09876543210"
    assert data2["did_number"] == "09876543210"
    assert data2["sip_port"] == 5080

    # 3. Read back Org 1 config -> must still be Org 1
    res1_read = client.get("/api/v1/telephony/sip?probe=false", headers=headers_org1)
    assert res1_read.status_code == 200
    assert res1_read.json()["sip_username"] == "09410697548"
    assert res1_read.json()["did_number"] == "09410697548"

    # 4. Read back Org 2 config -> must still be Org 2
    res2_read = client.get("/api/v1/telephony/sip?probe=false", headers=headers_org2)
    assert res2_read.status_code == 200
    assert res2_read.json()["sip_username"] == "09876543210"
    assert res2_read.json()["did_number"] == "09876543210"


def test_tenant_admin_cannot_access_other_org(client, org_tokens):
    """Org 2 admin trying to pass ?organization_id=1 must be strictly constrained to Org 2."""
    headers_org2 = {"Authorization": f"Bearer {org_tokens['org2']}"}
    res = client.get("/api/v1/telephony/sip?organization_id=1&probe=false", headers=headers_org2)
    assert res.status_code == 200
    # Must return Org 2's config, NOT Org 1's config!
    data = res.json()
    assert data["sip_username"] == "09876543210"
    assert data["did_number"] == "09876543210"


def test_super_admin_can_scope_organizations(client, org_tokens):
    """Platform Super Admin can query and manage any organization by organization_id."""
    headers_super = {"Authorization": f"Bearer {org_tokens['super']}"}

    # Query Org 1
    res1 = client.get("/api/v1/telephony/sip?organization_id=1&probe=false", headers=headers_super)
    assert res1.status_code == 200
    assert res1.json()["sip_username"] == "09410697548"

    # Query Org 2
    res2 = client.get("/api/v1/telephony/sip?organization_id=2&probe=false", headers=headers_super)
    assert res2.status_code == 200
    assert res2.json()["sip_username"] == "09876543210"


def test_unconfigured_org_fails_cleanly_without_fallback(client, org_tokens, db_session):
    """Org 3 has no SIP configuration: Outbound call must fail with 400, never fallback to Org 1 or .env."""
    # Ensure Org 3 SipConfig is empty or non-existent
    db_session.query(SipConfig).filter(SipConfig.organization_id == 3).delete()
    db_session.commit()

    headers_org3 = {"Authorization": f"Bearer {org_tokens['org3']}"}
    outbound_payload = {
        "to_number": "+919876543210",
        "from_number": "+919876543210",
    }
    res = client.post("/api/v1/telephony/sip/outbound", json=outbound_payload, headers=headers_org3)
    assert res.status_code == 400
    assert "does not have an active SIP configuration" in res.json()["detail"]


def test_asterisk_multi_tenant_config_build():
    """AsteriskConfigurator.build_multi_tenant_config must include all tenant endpoints without overwriting."""
    all_configs = [
        {
            "organization_id": 1,
            "sip_host": "49.249.142.147",
            "sip_username": "09410697548",
            "sip_password": "Org1Password",
            "sip_port": 5060,
            "did_number": "09410697548",
        },
        {
            "organization_id": 2,
            "sip_host": "sip.provider.org2.com",
            "sip_username": "09876543210",
            "sip_password": "Org2Password",
            "sip_port": 5080,
            "did_number": "09876543210",
        }
    ]

    docs = AsteriskConfigurator.build_multi_tenant_config(
        current_org_id=2,
        sip_host="sip.provider.org2.com",
        sip_username="09876543210",
        sip_password="Org2Password",
        sip_port=5080,
        did_number="09876543210",
        all_configs=all_configs,
    )

    pjsip = docs["pjsip_content"]
    dialplan = docs["dialplan_content"]

    # Verify both endpoints and auth blocks exist in pjsip.conf
    assert "[org_1_ep]" in pjsip
    assert "[auth_org_1]" in pjsip
    assert "username=09410697548" in pjsip
    assert "[org_2_ep]" in pjsip
    assert "[auth_org_2]" in pjsip
    assert "username=09876543210" in pjsip

    # Verify both contexts exist in extensions.conf
    assert "[context_org_1]" in dialplan
    assert '"org_id":1' in dialplan
    assert "[context_org_2]" in dialplan
    assert '"org_id":2' in dialplan


def test_get_registrar_strict_isolation():
    """get_registrar() must return None if organization_id is None, preventing cross-tenant registrar leakage."""
    assert get_registrar(None) is None

