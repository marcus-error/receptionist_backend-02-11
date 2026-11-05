"""
Test Suite: Inbound Greeting Exact Format Validation
Validates all strict requirements:
1. Inbound greeting format strictly matches:
   "Good [time of day]! Thank you for calling [Organisation Name]. How may I assist you?"
2. Never generates outbound phrases ("This is ... calling", "I'm calling from...", "I am calling on behalf of...") for inbound calls.
3. Organisation name is fetched dynamically from the database.
4. Correct time prefix:
   - 05:00 - 11:59: Good morning
   - 12:00 - 16:59: Good afternoon
   - 17:00 - 04:59: Good evening
5. Greeting is short, professional, and exactly matches the requested structure.
6. Script catalogues (services, doctors, hours) are stripped from the initial greeting.
7. Multi-tenant isolation across multiple organizations.
8. Fallback for organizations without uploaded scripts.
9. Outbound calls are not modified and retain outbound phrasing.
10. Follow-up conversation turns never repeat the greeting.
"""

import pytest
import re
from datetime import datetime
from zoneinfo import ZoneInfo
from unittest.mock import patch

from src.config.database import SessionLocal, init_db
from src.models.organization import Organization
from src.models.settings import AppSettings
from src.models.knowledge_doc import KnowledgeDocument
from src.services.rag_service import rag_service
from src.services.knowledge_service import KnowledgeService
from src.validators.knowledge_validator import KnowledgeDocCreate


@pytest.fixture(autouse=True)
def ensure_db():
    init_db()
    db = SessionLocal()
    org1 = db.query(Organization).filter(Organization.id == 1).first()
    if not org1:
        db.add(Organization(id=1, name="Default Organization"))
        db.commit()
    db.close()
    rag_service.invalidate_caches()


@pytest.mark.parametrize("hour,expected_prefix", [
    (5, "Good morning"),
    (8, "Good morning"),
    (11, "Good morning"),
    (12, "Good afternoon"),
    (14, "Good afternoon"),
    (16, "Good afternoon"),
    (17, "Good evening"),
    (20, "Good evening"),
    (23, "Good evening"),
    (0, "Good evening"),
    (4, "Good evening"),
])
def test_time_of_day_prefix_rules(hour, expected_prefix):
    """Verify exact hour boundaries for Good morning / Good afternoon / Good evening."""
    mock_now = datetime(2026, 11, 5, hour, 30, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    with patch("datetime.datetime") as mock_dt:
        mock_dt.now.return_value = mock_now
        mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
        prefix = rag_service._get_time_aware_greeting_prefix()
        assert prefix == expected_prefix, f"At {hour}:30, expected '{expected_prefix}', got '{prefix}'"


@pytest.mark.parametrize("org_name", [
    "Aminoid AI",
    "Sarans Hospitals",
    "Metro Health Clinic",
    "Apex Cardiology Center",
    "Blue Horizon Hotels",
    "Zenith Global Services LLC",
])
def test_exact_required_greeting_format_multiple_orgs(org_name):
    """
    Verify exact required greeting format:
    'Good [time of day]! Thank you for calling [Organisation Name]. How may I assist you?'
    """
    for time_prefix in ["Good morning", "Good afternoon", "Good evening"]:
        greeting = rag_service._condense_to_short_greeting(
            greeting="Welcome to our center. We provide comprehensive services.",
            resolved_org=org_name,
            time_prefix=time_prefix,
            scope="inbound"
        )

        expected = f"{time_prefix}! Thank you for calling {org_name}. How may I assist you?"
        assert greeting == expected, f"Expected '{expected}', got '{greeting}'"

        # Forbidden inbound phrases check
        forbidden_phrases = [
            "calling from",
            "on behalf of",
            f"this is {org_name.lower()} calling",
            "this is our team calling",
        ]
        for fp in forbidden_phrases:
            assert fp not in greeting.lower(), f"Forbidden phrase '{fp}' found in inbound greeting: '{greeting}'"


def test_dynamic_db_fetch_for_multiple_tenants():
    """Verify that multiple tenants dynamically resolve their exact DB organization name."""
    db = SessionLocal()
    test_org_ids = [401, 402, 403]
    try:
        org_a = Organization(id=401, name="Aminoid AI")
        org_b = Organization(id=402, name="Apollo Spectra Hospital")
        org_c = Organization(id=403, name="Care Diagnostics")
        db.add_all([org_a, org_b, org_c])
        db.commit()

        # Inbound greeting resolution with no script uploaded (DB fallback)
        g_a, _, _ = rag_service.get_active_greeting(401, allow_db_fallback=True)
        g_b, _, _ = rag_service.get_active_greeting(402, allow_db_fallback=True)
        g_c, _, _ = rag_service.get_active_greeting(403, allow_db_fallback=True)

        assert "Thank you for calling Aminoid AI. How may I assist you?" in g_a
        assert "Thank you for calling Apollo Spectra Hospital. How may I assist you?" in g_b
        assert "Thank you for calling Care Diagnostics. How may I assist you?" in g_c

        # Verify no cross-tenant leakage
        assert "Apollo Spectra" not in g_a
        assert "Care Diagnostics" not in g_a
        assert "Aminoid AI" not in g_b

    finally:
        db.query(Organization).filter(Organization.id.in_(test_org_ids)).delete()
        db.commit()
        db.close()


def test_script_with_lengthy_catalogue_stripped():
    """Verify uploaded script containing departments, doctors, and hours strips catalogue."""
    db = SessionLocal()
    org_id = 501
    try:
        org = Organization(id=org_id, name="Royal Palms Hotel & Spa")
        db.add(org)
        db.commit()

        ks = KnowledgeService(db)
        doc = ks.create_doc(KnowledgeDocCreate(
            title="Royal_Palms_Inbound_Script.txt",
            organization_id=org_id,
            document_type="SCRIPT",
            scope="inbound",
            ai_usage="Active",
            content="""# ROYAL PALMS HOTEL & SPA SCRIPT
## 1. Greeting & Call Opening
"Good morning! Welcome to Royal Palms Hotel & Spa. We offer luxury suites, spa therapies, fine dining, swimming pool access, and event hosting. How may I help you today?"

## 2. Room Rates & Amenities
Suites start from $250/night.
"""
        ), organization_id=org_id)

        # Retrieve greeting
        greeting, script_id, _ = rag_service.get_active_greeting(org_id)
        assert greeting is not None
        assert "Thank you for calling Royal Palms Hotel & Spa. How may I assist you?" in greeting
        assert "luxury suites" not in greeting
        assert "spa therapies" not in greeting
        assert "fine dining" not in greeting

        # Verify AppSettings is synced
        settings = db.query(AppSettings).filter(AppSettings.organization_id == org_id).first()
        assert settings is not None
        assert "Thank you for calling Royal Palms Hotel & Spa. How may I assist you?" in settings.default_greeting

    finally:
        db.query(KnowledgeDocument).filter(KnowledgeDocument.organization_id == org_id).delete()
        db.query(AppSettings).filter(AppSettings.organization_id == org_id).delete()
        db.query(Organization).filter(Organization.id == org_id).delete()
        db.commit()
        db.close()


def test_outbound_greeting_preserved():
    """Verify outbound call greetings are not changed to inbound 'Thank you for calling'."""
    raw_outbound = "Good morning. Am I speaking with Mr. Robert Smith regarding your appointment?"
    outbound = rag_service.extract_short_greeting(
        raw_greeting=raw_outbound,
        org_name="Sarans Hospitals",
        organization_id=1,
        scope="outbound"
    )

    assert "Am I speaking with" in outbound
    assert "Thank you for calling" not in outbound
