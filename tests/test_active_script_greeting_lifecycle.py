import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from src.models.organization import Organization
from src.models.knowledge_doc import KnowledgeDocument, KnowledgeDocStatus
from src.services.rag_service import rag_service
from src.services.sip_call_handler import SipCallSession, CallType


@pytest.fixture
def test_tenants_and_scripts(db_session):
    """Setup independent tenants Org 201 and Org 202 with scripts."""
    db_session.query(KnowledgeDocument).filter(KnowledgeDocument.organization_id.in_([201, 202])).delete()
    db_session.query(Organization).filter(Organization.id.in_([201, 202])).delete()
    db_session.commit()

    org1 = Organization(id=201, name="Apex Medical Center", organization_type="Healthcare", status="active")
    org2 = Organization(id=202, name="Zenith Dental Care", organization_type="Dental", status="active")
    db_session.add(org1)
    db_session.add(org2)
    db_session.commit()

    # Script A for Org 201 (ACTIVE)
    script_a = KnowledgeDocument(
        organization_id=201,
        title="Apex_Inbound_Script_A.txt",
        file_type="txt",
        document_type="SCRIPT",
        scope="inbound",
        ai_usage="Active",
        status=KnowledgeDocStatus.INDEXED,
        content="Greeting: Good morning, welcome to Apex Medical Center. How can I assist you with your health today?",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )

    # Script B for Org 201 (INACTIVE initially)
    script_b = KnowledgeDocument(
        organization_id=201,
        title="Apex_Inbound_Script_B.txt",
        file_type="txt",
        document_type="SCRIPT",
        scope="inbound",
        ai_usage="Inactive",
        status=KnowledgeDocStatus.INDEXED,
        content="GREETING & CALL OPENING\n“Hello and welcome to Apex Advanced Specialty Clinic! We are glad you called. How may I direct your call?”",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )

    # Script for Org 202 (ACTIVE)
    script_202 = KnowledgeDocument(
        organization_id=202,
        title="Zenith_Inbound_Script.txt",
        file_type="txt",
        document_type="SCRIPT",
        scope="inbound",
        ai_usage="Active",
        status=KnowledgeDocStatus.INDEXED,
        content="“Thank you for calling Zenith Dental Care. My name is Maya. How can I help you smile today?”",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )

    db_session.add(script_a)
    db_session.add(script_b)
    db_session.add(script_202)
    db_session.commit()
    db_session.refresh(script_a)
    db_session.refresh(script_b)
    db_session.refresh(script_202)

    rag_service.clear_organization_cache(201)
    rag_service.clear_organization_cache(202)

    return org1, org2, script_a, script_b, script_202


@pytest.mark.asyncio
async def test_full_active_script_lifecycle_and_switching(test_tenants_and_scripts, db_session):
    """
    Simulates the exact admin workflow:
    Scenario A: Script A is ACTIVE -> Inbound call speaks Script A.
    Scenario B: Deactivate Script A, activate Script B -> Inbound call speaks Script B.
    Scenario C: Update Script B content -> Inbound call speaks updated Script B.
    Scenario D: Deactivate all scripts -> Inbound call follows missing-script behavior.
    Scenario E: Multiple organizations -> Strict tenant isolation.
    """
    org1, org2, script_a, script_b, script_202 = test_tenants_and_scripts

    # --- SCENARIO A: Script A is ACTIVE on incoming call ---
    spoken_greetings_call_1 = []
    session1 = SipCallSession(
        caller_phone="9876543210",
        caller_name="Test Caller 1",
        organization_id=201,
        call_type=CallType.INBOUND,
    )
    with patch.object(session1, "_speak", new_callable=AsyncMock) as mock_speak:
        await session1._send_opening_greeting()
        assert mock_speak.called
        greeting_called = mock_speak.call_args[0][0]
        spoken_greetings_call_1.append(greeting_called)

    assert "Apex Medical Center" in spoken_greetings_call_1[0]
    assert "How can I assist you with your health today?" in spoken_greetings_call_1[0]
    assert session1.active_script_id == str(script_a.id)

    # --- SCENARIO B: Admin deactivates Script A and activates Script B ---
    script_a.ai_usage = "Inactive"
    script_b.ai_usage = "Active"
    script_b.updated_at = datetime.now(timezone.utc)
    db_session.commit()
    rag_service.clear_organization_cache(201)

    # Incoming Call 2 arrives
    spoken_greetings_call_2 = []
    session2 = SipCallSession(
        caller_phone="9876543211",
        caller_name="Test Caller 2",
        organization_id=201,
        call_type=CallType.INBOUND,
    )
    with patch.object(session2, "_speak", new_callable=AsyncMock) as mock_speak2:
        await session2._send_opening_greeting()
        assert mock_speak2.called
        greeting_called2 = mock_speak2.call_args[0][0]
        spoken_greetings_call_2.append(greeting_called2)

    assert "Apex Advanced Specialty Clinic" in spoken_greetings_call_2[0]
    assert "How may I direct your call?" in spoken_greetings_call_2[0]
    assert session2.active_script_id == str(script_b.id)

    # --- SCENARIO C: Admin updates active Script B greeting ---
    script_b.content = "“Welcome to Apex Care! We provide cardiology and neurology consultations. How can we help you today?”"
    script_b.updated_at = datetime.now(timezone.utc)
    db_session.commit()
    rag_service.clear_organization_cache(201)

    # Incoming Call 3 arrives
    spoken_greetings_call_3 = []
    session3 = SipCallSession(
        caller_phone="9876543212",
        caller_name="Test Caller 3",
        organization_id=201,
        call_type=CallType.INBOUND,
    )
    with patch.object(session3, "_speak", new_callable=AsyncMock) as mock_speak3:
        await session3._send_opening_greeting()
        assert mock_speak3.called
        greeting_called3 = mock_speak3.call_args[0][0]
        spoken_greetings_call_3.append(greeting_called3)

    assert "Apex Care" in spoken_greetings_call_3[0]
    assert "cardiology and neurology consultations" in spoken_greetings_call_3[0]
    assert session3.active_script_id == str(script_b.id)

    # --- SCENARIO D: Admin deactivates all scripts ---
    script_b.ai_usage = "Inactive"
    db_session.commit()
    rag_service.clear_organization_cache(201)

    # Incoming Call 4 arrives
    spoken_greetings_call_4 = []
    session4 = SipCallSession(
        caller_phone="9876543213",
        caller_name="Test Caller 4",
        organization_id=201,
        call_type=CallType.INBOUND,
    )
    with patch.object(session4, "_speak", new_callable=AsyncMock) as mock_speak4:
        await session4._send_opening_greeting()
        assert mock_speak4.called
        greeting_called4 = mock_speak4.call_args[0][0]
        spoken_greetings_call_4.append(greeting_called4)

    # Must follow missing script behavior without fabricating services or doctors
    assert "cardiology and neurology" not in spoken_greetings_call_4[0]
    assert "Apex Advanced Specialty" not in spoken_greetings_call_4[0]
    assert "How may I assist you today?" in spoken_greetings_call_4[0]

    # --- SCENARIO E: Multi-tenant isolation ---
    spoken_greetings_call_5 = []
    session5 = SipCallSession(
        caller_phone="9876543214",
        caller_name="Test Caller 5",
        organization_id=202,
        call_type=CallType.INBOUND,
    )
    with patch.object(session5, "_speak", new_callable=AsyncMock) as mock_speak5:
        await session5._send_opening_greeting()
        assert mock_speak5.called
        greeting_called5 = mock_speak5.call_args[0][0]
        spoken_greetings_call_5.append(greeting_called5)

    assert "Zenith Dental Care" in spoken_greetings_call_5[0]
    assert "How can I help you smile today?" in spoken_greetings_call_5[0]
    assert "Apex" not in spoken_greetings_call_5[0]
    assert session5.active_script_id == str(script_202.id)
