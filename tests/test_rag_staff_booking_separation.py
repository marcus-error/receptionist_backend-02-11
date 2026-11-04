import pytest
from unittest.mock import patch, MagicMock

from src.config.database import SessionLocal
from src.models.knowledge_doc import KnowledgeDocument, KnowledgeDocStatus
from src.models.staff import Staff, StaffStatus
from src.models.organization import Organization
from src.services.booking_state_service import (
    booking_state_service,
    BookingSession,
    BookingState,
)
from src.services.nlp_service import (
    nlp_service,
    INTENT_PRODUCT_INFORMATION,
    INTENT_SERVICE_INFORMATION,
    INTENT_STAFF_LOOKUP,
    INTENT_STAFF_AVAILABILITY,
    INTENT_DOCTOR_INFORMATION,
    INTENT_APPOINTMENT_BOOKING,
)
from src.services.rag_service import rag_service
from src.services.cache_service import cache_service
from src.services.ai_service import ai_service
from src.services.staff_service import StaffService


@pytest.fixture(autouse=True)
def setup_multi_tenant_data():
    """
    Sets up two isolated organizations with distinct products, staff, and departments:
    Org 1 (Tech Company):
      - RAG Doc: Products = Call Zenza, Galaxy Chat, AI Receptionist, Aminoid HRMS
      - Staff: Sanjay (department: CEO), Alice (department: HR), Bob (department: SALES), Dr. Harini (department: Dermatology)
    Org 2 (Green Valley School):
      - RAG Doc: Offerings = Primary Education, STEM Labs, Art & Music Studio
      - Staff: Mr. Kumar (department: Mathematics), Ms. Priya (department: Science)
    """
    rag_service.clear_cache()
    cache_service.clear_all()

    with SessionLocal() as db:
        # Clean previous test entries only (never touch production records id=1 or real staff)
        db.query(Staff).filter(Staff.id.in_([301, 302, 303, 304, 305, 306])).delete(synchronize_session=False)
        db.query(KnowledgeDocument).filter(KnowledgeDocument.id.in_([201, 202])).delete(synchronize_session=False)
        db.commit()

        # Seed Org 1 RAG Document
        doc1 = KnowledgeDocument(
            id=201,
            organization_id=1,
            title="Tech_Company_Products.pdf",
            document_type="RAG",
            scope="inbound",
            ai_usage="Active",
            status=KnowledgeDocStatus.INDEXED,
            content=(
                "Aminoid Tech Solutions\n\n"
                "# PRODUCT SUITE\n"
                "- Call Zenza: Intelligent Voice Agent for Inbound & Outbound Calling\n"
                "- Galaxy Chat: Enterprise Omnichannel Conversational Platform\n"
                "- AI Receptionist: Automated Medical and Enterprise Receptionist\n"
                "- Aminoid HRMS: Full-cycle human resource management software with payroll and attendance tracking\n"
            ),
        )
        # Seed Org 2 RAG Document
        doc2 = KnowledgeDocument(
            id=202,
            organization_id=2,
            title="School_Curriculum.pdf",
            document_type="RAG",
            scope="inbound",
            ai_usage="Active",
            status=KnowledgeDocStatus.INDEXED,
            content=(
                "Green Valley International School\n\n"
                "# SERVICES\n"
                "- Primary Education: Comprehensive early-learning program\n"
                "- STEM Labs: Hands-on robotics and coding workshops\n"
                "- Art & Music Studio: Creative visual and performing arts curriculum\n"
            ),
        )
        db.add_all([doc1, doc2])

        # Seed Org 1 Staff (with internal departments CEO, HR, SALES, Dermatology)
        staff1 = Staff(
            id=301,
            organization_id=1,
            name="Sanjay",
            department="CEO",
            online_booking_enabled=True,
            status=StaffStatus.AVAILABLE,
            working_hours="09:00-18:00",
            working_days='["Monday","Tuesday","Wednesday","Thursday","Friday"]'
        )
        staff2 = Staff(
            id=302,
            organization_id=1,
            name="Alice",
            department="HR",
            online_booking_enabled=True,
            status=StaffStatus.AVAILABLE,
            working_hours="09:00-18:00",
            working_days='["Monday","Tuesday","Wednesday","Thursday","Friday"]'
        )
        staff3 = Staff(
            id=303,
            organization_id=1,
            name="Bob",
            department="SALES",
            online_booking_enabled=True,
            status=StaffStatus.AVAILABLE,
            working_hours="09:00-18:00",
            working_days='["Monday","Tuesday","Wednesday","Thursday","Friday"]'
        )
        staff4 = Staff(
            id=304,
            organization_id=1,
            name="Dr. Harini",
            department="Dermatology",
            online_booking_enabled=True,
            status=StaffStatus.AVAILABLE,
            working_hours="09:00-18:00",
            working_days='["Monday","Tuesday","Wednesday","Thursday","Friday"]'
        )

        # Seed Org 2 Staff
        staff5 = Staff(
            id=305,
            organization_id=2,
            name="Mr. Kumar",
            department="Mathematics",
            online_booking_enabled=True,
            status=StaffStatus.AVAILABLE,
            working_hours="08:00-16:00",
            working_days='["Monday","Tuesday","Wednesday","Thursday","Friday"]'
        )
        staff6 = Staff(
            id=306,
            organization_id=2,
            name="Ms. Priya",
            department="Science",
            online_booking_enabled=True,
            status=StaffStatus.AVAILABLE,
            working_hours="08:00-16:00",
            working_days='["Monday","Tuesday","Wednesday","Thursday","Friday"]'
        )
        db.add_all([staff1, staff2, staff3, staff4, staff5, staff6])
        db.commit()

    rag_service.clear_cache()
    yield

    with SessionLocal() as db:
        db.query(Staff).filter(Staff.id.in_([301, 302, 303, 304, 305, 306])).delete(synchronize_session=False)
        db.query(KnowledgeDocument).filter(KnowledgeDocument.id.in_([201, 202])).delete(synchronize_session=False)
        # Ensure Doc 1 is preserved and active for Organization 1
        db.query(KnowledgeDocument).filter(KnowledgeDocument.id == 1).update({"ai_usage": "Active"})
        db.commit()
    rag_service.clear_cache()
    cache_service.clear_all()


class TestRAGStaffBookingSeparation:
    """
    Verifies strict data source separation:
    - Products / Services -> Active RAG Knowledge Base only (NEVER Staff Directory).
    - Staff Directory -> Staff lookup, availability, department mapping only.
    - Booking State Machine -> Smooth affirmative handling without re-querying or restarting.
    - Multi-Tenant Isolation -> Strict organization_id scoping.
    """

    # =========================================================================
    # 1. Product and Service Information — RAG Only
    # =========================================================================

    def test_products_retrieved_from_rag_not_staff_directory(self):
        """
        Customer asks: 'What products do you offer?'
        Must return products extracted from active RAG document for Organization 1.
        Must NEVER return staff departments (CEO, SALES, HR).
        Must NEVER say 'We currently offer CEO, SALES, or HR'.
        """
        products = booking_state_service.get_organization_products_and_services(organization_id=1)
        
        # Verify valid RAG products are returned
        assert "Call Zenza" in products
        assert "Galaxy Chat" in products
        assert "AI Receptionist" in products
        assert "Aminoid HRMS" in products

        # Verify staff departments are NEVER present in product list
        prod_lower = [p.lower() for p in products]
        assert "ceo" not in prod_lower
        assert "sales" not in prod_lower
        assert "hr" not in prod_lower

    def test_nlp_intent_routing_for_products_and_services(self):
        """
        Verify NLP accurately detects PRODUCT_INFORMATION and SERVICE_INFORMATION intents.
        """
        r1 = nlp_service.detect_intent("What products do you offer?")
        assert r1.intent == INTENT_PRODUCT_INFORMATION

        r2 = nlp_service.detect_intent("Tell me about Aminoid HRMS")
        assert r2.intent == INTENT_PRODUCT_INFORMATION

        r3 = nlp_service.detect_intent("What services do you provide?")
        assert r3.intent == INTENT_SERVICE_INFORMATION

        r4 = nlp_service.detect_intent("What are the features of your product?")
        assert r4.intent == INTENT_PRODUCT_INFORMATION

    def test_ai_response_for_products_uses_rag_only(self):
        """
        When asking about products, AI service response must describe products from RAG
        and NEVER include staff departments.
        """
        resp = ai_service.generate_response(
            text="What products do you offer?",
            context=(
                "Aminoid Tech Solutions\n# PRODUCT SUITE\n"
                "- Call Zenza: Intelligent Voice Agent\n"
                "- Galaxy Chat: Enterprise Conversational Platform\n"
                "- Aminoid HRMS: Full-cycle human resource management\n"
            ),
            organization_id=1,
            intent=INTENT_PRODUCT_INFORMATION
        )
        resp_lower = resp.lower()
        # Verify products are mentioned
        assert "call zenza" in resp_lower or "galaxy chat" in resp_lower or "hrms" in resp_lower
        # Verify staff department names are NOT mentioned as offerings
        assert "we currently offer ceo" not in resp_lower
        assert "we currently offer sales" not in resp_lower
        assert "we currently offer hr" not in resp_lower

    def test_unlisted_product_reports_unavailable_honestly(self):
        """
        When customer asks about an unlisted product, AI must state it is not available in current offerings
        based on active records; it must NOT invent fake products or substitute staff members.
        """
        resp = ai_service.generate_response(
            text="Tell me about SuperJet Quantum Product",
            context=(
                "Aminoid Tech Solutions\n# PRODUCT SUITE\n"
                "- Call Zenza: Intelligent Voice Agent\n"
                "- Galaxy Chat: Enterprise Conversational Platform\n"
            ),
            organization_id=1,
            intent=INTENT_PRODUCT_INFORMATION
        )
        assert "not available" in resp.lower() or "active records" in resp.lower()
        assert "ceo" not in resp.lower()

    # =========================================================================
    # 2. Staff Directory — Staff Lookup and Availability Only
    # =========================================================================

    def test_staff_lookup_by_department(self):
        """
        Customer asks: 'Who is in HR?' or 'Who handles sales?'
        Must map to actual staff members in DB Staff table for Organization 1.
        """
        with SessionLocal() as db:
            staff_svc = StaffService(db)
            hr_staff = staff_svc.get_staff_for_department("HR", organization_id=1)
            assert len(hr_staff) == 1
            assert hr_staff[0].name == "Alice"
            assert hr_staff[0].department == "HR"

            sales_staff = staff_svc.get_staff_for_department("SALES", organization_id=1)
            assert len(sales_staff) == 1
            assert sales_staff[0].name == "Bob"

            dermatology_staff = staff_svc.get_staff_for_department("Dermatology", organization_id=1)
            assert len(dermatology_staff) == 1
            assert dermatology_staff[0].name == "Dr. Harini"

    def test_staff_profile_retrieval_does_not_hallucinate(self):
        """
        Customer asks: 'What does Sanjay do?'
        Must retrieve actual role/department (CEO) from DB Staff table.
        Must NOT hallucinate medical qualifications.
        """
        with SessionLocal() as db:
            staff_svc = StaffService(db)
            profile = staff_svc.get_staff_profile("Sanjay", organization_id=1)
            assert profile is not None
            assert profile["name"] == "Sanjay"
            assert profile["department"] == "CEO"

            harini_profile = staff_svc.get_staff_profile("Dr. Harini", organization_id=1)
            assert harini_profile is not None
            assert harini_profile["name"] == "Dr. Harini"
            assert harini_profile["department"] == "Dermatology"

    def test_staff_availability_checks_real_database(self):
        """
        Customer asks: 'Is Sanjay available?'
        Must verify real availability in database for Sanjay in Organization 1.
        """
        session = BookingSession("test-avail-session", organization_id=1)
        avail = booking_state_service.check_doctor_availability_live("Sanjay", target_date_str="today", session=session)
        assert "available" in avail.lower() or "slot" in avail.lower()

        # Non-existent staff member
        avail_none = booking_state_service.check_doctor_availability_live("NonExistentPerson", target_date_str="today", session=session)
        assert "could not find" in avail_none.lower() or "not listed" in avail_none.lower()

    # =========================================================================
    # 3. Appointment Booking State Machine & Confirmations
    # =========================================================================

    def test_affirmative_response_to_offered_doctor_proceeds_cleanly(self):
        """
        When AI asks: 'Would you like to schedule with Dr. Harini?'
        Customer says: 'Sure'
        State machine must:
        - Lock Dr. Harini into session.doctor.
        - Advance immediately to WAITING_FOR_DATE without re-querying directory.
        - NEVER treat 'Sure' as an entity/doctor candidate.
        - NEVER say 'I could not verify sure in our directory'.
        - NEVER say 'We currently offer CEO, SALES, or HR'.
        """
        session = BookingSession("test-affirm-doctor", organization_id=1)
        session.department = "Dermatology"
        session._offered_dept_doctor = "Dr. Harini"
        session.state = BookingState.WAITING_FOR_DOCTOR

        resp = booking_state_service.process_booking_turn("Sure", session)

        assert session.doctor == "Dr. Harini"
        assert session.doctor_locked is True
        assert session.state == BookingState.WAITING_FOR_DATE
        assert "sure" not in resp.lower()
        assert "could not verify" not in resp.lower()
        assert "ceo" not in resp.lower()
        assert "which date" in resp.lower() or "date" in resp.lower()

    def test_affirmative_variations_all_proceed_correctly(self):
        """
        Verify variations: 'Yes', 'Okay', 'Proceed', 'Go ahead', 'Sure please'
        all confirm the offered doctor and advance to the next step.
        """
        for affirmative in ["Yes", "Okay", "Proceed", "Go ahead", "Sure please", "Yes, that's fine"]:
            session = BookingSession(f"test-affirm-{affirmative}", organization_id=1)
            session.department = "Dermatology"
            session._offered_dept_doctor = "Dr. Harini"
            session.state = BookingState.WAITING_FOR_DOCTOR

            resp = booking_state_service.process_booking_turn(affirmative, session)

            assert session.doctor == "Dr. Harini"
            assert session.doctor_locked is True
            assert session.state in (BookingState.WAITING_FOR_DATE, BookingState.WAITING_FOR_TIME)
            assert "could not verify" not in resp.lower()

    # =========================================================================
    # 4. Multi-Tenant Data Isolation
    # =========================================================================

    def test_multi_tenant_product_isolation(self):
        """
        Org 1 products must never appear in Org 2, and vice versa.
        """
        org1_prods = booking_state_service.get_organization_products_and_services(organization_id=1)
        org2_prods = booking_state_service.get_organization_products_and_services(organization_id=2)

        # Org 1 has software products
        assert "Call Zenza" in org1_prods
        assert "Galaxy Chat" in org1_prods
        assert "Primary Education" not in org1_prods

        # Org 2 has educational services
        assert "Primary Education" in org2_prods
        assert "STEM Labs" in org2_prods
        assert "Call Zenza" not in org2_prods
        assert "Galaxy Chat" not in org2_prods

    def test_multi_tenant_staff_isolation(self):
        """
        Org 1 staff (Sanjay, Dr. Harini) must not be accessible in Org 2 directory.
        Org 2 staff (Mr. Kumar, Ms. Priya) must not be accessible in Org 1 directory.
        """
        with SessionLocal() as db:
            staff_svc = StaffService(db)

            # Look up Sanjay in Org 1 -> found
            s1 = staff_svc.find_staff_by_name("Sanjay", organization_id=1)
            assert s1 is not None
            assert s1.name == "Sanjay"

            # Look up Sanjay in Org 2 -> NOT found
            s2 = staff_svc.find_staff_by_name("Sanjay", organization_id=2)
            assert s2 is None

            # Look up Mr. Kumar in Org 2 -> found
            k2 = staff_svc.find_staff_by_name("Mr. Kumar", organization_id=2)
            assert k2 is not None

            # Look up Mr. Kumar in Org 1 -> NOT found
            k1 = staff_svc.find_staff_by_name("Mr. Kumar", organization_id=1)
            assert k1 is None

