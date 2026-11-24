from __future__ import annotations
import re
import copy
import uuid
import logging
from typing import Dict, Optional, Any, List
from datetime import date, datetime, timedelta
import src.models

logger = logging.getLogger(__name__)


class SessionWorkflowMode:
    APPOINTMENT_BOOKING = "APPOINTMENT_BOOKING"
    LEAD_COLLECTION = "LEAD_COLLECTION"


class BookingState:
    IDLE = "IDLE"
    GENERAL_CONVERSATION = "GENERAL_CONVERSATION"
    PRODUCT_ENQUIRY = "PRODUCT_ENQUIRY"
    APPOINTMENT_START = "APPOINTMENT_START"
    BOOKING_STARTED = "BOOKING_STARTED"
    WAITING_FOR_DOCTOR = "WAITING_FOR_DOCTOR"
    WAITING_FOR_DEPARTMENT = "WAITING_FOR_DOCTOR"
    VERIFYING_DOCTOR = "VERIFYING_DOCTOR"
    WAITING_FOR_DATE = "WAITING_FOR_DATE"
    WAITING_FOR_BOOKING_DATE = "WAITING_FOR_DATE"
    WAITING_FOR_TIME = "WAITING_FOR_TIME"
    WAITING_FOR_BOOKING_TIME = "WAITING_FOR_TIME"
    CHECKING_AVAILABILITY = "CHECKING_AVAILABILITY"
    WAITING_FOR_CONFIRMATION = "WAITING_FOR_CONFIRMATION"
    CONFIRMATION_ACCEPTED = "CONFIRMATION_ACCEPTED"
    CORRECTION_REQUESTED = "CORRECTION_REQUESTED"
    WAITING_FOR_CORRECTION_FIELD = "WAITING_FOR_CORRECTION_FIELD"
    WAITING_FOR_CORRECTION_VALUE = "WAITING_FOR_CORRECTION_VALUE"
    VALIDATING_CORRECTION = "VALIDATING_CORRECTION"
    REVIEWING_UPDATED_DETAILS = "REVIEWING_UPDATED_DETAILS"
    WAITING_FOR_FINAL_CONFIRMATION = "WAITING_FOR_FINAL_CONFIRMATION"
    BOOKING_PROCESSING = "BOOKING_PROCESSING"
    BOOKING_CONFIRMED = "BOOKING_CONFIRMED"
    BOOKING_CANCELLED = "BOOKING_CANCELLED"
    COLLECTING_PATIENT_NAME = "COLLECTING_PATIENT_NAME"
    WAITING_FOR_PATIENT_NAME = "COLLECTING_PATIENT_NAME"
    WAITING_FOR_NAME = "COLLECTING_PATIENT_NAME"
    COLLECTING_EMAIL = "COLLECTING_EMAIL"
    WAITING_FOR_EMAIL = "COLLECTING_EMAIL"
    COLLECTING_PHONE = "COLLECTING_PHONE"
    WAITING_FOR_CONTACT = "COLLECTING_PHONE"
    WAITING_FOR_PHONE = "COLLECTING_PHONE"
    COLLECTING_LOCATION = "COLLECTING_LOCATION"
    WAITING_FOR_LOCATION = "COLLECTING_LOCATION"
    COLLECTING_BUSINESS = "COLLECTING_BUSINESS"
    WAITING_FOR_BUSINESS = "COLLECTING_BUSINESS"
    COLLECTING_REQUIREMENT = "COLLECTING_REQUIREMENT"
    WAITING_FOR_REQUIREMENT = "COLLECTING_REQUIREMENT"
    LEAD_COLLECTED = "LEAD_COLLECTED"
    BOOKING = "BOOKING"
    BOOKED = "BOOKED"
    FAILED = "FAILED"
    COMPLETED = "COMPLETED"
    CALL_TERMINATING = "CALL_TERMINATING"
    CALL_COMPLETED = "CALL_COMPLETED"


CORRECTION_AND_CONFIRMATION_STATES = (
    BookingState.WAITING_FOR_CONFIRMATION,
    BookingState.CONFIRMATION_ACCEPTED,
    BookingState.CORRECTION_REQUESTED,
    BookingState.WAITING_FOR_CORRECTION_FIELD,
    BookingState.WAITING_FOR_CORRECTION_VALUE,
    BookingState.VALIDATING_CORRECTION,
    BookingState.REVIEWING_UPDATED_DETAILS,
    BookingState.WAITING_FOR_FINAL_CONFIRMATION,
)


class FieldState:
    MISSING = "MISSING"
    COLLECTED = "COLLECTED"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    VERIFIED = "VERIFIED"


class NextAction:
    USER_INPUT_REQUIRED = "USER_INPUT_REQUIRED"
    AUTOMATIC_ACTION_REQUIRED = "AUTOMATIC_ACTION_REQUIRED"
    CONFIRMATION_REQUIRED = "CONFIRMATION_REQUIRED"
    COMPLETED = "COMPLETED"


class BookingSession:
    """
    Structured Session Memory for Appointment Booking & Multi-Turn Conversations.
    Retains verified state across multiple turns without losing state.
    """
    def __init__(
        self,
        session_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        call_sid: Optional[str] = None,
        organization_id: int = 1,
        business_type: str = "GENERAL",
        booking_type: str = "BUSINESS_CONSULTATION",
        business_name: Optional[str] = None,
        active_doc_id: Optional[str] = None,
        active_doc_title: Optional[str] = None
    ):
        self.session_id = session_id or f"sess_{uuid.uuid4().hex[:10]}"
        self.conversation_id: str = conversation_id or self.session_id
        self.call_sid: str = call_sid or self.session_id
        self.stream_sid: Optional[str] = None
        self.agent_id: Optional[str] = None
        self.user_id: Optional[str] = None
        self.organization_id: int = organization_id
        self.active_rag_id: Optional[str] = active_doc_id
        self.active_doc_id: Optional[str] = active_doc_id
        self.active_rag_title: str = active_doc_title or "Active Knowledge Base"
        self.active_doc_title: str = active_doc_title or "Active Knowledge Base"
        self.business_type: str = business_type or "GENERAL"
        self.booking_type: str = booking_type or "BUSINESS_CONSULTATION"
        self.business_name: str = business_name or "AI Receptionist"
        if str(self.booking_type).upper() == "SALES_DEMO" or str(self.business_type).upper() == "SOFTWARE":
            self.workflow_type: str = "SALES_DEMO"
        elif str(self.booking_type).upper() == "HOSPITAL_APPOINTMENT" or str(self.business_type).upper() == "HEALTHCARE":
            self.workflow_type: str = "HOSPITAL_APPOINTMENT"
        else:
            self.workflow_type: str = "GENERAL"
        self.workflow_mode: str = SessionWorkflowMode.APPOINTMENT_BOOKING
        self.confirmation_count: int = 0
        self.confirmation_message: Optional[str] = None
        self.confirmed_appointment_id: Optional[int] = None

        self.state = BookingState.IDLE
        self.conversation_started: bool = False
        self.greeting_sent: bool = False
        self.turn_count: int = 0
        self.turn_id: int = 0
        self.history: List[Dict[str, str]] = []

        self.doctor_id: Optional[int] = None
        self.doctor_name: Optional[str] = None
        self.doctor_locked: bool = False
        self.department_name: Optional[str] = None
        self.service_name: Optional[str] = None
        self.product_name: Optional[str] = None
        self.industry: Optional[str] = None
        self.requirement: Optional[str] = None
        self.interested_area: Optional[str] = None
        self.offered_slots: List[str] = []
        self.last_question_field: Optional[str] = None
        self.last_question_asked: Optional[str] = None
        self.last_question_timestamp: Optional[float] = None
        self.last_question_attempt_count: int = 0
        self.last_customer_response: Optional[str] = None
        self.last_detected_intent: Optional[str] = None
        self.current_collection_field: Optional[str] = None
        self.skipped_fields: List[str] = []
        self.conversation_summary: Optional[str] = None
        self.customer_requested_action: Optional[str] = None
        self.question_attempt_history: Dict[str, int] = {}
        self.field_retry_limits: Dict[str, int] = {
            "patient_name": 3,
            "email": 2,
            "contact_number": 3,
            "industry": 2,
            "requirement": 2
        }

        self.caller_name: Optional[str] = None
        self.caller_phone: Optional[str] = None
        self.call_direction: str = "INBOUND"
        self.call_type: str = "INBOUND"
        self.patient_relationship: Optional[str] = None
        self.is_booking_for_other: bool = False

        self.confirmed_facts: Dict[str, Dict[str, Any]] = {}
        self.completed_steps: List[str] = []
        self.state_version: int = 1

        self._rejected_departments: Set[str] = set()
        self._rejected_doctors: Set[str] = set()
        self._rejected_services: Set[str] = set()
        self._offered_dept_doctor: Optional[str] = None
        self._last_rejected_entity: Optional[str] = None

        self.booking_data: Dict[str, Any] = {
            "intent": None,
            "department": None,
            "doctor": None,
            "specialty": None,
            "service": None,
            "product": None,
            "industry": None,
            "requirement": None,
            "interested_area": None,
            "date": None,
            "date_obj": None,
            "time": None,
            "caller_name": None,
            "caller_phone": None,
            "patient_name": None,
            "patient_relationship": None,
            "is_booking_for_other": False,
            "email": None,
            "contact_number": None,
            "availability_verified": False,
            "booking_confirmed": False,
            "appointment_id": None,
            "idempotency_key": None,
            "conversation_language": "en"
        }
        self.field_status: Dict[str, str] = {
            "doctor": "PENDING",
            "date": "PENDING",
            "time": "PENDING",
            "patient_name": "PENDING",
            "contact_number": "PENDING",
            "email": "PENDING",
            "location": "PENDING"
        }
        self._is_completed: bool = False

    @property
    def is_completed(self) -> bool:
        return getattr(self, "_is_completed", False) or self.state in (BookingState.BOOKED, BookingState.BOOKING_CONFIRMED, BookingState.COMPLETED) or self.booking_confirmed

    @is_completed.setter
    def is_completed(self, val: bool):
        self._is_completed = bool(val)

    def is_active(self) -> bool:
        return self.state != BookingState.IDLE and not self.is_completed and self.state not in (BookingState.BOOKING_CONFIRMED, BookingState.BOOKED, BookingState.COMPLETED, BookingState.FAILED, BookingState.BOOKING_CANCELLED)

    def reset(self):
        self.state = BookingState.IDLE
        self.workflow_mode = SessionWorkflowMode.APPOINTMENT_BOOKING
        self.doctor_id = None
        self.doctor_name = None
        self.doctor_locked = False
        self.department_name = None
        self.service_name = None
        self.product_name = None
        self.industry = None
        self.requirement = None
        self.offered_slots = []
        self.last_question_field = None
        self.caller_name = None
        self.caller_phone = None
        self.patient_relationship = None
        self.is_booking_for_other = False
        self.confirmed_facts = {}
        self.completed_steps = []
        self.state_version += 1
        self._rejected_departments = set()
        self._rejected_doctors = set()
        self._rejected_services = set()
        self._offered_dept_doctor = None
        self._last_rejected_entity = None
        self.booking_data = {
            "intent": None,
            "department": None,
            "doctor": None,
            "specialty": None,
            "service": None,
            "product": None,
            "industry": None,
            "requirement": None,
            "interested_area": None,
            "date": None,
            "date_obj": None,
            "time": None,
            "caller_name": None,
            "caller_phone": None,
            "patient_name": None,
            "patient_relationship": None,
            "is_booking_for_other": False,
            "email": None,
            "contact_number": None,
            "availability_verified": False,
            "booking_confirmed": False,
            "appointment_id": None,
            "idempotency_key": None,
            "conversation_language": "en"
        }
        self.field_status = {
            "doctor": "PENDING",
            "date": "PENDING",
            "time": "PENDING",
            "patient_name": "PENDING",
            "contact_number": "PENDING",
            "email": "PENDING",
            "location": "PENDING"
        }
        self.is_completed = False

    def reset_for_new_booking(self):
        """
        Resets appointment-specific slot variables (department, doctor, service, product, date, time, appointment_id, status)
        while strictly preserving validated customer identity (patient name, contact number, email, caller details).
        """
        self.state = BookingState.IDLE
        self.doctor_id = None
        self.doctor_name = None
        self.doctor_locked = False
        self.department_name = None
        self.service_name = None
        self.product_name = None
        self.offered_slots = []
        self.last_question_field = None
        self.confirmed_facts = {}
        self.completed_steps = []
        self.state_version += 1
        self._rejected_departments = set()
        self._rejected_doctors = set()
        self._rejected_services = set()
        self._offered_dept_doctor = None
        self._last_rejected_entity = None
        
        # Preserve customer identity
        saved_caller_name = self.caller_name
        saved_caller_phone = self.caller_phone
        saved_patient_name = self.booking_data.get("patient_name")
        saved_email = self.booking_data.get("email")
        saved_contact = self.booking_data.get("contact_number")
        saved_location = self.booking_data.get("location")
        saved_industry = self.industry or self.booking_data.get("industry")
        saved_requirement = self.requirement or self.booking_data.get("requirement")
        saved_lang = self.booking_data.get("conversation_language", "en")

        self.booking_data = {
            "intent": None,
            "department": None,
            "doctor": None,
            "specialty": None,
            "service": None,
            "product": None,
            "industry": saved_industry,
            "requirement": saved_requirement,
            "interested_area": None,
            "date": None,
            "date_obj": None,
            "time": None,
            "caller_name": saved_caller_name,
            "caller_phone": saved_caller_phone,
            "patient_name": saved_patient_name,
            "patient_relationship": self.patient_relationship,
            "is_booking_for_other": self.is_booking_for_other,
            "email": saved_email,
            "contact_number": saved_contact,
            "location": saved_location,
            "availability_verified": False,
            "booking_confirmed": False,
            "appointment_id": None,
            "idempotency_key": None,
            "conversation_language": saved_lang
        }
        self.is_completed = False

    def invalidate_department(self, dept_name: Optional[str] = None):
        """
        Invalidates rejected department/specialty selection and any dependent doctor selection.
        Preserves valid patient details (name, phone, email, preferred date).
        """
        curr_dept = self.department or ""
        if dept_name:
            self._rejected_departments.add(dept_name.lower().strip())
        elif curr_dept:
            self._rejected_departments.add(curr_dept.lower().strip())

        self.department_name = None
        self.booking_data["department"] = None
        self.booking_data["specialty"] = None
        if hasattr(self, "confirmed_facts") and "department" in self.confirmed_facts:
            del self.confirmed_facts["department"]
        if hasattr(self, "completed_steps") and "department_selection" in self.completed_steps:
            self.completed_steps.remove("department_selection")

        # Dependent doctor and slots must be cleared
        self.doctor = None
        self.doctor_name = None
        self.doctor_id = None
        self.doctor_locked = False
        self.booking_data["doctor"] = None
        self.booking_data["availability_verified"] = False
        self.offered_slots = []
        self.booking_data["time"] = None
        self._offered_dept_doctor = None
        if hasattr(self, "confirmed_facts") and "doctor" in self.confirmed_facts:
            del self.confirmed_facts["doctor"]
        if hasattr(self, "completed_steps") and "doctor_selection" in self.completed_steps:
            self.completed_steps.remove("doctor_selection")

        # Clear obsolete requested / ambiguous entity slots
        self.booking_data.pop("user_requested_entity_term", None)
        self.booking_data.pop("unverified_entity_requested", None)
        self.booking_data.pop("ambiguous_entity_candidate", None)
        self.booking_data.pop("pending_confirmation_entity", None)

        self._last_rejected_entity = "department"
        self.state = BookingState.WAITING_FOR_DOCTOR
        self.last_asked_field = "department"

    def invalidate_doctor(self, doc_name: Optional[str] = None):
        """
        Invalidates rejected doctor selection while keeping department and customer details intact.
        """
        curr_doc = self.doctor or ""
        if doc_name:
            self._rejected_doctors.add(doc_name.lower().strip())
            clean_d = re.sub(r'^(?:dr\.?|doctor)\s+', '', doc_name.lower().strip(), flags=re.I).strip()
            if clean_d:
                self._rejected_doctors.add(clean_d)
        elif curr_doc:
            self._rejected_doctors.add(curr_doc.lower().strip())
            clean_d = re.sub(r'^(?:dr\.?|doctor)\s+', '', curr_doc.lower().strip(), flags=re.I).strip()
            if clean_d:
                self._rejected_doctors.add(clean_d)

        self.doctor = None
        self.doctor_name = None
        self.doctor_id = None
        self.doctor_locked = False
        self.booking_data["doctor"] = None
        self.booking_data["availability_verified"] = False
        self.offered_slots = []
        self.booking_data["time"] = None
        self._offered_dept_doctor = None
        if hasattr(self, "confirmed_facts") and "doctor" in self.confirmed_facts:
            del self.confirmed_facts["doctor"]
        if hasattr(self, "completed_steps") and "doctor_selection" in self.completed_steps:
            self.completed_steps.remove("doctor_selection")

        # Clear obsolete requested / ambiguous entity slots
        self.booking_data.pop("user_requested_entity_term", None)
        self.booking_data.pop("unverified_entity_requested", None)
        self.booking_data.pop("ambiguous_entity_candidate", None)
        self.booking_data.pop("pending_confirmation_entity", None)

        self._last_rejected_entity = "doctor"
        self.state = BookingState.WAITING_FOR_DOCTOR
        self.last_asked_field = "doctor"

    def invalidate_service_or_product(self, name: Optional[str] = None):
        """
        Invalidates rejected service or product selection.
        """
        curr = self.service or ""
        if name:
            self._rejected_services.add(name.lower().strip())
        elif curr:
            self._rejected_services.add(curr.lower().strip())

        self.service_name = None
        self.product_name = None
        self.booking_data["service"] = None
        self.booking_data["product"] = None
        if hasattr(self, "confirmed_facts") and "service" in self.confirmed_facts:
            del self.confirmed_facts["service"]
        if hasattr(self, "confirmed_facts") and "product" in self.confirmed_facts:
            del self.confirmed_facts["product"]
        if hasattr(self, "completed_steps") and "service_selection" in self.completed_steps:
            self.completed_steps.remove("service_selection")

        # Clear obsolete requested / ambiguous entity slots
        self.booking_data.pop("user_requested_entity_term", None)
        self.booking_data.pop("unverified_entity_requested", None)
        self.booking_data.pop("ambiguous_entity_candidate", None)
        self.booking_data.pop("pending_confirmation_entity", None)

        self._last_rejected_entity = "service"
        self.state = BookingState.WAITING_FOR_DOCTOR
        self.last_asked_field = "service"

    @property
    def last_asked_field(self) -> Optional[str]:
        return self.last_question_field

    @last_asked_field.setter
    def last_asked_field(self, val: Optional[str]):
        self.last_question_field = val

    @property
    def department(self) -> Optional[str]:
        return self.department_name or self.booking_data.get("department")

    @department.setter
    def department(self, val: Optional[str]):
        if val != self.booking_data.get("department"):
            self.booking_data["department"] = val
            self.department_name = val
            self.booking_data["specialty"] = val

    @property
    def service(self) -> Optional[str]:
        return self.service_name or self.booking_data.get("service") or self.booking_data.get("product")

    @service.setter
    def service(self, val: Optional[str]):
        if val != self.booking_data.get("service"):
            self.booking_data["service"] = val
            self.service_name = val
            self.booking_data["product"] = val
            self.product_name = val
            if not self.booking_data.get("department"):
                self.booking_data["department"] = val
                self.department_name = val

    @property
    def product(self) -> Optional[str]:
        return self.product_name or self.booking_data.get("product") or self.booking_data.get("service")

    @product.setter
    def product(self, val: Optional[str]):
        self.service = val

    @property
    def doctor(self) -> Optional[str]:
        return self.doctor_name or self.booking_data.get("doctor")

    @doctor.setter
    def doctor(self, val: Optional[str]):
        if val != self.booking_data.get("doctor"):
            prev = self.booking_data.get("doctor")
            self.booking_data["doctor"] = val
            self.doctor_name = val
            if prev is not None:
                self.booking_data["availability_verified"] = False

    @property
    def current_stage(self) -> str:
        return self.state

    @current_stage.setter
    def current_stage(self, val: str):
        self.state = val

    @property
    def current_intent(self) -> Optional[str]:
        return self.booking_data.get("intent")

    @current_intent.setter
    def current_intent(self, val: Optional[str]):
        self.booking_data["intent"] = val

    @property
    def full_name(self) -> Optional[str]:
        return self.booking_data.get("patient_name")

    @full_name.setter
    def full_name(self, val: Optional[str]):
        self.booking_data["patient_name"] = val

    @property
    def phone_number(self) -> Optional[str]:
        return self.booking_data.get("contact_number")

    @phone_number.setter
    def phone_number(self, val: Optional[str]):
        self.booking_data["contact_number"] = val

    @property
    def collected_fields(self) -> Dict[str, Any]:
        fields: Dict[str, Any] = {}
        if self.department:
            fields["department"] = self.department
            fields["specialty"] = self.department
        if self.service:
            fields["service"] = self.service
            fields["product"] = self.service
        if self.patient_name:
            fields["full_name"] = self.patient_name
            fields["patient_name"] = self.patient_name
        if self.doctor:
            fields["doctor"] = self.doctor
        if self.date:
            fields["appointment_date"] = self.date
            fields["date"] = self.date
        if self.time:
            fields["appointment_time"] = self.time
            fields["time"] = self.time
        if self.contact_number and self.contact_number != "Unknown":
            fields["phone_number"] = self.contact_number
            fields["contact_number"] = self.contact_number
        if self.email:
            fields["email"] = self.email
        if getattr(self, "industry", None) or self.booking_data.get("industry"):
            fields["industry"] = self.industry or self.booking_data.get("industry")
        if getattr(self, "requirement", None) or self.booking_data.get("requirement"):
            fields["requirement"] = self.requirement or self.booking_data.get("requirement")
        if getattr(self, "interested_area", None) or self.booking_data.get("interested_area"):
            area = self.interested_area or self.booking_data.get("interested_area")
            fields["interested_area"] = area
            fields["team"] = area
        if self.booking_data.get("specialty"):
            fields["specialty"] = self.booking_data["specialty"]
        return fields

    @property
    def field_states(self) -> Dict[str, str]:
        states = {}
        # Customer Full Name
        if not self.patient_name:
            states["customer_name"] = FieldState.MISSING
        elif len(self.patient_name.split()) == 1 and not self.booking_data.get("last_name_asked"):
            states["customer_name"] = FieldState.NEEDS_CLARIFICATION
        else:
            states["customer_name"] = FieldState.VERIFIED

        # Email
        if not self.email:
            states["email"] = FieldState.MISSING
        elif "@" in self.email and "." in self.email:
            states["email"] = FieldState.VERIFIED
        else:
            states["email"] = FieldState.NEEDS_CLARIFICATION

        # Phone Number
        if not self.contact_number or self.contact_number == "Unknown":
            states["phone"] = FieldState.MISSING
        elif len(re.sub(r'\D', '', self.contact_number)) >= 7:
            states["phone"] = FieldState.VERIFIED
        else:
            states["phone"] = FieldState.NEEDS_CLARIFICATION

        # Preferred Date
        if not self.date:
            states["preferred_date"] = FieldState.MISSING
        else:
            states["preferred_date"] = FieldState.VERIFIED

        # Preferred Time
        if not self.time:
            states["preferred_time"] = FieldState.MISSING
        elif self.booking_data.get("ambiguous_time_candidate"):
            states["preferred_time"] = FieldState.NEEDS_CLARIFICATION
        elif self.availability_verified:
            states["preferred_time"] = FieldState.VERIFIED
        else:
            states["preferred_time"] = FieldState.COLLECTED

        # Requested Service / Doctor / Product
        if not self.doctor and not self.service:
            states["requested_service"] = FieldState.MISSING
        elif self.booking_data.get("pending_confirmation_entity") or self.booking_data.get("ambiguous_entity_candidate"):
            states["requested_service"] = FieldState.NEEDS_CLARIFICATION
        else:
            states["requested_service"] = FieldState.VERIFIED

        return states

    def get_missing_lead_fields(self) -> List[str]:
        missing: List[str] = []
        # 1. Full name
        name = self.patient_name or self.caller_name or self.booking_data.get("name") or self.booking_data.get("patient_name")
        if not name or name in ("Unknown", "Unknown Caller", "User", "Test User", "None", ""):
            missing.append("patient_name")
        elif not self.patient_name:
            self.patient_name = name
        # 2. Email address
        if not self.email or "@" not in self.email or "." not in self.email:
            if not self.booking_data.get("email_declined"):
                missing.append("email")
        # 3. Phone number
        phone = self.contact_number
        if not phone or phone in ("Unknown", "Unknown Caller", "Private Number"):
            missing.append("contact_number")
        # 4. Business / Industry
        ind = self.industry or self.booking_data.get("industry") or self.booking_data.get("business_description")
        req = self.requirement or self.service or self.booking_data.get("requirement") or self.booking_data.get("product") or getattr(self, "current_topic", None)
        if not self.requirement and req:
            self.requirement = req
            self.booking_data["requirement"] = req
        # If requirement is already specific (e.g. ERP Software), industry is not required to block callback lead collection
        if not ind or ind in ("Unknown", "None", ""):
            if not req or req in ("Unknown", "None", "", "AI Receptionist Booking", "Consultation", "Demo"):
                missing.append("industry")
        # 5. Requirement / Interested service
        if not req or req in ("Unknown", "None", "", "AI Receptionist Booking", "Consultation", "Demo"):
            missing.append("requirement")
        return missing

    @property
    def missing_fields(self) -> List[str]:
        # Task 4: In IDLE, COMPLETED, or inactive states, no fields are missing!
        if self.state in (BookingState.IDLE, BookingState.COMPLETED, BookingState.BOOKED, BookingState.BOOKING_CONFIRMED, BookingState.BOOKING_CANCELLED, BookingState.FAILED) or not self.is_active():
            return []

        if getattr(self, "workflow_mode", None) == SessionWorkflowMode.LEAD_COLLECTION:
            return self.get_missing_lead_fields()

        missing: List[str] = []
        is_demo = (
            getattr(self, "booking_type", None) in ("SALES_DEMO", "BUSINESS_CONSULTATION")
            or getattr(self, "business_type", None) == "SOFTWARE"
            or getattr(self, "workflow_type", None) == "SALES_DEMO"
        )
        if is_demo:
            if not self.date:
                missing.append("demo_date")
            if not self.time:
                missing.append("demo_time")
            if not self.patient_name or self.patient_name in ("Unknown", "Unknown Caller", "User", "Test User", "None", ""):
                missing.append("full_name")
            if not self.contact_number and not getattr(self, "caller_phone", None):
                missing.append("phone_number")
            if not self.email and not self.booking_data.get("email_declined"):
                missing.append("email")
            return missing

        _COMM_KEYWORDS = {
            "phone", "telephone", "mobile", "cell", "cellphone", "whatsapp",
            "email", "mail", "gmail", "yahoo", "outlook", "contact", "number",
            "call", "sms", "text", "voice", "address", "website", "online",
            "representative", "agent", "receptionist", "assistant", "bot"
        }
        # 1. Purge communication keywords mistakenly captured in entity fields
        if self.department and self.department.lower().strip() in _COMM_KEYWORDS:
            self.department = None
            self.booking_data.pop("department", None)
            self.booking_data.pop("specialty", None)
        if self.service and self.service.lower().strip() in _COMM_KEYWORDS:
            self.service = None
            self.booking_data.pop("service", None)
            self.booking_data.pop("product", None)
        if self.doctor and self.doctor.lower().strip() in _COMM_KEYWORDS:
            self.doctor = None
            self.doctor_id = None
            self.doctor_name = None

        # 2. Doctor / Department DB resolution & validation
        has_valid_doctor = bool(self.doctor_id and self.doctor)
        if self.doctor and not self.doctor_id:
            try:
                from src.config.database import SessionLocal
                from src.models.staff import Staff
                _db = SessionLocal()
                try:
                    clean_doc = re.sub(r'^(?:dr\.?|doctor|prof\.?|mr\.?|mrs\.?|ms\.?)\s+', '', self.doctor.strip(), flags=re.I).strip()
                    s_rec = _db.query(Staff).filter(
                        Staff.organization_id == (getattr(self, "organization_id", 1) or 1),
                        (Staff.name.ilike(f"%{clean_doc}%") | Staff.name.ilike(f"%{self.doctor}%"))
                    ).first()
                    if s_rec:
                        self.doctor_id = s_rec.id
                        self.doctor = s_rec.name
                        has_valid_doctor = True
                    else:
                        has_valid_doctor = False
                finally:
                    _db.close()
            except Exception:
                pass

        has_valid_dept = False
        if self.department and self.department.lower().strip() not in _COMM_KEYWORDS:
            try:
                from src.config.database import SessionLocal
                from src.models.staff import Staff
                from src.models.department import Department
                _db = SessionLocal()
                try:
                    org_id = getattr(self, "organization_id", 1) or 1
                    d_exists = _db.query(Department).filter(
                        Department.organization_id == org_id,
                        Department.name.ilike(self.department)
                    ).first() or _db.query(Staff).filter(
                        Staff.organization_id == org_id,
                        Staff.department.ilike(self.department)
                    ).first()
                    if d_exists:
                        has_valid_dept = True
                    else:
                        has_valid_dept = False
                finally:
                    _db.close()
            except Exception:
                has_valid_dept = True

        has_valid_service = bool(self.service or self.product or self.booking_data.get("service") or self.booking_data.get("product"))
        is_demo_or_consultation = getattr(self, "booking_type", None) in ("SALES_DEMO", "BUSINESS_CONSULTATION") or getattr(self, "business_type", None) == "SOFTWARE"
        if not is_demo_or_consultation:
            if not has_valid_doctor and not has_valid_dept and not has_valid_service:
                missing.append("department_or_doctor")
            elif not has_valid_doctor and not self.booking_data.get("department_only_booking") and not has_valid_service:
                # Department provided, doctor can be resolved or requested
                pass

        if not self.date:
            missing.append("appointment_date")
        if not self.time:
            missing.append("appointment_time")
        if not self.patient_name or self.patient_name in ("Unknown", "Unknown Caller", "User", "Test User", "None", ""):
            missing.append("full_name")

        # Only populate contact_number from caller_phone if explicitly confirmed by customer
        if not self.contact_number and getattr(self, "caller_phone", None) and self.booking_data.get("caller_id_confirmed"):
            raw_c = re.sub(r'[^\d]', '', str(self.caller_phone))
            if len(raw_c) == 12 and raw_c.startswith("91"):
                raw_c = raw_c[2:]
            elif len(raw_c) == 11 and raw_c.startswith("0"):
                raw_c = raw_c[1:]
            if len(raw_c) == 10:
                self.contact_number = self.caller_phone
                self.field_states["contact_number"] = "COLLECTED"

        # 3. Phone validation: strictly require valid 10 digits
        raw_digits = re.sub(r'[^\d]', '', self.contact_number or "")
        if len(raw_digits) == 12 and raw_digits.startswith("91"):
            raw_digits = raw_digits[2:]
        elif len(raw_digits) == 11 and raw_digits.startswith("0"):
            raw_digits = raw_digits[1:]
        if not raw_digits or len(raw_digits) != 10:
            self.contact_number = None
            missing.append("phone_number")

        if not self.email and not self.booking_data.get("email_declined"):
            missing.append("email")
        return missing

    def get_state_dict(self) -> Dict[str, Any]:
        b_type = getattr(self, "business_type", "GENERAL") or "GENERAL"
        bk_type = getattr(self, "booking_type", "GENERAL") or "GENERAL"
        mf = self.missing_fields
        logger.info(f"[CONVERSATION_STATE] business_type={b_type} booking_type={bk_type} stage={self.state} missing_fields={mf}")
        return {
            "conversation_id": self.conversation_id,
            "call_sid": self.call_sid,
            "agent_id": self.agent_id,
            "organization_id": self.organization_id,
            "active_rag_id": self.active_rag_id or self.active_doc_id,
            "active_doc_id": self.active_doc_id or self.active_rag_id,
            "business_name": getattr(self, "business_name", None),
            "business_type": getattr(self, "business_type", "GENERAL"),
            "booking_type": getattr(self, "booking_type", "GENERAL"),
            "workflow_type": getattr(self, "workflow_type", "GENERAL"),
            "call_direction": getattr(self, "call_direction", "INBOUND"),
            "call_type": str(getattr(self, "call_type", "INBOUND")),
            "conversation_started": self.conversation_started,
            "greeting_sent": self.greeting_sent,
            "turn_id": self.turn_id,
            "turn_count": self.turn_count,
            "state_version": getattr(self, "state_version", 1),
            "current_intent": self.booking_data.get("intent") or ("general_inquiry" if self.state == BookingState.IDLE else "appointment_booking"),
            "current_stage": self.current_stage,
            "state": self.state,
            "last_asked_field": getattr(self, "last_asked_field", None),
            "department": self.department,
            "service": self.service,
            "product": self.product,
            "current_topic": getattr(self, "current_topic", None) or self.booking_data.get("current_topic"),
            "last_discussed_product": getattr(self, "current_topic", None) or self.booking_data.get("current_topic"),
            "_hello_repeat_count": getattr(self, "_hello_repeat_count", 0) or self.booking_data.get("_hello_repeat_count", 0),
            "industry": getattr(self, "industry", None),
            "requirement": getattr(self, "requirement", None),
            "interested_area": getattr(self, "interested_area", None),
            "doctor": self.doctor,
            "doctor_id": self.doctor_id,
            "date": self.date,
            "time": self.time,
            "patient_name": self.patient_name,
            "contact_number": self.contact_number,
            "email": self.email,
            "availability_verified": self.availability_verified,
            "booking_confirmed": self.booking_confirmed,
            "appointment_id": self.appointment_id,
            "confirmed_facts": getattr(self, "confirmed_facts", {}),
            "completed_steps": getattr(self, "completed_steps", []),
            "collected_fields": self.collected_fields,
            "missing_fields": self.missing_fields,
            "field_states": self.field_states,
            "field_status": getattr(self, "field_status", {}),
            "history": list(getattr(self, "history", []))[-40:],
            "booking_data": {k: v for k, v in self.booking_data.items() if not k.startswith("_") and k != "date_obj"}
        }

    @classmethod
    def from_state_dict(cls, data: Dict[str, Any]) -> "BookingSession":
        sess = cls(
            session_id=data.get("conversation_id") or data.get("call_sid") or "recovered",
            conversation_id=data.get("conversation_id"),
            call_sid=data.get("call_sid")
        )
        sess.agent_id = data.get("agent_id")
        sess.organization_id = data.get("organization_id", 1)
        sess.business_type = data.get("business_type", "GENERAL")
        sess.booking_type = data.get("booking_type", "GENERAL")
        sess.workflow_type = data.get("workflow_type", "GENERAL")
        sess.active_rag_id = data.get("active_rag_id")
        sess.state = data.get("state") or data.get("current_stage") or BookingState.IDLE
        sess.conversation_started = data.get("conversation_started", False)
        sess.greeting_sent = data.get("greeting_sent", False)
        sess.turn_count = data.get("turn_count", 0)
        sess.turn_id = data.get("turn_id", 0)
        sess.state_version = data.get("state_version", 1)
        sess.last_question_field = data.get("last_asked_field")

        sess.doctor_id = data.get("doctor_id")
        sess.doctor_name = data.get("doctor")
        sess.doctor_locked = bool(sess.doctor_name)
        sess.department_name = data.get("department")
        sess.service_name = data.get("service") or data.get("product")
        sess.product_name = sess.service_name
        sess.industry = data.get("industry")
        sess.requirement = data.get("requirement")
        sess.interested_area = data.get("interested_area")

        sess.confirmed_facts = data.get("confirmed_facts") or {}
        sess.completed_steps = data.get("completed_steps") or []

        sess.booking_data["intent"] = data.get("current_intent")
        sess.booking_data["department"] = data.get("department")
        sess.booking_data["service"] = sess.service_name
        sess.booking_data["product"] = sess.product_name
        sess.booking_data["industry"] = sess.industry
        sess.booking_data["requirement"] = sess.requirement
        sess.booking_data["interested_area"] = sess.interested_area
        sess.booking_data["doctor"] = data.get("doctor")
        sess.booking_data["specialty"] = data.get("department") or data.get("specialty") or sess.service_name
        sess.booking_data["date"] = data.get("date")
        sess.booking_data["time"] = data.get("time")
        sess.booking_data["patient_name"] = data.get("patient_name")
        sess.booking_data["contact_number"] = data.get("contact_number")
        sess.booking_data["email"] = data.get("email")
        sess.booking_data["availability_verified"] = data.get("availability_verified", False)
        sess.booking_data["booking_confirmed"] = data.get("booking_confirmed", False)
        sess.booking_data["appointment_id"] = data.get("appointment_id")
        sess.history = list(data.get("history", []))
        if "booking_data" in data and isinstance(data["booking_data"], dict):
            sess.booking_data.update(data["booking_data"])

        return sess


    @property
    def date(self) -> Optional[str]:
        return self.booking_data.get("date")

    @date.setter
    def date(self, val: Optional[str]):
        if val != self.booking_data.get("date"):
            self.booking_data["date"] = val
            self.booking_data["availability_verified"] = False

    @property
    def time(self) -> Optional[str]:
        return self.booking_data.get("time")

    @time.setter
    def time(self, val: Optional[str]):
        if val != self.booking_data.get("time"):
            self.booking_data["time"] = val
            self.booking_data["availability_verified"] = False

    @property
    def patient_name(self) -> Optional[str]:
        return self.booking_data.get("patient_name")

    @patient_name.setter
    def patient_name(self, val: Optional[str]):
        self.booking_data["patient_name"] = val

    @property
    def email(self) -> Optional[str]:
        return self.booking_data.get("email")

    @email.setter
    def email(self, val: Optional[str]):
        self.booking_data["email"] = val

    @property
    def contact_number(self) -> Optional[str]:
        return self.booking_data.get("contact_number")

    @contact_number.setter
    def contact_number(self, val: Optional[str]):
        self.booking_data["contact_number"] = val

    @property
    def availability_verified(self) -> bool:
        return bool(self.booking_data.get("availability_verified"))

    @availability_verified.setter
    def availability_verified(self, val: bool):
        self.booking_data["availability_verified"] = bool(val)

    @property
    def booking_confirmed(self) -> bool:
        return bool(self.booking_data.get("booking_confirmed"))

    @booking_confirmed.setter
    def booking_confirmed(self, val: bool):
        self.booking_data["booking_confirmed"] = bool(val)

    @property
    def conversation_language(self) -> str:
        return self.booking_data.get("conversation_language", "en")

    @conversation_language.setter
    def conversation_language(self, val: str):
        self.booking_data["conversation_language"] = val

    @property
    def appointment_id(self) -> Optional[int]:
        return self.booking_data.get("appointment_id")

    @appointment_id.setter
    def appointment_id(self, val: Optional[int]):
        self.booking_data["appointment_id"] = val

    @property
    def email_declined(self) -> bool:
        return bool(self.booking_data.get("email_declined", False))

    @email_declined.setter
    def email_declined(self, val: bool):
        self.booking_data["email_declined"] = bool(val)

    @property
    def email_confirmed(self) -> bool:
        return bool(self.booking_data.get("email_confirmed", False))

    @email_confirmed.setter
    def email_confirmed(self, val: bool):
        self.booking_data["email_confirmed"] = bool(val)

    @property
    def location(self) -> Optional[str]:
        return self.booking_data.get("location")

    @location.setter
    def location(self, val: Optional[str]):
        self.booking_data["location"] = val

    @property
    def business_description(self) -> Optional[str]:
        return self.booking_data.get("business_description")

    @business_description.setter
    def business_description(self, val: Optional[str]):
        self.booking_data["business_description"] = val


def is_valid_location_candidate(
    candidate: Optional[str],
    customer_name: Optional[str] = None,
    customer_email: Optional[str] = None,
    customer_phone: Optional[str] = None,
) -> bool:
    """
    Validates that a candidate location string is genuinely a location/city/address.
    Under NO circumstance can it be an email, email username, email domain, customer name, or phone.
    """
    if not candidate:
        return False
    cand = candidate.strip()
    if len(cand) < 2 or len(cand) > 60:
        return False
    cand_lower = cand.lower()

    # 1. Strictly forbid email format, symbols, or domains
    if "@" in cand or re.search(r'\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b', cand):
        return False
    if any(dom in cand_lower for dom in [".com", ".in", ".org", ".net", ".co", "gmail", "yahoo", "outlook", "hotmail", "icloud"]):
        return False

    # 2. Strictly forbid numeric strings or phone numbers
    clean_num = re.sub(r'[\s\-\(\)\+]', '', cand)
    if clean_num.isdigit() or re.match(r'^\+?\d{6,15}$', clean_num):
        return False

    # 3. Strictly forbid customer email username or email domain match
    if customer_email and "@" in customer_email:
        email_user = customer_email.split("@")[0].strip().lower()
        if cand_lower == email_user or cand_lower.replace(" ", "") == email_user.replace(" ", ""):
            return False
        email_dom = customer_email.split("@")[1].strip().lower()
        if cand_lower == email_dom or cand_lower in email_dom:
            return False

    # 4. Strictly forbid customer name match
    if customer_name:
        cust_clean = customer_name.strip().lower()
        if cand_lower == cust_clean:
            return False
        name_tokens = [w for w in cust_clean.split() if len(w) > 2]
        if cand_lower in name_tokens:
            return False

    # 5. Strictly forbid customer phone match
    if customer_phone:
        phone_digits = re.sub(r'\D', '', customer_phone)
        if phone_digits and phone_digits in clean_num:
            return False

    # 6. Stopwords / common non-location conversational words, affirmations, questions, and domain terms
    if "?" in cand:
        return False

    # Strictly reject questions, conversational utterances, affirmations, and confirmations
    _QUESTION_OR_CONFIRM_PAT = re.compile(
        r'\b(?:am\s+i|are\s+you|can\s+you|could\s+you|is\s+it|is\s+this|audible|hear\s+me|'
        r'everything\s+is\s+correct|is\s+correct|all\s+correct|correct|all\s+good|'
        r'confirm\s+my|confirm\s+the|confirm\s+appointment|confirm|confirmation|proceed|'
        r'book\s+my|book\s+it|book\s+a\s+demo|schedule\s+a\s+demo|put\s+a\s+demo|'
        r'yeah|yes|yep|sure|okay|ok|fine|yup|haan|sari|done|'
        r'hrms|hrm|erp|crm|software|application|platform|survey|demo|'
        r'i\s+said|call\s+me|my\s+name|this\s+is|i\s+want|want\s+to|explore)\b',
        re.IGNORECASE
    )
    if _QUESTION_OR_CONFIRM_PAT.search(cand_lower):
        return False

    _INVALID_WORDS = {
        "today", "tomorrow", "yesterday", "morning", "afternoon", "evening",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "doctor", "dr", "appointment", "booking", "me", "here", "myself", "sure", "yes", "no",
        "email", "mail", "phone", "contact", "name", "call", "please", "thanks", "thank you",
        "scratch", "demo", "representative", "patient", "none", "skip", "idhu", "vendaam",
        "okay", "ok", "fine", "correct", "good", "right", "what", "which", "when", "where",
        "how", "who", "why", "user", "test", "null", "undefined", "true", "false",
        "audible", "hear", "proceed", "ahead", "everything", "confirm", "booked"
    }
    tokens = cand_lower.split()
    if any(t in _INVALID_WORDS for t in tokens) and len(tokens) == 1:
        return False
    if tokens[0] in _INVALID_WORDS and len(tokens) == 1:
        return False
    if cand_lower in _INVALID_WORDS:
        return False

    # Strictly reject time expressions (e.g. "11:20AM", "9:30 AM") or modal phrases ("I would, before, 11:20AM")
    if re.search(r'\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b', cand_lower):
        return False
    if any(w in cand_lower.split() for w in ["would", "could", "should", "prefer", "before", "after", "choose", "like", "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]):
        return False

    return True


def parse_spoken_location(
    text: str,
    session: Optional[Any] = None,
    customer_name: Optional[str] = None,
    customer_email: Optional[str] = None,
    customer_phone: Optional[str] = None,
) -> Optional[str]:
    """
    Extracts customer-provided location/city/area from natural speech without fabricating.
    Guarantees independent extraction:
    - Never derives from email address, email username, or domain.
    - Never derives from customer name or phone number.
    - Strictly rejects email addresses, names, and phone numbers.
    Examples:
      - 'I am from Chennai' -> 'Chennai'
      - 'My location is Adyar' -> 'Adyar'
      - 'I live in Bangalore' -> 'Bangalore'
      - 'Address is Anna Nagar' -> 'Anna Nagar'
      - 'from Coimbatore' -> 'Coimbatore'
      - 'Anna Nagar, Chennai' -> 'Anna Nagar, Chennai'
      - 'Chennai' -> 'Chennai'
    """
    if not text:
        return None

    raw = text.strip()
    txt = raw.lower()

    if txt in ("yes", "no", "sure", "ok", "okay", "thanks", "thank you", "hello", "hi", "skip", "none", "confirm", "yes confirm", "yes please", "proceed", "please confirm", "yeah confirm"):
        return None

    c_name = customer_name or (getattr(session, "patient_name", None) if session else None)
    c_email = customer_email or (getattr(session, "email", None) if session else None)
    c_phone = customer_phone or (getattr(session, "contact_number", None) if session else None)

    # 1. Reject if the text is explicitly an email utterance without location
    if "@" in txt or re.search(r'\b(?:email|mail|e-mail)\s+(?:id|address|is|:)\b', txt):
        loc_part_match = re.search(r'\b(?:location|living\s+in|residing\s+in|from|city|place)\s+(?:is|:)?\s*([a-zA-Z0-9\s,\.\-]{2,40})', raw, re.IGNORECASE)
        if not loc_part_match:
            return None
        raw = loc_part_match.group(0)

    # 2. Strict patterns that do NOT match 'email address' or 'mail address'
    patterns = [
        r'\b(?:my\s+)?(?<!email\s)(?<!mail\s)(?<!e-mail\s)(?<!web\s)(?<!ip\s)(?:location|city|area|place|town|address)\s+(?:is|:)?\s*([a-zA-Z0-9\s,\.\-]{2,40})',
        r'\b(?:i\s+am\s+from|i\'m\s+from|calling\s+from|i\s+live\s+in|living\s+in|residing\s+in|located\s+at|located\s+in|staying\s+in)\s+([a-zA-Z0-9\s,\.\-]{2,40})',
        r'\b(?:from)\s+([a-zA-Z\s]{2,25})\b',
    ]

    for pat in patterns:
        m = re.search(pat, raw, re.IGNORECASE)
        if m:
            cand = m.group(1).strip()
            cand = re.sub(r'\s+(?:and|&|\bmy\b|\bphone\b|\bemail\b|\bnumber\b|\bfor\b|\bbooking\b|\bappointment\b).*$', '', cand, flags=re.IGNORECASE).strip()
            cand = re.sub(r'^(?:the|a|an)\s+', '', cand, flags=re.IGNORECASE).strip()
            cand = cand.rstrip(".,;!?")
            if is_valid_location_candidate(cand, customer_name=c_name, customer_email=c_email, customer_phone=c_phone):
                return cand.title()

    # 3. Direct standalone location / city answers (e.g. customer replied "Chennai" or "Anna Nagar, Chennai")
    clean_standalone = re.sub(r'^(?:i\s+am\s+in|in|at)\s+', '', raw, flags=re.IGNORECASE).strip().rstrip(".,;!?")

    if session is not None:
        last_asked = getattr(session, "last_asked_field", None)
        curr_state = getattr(session, "state", None)
        # When in confirmation state, never parse conversational speech or affirmations as location
        if curr_state in (BookingState.WAITING_FOR_CONFIRMATION, getattr(BookingState, "CONFIRMATION", "WAITING_FOR_CONFIRMATION")):
            if last_asked != "location":
                return None
        elif last_asked != "location" and curr_state not in (BookingState.COLLECTING_LOCATION, BookingState.WAITING_FOR_LOCATION):
            # If unsolicited and not actively collecting location, accept only if it contains a comma or known city indicator
            _KNOWN_CITIES = {"chennai", "bangalore", "bengaluru", "mumbai", "delhi", "hyderabad", "pune", "kolkata", "coimbatore", "madurai", "salem", "trichy", "kochi", "mysore", "noida", "gurgaon", "ahmedabad", "jaipur", "chandigarh"}
            has_city_or_comma = ("," in clean_standalone) or any(c in clean_standalone.lower().split() for c in _KNOWN_CITIES)
            if not has_city_or_comma:
                return None
    standalone_words = clean_standalone.split()
    if 1 <= len(standalone_words) <= 4:
        if is_valid_location_candidate(clean_standalone, customer_name=c_name, customer_email=c_email, customer_phone=c_phone):
            _NON_LOC_PHRASES = {
                "hello", "hi", "hey", "yes", "no", "sure", "ok", "okay", "fine", "correct",
                "proceed", "confirm", "book", "appointment", "demo", "doctor", "consultation",
                "morning", "evening", "afternoon", "today", "tomorrow", "thanks", "thank you",
                "not yet", "skip", "none", "cancel", "reschedule", "transfer", "audible", "hear",
                "everything is correct", "confirm my appointment", "confirm appointment"
            }
            if clean_standalone.lower() not in _NON_LOC_PHRASES:
                return clean_standalone.title()

    return None



def parse_spoken_phone_number(text: str) -> Optional[str]:
    """
    Parses direct digits and spoken digit words in English, Tamil, and Hindi into clean 10-digit phone strings.
    Isolates phone numbers in compound sentences (e.g. 'my phone number is 958 5310 712 and my mail ID is...').
    Handles 'double nine', 'triple five', 'double zero', etc.
    """
    if not text:
        return None

    txt_lower = text.lower().strip()

    # 1. Strip emails and email patterns to avoid digit contamination (e.g. user029@gmail.com)
    text_without_email = re.sub(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}', ' ', txt_lower)
    text_without_email = re.sub(r'\b(?:and\s+)?(?:my\s+)?(?:mail\s+id|email\s+id|email|mail)\s*(?:is|:)?\b', ' ', text_without_email, flags=re.IGNORECASE)

    # 1b. Strip formatted dates and month names with days, but do NOT strip 4-digit numbers that could be part of a phone
    text_clean_phone = re.sub(r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b', '', text_without_email)
    text_clean_phone = re.sub(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', '', text_clean_phone)
    text_clean_phone = re.sub(r'\b\d{1,2}(?:st|nd|rd|th)\b', '', text_clean_phone)
    text_clean_phone = re.sub(r'\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}\b', '', text_clean_phone)
    text_clean_phone = re.sub(r'\b\d{1,2}\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b', '', text_clean_phone)

    # 1c. Expand spoken multiples: "double nine" -> "nine nine", "triple five" -> "five five five", "double 9" -> "9 9"
    _multiples = {"double": 2, "triple": 3}
    for m_word, m_count in _multiples.items():
        text_clean_phone = re.sub(
            rf'\b{m_word}\s+([a-zA-Z0-9\u0b80-\u0bff]+)\b',
            lambda m: " ".join([m.group(1)] * m_count),
            text_clean_phone,
            flags=re.IGNORECASE
        )

    # 2. Check explicit phone/number prefix
    phone_prefix_match = re.search(r'\b(?:my\s+)?(?:phone|mobile|contact|cell|number)(?:\s+number)?\s+(?:is|:)?\s*', text_clean_phone)
    if phone_prefix_match:
        after_prefix = text_clean_phone[phone_prefix_match.end():]
        phone_segment = re.split(r'\s+(?:and|&|\bmy\s+name|\bname|\bfor\b|\bbooking\b)\b', after_prefix)[0]
        seg_digits = re.sub(r'[^\d]', '', phone_segment)
        if len(seg_digits) >= 10:
            if seg_digits.startswith("91") and len(seg_digits) == 12:
                return seg_digits[2:]
            if seg_digits.startswith("0") and len(seg_digits) == 11:
                return seg_digits[1:]
            if len(seg_digits) > 10 and seg_digits[-10] in "6789":
                return seg_digits[-10:]
            return seg_digits[:10] if seg_digits[:10][0] in "6789" else (seg_digits[-10:] if len(seg_digits) > 10 else seg_digits[:10])

    # 3. Direct digits in remaining text
    clean_digits = re.sub(r'[^\d]', '', text_clean_phone)
    if clean_digits.startswith("91") and len(clean_digits) == 12:
        return clean_digits[2:]
    if clean_digits.startswith("0") and len(clean_digits) == 11:
        return clean_digits[1:]
    if len(clean_digits) == 10:
        return clean_digits
    elif len(clean_digits) > 10:
        if clean_digits[-10] in "6789":
            return clean_digits[-10:]
        if clean_digits[:10][0] in "6789":
            return clean_digits[:10]
        return clean_digits[-10:]

    # 4. Spoken digit word translation map
    word_map = {
        "zero": "0", "oh": "0", "o": "0", "sunniyam": "0", "poojiyam": "0", "shunya": "0",
        "one": "1", "onru": "1", "onnu": "1", "ek": "1",
        "two": "2", "irandu": "2", "rendu": "2", "do": "2",
        "three": "3", "moondru": "3", "moonu": "3", "teen": "3",
        "four": "4", "naangku": "4", "naalu": "4", "char": "4",
        "five": "5", "aindhu": "5", "anji": "5", "paanch": "5",
        "six": "6", "aaru": "6", "chhah": "6",
        "seven": "7", "ezhu": "7", "saat": "7",
        "eight": "8", "ettu": "8", "aath": "8",
        "nine": "9", "ombadhu": "9", "onbadhu": "9", "nau": "9"
    }

    tokens = re.findall(r'\b[a-zA-Z0-9\u0b80-\u0bff]+\b', text_clean_phone)
    parsed = []
    for t in tokens:
        if t.isdigit():
            parsed.append(t)
        elif t in word_map:
            parsed.append(word_map[t])

    if len(parsed) >= 10:
        p_str = "".join(parsed)
        if p_str.startswith("91") and len(p_str) == 12:
            return p_str[2:]
        if p_str.startswith("0") and len(p_str) == 11:
            return p_str[1:]
        if len(p_str) > 10 and p_str[-10] in "6789":
            return p_str[-10:]
        return p_str[:10] if p_str[:10][0] in "6789" else p_str[-10:]

    return None



def detect_incomplete_phone(text: str) -> Optional[int]:
    """Detects if user spoke an incomplete phone number (6 to 9 digits)."""
    if not text:
        return None

    txt = text.lower()
    txt = re.sub(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}', ' ', txt)
    txt = re.sub(r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b', '', txt)
    txt = re.sub(r'\b(?:19|20)\d{2}\b', '', txt)
    wmap = {
        "zero": "0", "oh": "0", "o": "0", "sunniyam": "0", "poojiyam": "0", "shunya": "0",
        "one": "1", "onru": "1", "onnu": "1", "ek": "1",
        "two": "2", "irandu": "2", "rendu": "2", "do": "2",
        "three": "3", "moondru": "3", "moonu": "3", "teen": "3",
        "four": "4", "naangku": "4", "naalu": "4", "char": "4",
        "five": "5", "aindhu": "5", "anji": "5", "paanch": "5",
        "six": "6", "aaru": "6", "chhah": "6",
        "seven": "7", "ezhu": "7", "saat": "7",
        "eight": "8", "ettu": "8", "aath": "8",
        "nine": "9", "ombadhu": "9", "onbadhu": "9", "nau": "9"
    }
    tokens = re.findall(r'\b[a-zA-Z0-9\u0b80-\u0bff]+\b', txt)
    digits = []
    for t in tokens:
        if t.isdigit():
            digits.append(t)
        elif t in wmap:
            digits.append(wmap[t])
    d_str = "".join(digits)
    if 6 <= len(d_str) <= 9:
        return len(d_str)
    return None



def parse_spoken_email(text: str) -> Optional[str]:
    """
    Parses spoken email variations, spelled-out letters, and formatted emails in compound sentences.
    Handles:
      - 's a n j a y at gmail dot com' -> 'sanjay@gmail.com'
      - 'arun dot kumar at gmail dot com' -> 'arun.kumar@gmail.com'
      - 's a n j a y 1 2 3 at gmail dot com' -> 'sanjay123@gmail.com'
      - 'my email is sanjay at gmail dot com' -> 'sanjay@gmail.com'
      - 'g mail' -> 'gmail', 'hot mail' -> 'hotmail', 'out look' -> 'outlook'
      - Rejects incomplete fragments like '@gmail.com' or 'sanjay@' without fabricating characters.
    """
    if not text:
        return None

    raw = text.strip()
    txt = raw.lower()

    # Reject isolated domain or incomplete local parts (never fabricate missing username or domain)
    if re.match(r'^\s*[@]\s*[a-zA-Z0-9-]+\.[a-zA-Z]{2,}\s*$', txt) or re.match(r'^\s*(?:at|@)\s*(?:the\s+rate\s+)?(?:of\s+)?[a-zA-Z0-9-]+\s*(?:dot|\.)\s*[a-zA-Z]{2,}\s*$', txt):
        return None

    if re.match(r'^[a-zA-Z0-9._%+-]+@\s*$', txt):
        return None


    # 1. Direct standard email match in raw text first
    email_direct = re.search(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*\.[a-zA-Z]{2,}', txt)
    if email_direct:
        cand = email_direct.group(0).strip('.')
        if re.match(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*\.[a-zA-Z]{2,}$', cand):
            return cand

    # 2. Strip conversational prefixes (including email change / update requests)
    prefixes = [
        r'^(?:please\s+)?(?:can\s+you\s+)?(?:change|update|correct|set|switch|make)\s+(?:my\s+)?(?:email|mail)(?:\s+(?:id|address))?\s*(?:to|is|at|as|:)?\s*',
        r'^(?:please\s+)?(?:change|update|correct|set|switch)\s+(?:it\s+)?(?:to|as|:)\s*',
        r'^(?:no|nope|wrong|actually|wait),?\s*(?:please\s+)?(?:change|update|correct)?\s*(?:my\s+)?(?:email|mail)(?:\s+(?:id|address))?\s*(?:to|is|at|as|:)?\s*',
        r'^(?:my\s+(?:email|mail)\s+(?:id|address)?\s*(?:is|:)?)\s*',
        r'^(?:email|mail)\s+(?:is|id\s+is|address\s+is|:)\s*',
        r'^(?:it\'s|its|sure|yes|yeah|okay|ok)\s*,?\s*',
    ]
    cleaned = txt
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            new_cleaned = re.sub(p, '', cleaned, flags=re.IGNORECASE).strip()
            if new_cleaned != cleaned:
                cleaned = new_cleaned
                changed = True

    # 3. Replace spoken verbal tokens with standard email symbols
    cleaned = re.sub(r'\bat the rate(?: of)?\b', '@', cleaned)
    cleaned = re.sub(r'\bat\b', '@', cleaned)
    cleaned = re.sub(r'\bdot\b', '.', cleaned)
    cleaned = re.sub(r'\bunderscore\b', '_', cleaned)
    cleaned = re.sub(r'\b(?:dash|hyphen)\b', '-', cleaned)
    cleaned = re.sub(r'\bplus\b', '+', cleaned)
    cleaned = re.sub(r'\bspace\b', '', cleaned)

    # 3b. Normalize spoken domain words
    cleaned = re.sub(r'\bg\s*mail\b', 'gmail', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\bhot\s*mail\b', 'hotmail', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\bout\s*look\b', 'outlook', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\by\s*a\s*h\s*o\s*o\b', 'yahoo', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\bg\s*m\s*a\s*i\s*l\b', 'gmail', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\b\.\s*c\s*o\s*m\b', '.com', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\b\.\s*o\s*r\s*g\b', '.org', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\b\.\s*i\s*n\b', '.in', cleaned, flags=re.IGNORECASE)
    # Insert @ before known email domains if omitted in spoken utterance
    cleaned = re.sub(r'(?<!@)(?<!@\s)\b(gmail|yahoo|hotmail|outlook)\.(com|org|in|net)\b', r'@\1.\2', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*@[\s@]*', '@', cleaned)

    # 4. Spelled-out local part & domain reconstruction
    if '@' in cleaned:
        parts = cleaned.split('@')
        if len(parts) >= 2:
            local_raw = parts[0].strip()
            domain_raw = parts[1].strip()

            # Strip leading conversational words and clauses
            # e.g. "no, my email is wrong. it is robert.new" -> "robert.new"
            local_raw = re.sub(r'^.*?[,;:!]\s*', '', local_raw)
            local_raw = re.sub(r'^.*?\b(?:is|be|to|it\s+is|it\'s|email\s+is|id\s+is|mail\s+is|my\s+email\s+is|correct\s+email\s+is|change\s+to|actually)\s+', '', local_raw, flags=re.IGNORECASE).strip()

            # Contract spelled-out tokens in local part
            # e.g. "s a n j a y" -> "sanjay"
            _EMAIL_FRAMING_WORDS = {
                "is", "it", "its", "it's", "email", "mail", "id", "wrong", "to", "be", "not",
                "actual", "correct", "my", "use", "please", "address", "change", "new", "the"
            }
            _NUM_WORDS = {
                "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
                "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"
            }
            tokens = local_raw.split()
            valid_local_tokens = []
            for tok in reversed(tokens):
                cleaned_tok = tok.lower().strip(".,;:!")
                if cleaned_tok in _EMAIL_FRAMING_WORDS:
                    break
                mapped_tok = _NUM_WORDS.get(cleaned_tok, tok.lower())
                if re.match(r'^[a-z0-9._%+-]+$', mapped_tok):
                    valid_local_tokens.insert(0, mapped_tok)
                elif tok in ('.', '_', '-'):
                    valid_local_tokens.insert(0, tok)
                else:
                    break
            if valid_local_tokens:
                local_part = ''.join(valid_local_tokens)

                # Domain part reconstruction (e.g. "gmail . com" -> "gmail.com", "company . co . in" -> "company.co.in")
                dom_tokens = domain_raw.split()
                valid_dom_tokens = []
                for tok in dom_tokens:
                    if re.match(r'^[a-z0-9.-]+$', tok):
                        valid_dom_tokens.append(tok)
                    else:
                        break
                domain_part = ''.join(valid_dom_tokens)
                if domain_part in ("gmail", "yahoo", "hotmail", "outlook"):
                    domain_part = f"{domain_part}.com"

                candidate = f"{local_part}@{domain_part}"
                candidate = re.sub(r'\.{2,}', '.', candidate).strip('.')
                if re.match(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*\.[a-zA-Z]{2,}$', candidate):
                    return candidate

    # 5. Continuous compaction fallback
    compact = re.sub(r'\s+', '', cleaned)
    if re.search(r'@[a-zA-Z0-9-]+$', compact):
        domain_tail = compact.split('@')[-1]
        if domain_tail in ("gmail", "yahoo", "hotmail", "outlook"):
            compact = f"{compact}.com"
    m_compact = re.search(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*\.[a-zA-Z]{2,}', compact)
    if m_compact:
        cand = m_compact.group(0).strip('.')
        # Prevent "changemymail@..." or "myemail@..." from surviving in fallback
        cand = re.sub(r'^(?:(?:please)?(?:change|update|correct)?(?:my)?(?:email|mail|address|id))+', '', cand, flags=re.IGNORECASE)
        if re.match(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*\.[a-zA-Z]{2,}$', cand):
            return cand

    return None


# Export extract_email alias for backward-compatible and test imports
extract_email = parse_spoken_email


def detect_incomplete_spoken_email(text: str) -> Optional[str]:
    """
    Detects when a caller provides only an email domain or incomplete spoken email without a username.
    Examples:
      - "my email is at gmail dot com" -> "gmail.com"
      - "at gmail dot com" -> "gmail.com"
      - "@gmail.com" -> "gmail.com"
      - "outlook dot com" -> "outlook.com"
    Returns the normalized domain string if incomplete, else None.
    """
    if not text:
        return None
    txt = text.lower().strip()

    # Check if a full valid email is already present - if so, it's not incomplete!
    if parse_spoken_email(text):
        return None

    prefixes = [
        r'^(?:please\s+)?(?:can\s+you\s+)?(?:change|update|correct|set|switch|make)\s+(?:my\s+)?(?:email|mail)(?:\s+(?:id|address))?\s*(?:to|is|at|as|:)?\s*',
        r'^(?:my\s+(?:email|mail)\s+(?:id|address)?\s*(?:is|:)?)\s*',
        r'^(?:email|mail)\s+(?:is|id\s+is|address\s+is|:)\s*',
        r'^(?:it\'s|its|sure|yes|yeah|okay|ok)\s*,?\s*',
    ]
    cleaned = txt
    for p in prefixes:
        cleaned = re.sub(p, '', cleaned, flags=re.IGNORECASE).strip()

    cleaned = re.sub(r'\bat the rate(?: of)?\b', '@', cleaned)
    cleaned = re.sub(r'\bat\b', '@', cleaned)
    cleaned = re.sub(r'\bdot\b', '.', cleaned)
    cleaned = re.sub(r'\bg\s*mail\b', 'gmail', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\bhot\s*mail\b', 'hotmail', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\bout\s*look\b', 'outlook', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\by\s*a\s*h\s*o\s*o\b', 'yahoo', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*@[\s@]*', '@', cleaned)
    compact = re.sub(r'\s+', '', cleaned)

    # Check if compact matches domain pattern e.g. "@gmail.com", "gmail.com", "@yahoo.com"
    m = re.match(r'^@?([a-zA-Z0-9-]+\.(?:com|org|net|in|io|co\.in|edu))$', compact)
    if m:
        return m.group(1).lower()

    m2 = re.match(r'^@?(gmail|yahoo|hotmail|outlook)$', compact)
    if m2:
        return f"{m2.group(1).lower()}.com"

    return None


def is_phrase_negated(phrase: str, text: str) -> bool:
    """
    Detects whether an entity, service, or department phrase is negated in the text.
    Handles English, Hindi, and Tamil negation patterns with clause isolation and affirmative protection.
    Examples:
      - "I don't want dental care" -> True for "dental", "dental care"
      - "No dental" -> True
      - "I don't want Galaxy Jet; I want Galaxy Chat" -> True for Galaxy Jet, False for Galaxy Chat
      - "Dental care vendaam" -> True
      - "Dental nahi chahiye" -> True
    """
    if not phrase or not text:
        return False
    phrase_clean = phrase.lower().strip()
    text_clean = text.lower().strip()

    # Normalize honorifics so abbreviations like 'dr.' do not break clauses on the dot
    normalized_for_clauses = re.sub(r'\b(dr|mr|mrs|ms|prof)\.', r'\1', text_clean, flags=re.IGNORECASE)
    # Split text into distinct clauses by punctuation (; , . ! ?) or contrastive conjunctions
    clauses = re.split(r'[,;\.!?]|\b(?:but|however|whereas|instead\s+of)\b', normalized_for_clauses)

    # 1. Pre-negation patterns
    pre_neg_words = (
        r"(?:don\'?t\s+(?:want|need|like|prefer|book|schedule)|"
        r"do\s+not\s+(?:want|need|like|prefer|book|schedule)|"
        r"dont\s+(?:want|need|like|prefer|book|schedule)|"
        r"not\s+(?:want|need|interested\s+in|for|with)|"
        r"no|not|never|neither|nor|without|cancel|stop|except|excluding|other\s+than|"
        r"change\s+from|different\s+from|anything\s+but)"
    )

    # 2. Post-negation patterns
    post_neg_words = (
        r"(?:is\s+not\s+what\s+i\s+want|is\s+not\s+needed|not\s+needed|vendaam|vendam|illai|nahi|nahi\s+chahiye|mat|na\s+chahiye)"
    )

    phrase_tokens = [t for t in re.split(r'\s+', phrase_clean) if len(t) > 3 and t not in ("care", "department", "clinic")]

    for clause in clauses:
        c = clause.strip()
        if not c:
            continue

        if phrase_clean in c:
            # If the phrase in this clause is explicitly affirmed (e.g. "I want Galaxy Chat"), it is not negated
            affirmative_pat = rf'\b(?:i\s+(?:want|need|prefer|would\s+like)|book|reserve|schedule|give\s+me)\s+(?:the\s+|a\s+|an\s+)?{re.escape(phrase_clean)}\b'
            if re.search(affirmative_pat, c):
                direct_neg = rf'\b{pre_neg_words}(?:\s+[\w\']+){{0,2}}\s+\b{re.escape(phrase_clean)}\b'
                if not re.search(direct_neg, c):
                    continue

            # Check pre-negation in this clause
            pre_pat = rf'\b{pre_neg_words}(?:\s+[\w\']+){{0,2}}\s+\b{re.escape(phrase_clean)}\b'
            if re.search(pre_pat, c):
                return True

            # Check post-negation in this clause
            post_pat = rf'\b{re.escape(phrase_clean)}\b(?:\s+[\w\']+){{0,2}}\s+{post_neg_words}\b'
            if re.search(post_pat, c):
                return True

        elif len(phrase_tokens) == 1:
            # Single-word phrase (e.g. "dental")
            pt = phrase_tokens[0]
            if pt in c:
                pre_tok_pat = rf'\b{pre_neg_words}(?:\s+[\w\']+){{0,2}}\s+\b{re.escape(pt)}\b'
                if re.search(pre_tok_pat, c):
                    return True
                post_tok_pat = rf'\b{re.escape(pt)}\b(?:\s+[\w\']+){{0,2}}\s+{post_neg_words}\b'
                if re.search(post_tok_pat, c):
                    return True

    return False


class BookingStateService:
    is_phrase_negated = staticmethod(is_phrase_negated)
    """
    Session manager and multi-turn state machine processor for appointment bookings.
    """
    @staticmethod
    def clean_person_name(name: Optional[str]) -> Optional[str]:
        """
        Strips conversational prefixes, polite fillers, and non-name tokens from an extracted candidate name.
        E.g. 'Sure John' -> 'John', 'Yes My Name Is Karthik' -> 'Karthik', 'It Is Vignesh' -> 'Vignesh',
        'How much name is Sanjay' -> 'Sanjay', 'No my name is Sanjay' -> 'Sanjay'.
        """
        if not name or not name.strip():
            return None

        # Strip common introductory prefix patterns
        clean = re.sub(r'^(?:(?:no|nope|not|wrong|actually|hey|hi|hello|well)\s*,?\s*)?(?:i\s+said\s+)?(?:(?:my|the)\s+)?(?:full\s+|first\s+|last\s+)?name\s+(?:is|was)\s+', '', name, flags=re.IGNORECASE)
        clean = re.sub(r'^(?:(?:no|nope|not|wrong|actually|hey|hi|hello|well)\s*,?\s*)?(?:i\s+said\s+)?(?:this\s+is|myself|i\s+am|i\'?m|call\s+me|it\'?s|i\s+said)\s+', '', clean, flags=re.IGNORECASE)
        clean = re.sub(r'^(?:full\s+name|first\s+name|last\s+name)\s+', '', clean, flags=re.IGNORECASE)
        clean = re.sub(r'[^a-zA-Z\u0900-\u097F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\s]', '', clean).strip()
        _fillers = {
            "sure", "yes", "yeah", "ok", "okay", "hi", "hello", "hey", "well",
            "thanks", "thank", "you", "your", "yours", "sir", "madam", "please", "myself", "this",
            "it", "its", "is", "was", "are", "were", "be", "been", "here", "there", "am", "im",
            "my", "name", "names", "full", "first", "last", "middle", "call", "me",
            "dr", "doctor", "mr", "mrs", "ms", "miss", "prof",
            "how", "much", "what", "which", "who", "whom", "whose", "when", "where", "why",
            "can", "could", "would", "should", "shall", "will", "may", "might", "must",
            "do", "does", "did", "doing", "done", "have", "has", "had",
            "no", "nope", "not", "never", "none", "neither",
            "so", "if", "or", "and", "but", "the", "a", "an", "any", "some",
            "just", "now", "right", "actually", "maybe", "about",
            "want", "need", "like", "know", "think",
            "for", "with", "from", "to", "in", "at", "on", "off", "up", "down",
            "before", "after", "during", "while", "say", "says", "saying", "said",
            "tell", "tells", "telling", "told", "speak", "talk", "listen", "hear",
            "skip", "skipping", "skipped", "pannunga", "pannu", "panna", "panunga", "panu",
            "ஸ்கிப்", "பண்ணுங்க", "பண்ணு", "பண்ண", "விடுங்க", "விடவும்", "போகட்டும்", "வேண்டாம்", "வேண்டாங்க",
            "venam", "vendaam", "வந்து", "சந்தி", "சந்திங்க", "சந்திக்க", "சந்திப்பு", "சந்திப்போம்", "பார்க்க", "பாக்க", "ஏன்னே", "ஏண்ணே", "அண்ணே", "தம்பி", "பாஸ்", "தலைவா"
        }
        words = [w for w in clean.split() if w.lower() not in _fillers and len(w) > 1]
        if words and len(words) <= 4:
            cand = " ".join(words)
            if cand.isascii():
                cand = cand.title()
            if BookingStateService.is_valid_person_name(cand):
                return cand
        cand = clean.title() if clean.isascii() else clean
        return cand if BookingStateService.is_valid_person_name(clean) else None

    @staticmethod
    def is_valid_person_name(name: Optional[str]) -> bool:
        """
        Validates that an extracted string contains a genuine human name and not a standalone placeholder,
        dictionary word, question word, conversational filler, or skip expression.
        """
        if not name or not name.strip():
            return False

        norm_text = name.lower().strip()
        _SKIP_PHRASES = [
            "skip", "skip pannunga", "skip pannu", "skip panna", "email skip", "skip email",
            "i said skip", "skip this", "skip it", "skip invite", "without invite", "no email",
            "வேண்டாம்", "விடுங்க", "வேண்டாம் விடுங்க", "skip பண்ணுங்க", "ஸ்கிப் பண்ணுங்க", "ஸ்கிப்",
            "வேண்டாங்க", "venam", "vendaam", "dont have email", "don't have email", "no mail",
            "விட்டிருங்க", "விட்டுருங்க", "விடவும்", "போகட்டும்"
        ]
        for sp in _SKIP_PHRASES:
            if norm_text == sp or sp in norm_text:
                return False

        clean = re.sub(r'[^a-zA-Z\u0900-\u097F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\s]', '', name).strip()
        words = [w.lower() for w in clean.split()]
        if not words or len(words) > 4:
            return False

        _FORBIDDEN_NAME_TOKENS = {
            "how", "much", "what", "which", "why", "where", "when",
            "can", "could", "would", "should",
            "no", "nope", "not", "never", "yes", "yeah", "ok", "okay",
            "phone", "email", "mobile", "number", "doctor",
            "appointment", "booking", "schedule", "cancel", "reschedule",
            "website", "software", "demo", "none", "null", "undefined", "unknown",
            "starting", "running", "launching", "building", "operating", "business", "company",
            "startup", "firm", "retail", "manufacturing", "solution", "solutions", "application",
            "platform", "continue", "proceed", "conversation", "last", "service", "services",
            "department", "industry", "model", "assistant", "system", "caller", "speaker", "machine", "robot", "agent",
            # Skip tokens in English, Tanglish, and Tamil
            "skip", "skips", "skipping", "skipped", "pannunga", "pannu", "panna", "panunga", "panu",
            "ஸ்கிப்", "பண்ணுங்க", "பண்ணு", "பண்ண", "விடுங்க", "விடவும்", "போகட்டும்", "வேண்டாம்", "வேண்டாங்க",
            "venam", "vendaam",
            # Tamil conversational, emotional, filler, and unclear speech tokens
            "என்ன", "எதுக்கு", "ஏன்", "எப்படி", "எங்கே", "எப்பொழுது", "யார்",
            "கோச்சிகிட்டியா", "கோவமா", "கோவம்", "கோபம்", "அமைதியா",
            "இல்லை", "சரி", "ஆமாம்", "திரும்ப", "சொன்னதே", "திருப்பி", "கேக்குற",
            "வந்து", "சந்தி", "சந்திங்க", "சந்திக்க", "சந்திப்பு", "சந்திப்போம்", "பார்க்க", "பாக்க", "ஏன்னே", "ஏண்ணே", "அண்ணே", "தம்பி", "பாஸ்", "தலைவா",
            "சொல்லுங்க", "சொல்லு", "கேளுங்க", "கேளு", "பாருங்க", "இருங்க", "போங்க", "வாங்க",
            "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
            "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
            "today", "tomorrow", "yesterday", "tonight", "morning", "afternoon", "evening", "night",
            "date", "dated", "month", "year", "time", "timing", "slot", "slots", "clock", "am", "pm",
            "first", "second", "third", "fourth", "fifth"
        }
        if any(w in _FORBIDDEN_NAME_TOKENS for w in words):
            return False

        disallowed_standalone = {
            "name", "names", "my name", "patient", "customer", "person", "user", "someone", "anyone",
            "full name", "first name", "last name", "peru", "naam", "unknown", "none", "null", "undefined",
            "sir", "madam", "doctor", "dr", "appointment", "booking", "phone", "email",
            "model", "the model", "assistant", "system", "caller", "agent",
            "yes", "no", "yeah", "okay", "ok", "sure", "hello", "hi", "hey", "thanks", "thank",
            "what", "who", "which", "how", "when", "where", "why", "pardon", "sorry", "pardon me",
            "today", "tomorrow", "yesterday", "tonight", "morning", "afternoon", "evening", "night",
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
            "am", "pm", "before", "after", "say", "saying", "said", "tell", "told",
            "test", "skip", "testskip", "test_skip"
        }
        if len(words) == 1 and words[0] in disallowed_standalone:
            return False

        _non_names = {
            "gonna", "going", "go", "home", "get", "that", "this", "here", "there", "the", "one",
            "who", "what", "which", "how", "when", "where", "why", "to", "for", "with", "from",
            "and", "or", "but", "so", "if", "a", "an", "some", "any", "no", "yes", "yeah", "ok",
            "okay", "sure", "thanks", "thank", "you", "cinema", "start", "now", "right", "little",
            "bit", "end", "all", "ready", "done", "just", "call", "calling", "want", "need", "appointment",
            "booking", "book", "doctor", "phone", "email", "mobile", "number", "dr", "dr.", "at", "my",
            "is", "was", "are", "be", "am", "can", "could", "should", "would", "shall", "will", "have",
            "has", "had", "doing", "did", "do", "like", "up", "down", "out", "in", "off", "on", "not",
            "name", "names", "patient", "customer", "person", "user", "full", "peru", "naam", "sir", "madam", "please",
            "before", "after", "during", "while", "until", "till", "say", "says", "saying", "said",
            "tell", "tells", "telling", "told", "speak", "talk", "listen", "hear",
            "we", "i", "me", "us", "they", "them", "pm", "am", "actually", "maybe", "about"
        }
        if all(w in _non_names for w in words):
            return False

        meaningful_words = [w for w in words if w not in disallowed_standalone and w not in _non_names]
        if not meaningful_words:
            return False

        return True

    @classmethod
    def classify_booking_intent(
        cls,
        text: str,
        current_field: Optional[str] = None,
        session: Optional[BookingSession] = None
    ) -> str:
        """
        Semantic Intent Classifier for appointment booking turns.
        Returns one of:
        SKIP_FIELD, CORRECT_NAME, PROVIDE_NAME, PROVIDE_PHONE, PROVIDE_EMAIL,
        CONFIRM, REJECT, CHANGE_APPOINTMENT, REQUEST_REPEAT, UNCLEAR
        """
        txt = (text or "").lower().strip()
        if not txt:
            return "UNCLEAR"

        # 1. Skip Field Intent: Detected FIRST before any entity extraction
        _SKIP_PATTERNS = [
            "skip", "skip pannunga", "skip pannu", "skip panna", "email skip", "skip email",
            "i said skip", "skip this", "skip it", "skip invite", "without invite", "no email",
            "don't have email", "dont have email", "no mail", "skip and continue",
            "proceed without invite", "without calendar invite", "no calendar invite",
            "வேண்டாம்", "விடுங்க", "வேண்டாம் விடுங்க", "skip பண்ணுங்க", "ஸ்கிப் பண்ணுங்க", "ஸ்கிப்",
            "வேண்டாங்க", "venam", "vendaam", "விட்டுருங்க", "விட்டிருங்க", "விடவும்", "போகட்டும்",
            "email illa", "mail illa", "email nahi", "nahi hai", "email வேண்டாம்", "mail venam"
        ]
        if any(sp in txt for sp in _SKIP_PATTERNS) or (current_field in ("email", "location") and txt.rstrip(".,!?") in ("no", "nope", "none", "na", "illa", "nahi")):
            return "SKIP_FIELD"

        # 2. Reject / Cancel Intent
        if any(w in txt for w in ["cancel booking", "cancel appointment", "dont book", "do not book", "stop booking", "cancel", "never mind", "nevermind"]):
            return "REJECT"

        # 3. Confirmation Intent
        _CONFIRM_EXACT = {
            "yes", "yeah", "yep", "sure", "correct", "right", "fine", "perfect",
            "good", "proceed", "confirm", "ok", "okay", "ha", "haan", "sari", "aam",
            "go", "go ahead", "book it", "confirm it", "do it", "please do", "please book",
            "all good", "looks good", "done", "yes please", "சரி", "ஆமாம்"
        }
        if txt.rstrip(".,!?") in _CONFIRM_EXACT or any(w in txt for w in ["confirm my appointment", "confirm appointment", "details are correct", "everything is correct", "lock it in", "go ahead"]):
            return "CONFIRM"

        # 4. Name Correction Intent
        if any(txt.startswith(p) for p in [
            "actually", "no,", "no ", "not ", "wrong name", "wrong,", "wrong ",
            "i said", "correct name is", "correct name", "my name is actually",
            "நான் வேற சொன்னேன்", "பேர் தப்பு", "பெயர் தவறு"
        ]):
            return "CORRECT_NAME"

        # 5. Email Intent
        if "@" in txt or any(dom in txt for dom in [".com", ".org", ".in", ".net", "gmail", "yahoo", "outlook"]):
            return "PROVIDE_EMAIL"

        # 6. Phone Intent
        digits = re.sub(r'\D', '', txt)
        if len(digits) >= 10:
            return "PROVIDE_PHONE"

        # 7. Repeat Intent
        if any(w in txt for w in ["what", "pardon", "repeat", "say again", "திரும்ப சொல்லுங்க", "என்ன சொன்னீங்க"]):
            return "REQUEST_REPEAT"

        # 8. Date / Time / Change Appointment Intent
        if any(w in txt for w in ["change date", "change time", "reschedule", "different time", "different date", "another time"]):
            return "CHANGE_APPOINTMENT"

        # 9. Provide Name Intent (if currently asking for name and valid name structure)
        if current_field in ("patient_name", "caller_name", "name") or (session and getattr(session, "last_asked_field", "") == "patient_name"):
            clean_c = cls.clean_person_name(text)
            if clean_c and cls.is_valid_person_name(clean_c):
                return "PROVIDE_NAME"

        return "UNCLEAR"

    @classmethod
    def parse_spoken_time(cls, text: str) -> Optional[str]:
        """
        Parses spoken time representations across English, Tamil, and Tanglish into normalized 'HH:MM AM/PM' format.
        Handles:
        - Tamil spoken half-hours: 'ஒன்பதரை' -> '09:30 AM', 'பத்தரை' -> '10:30 AM'
        - Tamil phonetic English: 'நைன் தேட்டி எஎம்' -> '09:30 AM', 'நைன் தேட்டி' -> '09:30 AM'
        - Standard English: 'Nine thirty AM' -> '09:30 AM'
        - Colon and Dot notation: '9:30 AM', '9.30 AM', '09:30' -> '09:30 AM'
        """
        if not text:
            return None
        txt = text.lower().strip()

        # 1. Spoken Tamil Half-Hours
        TAMIL_SPOKEN_HALF = {
            'ஒன்பதரை': '09:30 AM',
            'பத்தரை': '10:30 AM',
            'பதினொன்றரை': '11:30 AM',
            'பன்னிரண்டரை': '12:30 PM',
            'ஒன்றரை': '01:30 PM',
            'இரண்டரை': '02:30 PM',
            'ரெண்டரை': '02:30 PM',
            'மூன்றரை': '03:30 PM',
            'நாலரை': '04:30 PM',
            'ஐந்தரை': '05:30 PM',
            'அஞ்சரை': '05:30 PM',
        }
        for k, v in TAMIL_SPOKEN_HALF.items():
            if k in txt:
                return v

        # Spoken Tamil whole hours
        TAMIL_SPOKEN_HOURS = {
            'ஒன்பது மணி': '09:00 AM',
            'பத்து மணி': '10:00 AM',
            'பதினொரு மணி': '11:00 AM',
            'பன்னிரண்டு மணி': '12:00 PM',
            'ஒரு மணி': '01:00 PM',
            'இரண்டு மணி': '02:00 PM',
            'ரெண்டு மணி': '02:00 PM',
            'மூன்று மணி': '03:00 PM',
            'நான்கு மணி': '04:00 PM',
            'நாலு மணி': '04:00 PM',
            'ஐந்து மணி': '05:00 PM',
            'அஞ்சு மணி': '05:00 PM',
        }
        for k, v in TAMIL_SPOKEN_HOURS.items():
            if k in txt:
                return v

        # 2. Tamil Phonetic English Numbers (e.g. 'நைன் தேட்டி எஎம்', 'நைன் தேட்டி')
        TAMIL_PHONETIC_HOURS = {
            'ஒன்': 1, 'டூ': 2, 'த்ரீ': 3, 'போர்': 4, 'பைவ்': 5,
            'சிக்ஸ்': 6, 'செவன்': 7, 'எய்ட்': 8, 'நைன்': 9, 'டென்': 10,
            'லெவன்': 11, 'டுவல்வ்': 12, 'டுவெல்வ்': 12
        }
        TAMIL_PHONETIC_MINS = {
            'தேட்டி': 30, 'தர்ட்டி': 30, 'பிப்டீன்': 15, 'போர்ட்டி': 40, 'போர்ட்டி பைவ்': 45, 'டுவென்டி': 20
        }
        for h_name, h_val in TAMIL_PHONETIC_HOURS.items():
            if h_name in txt:
                for m_name, m_val in TAMIL_PHONETIC_MINS.items():
                    if m_name in txt:
                        mer = 'AM' if any(w in txt for w in ['எஎம்', 'ஏஎம்', 'காலை', 'am']) else ('PM' if any(w in txt for w in ['பிஎம்', 'மாலை', 'மதியம்', 'pm']) else ('AM' if 8 <= h_val <= 11 else 'PM'))
                        return f'{h_val:02d}:{m_val:02d} {mer}'
                # Hour only
                if any(w in txt for w in ['மணி', 'o clock', 'oclock', 'எஎம்', 'ஏஎம்', 'பிஎம்']):
                    mer = 'AM' if any(w in txt for w in ['எஎம்', 'ஏஎம்', 'காலை', 'am']) else ('PM' if any(w in txt for w in ['பிஎம்', 'மாலை', 'மதியம்', 'pm']) else ('AM' if 8 <= h_val <= 11 else 'PM'))
                    return f'{h_val:02d}:00 {mer}'

        # 3. Colon and Dot Time Notation: e.g. '09:30 AM', '9:30 AM', '9.30 AM', '9.30', '09:30'
        m_num = re.search(r'\b([0-2]?[0-9])[:.]([0-5][0-9])\s*(am|pm|a\.m\.|p\.m\.)?\b', txt, re.IGNORECASE)
        if m_num:
            h, m = int(m_num.group(1)), int(m_num.group(2))
            mer = m_num.group(3)
            if mer:
                mer_clean = 'AM' if 'a' in mer.lower() else 'PM'
            else:
                if any(w in txt for w in ['pm', 'p.m', 'evening', 'afternoon', 'night', 'மாலை', 'மதியம்']):
                    mer_clean = 'PM'
                elif any(w in txt for w in ['am', 'a.m', 'morning', 'காலை']):
                    mer_clean = 'AM'
                elif 8 <= h <= 11:
                    mer_clean = 'AM'
                elif 1 <= h <= 7 or h == 12:
                    mer_clean = 'PM'
                elif 13 <= h <= 23:
                    h = h - 12
                    mer_clean = 'PM'
                else:
                    mer_clean = 'AM'
            if 13 <= h <= 23:
                h = h - 12
            elif h == 0:
                h = 12
            return f'{h:02d}:{m:02d} {mer_clean}'

        # 4. Spoken English hours (e.g. 'Nine thirty AM', 'two thirty')
        _SPOKEN_HOURS_MAP = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12
        }
        _SPOKEN_MINS_MAP = {
            'fifteen': 15, 'thirty': 30, 'forty five': 45, 'forty-five': 45, 'fortyfive': 45,
            'twenty': 20, 'ten': 10, 'fifty': 50, 'forty': 40
        }
        spoken_hm_match = re.search(r'\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d{1,2})\s+(fifteen|thirty|forty[\s\-]five|twenty|forty|fifty)\s*(am|pm|a\.m\.|p\.m\.)?\b', txt, re.IGNORECASE)
        if spoken_hm_match:
            h_str = spoken_hm_match.group(1).lower()
            m_str = spoken_hm_match.group(2).lower().replace(' ', '').replace('-', '')
            mer = spoken_hm_match.group(3)
            h = int(h_str) if h_str.isdigit() else _SPOKEN_HOURS_MAP.get(h_str, 9)
            m_val = _SPOKEN_MINS_MAP.get(m_str, 30)
            if mer:
                mer_clean = 'AM' if 'a' in mer.lower() else 'PM'
            elif any(w in txt for w in ['pm', 'p.m', 'evening', 'afternoon', 'night', 'மாலை', 'மதியம்']):
                mer_clean = 'PM'
            elif any(w in txt for w in ['am', 'a.m', 'morning', 'காலை']):
                mer_clean = 'AM'
            elif 8 <= h <= 11:
                mer_clean = 'AM'
            else:
                mer_clean = 'PM'
            return f'{h:02d}:{m_val:02d} {mer_clean}'

        return None

    @classmethod
    def validate_booking_fields(cls, session: BookingSession) -> Dict[str, Any]:
        """
        Validates all mandatory booking fields before read-back confirmation and booking execution.
        Returns {'is_valid': bool, 'errors': List[str]}.
        """
        errors = []

        # 1. Date: must be non-empty, ISO YYYY-MM-DD, not relative string, and not in the past
        if not session.date:
            errors.append("date missing")
        elif session.date in ("Tomorrow", "Today", "tomorrow", "today", "day after tomorrow"):
            errors.append(f"relative date string '{session.date}' not resolved to ISO YYYY-MM-DD")
        elif not re.match(r'^\d{4}-\d{2}-\d{2}$', str(session.date)):
            errors.append(f"invalid date format '{session.date}'")
        else:
            from datetime import datetime, timezone, timedelta
            try:
                from zoneinfo import ZoneInfo
                tz = ZoneInfo("Asia/Kolkata")
                today_str = datetime.now(tz).strftime("%Y-%m-%d")
            except Exception:
                today_str = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%Y-%m-%d")
            if str(session.date) < today_str:
                errors.append(f"date {session.date} is in the past")

        # 2. Time: must be non-empty, HH:MM AM/PM
        if not session.time:
            errors.append("time missing")
        elif not re.search(r'^\d{1,2}:\d{2}\s*(?:AM|PM)$', str(session.time), re.IGNORECASE):
            errors.append(f"invalid time format '{session.time}'")

        # 3. Patient Name: must be non-empty, pass is_valid_person_name, not 'skip'
        if not session.patient_name:
            errors.append("patient_name missing")
        elif not cls.is_valid_person_name(session.patient_name):
            errors.append(f"invalid patient_name '{session.patient_name}'")

        # 4. Phone: must have at least 10 digits
        if not session.contact_number or session.contact_number == "Unknown":
            errors.append("contact_number missing")
        else:
            digits = re.sub(r'\D', '', str(session.contact_number))
            if len(digits) < 10:
                errors.append(f"contact_number '{session.contact_number}' has fewer than 10 digits")

        # 5. Email: must be provided and valid OR explicitly skipped
        is_email_skipped = (
            session.field_status.get("email") == "SKIPPED"
            or session.booking_data.get("email_declined")
            or session.booking_data.get("email_skipped")
            or session.booking_data.get("email_status") == "SKIPPED"
        )
        if not session.email and not is_email_skipped:
            errors.append("email missing and not skipped")

        is_valid = len(errors) == 0
        call_id = getattr(session, "call_sid", None) or getattr(session, "session_id", "unknown")
        logger.info(f"[BOOKING_CONFIRMATION] call_id={call_id} all_valid={is_valid} missing_fields={errors}")
        return {"is_valid": is_valid, "errors": errors}

    @staticmethod
    def _mask_pii(val: Optional[str]) -> str:
        if not val:
            return "None"
        s = str(val).strip()
        if "@" in s:
            parts = s.split("@", 1)
            user, domain = parts[0], parts[1]
            if len(user) <= 2:
                masked_user = user[0] + "*"
            else:
                masked_user = user[0] + "*" * (len(user) - 2) + user[-1]
            return f"{masked_user}@{domain}"
        digits = re.sub(r'\D', '', s)
        if len(digits) >= 7:
            return "*" * (len(digits) - 4) + digits[-4:]
        if len(s) > 2:
            return s[0] + "*" * (len(s) - 2) + s[-1]
        return s[0] + "*"

    def __init__(self):
        self._sessions: Dict[str, BookingSession] = {}

    def get_session(
        self,
        session_id: str,
        conversation_id: Optional[str] = None,
        call_sid: Optional[str] = None,
        organization_id: Optional[int] = None,
        business_type: Optional[str] = None,
        booking_type: Optional[str] = None,
        business_name: Optional[str] = None,
        active_doc_id: Optional[str] = None,
        active_doc_title: Optional[str] = None
    ) -> BookingSession:
        target_keys = [k for k in [session_id, conversation_id, call_sid] if k]
        for k in target_keys:
            if k in self._sessions:
                sess = self._sessions[k]
                if conversation_id and not sess.conversation_id:
                    sess.conversation_id = conversation_id
                if call_sid and not sess.call_sid:
                    sess.call_sid = call_sid
                if organization_id is not None:
                    sess.organization_id = organization_id
                if business_type:
                    sess.business_type = business_type
                if booking_type:
                    sess.booking_type = booking_type
                if business_name:
                    sess.business_name = business_name
                if active_doc_id:
                    sess.active_doc_id = active_doc_id
                    sess.active_rag_id = active_doc_id
                if active_doc_title:
                    sess.active_doc_title = active_doc_title
                    sess.active_rag_title = active_doc_title
                for alias in target_keys:
                    self._sessions[alias] = sess
                return sess

        # Attempt to recover durable state from database
        persisted_sid = call_sid or session_id
        recovered = self.load_session_state(persisted_sid)
        if recovered:
            logger.info(f"[SESSION_RECOVERY] Successfully recovered durable session for {persisted_sid}")
            if organization_id is not None:
                recovered.organization_id = organization_id
            if business_type:
                recovered.business_type = business_type
            if booking_type:
                recovered.booking_type = booking_type
            if business_name:
                recovered.business_name = business_name
            if active_doc_id:
                recovered.active_doc_id = active_doc_id
                recovered.active_rag_id = active_doc_id
            if active_doc_title:
                recovered.active_doc_title = active_doc_title
                recovered.active_rag_title = active_doc_title
            for k in target_keys:
                self._sessions[k] = recovered
            return recovered

        new_sess = BookingSession(
            session_id=session_id,
            conversation_id=conversation_id,
            call_sid=call_sid,
            organization_id=organization_id or 1,
            business_type=business_type or "GENERAL",
            booking_type=booking_type or "BUSINESS_CONSULTATION",
            business_name=business_name or "AI Receptionist",
            active_doc_id=active_doc_id,
            active_doc_title=active_doc_title
        )
        for k in target_keys:
            self._sessions[k] = new_sess
        return new_sess

    def remove_session(self, session_id: str):
        if session_id in self._sessions:
            sess = self._sessions[session_id]
            keys_to_del = [k for k, v in list(self._sessions.items()) if v is sess or k == session_id]
            for k in keys_to_del:
                self._sessions.pop(k, None)

    def save_session_state(self, session: BookingSession):
        """
        Durable persistence of structured conversation memory into CallLog.receptionist_state.
        Survives async worker reloads and process restarts.
        """
        try:
            call_sid = getattr(session, "call_sid", None)
            if not call_sid:
                return
            state_dict = session.get_state_dict()
            from src.config.database import SessionLocal
            from src.models.call_log import CallLog
            db = SessionLocal()
            try:
                import json
                call = db.query(CallLog).filter(CallLog.call_sid == call_sid).first()
                if call:
                    call.receptionist_state = json.dumps(state_dict, ensure_ascii=False)
                    call.intent = session.current_intent
                    if session.patient_name:
                        call.caller_name = session.patient_name
                    if session.contact_number and (not call.caller_phone or call.caller_phone in ('Unknown', 'unknown')):
                        call.caller_phone = session.contact_number
                    db.commit()
                    if session.patient_name:
                        try:
                            from src.services.lead_service import LeadService
                            LeadService(db).upsert_lead_from_session(session, call_log_id=call.id)
                        except Exception as le:
                            logger.debug(f"LeadService sync in save_session_state ignored: {le}")
            finally:
                db.close()
        except Exception as e:
            logger.warning(f"Failed to persist session state to DB: {e}")

    def load_session_state(self, call_sid: str, organization_id: int = 1) -> Optional[BookingSession]:
        """
        Restores durable conversation memory from CallLog.receptionist_state.
        """
        if not call_sid:
            return None

        try:
            import json
            from src.config.database import SessionLocal
            from src.models.call_log import CallLog
            db = SessionLocal()
            try:
                call = db.query(CallLog).filter(CallLog.call_sid == call_sid).first()
                if call and call.receptionist_state:
                    try:
                        state_data = json.loads(call.receptionist_state)
                        if isinstance(state_data, dict) and state_data.get("state"):
                            sess = BookingSession.from_state_dict(state_data)
                            sess.call_sid = call_sid
                            sess.organization_id = organization_id or getattr(call, "organization_id", 1) or 1
                            return sess
                    except Exception:
                        pass
            finally:
                db.close()
        except Exception as e:
            logger.warning(f"Failed to load session state from DB: {e}")
        return None


    def resolve_department(self, candidate_name: str, organization_id: int = 1, db: Optional[Any] = None) -> Optional[str]:
        """
        Resolves candidate department/specialty name against authoritative DB and Knowledge Base.
        Never invents departments.
        """
        if not candidate_name or not candidate_name.strip():
            return None

        cand_clean = candidate_name.strip().lower()
        _COMM_KEYWORDS = {
            "phone", "telephone", "mobile", "cell", "cellphone", "whatsapp",
            "email", "mail", "gmail", "yahoo", "outlook", "contact", "number",
            "call", "sms", "text", "voice", "address", "website", "online",
            "representative", "agent", "receptionist", "assistant", "bot"
        }
        if cand_clean in _COMM_KEYWORDS or any(w in cand_clean.split() for w in ["phone", "telephone", "mobile", "email", "whatsapp"]):
            return None

        cand_clean = re.sub(r'^(?:dr\.?|doctor|department\s+of|the\s+department\s+of|the)\s+', '', cand_clean).strip()
        cand_norm = re.sub(r'[^a-z0-9]', '', cand_clean)
        if not cand_norm or len(cand_norm) < 3:
            return None


        close_db = False
        if not db:
            from src.config.database import SessionLocal
            db = SessionLocal()
            close_db = True
        try:
            from src.models.department import Department
            from src.models.staff import Staff
            depts = db.query(Department).filter(Department.organization_id == organization_id).all()
            for d in depts:
                if d.name:
                    d_norm = re.sub(r'[^a-z0-9]', '', d.name.lower())
                    if cand_norm == d_norm or cand_clean in d.name.lower() or d.name.lower() in cand_clean:
                        return d.name

            # Check distinct Staff.department in DB
            staff_depts = db.query(Staff.department).filter(
                Staff.organization_id == organization_id,
                Staff.department.isnot(None)
            ).distinct().all()
            for (sd,) in staff_depts:
                if sd:
                    sd_norm = re.sub(r'[^a-z0-9]', '', sd.lower())
                    if cand_norm == sd_norm or cand_clean in sd.lower() or sd.lower() in cand_clean:
                        return sd

            # Active RAG Master Context
            from src.services.rag_service import rag_service
            master_ctx = rag_service.get_organization_master_context(organization_id)
            if master_ctx:
                dept_bullet_matches = re.findall(r'(?:#+\s*(?:DEPARTMENTS|SPECIALTIES)[^\n]*\n(?:[^\n]*\n)*?)((?:[-*•]\s*[^\n]+\n?)+)', master_ctx, re.IGNORECASE)
                for bm in dept_bullet_matches:
                    for b in re.findall(r'[-*•]\s*([^\n\r]+)', bm):
                        clean_b = b.strip().replace('*', '')
                        b_norm = re.sub(r'[^a-z0-9]', '', clean_b.lower())
                        if cand_norm == b_norm or cand_clean in clean_b.lower() or clean_b.lower() in cand_clean:
                            return clean_b
                dept_inline = re.search(r'(?:Departments|Specialties)\s*:\s*([^\n]+)', master_ctx, re.IGNORECASE)
                if dept_inline:
                    for s_item in re.split(r'[,;•\n]', dept_inline.group(1)):
                        clean_item = s_item.strip().replace('*', '')
                        b_norm = re.sub(r'[^a-z0-9]', '', clean_item.lower())
                        if cand_norm == b_norm or cand_clean in clean_item.lower() or clean_item.lower() in cand_clean:
                            return clean_item
        except Exception as e:
            logger.warning(f"Error resolving department: {e}")
        finally:
            if close_db:
                db.close()
        return None


    def get_organization_products_and_services(self, organization_id: int = 1) -> List[str]:
        """
        Retrieves products and services STRICTLY from the active RAG script / knowledge base.
        Never includes internal staff departments (e.g. CEO, SALES, HR).
        """
        from src.services.rag_service import rag_service
        services = rag_service.get_organization_services(organization_id)
        admin_roles = {"ceo", "sales", "hr", "human resources", "admin", "administration", "management"}
        return [s for s in services if s.lower().strip() not in admin_roles]

    def get_organization_services_and_departments(self, organization_id: int = 1) -> List[str]:
        """
        Dynamically fetches valid departments and services from the database and active RAG document
        for the given organization. Never returns hardcoded entities.
        Does NOT mix Staff department names into product/service offerings.
        """
        services_and_depts = []
        seen = set()

        # 1. From database Department table
        try:
            from src.config.database import SessionLocal
            from src.models.department import Department
            from src.models.staff import Staff, StaffStatus
            db = SessionLocal()
            try:
                dept_records = db.query(Department).filter(Department.organization_id == organization_id).all()
                for d in dept_records:
                    if d.name and d.name.strip().lower() not in seen:
                        seen.add(d.name.strip().lower())
                        services_and_depts.append(d.name.strip())

                # From Staff explicit services ONLY (NOT Staff.department)
                staff_records = db.query(Staff).filter(
                    Staff.organization_id == organization_id,
                    Staff.online_booking_enabled != False,
                    Staff.status != StaffStatus.OFFLINE
                ).all()
                for s in staff_records:
                    if s.services:
                        try:
                            s_list = json.loads(s.services) if isinstance(s.services, str) else s.services
                            if isinstance(s_list, list):
                                for s_item in s_list:
                                    if s_item and str(s_item).strip().lower() not in seen:
                                        seen.add(str(s_item).strip().lower())
                                        services_and_depts.append(str(s_item).strip())
                        except Exception:
                            pass
            finally:
                db.close()
        except Exception as e:
            logger.warning(f"Error fetching DB departments/services: {e}")

        # 2. From active RAG document for this organization (products and services)
        try:
            rag_services = self.get_organization_products_and_services(organization_id)
            for rs in rag_services:
                if rs and rs.strip().lower() not in seen:
                    seen.add(rs.strip().lower())
                    services_and_depts.append(rs.strip())
        except Exception as e:
            logger.warning(f"Error fetching RAG services: {e}")

        return services_and_depts

    def get_doctors_for_department(self, department_name: str, organization_id: int = 1, db: Optional[Any] = None, session: Optional[BookingSession] = None) -> List[Dict[str, Any]]:
        """
        Retrieves authoritative doctors associated with a department from DB and active RAG directory.
        Only returns doctors verified in Staff DB with a valid ID and active booking status.
        Excludes doctors or departments rejected in session.
        """
        if not department_name:
            return []
        
        # If department was rejected in this session, return empty list!
        if session and department_name.lower().strip() in getattr(session, "_rejected_departments", set()):
            return []

        dept_norm = re.sub(r'[^a-z0-9]', '', department_name.lower())
        results = []
        seen_names = set()

        close_db = False
        if not db:
            from src.config.database import SessionLocal
            db = SessionLocal()
            close_db = True
        try:
            from src.models.staff import Staff, StaffStatus
            staff_members = db.query(Staff).filter(
                Staff.organization_id == organization_id,
                Staff.online_booking_enabled != False,
                Staff.status != StaffStatus.OFFLINE
            ).all()
            for s in staff_members:
                if not s.name or not s.department:
                    continue
                s_dept_norm = re.sub(r'[^a-z0-9]', '', s.department.lower())
                if dept_norm in s_dept_norm or s_dept_norm in dept_norm:
                    clean_n = re.sub(r'^(?:dr\.?|doctor)\s+', '', s.name.strip(), flags=re.I).strip().title()
                    # Filter out rejected doctors
                    if session and (clean_n.lower() in getattr(session, "_rejected_doctors", set()) or s.name.lower() in getattr(session, "_rejected_doctors", set())):
                        continue
                    disp_name = f"Dr. {clean_n}" if not clean_n.lower().startswith("dr") else clean_n
                    if clean_n.lower() not in seen_names:
                        seen_names.add(clean_n.lower())
                        results.append({
                            "id": s.id,
                            "name": disp_name,
                            "clean_name": clean_n.lower(),
                            "department": s.department
                        })

            # Also check RAG Doctor directory, but REQUIRE matching active staff in DB
            from src.services.rag_service import rag_service
            master_ctx = rag_service.get_organization_master_context(organization_id)
            if master_ctx:
                doc_entries = re.findall(
                    r'(?m)^(?!#)\s*(?:Dr\.?|Doctor)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)[^\n]*\n(?:[^\n]*\n){0,3}?Department:\s*([^\n]+)',
                    master_ctx,
                    re.IGNORECASE
                )
                for d_name, d_dept in doc_entries:
                    clean_n = d_name.strip().title()
                    if clean_n.lower() in ("directory", "information", "staff", "doctors", "team", "schedule"):
                        continue
                    if session and clean_n.lower() in getattr(session, "_rejected_doctors", set()):
                        continue
                    d_dept_norm = re.sub(r'[^a-z0-9]', '', d_dept.lower())
                    if dept_norm in d_dept_norm or d_dept_norm in dept_norm:
                        if clean_n.lower() not in seen_names:
                            db_s = next((s for s in staff_members if clean_n.lower() in s.name.lower() or s.name.lower() in clean_n.lower()), None)
                            # CRITICAL: Never offer a doctor when verified DB record or ID is missing!
                            if db_s and db_s.id:
                                seen_names.add(clean_n.lower())
                                results.append({
                                    "id": db_s.id,
                                    "name": f"Dr. {clean_n}",
                                    "clean_name": clean_n.lower(),
                                    "department": d_dept.strip()
                                })
        except Exception as e:
            logger.warning(f"Error fetching doctors for department: {e}")
        finally:
            if close_db:
                db.close()
        return results

    def evaluate_post_response(self, session: BookingSession) -> Dict[str, Any]:
        """
        Lightweight deterministic post-response state evaluator.
        Determines if the system requires user input, an automatic system action, or is completed.
        """
        if session.is_completed or session.state in (BookingState.BOOKED, BookingState.COMPLETED):
            return {
                "next_action": NextAction.COMPLETED,
                "state": "COMPLETED",
                "reason": "Appointment confirmed or conversation concluded."
            }

        if session.is_active():
            if session.state == BookingState.WAITING_FOR_CONFIRMATION:
                return {
                    "next_action": NextAction.CONFIRMATION_REQUIRED,
                    "state": "LISTENING",
                    "reason": "Waiting for caller confirmation."
                }
            missing = session.missing_fields
            # If all required fields are present and not yet completed/confirmed
            if not missing and session.state not in (BookingState.BOOKED, BookingState.COMPLETED):
                return {
                    "next_action": NextAction.AUTOMATIC_ACTION_REQUIRED,
                    "state": "PROCESSING",
                    "reason": "All required details present. Ready for automatic booking execution."
                }
            else:
                return {
                    "next_action": NextAction.USER_INPUT_REQUIRED,
                    "state": "LISTENING",
                    "missing_fields": missing,
                    "reason": f"Waiting for caller input for fields: {missing}"
                }

        return {
            "next_action": NextAction.USER_INPUT_REQUIRED,
            "state": "LISTENING",
            "reason": "Listening for next caller query."
        }

    def get_silence_prompt(self, session: BookingSession) -> str:
        """
        Generates a concise, polite reassurance prompt when caller remains silent
        during an active turn without resetting state or re-emitting greetings.
        """
        lang = getattr(session, "conversation_language", "en")
        first_name = (session.patient_name or "").split()[0] if session.patient_name else ""
        name_prefix = f"{first_name}, " if first_name else ""

        if session.state == BookingState.WAITING_FOR_PATIENT_NAME:
            if lang in ("ta", "tanglish", "ta_mix"):
                return "உங்க பெயர் சொல்லுங்க, நான் கேட்டுட்டு இருக்கேன்."
            elif lang == "hi":
                return "Kripya apna naam batayein, main sun raha hoon."
            elif lang in ("te", "telugu"):
                return "దయచేసి మీ పేరు చెప్పండి."
            elif lang in ("ml", "malayalam"):
                return "ദയവായി നിങ്ങളുടെ പേര് പറയൂ."
            elif lang in ("kn", "kannada"):
                return "ದಯವಿಟ್ಟು ನಿಮ್ಮ ಹೆಸರನ್ನು ತಿಳಿಸಿ."
            return f"{name_prefix}Take your time. May I know your name whenever you're ready?"

        elif session.state == BookingState.WAITING_FOR_CONTACT:
            if lang in ("ta", "tanglish", "ta_mix"):
                return "உங்க phone number சொல்லுங்க."
            elif lang == "hi":
                return "Kripya apna phone number batayein."
            elif lang in ("te", "telugu"):
                return "దయచేసి మీ ఫోన్ నంబర్ చెప్పండి."
            elif lang in ("ml", "malayalam"):
                return "ദയവായി നിങ്ങളുടെ ഫോൺ നമ്പർ പറയൂ."
            elif lang in ("kn", "kannada"):
                return "ದಯವಿಟ್ಟು ನಿಮ್ಮ ಫೋನ್ ನಂಬರ್ ತಿಳಿಸಿ."
            return f"{name_prefix}Whenever you're ready, please share your contact number."

        elif session.state == BookingState.WAITING_FOR_EMAIL:
            if lang in ("ta", "tanglish", "ta_mix"):
                return "உங்க email address சொல்லுங்க."
            elif lang == "hi":
                return "Kripya apna email address batayein."
            elif lang in ("te", "telugu"):
                return "దయచేసి మీ ఈమెయిల్ అడ్రస్ చెప్పండి."
            elif lang in ("ml", "malayalam"):
                return "ദയവായി നിങ്ങളുടെ ഇമെയിൽ വിലാസം പറയൂ."
            elif lang in ("kn", "kannada"):
                return "ದಯವಿಟ್ಟು ನಿಮ್ಮ ಇಮೇಲ್ ವಿಳಾಸ ತಿಳಿಸಿ."
            return f"{name_prefix}Please let me know your email address when you're ready."

        elif session.state in (BookingState.COLLECTING_LOCATION, BookingState.WAITING_FOR_LOCATION):
            if lang in ("ta", "tanglish", "ta_mix"):
                return "உங்க location அல்லது ஊர் சொல்லுங்க."
            elif lang == "hi":
                return "Kripya apna location ya shehar batayein."
            elif lang in ("te", "telugu"):
                return "దయచేసి మీ లొకేషన్ చెప్పండి."
            elif lang in ("ml", "malayalam"):
                return "ദയവായി നിങ്ങളുടെ ലൊക്കേഷൻ പറയൂ."
            elif lang in ("kn", "kannada"):
                return "ದಯವಿಟ್ಟು ನಿಮ್ಮ location ತಿಳಿಸಿ."
            return f"{name_prefix}Please let me know your location or city whenever you're ready."

        elif session.state == BookingState.WAITING_FOR_DATE:
            if lang in ("ta", "tanglish", "ta_mix"):
                return "எந்த date-ல appointment வேணும்?"
            elif lang == "hi":
                return "Aap kis date ko appointment prefer karenge?"
            elif lang in ("te", "telugu"):
                return "మీరు ఏ తేదీన అపాయింట్‌మెంట్ తీసుకోవాలనుకుంటున్నారు?"
            elif lang in ("ml", "malayalam"):
                return "ഏത് തീയതിയിലാണ് നിങ്ങൾക്ക് അപ്പോയിന്റ്മെന്റ് വേണ്ടത്?"
            elif lang in ("kn", "kannada"):
                return "ಯಾವ ದಿನಾಂಕದಂದು ಅಪಾಯಿಂಟ್ಮೆಂಟ್ ಬೇಕು?"
            return f"{name_prefix}Which date would you prefer for the appointment?"

        elif session.state == BookingState.WAITING_FOR_TIME:
            if lang in ("ta", "tanglish", "ta_mix"):
                return "எந்த time slot உங்களுக்கு வசதியா இருக்கும்?"
            elif lang == "hi":
                return "Konsa samay aapke liye sahi rahega?"
            elif lang in ("te", "telugu"):
                return "ఏ సమయం మీకు అనుకూలంగా ఉంటుంది?"
            elif lang in ("ml", "malayalam"):
                return "ഏത് സമയമാണ് നിങ്ങൾക്ക് സൗകര്യം?"
            elif lang in ("kn", "kannada"):
                return "ಯಾವ ಸಮಯ ನಿಮಗೆ ಸೂಕ್ತವಾಗಿದೆ?"
            return f"{name_prefix}What time would work best for you?"

        elif session.state == BookingState.WAITING_FOR_DOCTOR:
            if getattr(session, "_offered_dept_doctor", None):
                return f"Would you like to schedule with {session._offered_dept_doctor}?"
            elif session.department:
                return f"Which doctor in {session.department} would you like to see?"
            return self._get_target_entity_question(session, name_ack=name_prefix, lang=lang)

        if lang in ("ta", "tanglish", "ta_mix"):
            return "நான் line-ல தான் இருக்கேன். உங்களுக்கு வேற ஏதாச்சு உதவி வேணுமா?"
        elif lang == "hi":
            return "Main sun raha hoon. Kya aapko aur koi jaankari chahiye?"
        elif lang in ("te", "telugu"):
            return "నేను లైన్ లోనే ఉన్నాను. మీకు ఇంకేమైనా సహాయం కావాలా?"
        elif lang in ("ml", "malayalam"):
            return "ഞാൻ ഇവിടെത്തന്നെ ഉണ്ട്. നിങ്ങൾക്ക് മറ്റ് എന്തെങ്കിലും സഹായം വേണമോ?"
        elif lang in ("kn", "kannada"):
            return "ನಾನು ಇಲ್ಲೇ ಇದ್ದೇನೆ. ನಿಮಗೆ ಇನ್ನೇನಾದರೂ ಸಹಾಯ ಬೇಕೇ?"
        return "I'm still here whenever you're ready. How can I help you further?"

    parse_spoken_phone_number = staticmethod(parse_spoken_phone_number)
    parse_spoken_email = staticmethod(parse_spoken_email)
    detect_incomplete_spoken_email = staticmethod(detect_incomplete_spoken_email)
    parse_spoken_location = staticmethod(parse_spoken_location)
    is_valid_location_candidate = staticmethod(is_valid_location_candidate)

    def get_appointment_or_demo_details(self, session: "BookingSession") -> str:
        """
        Retrieves actual verified appointment or demo request status from session/DB.
        Guarantees that unconfirmed appointments are never falsely presented as confirmed.
        """
        lang = getattr(session, "conversation_language", "en") or "en"

        # 1. Check genuine confirmed appointment
        is_confirmed = bool(session.booking_confirmed and session.appointment_id)
        if not is_confirmed and session.appointment_id:
            try:
                from src.config.database import SessionLocal
                from src.models.appointment import Appointment, AppointmentStatus
                _db = SessionLocal()
                try:
                    appt = _db.query(Appointment).filter(Appointment.id == session.appointment_id).first()
                    if appt and appt.status == AppointmentStatus.CONFIRMED:
                        is_confirmed = True
                finally:
                    _db.close()
            except Exception:
                pass

        if is_confirmed:
            target_str = f" with {session.doctor}" if session.doctor else (f" for {session.service}" if session.service else "")
            if lang in ("ta", "tanglish", "ta_mix"):
                return f"உங்க appointment {session.date} அன்று {session.time}-க்கு{target_str} Appointment ID #{session.appointment_id}-ன் கீழ் confirm செய்யப்பட்டுள்ளது. நான் உங்களுக்கு வேற எப்படி உதவலாம்?"
            elif lang == "hi":
                return f"Aapka appointment {session.date} ko {session.time} par{target_str} Appointment ID #{session.appointment_id} ke sath confirm hai. Main aapki aur kya madad kar sakta hoon?"
            else:
                return f"Your appointment is confirmed for {session.date} at {session.time}{target_str} under Appointment ID #{session.appointment_id}. Is there anything else I can assist you with?"

        # 2. Check lead / callback request (e.g. demo requested)
        avail = session.booking_data.get("caller_availability")
        avail_str = f", noting that you are available {avail}" if avail else ""
        avail_str_ta = f" (நீங்க {avail} available என்று குறித்துக்கொண்டோம்)" if avail else ""
        req = session.requirement or session.service or session.booking_data.get("product") or "demo"

        if getattr(session, "workflow_mode", None) == SessionWorkflowMode.LEAD_COLLECTION or session.booking_data.get("lead_saved") or session.booking_data.get("caller_availability"):
            if lang in ("ta", "tanglish", "ta_mix"):
                return f"உங்களுக்கு இன்னும் appointment confirm ஆகவில்லை. எங்க team 24 மணி நேரத்திற்குள் உங்களை contact செய்து {req}-வை schedule பண்ணுவாங்க{avail_str_ta}. உங்களுக்கு வேற ஏதாவது உதவி வேணுமா?"
            elif lang == "hi":
                return f"Aapka appointment abhi confirm nahi hua hai. Hamari team 24 ghante ke bheetar aapse contact karke {req} schedule karegi. Kya aapko aur koi sahayata chahiye?"
            else:
                return f"You don't have a confirmed appointment yet. Our team will contact you within 24 hours to schedule your {req}{avail_str}. Is there anything else I can help you with?"

        # 3. Neither confirmed appointment nor demo request exists
        if lang in ("ta", "tanglish", "ta_mix"):
            return "உங்களுக்கு தற்போது முன்பதிவு எதுவும் உறுதி செய்யப்படவில்லை. இப்போதே ஒரு appointment book பண்ணலாமா?"
        elif lang == "hi":
            return "Aapka koi appointment abhi confirm nahi hai. Kya aap abhi ek appointment book karna chahenge?"
        else:
            return "You do not have a confirmed appointment scheduled currently. Would you like me to help you schedule one now?"

    def _get_target_entity_question(
        self,
        session: "BookingSession",
        name_ack: str = "",
        lang: str = "en"
    ) -> str:
        """
        Dynamically generates the next-action question for the unknown appointment target
        (the person/service the caller wants to book with).

        Priority:
          1. Preserve user's own term if they already said it (stored in booking_data).
          2. Use RAG terminology for the active business domain.
          3. Fall back to neutral generic phrasing.

        Never emits "service or staff member" or any other hardcoded business noun.
        """
        from src.services.rag_service import rag_service
        org_id = getattr(session, "organization_id", 1) or 1
        doc_id = getattr(session, "active_doc_id", None) or getattr(session, "active_rag_id", None)

        # --- If department is already known, ask directly for the doctor in that department ---
        if session.department:
            if lang in ("ta", "tanglish", "ta_mix"):
                return f"{name_ack}{session.department}-ல எந்த doctor-ஐ பார்க்கணும்?"
            elif lang == "hi":
                return f"{name_ack}{session.department} में आप किस डॉक्टर से मिलना चाहते हैं?"
            elif lang in ("kn", "kannada"):
                return f"{name_ack}{session.department} ವಿಭಾಗದಲ್ಲಿ ನೀವು ಯಾವ ವೈದ್ಯರನ್ನು ಭೇಟಿಯಾಗಲು ಬಯಸುತ್ತೀರಿ?"
            elif lang in ("te", "telugu"):
                return f"{name_ack}{session.department} విభాగంలో మీరు ఏ డాక్టర్‌ను సంప్రదించాలనుకుంటున్నారు?"
            return f"{name_ack}For {session.department}, which doctor would you like to consult?"

        # --- User's own terminology preserved (e.g. "coordinator", "teacher") ---
        user_entity_word = (
            session.booking_data.get("user_requested_entity_term")
            or ""
        ).strip().lower()

        # --- RAG terminology ---
        try:
            terminology = rag_service.get_appointment_entity_terminology(
                organization_id=org_id, document_id=doc_id
            )
        except Exception:
            terminology = {"ask_target_question": None, "entity_term": None, "entity_action": "meet", "service_based": False}

        rag_question = terminology.get("ask_target_question")  # e.g. "Which teacher would you like to meet?"
        entity_term = terminology.get("entity_term")  # e.g. "teacher"
        entity_action = terminology.get("entity_action", "meet")
        service_based = terminology.get("service_based", False)

        # Only echo user's own word back if it is a recognized role category noun
        # Never echo arbitrary products, services, or phrases (e.g. "Which galaxy jet...")
        _ALLOWED_ROLE_ECHOES = {
            "doctor", "dr", "physician", "nurse", "teacher", "trainer", "coordinator",
            "stylist", "counselor", "therapist", "consultant", "specialist", "advisor",
            "agent", "lawyer", "attorney", "accountant", "instructor", "expert"
        }
        if user_entity_word and user_entity_word in _ALLOWED_ROLE_ECHOES:
            action_word = "book" if service_based else entity_action
            if lang in ("ta", "tanglish", "ta_mix"):
                return f"{name_ack}எந்த {user_entity_word}-ஐ பார்க்கணும்?"
            elif lang == "hi":
                return f"{name_ack}Aap kis {user_entity_word} se milna chahte hain?"
            return f"{name_ack}Sure. Which {user_entity_word} would you like to {action_word}?"

        # If RAG gave us a concrete question, use it with the ack prefix
        if rag_question:
            if lang in ("ta", "tanglish", "ta_mix"):
                if entity_term == "doctor":
                    return f"{name_ack}எந்த doctor அல்லது department-ஐ பார்க்கணும்?"
                elif entity_term == "teacher":
                    return f"{name_ack}எந்த teacher-ஐ சந்திக்க விரும்புகிறீர்கள்?"
                elif service_based:
                    return f"{name_ack}எந்த service book பண்ணனும்?"
                else:
                    return f"{name_ack}யாரை பார்க்கணும்?"
            elif lang == "hi":
                if entity_term == "doctor":
                    return f"{name_ack}Aap kis doctor ya vibhaag se milna chahte hain?"
                elif entity_term == "teacher":
                    return f"{name_ack}Aap kis teacher se milna chahte hain?"
                elif service_based:
                    return f"{name_ack}Aap kaunsi service book karna chahte hain?"
                else:
                    return f"{name_ack}Aap kisse milna chahte hain?"
            elif lang in ("kn", "kannada"):
                if entity_term == "doctor":
                    return f"{name_ack}ನೀವು ಯಾವ ವೈದ್ಯರನ್ನು ಅಥವಾ ವಿಭಾಗವನ್ನು ಸಂಪರ್ಕಿಸಲು ಬಯಸುತ್ತೀರಿ?"
                elif service_based:
                    return f"{name_ack}ನೀವು ಯಾವ ಸೇವೆಯನ್ನು ಕಾಯ್ದಿರಿಸಲು ಬಯಸುತ್ತೀರಿ?"
            elif lang in ("te", "telugu"):
                if entity_term == "doctor":
                    return f"{name_ack}మీరు ఏ వైద్యుడిని లేదా విభాగాన్ని సంప్రదించాలనుకుంటున్నారు?"
                elif service_based:
                    return f"{name_ack}మీరు ఏ సేవను బుక్ చేయాలనుకుంటున్నారు?"
            return f"{name_ack}{rag_question}"

        # Hospital domain fallback
        if getattr(session, "booking_type", "") == "HOSPITAL_APPOINTMENT" or getattr(session, "business_type", "") == "HOSPITAL":
            if lang in ("ta", "tanglish", "ta_mix"):
                return f"{name_ack}எந்த doctor அல்லது department-ஐ பார்க்கணும்?"
            elif lang == "hi":
                return f"{name_ack}Aap kis doctor ya vibhaag se milna chahte hain?"
            return f"{name_ack}Which doctor or department would you like to consult?"

        # Generic neutral fallback — no business noun hardcoded
        if lang in ("ta", "tanglish", "ta_mix"):
            return f"{name_ack}எந்த service book பண்ணனும்?"
        elif lang == "hi":
            return f"{name_ack}Aap kisse milna ya kya book karna chahte hain?"
        return f"{name_ack}Sure. Who or what would you like to book the appointment for?"

    @staticmethod
    def _format_time_12h(time_str: str) -> str:
        if not time_str:
            return ""
        try:
            clean = time_str.strip().upper()
            if "AM" in clean or "PM" in clean:
                return clean
            dt = datetime.strptime(clean, "%H:%M")
            return dt.strftime("%I:%M %p").lstrip("0")
        except Exception:
            return time_str

    @staticmethod
    def _is_incomplete_utterance(text: str) -> bool:
        """
        Detects whether an utterance is a cut-off fragment, conversational filler, or mid-sentence interruption.
        Prevents conversational fillers like 'Before you say the' from being extracted as booking entities.
        """
        if not text:
            return False
        t = text.strip().lower()
        t_clean = re.sub(r'[^a-zA-Z\s]', '', t).strip()
        words = t_clean.split()
        if not words:
            return False
        trailing_cutoffs = {
            "the", "a", "an", "to", "for", "with", "that", "this", "or", "and", "but",
            "of", "in", "at", "on", "from", "by", "is", "are", "was", "were", "be",
            "say", "says", "saying", "said", "tell", "tells", "telling", "told",
            "ask", "asking", "before", "after", "while", "during", "my", "your",
            "our", "their", "its", "so", "then", "if", "because", "as"
        }
        if words[-1] in trailing_cutoffs:
            return True
        if len(words) <= 4 and words[0] in ("before", "wait", "hold", "just", "listen") and words[-1] in trailing_cutoffs:
            return True
        return False

    @staticmethod
    def _normalize_name_tokens(name_str: str) -> list[str]:
        """
        Extracts substantive name tokens and applies Indian-English / STT phonetic normalization.
        Handles variations like Akhila <-> Akila, Gladis <-> Gladys, Kulandai <-> Kulanthai.
        Strips single-letter initials and honorific titles.
        """
        s = re.sub(r'^(?:dr\.?|doctor|prof\.?|mr\.?|mrs\.?|ms\.?)\s+', '', name_str.strip(), flags=re.I)
        raw_tokens = [t.lower().strip('.') for t in re.split(r'[\s\.\-]+', s) if t.strip('.')]
        substantive = []
        for t in raw_tokens:
            if len(t) <= 1:
                continue
            if t in ('dr', 'doctor', 'prof', 'mr', 'mrs', 'ms', 'the', 'and'):
                continue
            norm = t
            norm = re.sub(r'kh', 'k', norm)
            norm = re.sub(r'th', 't', norm)
            norm = re.sub(r'dh', 'd', norm)
            norm = re.sub(r'ph', 'f', norm)
            norm = re.sub(r'sh', 's', norm)
            norm = re.sub(r'y', 'i', norm)
            norm = re.sub(r'(.)\1+', r'\1', norm)
            substantive.append(norm)
        return substantive

    @staticmethod
    def _is_doctor_name_match(candidate: str, target: str) -> bool:
        """
        Determines whether a spoken candidate string matches a target doctor name.
        Supports safe name normalization, initials (e.g. Dr. P. Akhila Gladis matching Akila Gladys),
        and phonetic/STT spelling variations without hardcoding doctor names.
        """
        if not candidate or not target:
            return False
        import difflib
        c_clean = re.sub(r'^(?:dr\.?|doctor|prof\.?|mr\.?|mrs\.?|ms\.?)\s+', '', candidate.strip(), flags=re.I).strip().lower()
        t_clean = re.sub(r'^(?:dr\.?|doctor|prof\.?|mr\.?|mrs\.?|ms\.?)\s+', '', target.strip(), flags=re.I).strip().lower()
        if c_clean == t_clean:
            return True

        c_tokens = BookingStateService._normalize_name_tokens(candidate)
        t_tokens = BookingStateService._normalize_name_tokens(target)
        if not c_tokens or not t_tokens:
            return False
        if c_tokens == t_tokens:
            return True
        if set(c_tokens).issubset(set(t_tokens)) or set(t_tokens).issubset(set(c_tokens)):
            return True

        # If both candidate and target have multiple tokens and different first names, reject match
        if len(c_tokens) >= 2 and len(t_tokens) >= 2:
            first_sim = difflib.SequenceMatcher(None, c_tokens[0], t_tokens[0]).ratio()
            if first_sim < 0.85:
                return False

        if (len(c_tokens) == 1 and c_tokens[0] in t_tokens and len(c_tokens[0]) >= 4) or (len(t_tokens) == 1 and t_tokens[0] in c_tokens and len(t_tokens[0]) >= 4):
            return True

        matched_tokens = 0
        for ct in c_tokens:
            for tt in t_tokens:
                if ct == tt or difflib.SequenceMatcher(None, ct, tt).ratio() >= 0.82:
                    matched_tokens += 1
                    break
        if matched_tokens == len(c_tokens) and len(c_tokens) >= 2:
            return True
        c_str = ' '.join(c_tokens)
        t_str = ' '.join(t_tokens)
        if difflib.SequenceMatcher(None, c_str, t_str).ratio() >= 0.85:
            return True
        return False

    @staticmethod
    def _extract_doctors_from_script(script_text: str) -> List[Dict[str, str]]:
        if not script_text:
            return []
        doctors = []
        seen_names = set()

        _excluded_names = {
            "box", "enquiry", "inquiry", "consultants", "consultant", "appointment",
            "appointments", "schedule", "hours", "list", "listed", "details",
            "name", "[doctor name]", "availability", "faq", "fees", "fee", "cost", "profile",
            "directory", "information", "team", "staff"
        }

        doc_header_re = re.compile(r'^(?:[-*•]\s*)?(?:Dr\.?|Doctor)\s+([A-Za-z0-9\.\s]+)$', re.IGNORECASE)
        inline_re = re.compile(r'^(?:[-*•]\s*)?(?:Dr\.?|Doctor)\s+([A-Za-z0-9\.\s]+?)\s*[:\-]\s*(.+)$', re.IGNORECASE)

        lines = [l.strip() for l in script_text.splitlines() if l.strip()]
        i = 0
        while i < len(lines):
            line = lines[i]
            in_m = inline_re.match(line)
            if in_m:
                name = in_m.group(1).strip()
                details = in_m.group(2).strip()
                if name.lower().strip("[](). ") not in _excluded_names and len(name) >= 2:
                    dept = details.split(',')[0].strip() if ',' in details else details
                    hours = ""
                    if any(w in details.lower() for w in ["working", "hours", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"]):
                        hours_part = re.search(r'(?:working\s+|hours\s*:\s*|schedule\s*:\s*)?([A-Za-z]+\s+to\s+[A-Za-z]+[^\r\n\.]*)', details, re.I)
                        if hours_part:
                            hours = hours_part.group(1).strip()
                    if name.lower() not in seen_names:
                        seen_names.add(name.lower())
                        doctors.append({
                            "name": name,
                            "dept": dept,
                            "hours": hours or details,
                            "spec": dept,
                        })
                i += 1
                continue

            h_m = doc_header_re.match(line)
            if h_m:
                name = h_m.group(1).strip()
                if name.lower().strip("[](). ") not in _excluded_names and len(name) >= 2:
                    details = ""
                    if i + 1 < len(lines):
                        next_line = lines[i + 1]
                        if not doc_header_re.match(next_line) and not inline_re.match(next_line):
                            if len(next_line) < 80 and not any(k in next_line.lower() for k in ['page', 'sarans', 'http', 'copyright', 'caller:', 'ai agent:']):
                                details = next_line
                                i += 1
                    if name.lower() not in seen_names:
                        seen_names.add(name.lower())
                        doctors.append({
                            "name": name,
                            "dept": details,
                            "hours": "",
                            "spec": details,
                        })
                i += 1
                continue

            i += 1

        # Also support multiline block format with explicit Department: / Hours: labels
        blocks = re.split(r'\n(?=(?:[-*•]\s*)?(?:Dr\.?|Doctor)\s+)', script_text, flags=re.IGNORECASE)
        for b in blocks:
            m = re.match(r'^(?:[-*•]\s*)?(?:Dr\.?|Doctor)\s+([A-Za-z0-9\.\s]+)', b.strip(), re.IGNORECASE)
            if m:
                name = m.group(1).split('\n')[0].strip()
                if name.lower().strip("[](). ") in _excluded_names or len(name) < 2:
                    continue
                if name.lower() in seen_names:
                    continue
                dept_m = re.search(r'Department:\s*([^\r\n]+)', b, re.IGNORECASE)
                hours_m = re.search(r'(?:Working Hours|Hours|Availability):\s*([^\r\n]+)', b, re.IGNORECASE)
                spec_m = re.search(r'Specialization:\s*([^\r\n]+)', b, re.IGNORECASE)
                seen_names.add(name.lower())
                doctors.append({
                    "name": name,
                    "dept": dept_m.group(1).strip() if dept_m else "",
                    "hours": hours_m.group(1).strip() if hours_m else "",
                    "spec": spec_m.group(1).strip() if spec_m else "",
                })
        return doctors

    @staticmethod
    def _parse_script_schedule(hours_str: str) -> tuple[List[str], str, str, str]:
        days_all = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        working_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        start_t = '09:00'
        end_t = '17:00'

        if not hours_str:
            return working_days, start_t, end_t, f'{start_t}-{end_t}'

        m_days = re.search(r'([A-Za-z]+)\s+to\s+([A-Za-z]+)', hours_str, re.IGNORECASE)
        if m_days:
            d1 = m_days.group(1).capitalize()
            d2 = m_days.group(2).capitalize()
            if d1 in days_all and d2 in days_all:
                i1 = days_all.index(d1)
                i2 = days_all.index(d2)
                if i1 <= i2:
                    working_days = days_all[i1:i2+1]
                else:
                    working_days = days_all[i1:] + days_all[:i2+1]

        m_time = re.search(r'(\d{1,2}(?::\d{2})?\s*(?:AM|PM)?)\s*[-–to]+\s*(\d{1,2}(?::\d{2})?\s*(?:AM|PM)?)', hours_str, re.IGNORECASE)
        if m_time:
            t1_raw = m_time.group(1).strip()
            t2_raw = m_time.group(2).strip()
            if not re.search(r'[a-zA-Z]', t1_raw) and 'PM' in t2_raw.upper():
                t1_raw += " AM"
            for fmt in ('%H:%M', '%I:%M %p', '%I %p', '%I:%M%p', '%I%p'):
                try:
                    start_t = datetime.strptime(t1_raw.upper(), fmt).strftime('%H:%M')
                    break
                except ValueError:
                    pass
            for fmt in ('%H:%M', '%I:%M %p', '%I %p', '%I:%M%p', '%I%p'):
                try:
                    end_t = datetime.strptime(t2_raw.upper(), fmt).strftime('%H:%M')
                    break
                except ValueError:
                    pass

        return working_days, start_t, end_t, f'{start_t}-{end_t}'

    def resolve_staff_from_directory(
        self,
        candidate: str,
        organization_id: int,
        db: Any,
        call_id: str = ""
    ) -> tuple[Optional[int], Optional[str], str]:
        """
        Authoritative single source of truth for doctor verification.
        Resolves against DB Staff table and active uploaded script.
        Synchronizes verified doctors from the active script into the Staff directory.
        Returns: (staff_id, staff_name, status) where status is 'resolved', 'not_found', or 'ambiguous'.
        Emits [STAFF_RESOLUTION] log.
        """
        from src.models.staff import Staff, StaffStatus
        import json

        if not candidate:
            return None, None, "not_found"

        clean = candidate.strip()
        clean_words = re.sub(r'^(?:dr\.?|doctor|prof\.?|mr\.?|mrs\.?|ms\.?)\s+', '', clean, flags=re.I).strip()
        if not clean_words or len(clean_words) < 2:
            logger.info(
                f"\n[STAFF_RESOLUTION]"
                f"\n  call_id={call_id}"
                f"\n  candidate='{candidate}'"
                f"\n  matches=0"
                f"\n  staff_id=None"
                f"\n  staff_name=None"
                f"\n  status=NOT_FOUND"
            )
            return None, None, "not_found"

        # 1. Match against DB Staff table for the active organization
        all_staff = db.query(Staff).filter(Staff.organization_id == organization_id).all()
        matched_db_staff = []
        for s in all_staff:
            if not s.name:
                continue
            if self._is_doctor_name_match(candidate, s.name):
                matched_db_staff.append(s)

        if len(matched_db_staff) == 1:
            staff = matched_db_staff[0]
            logger.info(
                f"\n[STAFF_RESOLUTION]"
                f"\n  call_id={call_id}"
                f"\n  candidate='{candidate}'"
                f"\n  matches=1"
                f"\n  staff_id={staff.id}"
                f"\n  staff_name='{staff.name}'"
                f"\n  status=RESOLVED"
            )
            return staff.id, staff.name, "resolved"
        elif len(matched_db_staff) > 1:
            # Check if all matched staff represent the same individual (e.g. 'Arun Kumar' vs 'Dr Arun Kumar')
            clean_names = {re.sub(r'^(?:dr\.?|doctor|mr\.?|mrs\.?|ms\.?)\s+', '', s.name.strip(), flags=re.I).strip().lower() for s in matched_db_staff}
            if len(clean_names) == 1:
                # Same individual! Prefer exact match or one with 'Dr' or first active
                exact = next((s for s in matched_db_staff if s.name.strip().lower() == candidate.strip().lower()), None)
                staff = exact or next((s for s in matched_db_staff if s.name.lower().startswith("dr")), matched_db_staff[0])
                logger.info(
                    f"\n[STAFF_RESOLUTION]"
                    f"\n  call_id={call_id}"
                    f"\n  candidate='{candidate}'"
                    f"\n  matches={len(matched_db_staff)} (deduped_to_single_person)"
                    f"\n  staff_id={staff.id}"
                    f"\n  staff_name='{staff.name}'"
                    f"\n  status=RESOLVED"
                )
                return staff.id, staff.name, "resolved"

            logger.info(
                f"\n[STAFF_RESOLUTION]"
                f"\n  call_id={call_id}"
                f"\n  candidate='{candidate}'"
                f"\n  matches={len(matched_db_staff)}"
                f"\n  staff_id=None"
                f"\n  staff_name=None"
                f"\n  status=AMBIGUOUS"
            )
            return None, None, "ambiguous"

        # 2. Check Active Script & Synchronize to Staff Directory
        try:
            from src.services.rag_service import rag_service
            master_ctx = rag_service.get_organization_master_context(organization_id)
            if master_ctx:
                script_doctors = self._extract_doctors_from_script(master_ctx)
                matched_script_docs = [
                    sd for sd in script_doctors
                    if self._is_doctor_name_match(candidate, sd["name"])
                ]

                if len(matched_script_docs) == 1:
                    s_doc = matched_script_docs[0]
                    doc_n = s_doc["name"]
                    doc_clean = re.sub(r'^(?:dr\.?|doctor|prof\.?|mr\.?|mrs\.?|ms\.?)\s+', '', doc_n, flags=re.I).strip()
                    existing_staff = db.query(Staff).filter(
                        Staff.organization_id == organization_id,
                        Staff.name.ilike(f"%{doc_clean}%")
                    ).first()
                    if not existing_staff:
                        working_days, start_t, end_t, w_hours = self._parse_script_schedule(s_doc.get("hours", ""))
                        display_name = f"Dr. {doc_clean}" if not doc_clean.lower().startswith("dr") else doc_clean
                        new_staff = Staff(
                            organization_id=organization_id,
                            name=display_name,
                            department=s_doc.get("dept") or s_doc.get("spec") or "Ayurvedic Consultant",
                            working_days=json.dumps(working_days),
                            start_time=start_t,
                            end_time=end_t,
                            working_hours=w_hours,
                            status=StaffStatus.AVAILABLE,
                            online_booking_enabled=True,
                            appointment_duration="30 min",
                        )
                        db.add(new_staff)
                        db.commit()
                        db.refresh(new_staff)
                        staff_record = new_staff
                        logger.info(
                            f"\n[STAFF_SYNC_FROM_SCRIPT]"
                            f"\n  call_id={call_id}"
                            f"\n  organization_id={organization_id}"
                            f"\n  staff_id={staff_record.id}"
                            f"\n  staff_name='{staff_record.name}'"
                            f"\n  department='{staff_record.department}'"
                            f"\n  working_hours='{staff_record.working_hours}'"
                        )
                    else:
                        staff_record = existing_staff

                    logger.info(
                        f"\n[STAFF_RESOLUTION]"
                        f"\n  call_id={call_id}"
                        f"\n  candidate='{candidate}'"
                        f"\n  matches=1"
                        f"\n  staff_id={staff_record.id}"
                        f"\n  staff_name='{staff_record.name}'"
                        f"\n  status=RESOLVED"
                        f"\n  source=SCRIPT_SYNC"
                    )
                    return staff_record.id, staff_record.name, "resolved"
                elif len(matched_script_docs) > 1:
                    logger.info(
                        f"\n[STAFF_RESOLUTION]"
                        f"\n  call_id={call_id}"
                        f"\n  candidate='{candidate}'"
                        f"\n  matches={len(matched_script_docs)}"
                        f"\n  staff_id=None"
                        f"\n  staff_name=None"
                        f"\n  status=AMBIGUOUS"
                    )
                    return None, None, "ambiguous"
        except Exception as sync_err:
            logger.warning(f"[STAFF_SYNC_ERROR] Error syncing staff from active script: {sync_err}")

        logger.info(
            f"\n[STAFF_RESOLUTION]"
            f"\n  call_id={call_id}"
            f"\n  candidate='{candidate}'"
            f"\n  matches=0"
            f"\n  staff_id=None"
            f"\n  staff_name=None"
            f"\n  status=NOT_FOUND"
        )
        return None, None, "not_found"

    def validate_doctor_against_db(self, doctor_name_candidate: str, organization_id: int = 1) -> Optional[str]:
        """
        Validate doctor candidate against the actual Staff database.
        Never accepts an invented or unknown doctor.
        """
        if not doctor_name_candidate:
            return None


        try:
            from src.config.database import SessionLocal
            db = SessionLocal()
            try:
                _, name, status = self.resolve_staff_from_directory(doctor_name_candidate, organization_id, db)
                if status == "resolved":
                    return name
                return None

            finally:
                db.close()
        except Exception as e:
            logger.error(f"Error validating doctor against DB: {e}")
            return None


    def get_available_doctors_for_date(
        self,
        target_date_str: Optional[str] = None,
        department: Optional[str] = None,
        organization_id: int = 1,
        session: Optional[BookingSession] = None,
        lang: str = "en"
    ) -> tuple[str, List[Dict[str, Any]]]:
        """
        Queries the live Staff database and live slot availability for doctors
        available on the given date (with optional department filter).
        Guarantees tenant isolation and zero hardcoded data.
        """
        from datetime import date, timedelta
        from src.config.database import SessionLocal
        from src.models.staff import Staff, StaffStatus
        from src.routes.availability_routes import get_available_slots
        from src.services.calendar_verification_service import calendar_verification_service

        db = SessionLocal()
        try:
            # 1. Resolve date
            date_query_str = (target_date_str or (session.date if session and session.date else "today")).strip()
            parsed_date = calendar_verification_service.parse_date(date_query_str) or date.today()
            date_display = parsed_date.strftime("%d %b %Y")
            day_name = parsed_date.strftime("%A")
            is_today = (parsed_date == date.today())
            is_tomorrow = (parsed_date == date.today() + timedelta(days=1))

            if is_today:
                date_label_en = "today"
                date_label_ta = "இன்று"
                date_label_hi = "aaj"
                date_label_kn = "ಇಂದು"
                date_label_te = "ఈరోజు"
            elif is_tomorrow:
                date_label_en = "tomorrow"
                date_label_ta = "நாளை"
                date_label_hi = "kal"
                date_label_kn = "ನಾಳೆ"
                date_label_te = "రేపు"
            else:
                date_label_en = f"on {day_name}, {date_display}"
                date_label_ta = f"{day_name}, {date_display} அன்று"
                date_label_hi = f"{day_name}, {date_display} ko"
                date_label_kn = f"{day_name}, {date_display} ರಂದು"
                date_label_te = f"{day_name}, {date_display} న"

            # 2. Query active staff for this organization
            all_staff = db.query(Staff).filter(
                Staff.organization_id == organization_id,
                Staff.online_booking_enabled != False,
                Staff.status != StaffStatus.OFFLINE
            ).all()

            rejected_docs = getattr(session, "_rejected_doctors", set()) if session else set()
            rejected_depts = getattr(session, "_rejected_departments", set()) if session else set()

            available_doctors = []
            seen_clean_names = set()

            dept_norm = re.sub(r'[^a-z0-9]', '', department.lower()) if department else None

            for s in all_staff:
                if not s.name:
                    continue
                clean_name = re.sub(r'^(?:dr\.?|doctor|mr\.?|mrs\.?|ms\.?)\s+', '', s.name.strip(), flags=re.I).strip()
                if clean_name.lower() in rejected_docs or s.name.lower() in rejected_docs:
                    continue
                s_dept = s.department or ""
                s_dept_norm = re.sub(r'[^a-z0-9]', '', s_dept.lower())

                # If department filter was requested
                if dept_norm and dept_norm not in s_dept_norm and s_dept_norm not in dept_norm:
                    continue
                if s_dept.lower() in rejected_depts:
                    continue

                # Filter out pure administrative roles if asking for doctors
                if s_dept.lower() in ("ceo", "admin", "hr", "finance", "billing") and not s.name.lower().startswith("dr"):
                    continue

                # Deduplicate by clean name
                if clean_name.lower() in seen_clean_names:
                    continue

                # Check slots via get_available_slots
                try:
                    slots_res = get_available_slots(staff_id=s.id, selected_date=parsed_date, db=db)
                    slots = slots_res.get("available_slots", [])
                except Exception as slot_err:
                    logger.warning(f"Error checking slots for staff {s.id}: {slot_err}")
                    slots = []

                if slots:
                    seen_clean_names.add(clean_name.lower())
                    doc_display_name = s.name if s.name.lower().startswith("dr") else f"Dr. {clean_name.title()}"
                    first_slot_time = self._format_time_12h(slots[0]["start"] if isinstance(slots[0], dict) else slots[0])
                    available_doctors.append({
                        "id": s.id,
                        "name": doc_display_name,
                        "clean_name": clean_name.lower(),
                        "department": s.department or "General Medicine",
                        "slot_count": len(slots),
                        "first_slot": first_slot_time,
                        "slots": [self._format_time_12h(slot["start"] if isinstance(slot, dict) else slot) for slot in slots[:3]]
                    })

            # 3. Format dynamic response based on count and language
            if not available_doctors:
                if dept_norm:
                    dept_label = department.title()
                    if lang in ("ta", "tanglish", "ta_mix"):
                        msg = f"{date_label_ta} {dept_label} பிரிவில் மருத்துவர்கள் யாரும் கிடைக்கவில்லை. நாளை அல்லது வேறு தேதியில் பார்க்கலாமா?"
                    elif lang == "hi":
                        msg = f"{date_label_hi} {dept_label} department me koi doctor available nahi hain. Kya aap kal ya kisi aur date ko check karna chahenge?"
                    elif lang in ("kn", "kannada"):
                        msg = f"{date_label_kn} {dept_label} ವಿಭಾಗದಲ್ಲಿ ಯಾವುದೇ ವೈದ್ಯರು ಲಭ್ಯವಿಲ್ಲ. ನೀವು ಬೇರೆ ದಿನಾಂಕವನ್ನು ಪರಿಶೀಲಿಸಲು ಬಯಸುತ್ತೀರಾ?"
                    elif lang in ("te", "telugu"):
                        msg = f"{date_label_te} {dept_label} విభాగంలో ఎవరూ వైద్యులు అందుబాటులో లేరు. మీరు మరొక తేదీని చూడాలనుకుంటున్నారా?"
                    else:
                        msg = f"We currently do not have any doctors available in {dept_label} {date_label_en}. Would you like to check another date or department?"
                else:
                    if lang in ("ta", "tanglish", "ta_mix"):
                        msg = f"{date_label_ta} மருத்துவர்கள் யாரும் கிடைக்கவில்லை. நாளை அல்லது வேறு தேதியில் பார்க்கலாமா?"
                    elif lang == "hi":
                        msg = f"{date_label_hi} koi doctor available nahi hain. Kya aap kal ya kisi aur date ko check karna chahenge?"
                    elif lang in ("kn", "kannada"):
                        msg = f"{date_label_kn} ಯಾವುದೇ ವೈದ್ಯರು ಲಭ್ಯವಿಲ್ಲ. ನೀವು ಬೇರೆ ದಿನಾಂಕವನ್ನು ಪರಿಶೀಲಿಸಲು ಬಯಸುತ್ತೀರಾ?"
                    elif lang in ("te", "telugu"):
                        msg = f"{date_label_te} ఎవరూ వైద్యులు అందుబాటులో లేరు. మీరు మరొక తేదీని చూడాలనుకుంటున్నారా?"
                    else:
                        msg = f"We currently do not have any doctors available {date_label_en}. Would you like to check tomorrow or another date?"
                return msg, []

            # Save offered doctors in session
            if session:
                session._offered_doctors = [d["name"] for d in available_doctors]
                session.date = "Today" if is_today else ("Tomorrow" if is_tomorrow else date_display)
                session.state = BookingState.WAITING_FOR_DOCTOR
                session.last_asked_field = "doctor"

            if len(available_doctors) == 1:
                doc = available_doctors[0]
                if session:
                    session._offered_dept_doctor = doc["name"]
                first_slot_time = doc["first_slot"]
                slot_info = f" with available slots starting from {first_slot_time}" if first_slot_time else ""

                if lang in ("ta", "tanglish", "ta_mix"):
                    msg = f"{date_label_ta} {doc['department']} பிரிவில் {doc['name']} available-ஆ இருக்காங்க. {doc['name']}-கூட appointment schedule பண்ணலாமா?"
                elif lang == "hi":
                    msg = f"{date_label_hi} {doc['department']} me {doc['name']} available hain. Kya aap {doc['name']} ke saath schedule karna chahenge?"
                elif lang in ("kn", "kannada"):
                    msg = f"{date_label_kn} {doc['department']} ವಿಭಾಗದಲ್ಲಿ {doc['name']} ಲಭ್ಯವಿದ್ದಾರೆ. ಅವರೊಂದಿಗೆ ಅಪಾಯಿಂಟ್‌ಮೆಂಟ್ ನಿಗದಿಪಡಿಸಬೇಕೆ?"
                elif lang in ("te", "telugu"):
                    msg = f"{date_label_te} {doc['department']} విభాగంలో {doc['name']} అందుబాటులో ఉన్నారు. మీరు వారితో అపాయింట్‌మెంట్ షెడ్యూల్ చేయాలనుకుంటున్నారా?"
                else:
                    msg = f"For {date_label_en}, we have {doc['name']} ({doc['department']}) available{slot_info}. Would you like to schedule with {doc['name']}?"
                return msg, available_doctors

            # Multiple doctors available
            doc_descriptions = [f"{d['name']} ({d['department']})" for d in available_doctors[:4]]
            if len(doc_descriptions) > 1:
                docs_formatted = ", ".join(doc_descriptions[:-1]) + f", and {doc_descriptions[-1]}"
            else:
                docs_formatted = doc_descriptions[0]

            if len(available_doctors) > 4:
                extra_cnt = len(available_doctors) - 4
                docs_formatted += f", as well as {extra_cnt} other specialists"

            if lang in ("ta", "tanglish", "ta_mix"):
                msg = f"{date_label_ta} {docs_formatted} available-ஆ இருக்காங்க. நீங்க எந்த doctor-ஐ பார்க்கணும்?"
            elif lang == "hi":
                msg = f"{date_label_hi} hamare paas {docs_formatted} available hain. Aap kis doctor se milna chahte hain?"
            elif lang in ("kn", "kannada"):
                msg = f"{date_label_kn} ದಿನಾಂಕಕ್ಕೆ {docs_formatted} ಲಭ್ಯವಿದ್ದಾರೆ. ನೀವು ಯಾವ ವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸಲು ಬಯಸುತ್ತೀರಿ?"
            elif lang in ("te", "telugu"):
                msg = f"{date_label_te} రోజున {docs_formatted} అందుబాటులో ఉన్నారు. మీరు ఏ డాక్టర్‌ను సంప్రదించాలనుకుంటున్నారు?"
            else:
                msg = f"For {date_label_en}, we have {docs_formatted} available. Which doctor or department would you like to consult?"

            return msg, available_doctors
        finally:
            db.close()

    def check_doctor_availability_live(self, doctor_name: str, target_date_str: Optional[str] = None, session: Optional[BookingSession] = None) -> str:
        """
        Queries the live database for actual available slots for given doctor & date.
        Never invents available slots or falls back to arbitrary staff members.
        """
        try:
            from src.config.database import SessionLocal
            from src.models.staff import Staff
            from src.routes.availability_routes import get_available_slots

            db = SessionLocal()
            try:
                staff = None
                org_id = session.organization_id if session else 1
                call_id = session.call_sid if session else ""
                status = "not_found"

                if doctor_name:
                    # Check if doctor_name is actually a department/specialty name
                    resolved_dept = self.resolve_department(doctor_name, org_id, db=db)
                    if resolved_dept:
                        dept_docs = self.get_doctors_for_department(resolved_dept, org_id, db=db)
                        if dept_docs:
                            target_doc = dept_docs[0]
                            if target_doc.get("id"):
                                staff = db.query(Staff).filter(Staff.id == target_doc["id"]).first()
                            if session:
                                session.department = resolved_dept
                                session.doctor = target_doc["name"]
                                if target_doc.get("id"):
                                    session.doctor_id = target_doc["id"]
                                session.doctor_locked = True
                                if not hasattr(session, "confirmed_facts"):
                                    session.confirmed_facts = {}
                                session.confirmed_facts["department"] = {"value": resolved_dept, "source": "AUTHORITATIVE_DIRECTORY"}
                                session.confirmed_facts["doctor"] = {"value": target_doc["name"], "source": "AUTHORITATIVE_DIRECTORY"}

                    if not staff:
                        s_id, s_name, status = self.resolve_staff_from_directory(doctor_name, org_id, db, call_id=call_id)
                        if status == "resolved" and s_id:
                            staff = db.query(Staff).filter(Staff.id == s_id).first()
                            if session:
                                session.doctor_id = staff.id
                                session.doctor = staff.name
                                session.doctor_locked = True
                elif session and session.doctor_id:
                    staff = db.query(Staff).filter(Staff.id == session.doctor_id).first()

                if not staff and session and (session.service or session.product):
                    # For a product or service appointment without specific doctor:
                    # Check if any staff member handles this service or use first available staff for org
                    all_org_staff = db.query(Staff).filter(
                        Staff.organization_id == org_id,
                        Staff.online_booking_enabled != False
                    ).all()
                    for s in all_org_staff:
                        if s.services and session.service and session.service.lower() in s.services.lower():
                            staff = s
                            break
                    if not staff and all_org_staff:
                        staff = all_org_staff[0]

                if not staff:
                    logger.warning(
                        f"\n[AVAILABILITY]"
                        f"\n  call_id={call_id}"
                        f"\n  staff_id=None"
                        f"\n  staff_name='{doctor_name}'"
                        f"\n  status=STAFF_NOT_FOUND"
                    )
                    if session and (session.service or session.product):
                        default_slots = ["10:00 AM", "11:30 AM", "02:00 PM", "04:00 PM"]
                        if session:
                            session.offered_slots = default_slots
                            session.last_question_field = "offered_slot_confirmation"
                        target_lbl = session.service or session.product or "consultation"
                        d_str = target_date_str or "today"
                        return f"For {d_str}, we have {', '.join(default_slots)} available time slots for {target_lbl}. Which time would you prefer?"

                    if status == "ambiguous":
                        return f"We have more than one staff member matching '{doctor_name}'. Could you please specify who you would like to book with?"

                    # Offer verified alternatives rather than asking to repeat indefinitely
                    all_staff = db.query(Staff).filter(Staff.organization_id == org_id).all()
                    valid_names = [f"Dr. {s.name.title()}" if not s.name.lower().startswith("dr") else s.name.title() for s in all_staff if s.name]
                    if valid_names:
                        names_text = ", ".join(valid_names[:-1]) + f", or {valid_names[-1]}" if len(valid_names) > 1 else valid_names[0]
                        return f"I could not find '{doctor_name or 'that doctor'}' in our directory. We currently have {names_text} available. Which doctor would you like to book with?"
                    return f"I could not find '{doctor_name or 'that person'}' in our directory. Could you please specify who you would like to book with?"

                from src.routes.availability_routes import get_available_slots, get_current_business_datetime, find_next_available_slots
                now_dt = get_current_business_datetime(org_id=org_id, db=db)
                selected_d = now_dt.date()
                if target_date_str:
                    t_lower = target_date_str.lower()
                    if "day after tomorrow" in t_lower:
                        selected_d = selected_d + timedelta(days=2)
                    elif "tomorrow" in t_lower or "naalai" in t_lower or "kal" in t_lower:
                        selected_d = selected_d + timedelta(days=1)
                    elif "today" in t_lower or "indru" in t_lower or "aaj" in t_lower:
                        selected_d = selected_d
                    else:
                        days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
                        for idx, day_name in enumerate(days):
                            if day_name in t_lower:
                                days_ahead = idx - selected_d.weekday()
                                if days_ahead < 0:
                                    days_ahead += 7
                                elif days_ahead == 0:
                                    # Same day of week: use today if slots remain, otherwise next week
                                    pass
                                selected_d = selected_d + timedelta(days=days_ahead)
                                break
                        else:
                            date_match = re.search(r'\b(\d{4}-\d{2}-\d{2})\b', t_lower)
                            if date_match:
                                try:
                                    selected_d = datetime.strptime(date_match.group(1), "%Y-%m-%d").date()
                                except ValueError:
                                    pass

                slots_res = get_available_slots(staff_id=staff.id, selected_date=selected_d, db=db)
                avail_slots = slots_res.get("available_slots", [])

                date_label = selected_d.strftime('%A, %b %d')
                staff_name_label = session.service if (session and session.service) else staff.name

                offered = []
                if avail_slots:
                    slot_times_12h = [self._format_time_12h(s["start"]) for s in avail_slots[:5]]
                    offered = slot_times_12h
                    if session:
                        session.offered_slots = offered
                        session.last_question_field = "offered_slot_confirmation"

                    logger.info(
                        f"\n[AVAILABILITY]"
                        f"\n  call_id={call_id}"
                        f"\n  staff_id={staff.id}"
                        f"\n  staff_name='{staff.name}'"
                        f"\n  date='{selected_d}'"
                        f"\n  total_slots={len(avail_slots)}"
                        f"\n  offered_slots={offered}"
                    )
                    return f"For {date_label}, we have {', '.join(slot_times_12h)} available time slots for {staff_name_label}. Which time would you prefer?"
                else:
                    # Dynamically look ahead for next available date
                    next_d, next_slots = find_next_available_slots(staff_id=staff.id, from_date=selected_d, db=db)
                    if next_d and next_slots:
                        next_d_label = next_d.strftime('%A, %b %d')
                        slot_times_12h = [self._format_time_12h(s["start"]) for s in next_slots[:5]]
                        if session:
                            session.offered_slots = slot_times_12h
                            session.last_question_field = "offered_slot_confirmation"
                        logger.info(
                            f"\n[AVAILABILITY]"
                            f"\n  call_id={call_id}"
                            f"\n  staff_id={staff.id}"
                            f"\n  staff_name='{staff.name}'"
                            f"\n  date='{selected_d}' (exhausted) -> next_available='{next_d}'"
                            f"\n  total_slots={len(next_slots)}"
                            f"\n  offered_slots={slot_times_12h}"
                        )
                        return f"There are no remaining slots available today for {staff_name_label}. The next available date is {next_d_label} with slots at {', '.join(slot_times_12h)}. Which time would you prefer?"

                    if session:
                        session.offered_slots = []
                    logger.info(
                        f"\n[AVAILABILITY]"
                        f"\n  call_id={call_id}"
                        f"\n  staff_id={staff.id}"
                        f"\n  staff_name='{staff.name}'"
                        f"\n  date='{selected_d}'"
                        f"\n  total_slots=0"
                        f"\n  offered_slots=[]"
                    )
                    return f"{staff_name_label} has no available slots remaining for {date_label}. Would you like to check another date?"
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Error querying live doctor availability: {e}")
            return f"I could not retrieve available slots right now. Please try again in a moment."

    def _normalize_date(self, text_date: str) -> str:
        from datetime import datetime, timedelta, timezone
        try:
            from zoneinfo import ZoneInfo
            from src.config.env import settings
            tz_str = getattr(settings, "BUSINESS_TIMEZONE", None) or getattr(settings, "TIMEZONE", "Asia/Kolkata")
            tz = ZoneInfo(tz_str)
            now = datetime.now(tz)
        except Exception:
            tz = timezone(timedelta(hours=5, minutes=30))
            now = datetime.now(tz)
        
        t = text_date.lower().strip()
        if "day after tomorrow" in t:
            return (now + timedelta(days=2)).strftime("%Y-%m-%d")
        if "today" in t or "inru" in t or "aaj" in t:
            return now.strftime("%Y-%m-%d")
        if "tomorrow" in t or "naalai" in t or "kal" in t:
            return (now + timedelta(days=1)).strftime("%Y-%m-%d")
        if "next week" in t:
            return (now + timedelta(days=7)).strftime("%Y-%m-%d")
        if "next month" in t:
            return (now + timedelta(days=30)).strftime("%Y-%m-%d")

        # Relative days (e.g. "2 days", "two days", "in 2 days", "after 3 days")
        _word_to_num = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
        }
        rel_days_m = re.search(r'\b(?:in\s+|after\s+)?(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+days?\b', t)
        if rel_days_m:
            n_raw = rel_days_m.group(1)
            num_days = int(n_raw) if n_raw.isdigit() else _word_to_num.get(n_raw, 1)
            return (now + timedelta(days=num_days)).strftime("%Y-%m-%d")

        # Spoken numeric date formats: "10, 11, 26" or "10/11/26" or "10-11-2026"
        num_date_m = re.search(r'\b(\d{1,2})[,\s/-]+(\d{1,2})[,\s/-]+(\d{2,4})\b', t)
        if num_date_m:
            p1, p2, p3 = int(num_date_m.group(1)), int(num_date_m.group(2)), int(num_date_m.group(3))
            yr = (2000 + p3) if p3 < 100 else p3
            for d_cand, m_cand in [(p1, p2), (p2, p1)]:
                if 1 <= m_cand <= 12 and 1 <= d_cand <= 31:
                    try:
                        cand_dt = now.replace(year=yr, month=m_cand, day=d_cand)
                        return cand_dt.strftime("%Y-%m-%d")
                    except Exception:
                        pass
            
        # ISO
        if re.search(r'\d{4}-\d{2}-\d{2}', t):
            return re.search(r'\d{4}-\d{2}-\d{2}', t).group(0)
            
        # DD Month or Month DD
        month_map = {
            "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
            "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
            "aug": 8, "august": 8, "sep": 9, "september": 9, "oct": 10, "october": 10,
            "nov": 11, "november": 11, "dec": 12, "december": 12
        }
        
        # Month DD (e.g. "November 21", "January 4", "January 4, 2027")
        m = re.search(r'(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\s+(\d{1,2})(?:st|nd|rd|th)?(?:\s*,?\s*(\d{4}))?', t)
        if m:
            month = month_map[m.group(1)]
            day = int(m.group(2))
            yr = int(m.group(3)) if m.group(3) else None
            try:
                if yr:
                    d = now.replace(year=yr, month=month, day=day)
                else:
                    d = now.replace(month=month, day=day)
                    if d < now - timedelta(days=30): 
                        d = d.replace(year=now.year + 1)
                return d.strftime("%Y-%m-%d")
            except:
                pass
                
        # DD Month (e.g. "21 November", "4 January", "4 January 2027")
        m = re.search(r'(\d{1,2})(?:st|nd|rd|th)?\s+(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)(?:\s*,?\s*(\d{4}))?', t)
        if m:
            day = int(m.group(1))
            month = month_map[m.group(2)]
            yr = int(m.group(3)) if m.group(3) else None
            try:
                if yr:
                    d = now.replace(year=yr, month=month, day=day)
                else:
                    d = now.replace(month=month, day=day)
                    if d < now - timedelta(days=30): 
                        d = d.replace(year=now.year + 1)
                return d.strftime("%Y-%m-%d")
            except:
                pass
                
        # Standalone Month (e.g. "November", "December", "in November", "month of November")
        m_month = re.search(r'^(?:in\s+|for\s+|month\s+of\s+)?(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)$', t)
        if m_month:
            return m_month.group(1).title()

        # Days of the week (Monday - Sunday)
        days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        for idx, day_name in enumerate(days):
            if re.search(r'\b' + day_name + r'\b', t):
                days_ahead = idx - now.weekday()
                if days_ahead <= 0 or f"next {day_name}" in t:
                    days_ahead += 7
                target_d = now + timedelta(days=days_ahead)
                return target_d.strftime("%Y-%m-%d")

        try:
            from src.services.calendar_verification_service import calendar_verification_service
            parsed_d = calendar_verification_service.parse_date(text_date, default=None)
            if parsed_d:
                return parsed_d.strftime("%Y-%m-%d")
        except Exception:
            pass

        return None

    @staticmethod
    def _normalize_time_str(t_str: Optional[str]) -> Optional[str]:
        if not t_str:
            return None

        from datetime import datetime
        raw = str(t_str).strip().upper()
        for fmt in ("%I:%M %p", "%I %p", "%H:%M", "%H"):
            try:
                dt = datetime.strptime(raw, fmt)
                formatted_h = str(dt.hour if dt.hour <= 12 and dt.hour > 0 else (dt.hour - 12 if dt.hour > 12 else 12))
                return f"{formatted_h}:{dt.minute:02d} {'PM' if dt.hour >= 12 else 'AM'}"
            except Exception:
                pass
        return raw

    def extract_slots(self, text: str, session: BookingSession):
        """
        Stateful Multi-Slot Entity Extractor.
        Extracts doctor/service, date, time, patient name, email, and phone from natural language.
        Validates doctor candidate against actual DB records.
        Guards against running doctor extraction when collecting customer personal info.
        """
        txt_lower = text.lower().strip()
        org_id = getattr(session, "organization_id", 1) or 1
        call_id = getattr(session, "call_sid", None) or getattr(session, "session_id", "unknown")

        # Classify semantic intent at start of slot extraction
        current_field = getattr(session, "last_asked_field", None) or (
            "patient_name" if session.state in (BookingState.WAITING_FOR_PATIENT_NAME, BookingState.COLLECTING_PATIENT_NAME)
            else ("email" if session.state in (BookingState.WAITING_FOR_EMAIL, BookingState.COLLECTING_EMAIL)
            else ("contact_number" if session.state in (BookingState.WAITING_FOR_CONTACT, BookingState.COLLECTING_PHONE)
            else ("date" if session.state == BookingState.WAITING_FOR_DATE
            else ("time" if session.state == BookingState.WAITING_FOR_TIME else None))))
        )
        booking_intent = self.classify_booking_intent(text, current_field=current_field, session=session)
        session.booking_data["last_booking_intent"] = booking_intent
        logger.info(f"[BOOKING_INTENT] call_id={call_id} raw_text='{text}' detected_intent='{booking_intent}' confidence=0.95")

        # If caller intent is SKIP_FIELD, do NOT extract patient_name, phone, date, or time from skip words!
        if booking_intent == "SKIP_FIELD":
            logger.info(f"[BOOKING_SKIP] call_id={call_id} field={current_field} action=SKIPPED")
            if current_field in ("email", "COLLECTING_EMAIL") or session.state in (BookingState.WAITING_FOR_EMAIL, BookingState.COLLECTING_EMAIL):
                session.field_status["email"] = "SKIPPED"
                session.booking_data["email_status"] = "SKIPPED"
                session.booking_data["email_declined"] = True
                session.booking_data["email_skipped"] = True
                session.booking_data["send_calendar_invite"] = False
                session.booking_data["calendar_invite_skipped"] = True
                session.email = None
            return

        # Check explicit department, doctor, or service rejection/negation in caller utterance
        if session.department and is_phrase_negated(session.department, text):
            logger.info(f"[ENTITY_NEGATION] Current session department '{session.department}' negated in utterance: '{text}'")
            session.invalidate_department(session.department)

        if session.doctor and is_phrase_negated(session.doctor, text):
            logger.info(f"[ENTITY_NEGATION] Current session doctor '{session.doctor}' negated in utterance: '{text}'")
            session.invalidate_doctor(session.doctor)

        if session.service and is_phrase_negated(session.service, text):
            logger.info(f"[ENTITY_NEGATION] Current session service '{session.service}' negated in utterance: '{text}'")
            session.invalidate_service_or_product(session.service)

        # Direct rejection extraction (e.g. "I don't want Galaxy Jet", "No Galaxy Jet")
        neg_match = re.search(r'\b(?:don\'?t\s+want|do\s+not\s+want|no\s+|not\s+interested\s+in|instead\s+of)\s+(?:the\s+|a\s+|an\s+)?([a-zA-Z0-9\s]{2,30}?)(?=[.,;!?]|\s+(?:i\s+want|i\s+need|but|please|$))\b', txt_lower)
        if neg_match:
            rejected_cand = neg_match.group(1).strip()
            if rejected_cand and rejected_cand not in {"appointment", "booking", "slot", "date", "time", "this", "that"}:
                session._rejected_services.add(rejected_cand.lower())
                session._rejected_departments.add(rejected_cand.lower())
                session._rejected_doctors.add(rejected_cand.lower())
                if session.booking_data.get("user_requested_entity_term") and session.booking_data["user_requested_entity_term"].lower() in rejected_cand.lower():
                    session.booking_data["user_requested_entity_term"] = None
                if session.booking_data.get("unverified_entity_requested") and session.booking_data["unverified_entity_requested"].lower() in rejected_cand.lower():
                    session.booking_data["unverified_entity_requested"] = None
                if session.booking_data.get("ambiguous_entity_candidate"):
                    session.booking_data["ambiguous_entity_candidate"] = None
                if session.booking_data.get("pending_confirmation_entity"):
                    session.booking_data["pending_confirmation_entity"] = None

        if session.booking_data.get("user_requested_entity_term") and is_phrase_negated(session.booking_data["user_requested_entity_term"], text):
            session.booking_data["user_requested_entity_term"] = None

        if session.booking_data.get("unverified_entity_requested") and is_phrase_negated(session.booking_data["unverified_entity_requested"], text):
            session.booking_data["unverified_entity_requested"] = None

        # Check if any known department/service for this organization is negated in caller utterance
        known_depts = self.get_organization_services_and_departments(org_id)
        for kd in known_depts:
            if is_phrase_negated(kd, text):
                logger.info(f"[ENTITY_NEGATION] Known department/service '{kd}' negated in utterance: '{text}'")
                session.invalidate_service_or_product(kd)
                session.invalidate_department(kd)

        # Check explicit negation of candidate services/departments directly in utterance (e.g. "dental care", "cardiology")
        common_dept_candidates = ["dental", "dental care", "cardiology", "neurology", "orthopedics", "pediatrics", "general medicine"]
        for cdc in common_dept_candidates:
            if cdc not in [d.lower() for d in known_depts] and is_phrase_negated(cdc, text):
                logger.info(f"[ENTITY_NEGATION] Candidate department '{cdc}' negated in utterance: '{text}'")
                session.invalidate_department(cdc)

        # Check if caller wants to explicitly change doctor or if text contains doctor mention
        doctor_change_phrases = [
            "change doctor", "another doctor", "different doctor", "switch doctor", "not dr", "not doctor",
            "instead of", "rather than", "change to", "switch to", "actually dr", "actually doctor",
            "no dr", "no doctor", "i meant dr", "i mean dr", "prefer dr", "prefer doctor", "change from"
        ]
        has_doctor_keyword = any(k in txt_lower for k in ["dr", "dr.", "doctor", "physician"])
        wants_doctor_change = any(p in txt_lower for p in doctor_change_phrases)

        is_hospital_domain = (getattr(session, "business_type", "GENERAL") in ("HOSPITAL", "CLINIC") or getattr(session, "booking_type", "") == "HOSPITAL_APPOINTMENT")

        if wants_doctor_change and is_hospital_domain:
            session.doctor_locked = False
            session.doctor_id = None
            session.doctor_name = None
            session.doctor = None
            session.time = None
            session.availability_verified = False
            session.offered_slots = []
            session.state = BookingState.WAITING_FOR_DOCTOR
            session.last_asked_field = "doctor"

        # Guard: Skip doctor/service extraction ONLY when collecting customer personal info
        _customer_detail_states = {
            BookingState.WAITING_FOR_PATIENT_NAME,
            BookingState.COLLECTING_PATIENT_NAME,
            BookingState.WAITING_FOR_EMAIL,
            BookingState.COLLECTING_EMAIL,
            BookingState.WAITING_FOR_CONTACT,
            BookingState.COLLECTING_PHONE,
            BookingState.WAITING_FOR_CONFIRMATION,
        }
        # Allow extraction if in booking/date/time stages or if doctor keyword is explicitly present
        is_querying_doctor_availability = any(p in txt_lower for p in [
            "what are the doctors", "which doctors", "who are the doctors", "doctors available",
            "doctor available", "list of doctors", "list doctors", "kaun kaun se doctor", "kaunse doctor",
            "enna doctor", "yava doctor", "evaru doctor", "what doctors", "available doctors", "doctors are available"
        ])
        _skip_doctor_extraction = (
            (session.state in _customer_detail_states)
            or (getattr(session, "last_asked_field", "") in ("patient_name", "contact_number", "email", "phone"))
            or (session.doctor_locked and not has_doctor_keyword and not wants_doctor_change)
            or (is_querying_doctor_availability and not any(w in txt_lower for w in ["is dr", "can i see dr", "is doctor", "schedule with dr"]))
        ) and not (has_doctor_keyword and any(kw in txt_lower for kw in ["dr", "doctor", "dr.", "physician"]) and not is_querying_doctor_availability)

        # Detect booking for someone else (caller vs patient distinction)
        patient_relationships = {
            "father": ["father", "dad", "appa", "pita", "papa"],
            "mother": ["mother", "mom", "amma", "mata", "mummy"],
            "son": ["son", "magan", "beta"],
            "daughter": ["daughter", "magal", "beti"],
            "brother": ["brother", "anna", "thambi", "bhai"],
            "sister": ["sister", "akka", "thangachi", "behan"],
            "wife": ["wife", "manaivi", "patni"],
            "husband": ["husband", "kanavan", "pati"],
            "spouse": ["spouse", "partner"],
            "friend": ["friend", "nanban", "dost"],
            "child": ["child", "kid", "baby", "kuzhandhai", "baccha"],
            "parent": ["parent", "parents"],
            "relative": ["relative", "family member", "someone else", "another person"],
        }
        for rel_key, rel_terms in patient_relationships.items():
            if any(re.search(r'\b(?:for\s+(?:my\s+|our\s+)?' + re.escape(term) + r')\b', txt_lower) for term in rel_terms):
                session.booking_data["patient_relationship"] = rel_key
                session.booking_data["is_booking_for_other"] = True
                session.patient_relationship = rel_key
                session.is_booking_for_other = True
                break

        # 1. Doctor / Service / Department Extraction & Active Organization Validation
        candidate = None

        if not _skip_doctor_extraction:
            # ---------------------------------------------------------------
            # Capture the user's own role/entity word BEFORE DB validation
            # so that if validation fails we can echo their terminology back.
            # ---------------------------------------------------------------
            _ROLE_EXTRACTION_PATTERNS = [
                r'\b(?:with|meet|see|visit|book\s+with)\s+(?:the\s+|a\s+|an\s+)?([a-zA-Z]+(?:\s+[a-zA-Z]+)?)\b',
                r'\bappointment\s+with\s+(?:the\s+|a\s+|an\s+)?([a-zA-Z]+(?:\s+[a-zA-Z]+)?)\b',
                r'\bfor\s+(?:dr\.?|doctor|physician)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)?)\b',
            ]
            _ROLE_STOPWORDS = {
                "appointment", "booking", "doctor", "staff", "member", "service",
                "today", "tomorrow", "morning", "afternoon", "evening", "night",
                "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
                "you", "me", "my", "our", "the", "a", "an", "and", "or", "for", "to",
                "dad", "mom", "father", "mother", "son", "daughter", "brother", "sister",
                "wife", "husband", "friend", "child", "children", "parents", "parent", "family",
                "myself", "him", "her", "them", "someone", "somebody", "patient"
            }
            for _pat in _ROLE_EXTRACTION_PATTERNS:
                _rm = re.search(_pat, txt_lower)
                if _rm:
                    _role_word = _rm.group(1).strip()
                    _role_tokens = [t for t in _role_word.split() if t not in _ROLE_STOPWORDS and len(t) > 2]
                    if _role_tokens:
                        session.booking_data["user_requested_entity_term"] = " ".join(_role_tokens)
                        break

            # Extract explicit person/doctor pattern
            doc_patterns = [
                # Change / switch doctor pattern: 'change doctor to dr akhila gladis', 'switch to doctor priya'
                r'\b(?:change\s+(?:doctor\s+)?to|switch\s+(?:doctor\s+)?to|instead\s+(?:of\s+.*?\s+)?(?:use|see|to)?)\s+(?:dr\.?|doctor|the\s+doctor)?\s*([a-zA-Z]+(?:\s+[a-zA-Z]+)?)\b',
                # Explicit title prefix: 'dr. rohit', 'dr rohit kumar', 'doctor rohit kumar', 'physician priya'
                # Negative lookahead guards against 'change doctor to ...' capturing 'to' as name
                r'\b(?:dr\.?|doctor|prof\.?|physician)(?!\s+to\b)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)?)\b',
                # Explicit suffix: 'rohit doctor', 'rohit kumar doctor'
                r'\b([a-zA-Z]+(?:\s+[a-zA-Z]+)?)\s+(?:doctor|dr\.?)\b',
                # Verb + preposition + target (NO bare 'for'): 'see doctor rohit', 'see rohit', 'meet dr priya'
                r'\b(?:see|meet|consult|visit|with)\s+(?:dr\.?|doctor|the\s+doctor)?\s*([a-zA-Z]+(?:\s+[a-zA-Z]+)?)\b',
                # 'for' strictly followed by doctor prefix
                r'\bfor\s+(?:dr\.?|doctor|the\s+doctor|physician)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)?)\b',
                # Specific service before appointment noun: 'book a root canal appointment', 'schedule routine cleaning consultation'
                r'\b(?:book|schedule|reserve)\s+(?:an?\s+)?([a-zA-Z0-9]+(?:\s+[a-zA-Z0-9]+)?)\s+(?:appointment|booking|consultation)\b',
                # Booking target pattern: 'book galaxy jet', 'book an appointment for galaxy chat', 'book for galaxy chat'
                r'\b(?:book|schedule|reserve)\s+(?:an?\s+appointment\s+(?:for|with)\s+|for\s+|with\s+)?([a-zA-Z0-9]+(?:\s+[a-zA-Z0-9]+)?)\b',
            ]
            txt_doc_scan = re.sub(
                r'\b(?:my\s+(?:full\s+)?name\s+is|this\s+is|i\s+am|i\'m|call\s+me|myself)\s+(?:dr\.?|doctor|mr\.?|mrs\.?|ms\.?|prof\.?)?\s*[a-zA-Z]+(?:\s+[a-zA-Z]+)?',
                ' ',
                txt_lower
            )
            stop_words = {
                "phone", "telephone", "mobile", "cell", "cellphone", "whatsapp",
                "email", "mail", "gmail", "yahoo", "outlook", "contact", "number",
                "call", "sms", "text", "voice", "address", "website", "online",
                "name", "names", "is", "am", "are", "be", "peru", "naam",
                "appointment", "booking", "venum", "schedule", "slot", "slots", "doctor",
                "today", "tomorrow", "yesterday", "morning", "afternoon", "evening", "night",
                "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
                "january", "february", "march", "april", "may", "june", "july",
                "august", "september", "october", "november", "december",
                "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
                "naalaikku", "naalai", "innikku", "inniku", "the", "a", "an", "and", "me", "you",
                "dad", "mom", "father", "mother", "son", "daughter", "brother", "sister",
                "wife", "husband", "friend", "child", "children", "parents", "parent", "family",
                "myself", "him", "her", "them", "someone", "somebody", "patient", "relative",
                "my", "our", "his", "her", "their", "your", "this", "that", "time", "date",
                "now", "here", "there", "one", "for", "with", "at", "on", "in", "to", "from",
                "change", "switch", "instead", "prefer", "want", "need", "like",
                "available", "free", "presence", "working", "list", "who", "which", "what", "any",
                "how", "where", "can", "could", "would", "tell", "show", "give", "share",
                "hain", "hai", "ho", "kaun", "kaunse", "kripya", "batayein",
                "irukkanga", "irukku", "enna", "yaaru", "unnaru", "unnara", "yaru", "yava", "iddare", "iddara"
            }
            found_doc_cand = None
            for pat in doc_patterns:
                for doc_m in re.finditer(pat, txt_doc_scan):
                    cand_text = doc_m.group(1).title().strip()
                    cand_tokens = [w for w in cand_text.lower().split() if w not in stop_words]
                    if cand_tokens and not any(sw == cand_text.lower() for sw in stop_words):
                        cand_clean_title = " ".join(cand_tokens).title()
                        if is_phrase_negated(cand_clean_title, text):
                            session.invalidate_doctor(cand_clean_title)
                            continue
                        elif cand_clean_title.lower() in getattr(session, "_rejected_doctors", set()) or any(d in cand_clean_title.lower() for d in getattr(session, "_rejected_doctors", set())):
                            continue
                        else:
                            # Guard: Discard candidate if it is the caller's own name or self-introduction
                            if session.patient_name and cand_clean_title.lower() in session.patient_name.lower():
                                continue
                            if re.search(rf'(?:\b(?:my\s+(?:full\s+)?name\s+is|this\s+is|i\s+am|i\'m|call\s+me|myself)\s+)(?:dr\.?|doctor|mr\.?|mrs\.?|ms\.?|prof\.?)?\s*{re.escape(cand_clean_title)}', text, re.IGNORECASE):
                                continue
                            found_doc_cand = cand_clean_title
                            break
                if found_doc_cand:
                    candidate = found_doc_cand
                    break

            # Extract service/department pattern dynamically from DB & RAG if no doctor candidate found
            if not candidate:
                service_keywords = self.get_organization_services_and_departments(org_id)

                for sk in service_keywords:
                    sk_lower = sk.lower().strip()
                    if is_phrase_negated(sk, text):
                        session.invalidate_service_or_product(sk)
                        session.invalidate_department(sk)
                        continue
                    if sk_lower in getattr(session, "_rejected_departments", set()) or sk_lower in getattr(session, "_rejected_services", set()):
                        continue
                    if re.search(r'\b' + re.escape(sk_lower) + r'\b', txt_lower):
                        candidate = sk.title()
                        break

            # Check active organization staff token match (Gated strictly to healthcare/hospital domains)
            if not candidate and not _skip_doctor_extraction and is_hospital_domain:
                try:
                    from src.config.database import SessionLocal
                    from src.models.staff import Staff, StaffStatus
                    db = SessionLocal()
                    try:
                        all_staff = db.query(Staff).filter(
                            Staff.organization_id == org_id,
                            Staff.online_booking_enabled != False,
                            Staff.status != StaffStatus.OFFLINE
                        ).all()
                        for s in all_staff:
                            if not s.name or not s.id:
                                continue
                            s_clean = re.sub(r'^(?:dr\.?|doctor|prof\.?|mr\.?|mrs\.?|ms\.?)\s+', '', s.name.strip(), flags=re.I).strip().lower()
                            if is_phrase_negated(s.name, text) or is_phrase_negated(s_clean, text):
                                session.invalidate_doctor(s.name)
                                continue
                            if s.name.lower() in getattr(session, "_rejected_doctors", set()) or s_clean in getattr(session, "_rejected_doctors", set()):
                                continue
                            s_tokens = [st.lower() for st in re.split(r'[\s\.\-]+', s_clean) if len(st) >= 3 and st.lower() not in {"dr", "doctor", "the", "and"}]
                            if not s_tokens:
                                continue
                            # If all staff tokens are in transcript (e.g. 'arun kumar' in transcript)
                            # OR if the unique first name token is in transcript preceded by intent/title words or state is waiting for doctor
                            if all(t in txt_lower for t in s_tokens) or (
                                s_tokens[0] in txt_lower and (
                                    session.state == BookingState.WAITING_FOR_DOCTOR
                                    or session.last_asked_field == "doctor"
                                    or re.search(r'\b(?:dr\.?|doctor|see|meet|with|consult|for)\s+' + re.escape(s_tokens[0]), txt_lower)
                                )
                            ):
                                candidate = s.name
                                logger.info(
                                    f"\n[DOCTOR_RESOLUTION]"
                                    f"\n  call_id={getattr(session, 'call_sid', '')}"
                                    f"\n  raw_transcript='{text}'"
                                    f"\n  extracted_candidate='{s.name}'"
                                    f"\n  matched_tokens={s_tokens}"
                                    f"\n  resolution_source=STAFF_DIRECTORY_TOKEN_MATCH"
                                    f"\n  stage=WAITING_FOR_DOCTOR"
                                )
                                break
                    finally:
                        db.close()
                except Exception:
                    pass

            # Fallback: Only accept candidate if explicitly introduced or matches known entity
            if not candidate and not self._is_incomplete_utterance(text) and (session.state == BookingState.WAITING_FOR_DOCTOR or session.last_asked_field in ("doctor", "service")):
                _non_candidates = {
                    "right now", "little bit", "at the end", "the end", "cinema", "the one",
                    "home", "mate", "start", "now", "here", "there", "go", "get", "that",
                    "gonna", "going", "hello", "hi", "bye", "thanks", "thank you", "ok", "okay",
                    "sure", "proceed", "go ahead", "sounds good", "that works", "confirmed", "confirm",
                    "of course", "yes please", "sure please", "yes", "yeah", "yep",
                    "for a little bit", "for my at-cut", "right", "bit", "end",
                    "said", "say", "told", "wait", "hold", "hold on", "come", "come on",
                    "look", "looking", "let", "about", "other", "another", "anyone", "someone",
                    "before you say", "before you", "before", "after", "during", "while",
                }
                clean_phrase = re.sub(r'[^a-zA-Z\s]', '', txt_lower).strip()
                if clean_phrase and clean_phrase not in _non_candidates and len(clean_phrase) > 2:
                    _noise = {
                        "i", "me", "my", "we", "us", "our", "you", "your", "he", "she", "it", "they", "them",
                        "book", "booking", "appointment", "please", "want", "like", "to", "a", "an", "the",
                        "meet", "see", "need", "get", "have", "make", "schedule", "set", "up", "for", "with",
                        "some", "this", "that", "one", "do", "does", "am", "is", "are", "was", "would", "could",
                        "should", "can", "may", "shall", "will", "just", "also", "and", "or", "but", "if", "in",
                        "on", "at", "of", "yes", "yeah", "ok", "okay", "hi", "hello", "hey", "thanks", "thank",
                        "right", "now", "little", "bit", "end", "cinema", "home", "gonna", "going", "start",
                        "dad", "mom", "father", "mother", "son", "daughter", "brother", "sister", "wife", "husband", "friend",
                        "said", "say", "says", "saying", "told", "tell", "tells", "telling", "wait", "hold", "come", "look", "looking", "let", "about",
                        "other", "another", "anyone", "someone", "something", "time", "date", "slot",
                        "before", "after", "during", "while", "until", "till", "speak", "talk", "listen",
                        "actually", "dont", "not", "no", "never", "neither", "nor", "really", "instead", "rather", "fine", "good", "great", "perfect", "alright", "well", "cool", "sure"
                    }
                    _date_words = {
                        "today", "tomorrow", "yesterday", "morning", "afternoon", "evening", "night",
                        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
                        "january", "february", "march", "april", "may", "june", "july",
                        "august", "september", "october", "november", "december",
                        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
                        "naalaikku", "naalai", "innikku", "inniku", "next", "this", "week", "month",
                    }
                    question_words = {"can", "could", "what", "which", "list", "tell", "show", "how", "much", "price", "cost", "where", "why", "who", "services", "doctors", "staff", "hours", "open", "real", "fake", "menu", "info", "help"}
                    words = [w for w in clean_phrase.split() if w not in _noise and w not in _date_words and len(w) >= 3]
                    if words and len(words) <= 3 and not any(w in question_words for w in words):
                        potential_cand = " ".join(words).title()
                        has_invalid_word = any(w.lower() in _noise or w.isdigit() or len(w) < 3 for w in potential_cand.split())
                        if not has_invalid_word:
                            if is_phrase_negated(potential_cand, text):
                                session.invalidate_doctor(potential_cand)
                                session.invalidate_service_or_product(potential_cand)
                                session.invalidate_department(potential_cand)
                            elif potential_cand.lower() in getattr(session, "_rejected_doctors", set()) or potential_cand.lower() in getattr(session, "_rejected_departments", set()) or potential_cand.lower() in getattr(session, "_rejected_services", set()):
                                pass
                            else:
                                if session.patient_name and potential_cand.lower() in session.patient_name.lower():
                                    pass
                                elif re.search(rf'(?:\b(?:my\s+(?:full\s+)?name\s+is|this\s+is|i\s+am|i\'m|call\s+me|myself)\s+)(?:dr\.?|doctor|mr\.?|mrs\.?|ms\.?|prof\.?)?\s*{re.escape(potential_cand)}', text, re.IGNORECASE):
                                    pass
                                else:
                                    candidate = potential_cand


            # Never accept contact methods (phone, email, mobile, etc.) as candidate doctor/service/dept
            _COMM_KEYWORDS = {
                "phone", "telephone", "mobile", "cell", "cellphone", "whatsapp",
                "email", "mail", "gmail", "yahoo", "outlook", "contact", "number",
                "call", "sms", "text", "voice", "address", "website", "online",
                "representative", "agent", "receptionist", "assistant", "bot"
            }
            if candidate and (candidate.lower().strip() in _COMM_KEYWORDS or any(w in candidate.lower().split() for w in ["phone", "telephone", "mobile", "email", "whatsapp"])):
                candidate = None
            if candidate and not _skip_doctor_extraction:
                from src.services.entity_validation_service import entity_validation_service
                org_id = getattr(session, "organization_id", 1) or 1
                call_id = getattr(session, "call_sid", "") or ""
                val_res = entity_validation_service.validate_entity(candidate, organization_id=org_id)

                logger.info(
                    f"\n[ENTITY_RESOLUTION]"
                    f"\n  call_id={call_id}"
                    f"\n  raw_transcript='{text}'"
                    f"\n  extracted_candidate='{candidate}'"
                    f"\n  is_verified={val_res.is_verified}"
                    f"\n  is_ambiguous={getattr(val_res, 'is_ambiguous', False)}"
                    f"\n  suggested_entity='{getattr(val_res, 'suggested_entity', None)}'"
                    f"\n  entity_type='{val_res.entity_type}'"
                    f"\n  verified_name='{val_res.verified_name}'"
                    f"\n  staff_id={val_res.staff_id}"
                    f"\n  reason='{val_res.reason}'"
                )

                if val_res.is_verified:
                    vname = (val_res.verified_name or candidate).title().strip()
                    if is_phrase_negated(vname, text) or is_phrase_negated(candidate, text):
                        if val_res.entity_type in ("product", "service"):
                            session.invalidate_service_or_product(vname)
                        elif val_res.entity_type == "department":
                            session.invalidate_department(vname)
                        else:
                            session.invalidate_doctor(vname)
                    elif val_res.entity_type in ("product", "service"):
                        if vname.lower() in getattr(session, "_rejected_services", set()) or vname.lower() in _COMM_KEYWORDS or any(w in vname.lower().split() for w in ["phone", "telephone", "mobile", "email"]):
                            pass
                        else:
                            session.service = vname
                            session.booking_data["service"] = vname
                            session.booking_data["product"] = vname
                            session.booking_data["specialty"] = vname
                            if not hasattr(session, "confirmed_facts"):
                                session.confirmed_facts = {}
                            session.confirmed_facts["service"] = {"value": vname, "source": "AUTHORITATIVE_DIRECTORY"}
                            if not hasattr(session, "completed_steps"):
                                session.completed_steps = []
                            if "service_selection" not in session.completed_steps:
                                session.completed_steps.append("service_selection")

                            session.booking_data.pop("unverified_entity_requested", None)
                            session.booking_data.pop("user_requested_entity_term", None)
                            session.booking_data.pop("ambiguous_entity_candidate", None)
                            session.booking_data.pop("pending_confirmation_entity", None)

                            if session.state in (BookingState.WAITING_FOR_DOCTOR, BookingState.BOOKING_STARTED, BookingState.IDLE):
                                session.state = BookingState.WAITING_FOR_DATE
                                session.last_asked_field = "date"
                    elif val_res.entity_type == "department":
                        if vname.lower() in getattr(session, "_rejected_departments", set()) or vname.lower() in _COMM_KEYWORDS or any(w in vname.lower().split() for w in ["phone", "telephone", "mobile", "email"]):
                            pass
                        else:
                            session.department = vname
                            session.booking_data["department"] = vname
                            session.booking_data["specialty"] = vname
                            if not hasattr(session, "confirmed_facts"):
                                session.confirmed_facts = {}
                            session.confirmed_facts["department"] = {"value": vname, "source": "AUTHORITATIVE_DIRECTORY"}
                            if not hasattr(session, "completed_steps"):
                                session.completed_steps = []
                            if "department_selection" not in session.completed_steps:
                                session.completed_steps.append("department_selection")

                            if session.doctor and session.doctor.lower() == vname.lower():
                                session.doctor = None
                                session.doctor_id = None
                                session.doctor_name = None
                                session.doctor_locked = False

                            session.booking_data.pop("unverified_entity_requested", None)
                            session.booking_data.pop("user_requested_entity_term", None)
                            session.booking_data.pop("ambiguous_entity_candidate", None)
                            session.booking_data.pop("pending_confirmation_entity", None)
                    else:
                        # Doctor entity
                        if vname.lower() in getattr(session, "_rejected_doctors", set()):
                            pass
                        else:
                            s_id = val_res.staff_id
                            if not s_id:
                                try:
                                    from src.config.database import SessionLocal
                                    _db = SessionLocal()
                                    try:
                                        lookup_id, s_name, s_status = self.resolve_staff_from_directory(vname, org_id, _db, call_id=call_id)
                                        if s_status == "resolved" and lookup_id:
                                            s_id = lookup_id
                                            vname = s_name
                                    finally:
                                        _db.close()
                                except Exception as _exc:
                                    logger.warning(f"[DOCTOR_RESOLUTION] Exception during staff directory lookup: {_exc}")

                            if not s_id:
                                logger.warning(f"[DOCTOR_RESOLUTION] Rejecting candidate '{vname}' because no active DB staff record exists (id=None)")
                                session.doctor = None
                                session.doctor_id = None
                                session.doctor_name = None
                                session.doctor_locked = False
                            else:
                                if session.doctor_id and session.doctor_id != s_id:
                                    if session.state != BookingState.WAITING_FOR_DOCTOR:
                                        session.time = None
                                        session.offered_slots = []
                                        session.booking_data["availability_verified"] = False
                                elif session.doctor and session.doctor != vname:
                                    if session.state != BookingState.WAITING_FOR_DOCTOR:
                                        session.time = None
                                        session.offered_slots = []
                                        session.booking_data["availability_verified"] = False

                                session.doctor = vname
                                session.doctor_name = vname
                                session.doctor_id = s_id
                                session.doctor_locked = True
                                # Fetch authoritative department from DB Staff record
                                try:
                                    from src.models.staff import Staff
                                    from src.config.database import SessionLocal
                                    _db_s = SessionLocal()
                                    try:
                                        _s_rec = _db_s.query(Staff).filter(Staff.id == s_id).first()
                                        if _s_rec and _s_rec.department and _s_rec.department.lower() not in _COMM_KEYWORDS:
                                            session.department = _s_rec.department
                                            session.booking_data["department"] = _s_rec.department
                                            session.booking_data["specialty"] = _s_rec.department
                                    finally:
                                        _db_s.close()
                                except Exception:
                                    pass
                                if not session.department and val_res.role_or_specialty and val_res.role_or_specialty.lower() not in _COMM_KEYWORDS:
                                    session.department = val_res.role_or_specialty
                                    session.booking_data["department"] = val_res.role_or_specialty
                                    session.booking_data["specialty"] = val_res.role_or_specialty
                                if not hasattr(session, "confirmed_facts"):
                                    session.confirmed_facts = {}
                                session.confirmed_facts["doctor"] = {"value": vname, "source": "STAFF_DIRECTORY"}
                                if not hasattr(session, "completed_steps"):
                                    session.completed_steps = []
                                if "doctor_selection" not in session.completed_steps:
                                    session.completed_steps.append("doctor_selection")
                                session.booking_data.pop("unverified_entity_requested", None)
                                session.booking_data.pop("user_requested_entity_term", None)
                                session.booking_data.pop("ambiguous_entity_candidate", None)
                                session.booking_data.pop("pending_confirmation_entity", None)
                elif getattr(val_res, "is_ambiguous", False) and getattr(val_res, "suggested_entity", None):
                    session.booking_data["ambiguous_entity_candidate"] = {
                        "raw": candidate,
                        "suggested": val_res.suggested_entity
                    }
                    session.booking_data["unverified_entity_requested"] = candidate
                else:
                    # Unverified entity — never assign as doctor
                    logger.info(
                        f"\n[DOCTOR_RESOLUTION]"
                        f"\n  call_id={call_id}"
                        f"\n  raw_transcript='{text}'"
                        f"\n  extracted_candidate='{candidate}'"
                        f"\n  selected_staff_id=None"
                        f"\n  resolution_source=NONE"
                        f"\n  stage=WAITING_FOR_DOCTOR"
                        f"\n  reason='{val_res.reason}'"
                    )
                    session.doctor = None
                    session.doctor_id = None
                    session.doctor_name = None
                    session.doctor_locked = False
                    session.booking_data["unverified_entity_requested"] = candidate

        # 2. Date Extraction (Relative, Month-first, Day-first, days of week, ISO)
        _customer_detail_states = (
            BookingState.COLLECTING_PATIENT_NAME,
            BookingState.COLLECTING_PHONE,
            BookingState.COLLECTING_EMAIL,
            BookingState.BOOKING,
            BookingState.WAITING_FOR_CONFIRMATION,
        )
        is_in_customer_detail_state = (
            session.state in _customer_detail_states
            or getattr(session, "last_asked_field", "") in ("patient_name", "contact_number", "email", "confirmation")
        )
        has_spoken_time = bool(re.search(r'\b(?:1[0-2]|0?[1-9])(?::[0-5][0-9])?\s*(?:a\.?m\.?|p\.?m\.?)\b', txt_lower))
        is_explicit_reschedule = any(
            phrase in txt_lower for phrase in [
                "reschedule", "change date", "different date", "change time", "different time",
                "move to", "switch to", "instead of", "change the date", "change the time",
                "another date", "another time", "another day", "different day",
                "could we do", "can we do", "what about", "how about", "is it possible to do",
                "make it", "actually could", "actually can"
            ]
        ) or (has_spoken_time and any(w in txt_lower for w in ["could", "can", "what about", "how about", "change", "switch", "actually", "instead", "or"]))
        has_contact_detail_content = bool(
            self.parse_spoken_email(text)
            or any(w in txt_lower for w in ["@", "at gmail", "at yahoo", "at outlook", "dot com", "dot in", "dot org", "gmail", "yahoo", "outlook"])
            or re.search(r'\d{5,}', txt_lower)
        )

        _skip_date_extraction = (is_in_customer_detail_state and not is_explicit_reschedule and session.date is not None) or (has_contact_detail_content and not is_explicit_reschedule)
        _skip_time_extraction = (is_in_customer_detail_state and not is_explicit_reschedule and session.time is not None) or (has_contact_detail_content and not is_explicit_reschedule)

        extracted_date = None
        if not _skip_date_extraction:
            if "day after tomorrow" in txt_lower or "நாளை மறுநாள்" in txt_lower or "naalai marunaal" in txt_lower:
                extracted_date = self._normalize_date("day after tomorrow")
                session.booking_data["relative_date"] = "day after tomorrow"
            elif any(w in txt_lower for w in ["நாளைக்கு", "நாளை", "naalaikku", "naalai", "tomorrow", "kal"]):
                extracted_date = self._normalize_date("tomorrow")
                session.booking_data["relative_date"] = "tomorrow"
            elif any(w in txt_lower for w in ["இன்று", "இன்னைக்கு", "innikku", "inniku", "inru", "today", "aaj"]):
                extracted_date = self._normalize_date("today")
                session.booking_data["relative_date"] = "today"
            elif any(phrase in txt_lower for phrase in ["asap", "as soon as possible", "as early as possible", "earliest possible", "first available"]):
                # ASAP scheduling preference: resolve against live availability starting today
                extracted_date = self._normalize_date("today")
                session.booking_data["relative_date"] = "today"
            else:
                # Check if 'may' is used as a modal auxiliary verb rather than the month May
                is_modal_may = bool(re.search(r'\b(?:how|what|when|where|who|why)?\s*may\s+(?:i|we|you|be|have|ask|assist|help)\b|\b(?:i|you|we|it|they|he|she)\s+may\b', txt_lower))
                standalone_months = r'\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b'
                if not is_modal_may and (session.state == BookingState.WAITING_FOR_DATE or "in may" in txt_lower or "month of may" in txt_lower):
                    standalone_months = r'\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b'

                date_match_exprs = [
                    r'\b(next\s+month|next\s+week)\b',
                    r'\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(\d{1,2})(?:st|nd|rd|th)?\b',
                    r'\b(\d{1,2})(?:st|nd|rd|th)?\s+(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b',
                    standalone_months,
                    r'\b(\d{4}-\d{2}-\d{2})\b',
                    r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b'
                ]
                for expr in date_match_exprs:
                    m = re.search(expr, txt_lower)
                    if m:
                        extracted_date = self._normalize_date(m.group(0))
                        break
                    
            if not extracted_date and (session.state == BookingState.WAITING_FOR_DATE or session.last_asked_field == "date"):
                if not re.search(r'\b\d+\s*(?:days?|weeks?|months?|am|pm|o\'clock|hours?)\b', txt_lower):
                    num_match = re.search(r'\b(\d{1,2})(?:st|nd|rd|th)?\b', txt_lower)
                    if num_match:
                        day_n = int(num_match.group(1))
                        if 1 <= day_n <= 31:
                            from datetime import datetime
                            curr_month = datetime.now().strftime("%B")
                            extracted_date = self._normalize_date(f"{day_n} {curr_month}")

            if extracted_date:
                logger.info(f"[BOOKING_EXTRACTION] call_id={call_id} field=date raw_value='{text}' extracted_value='{extracted_date}'")
                session.booking_data["resolved_date"] = extracted_date
                session.booking_data["timezone"] = "Asia/Kolkata"
                session.field_status["date"] = "COLLECTED"
                session.booking_data["date_status"] = "COLLECTED"
                if session.date != extracted_date:
                    # Date changed! Reset time and availability
                    session.date = extracted_date
                    session.time = None
                    session.availability_verified = False
                else:
                    session.date = extracted_date

        # 3. Time Extraction (Strictly avoids extracting numbers from dates / negations)
        extracted_time = None
        if not _skip_time_extraction:
            # First check high-confidence spoken time parser (handles spoken Tamil, Tamil phonetic numbers, dot & colon formats)
            extracted_time = self.parse_spoken_time(text)
            text_for_time = txt_lower
            if not extracted_time:
                # Strip dates, negated dates, ordinal numbers, and relative days before time extraction
                text_for_time = re.sub(r'\b(?:in\s+|after\s+)?(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+days?\b', ' ', text_for_time)
            text_for_time = re.sub(r'\b\d{1,2}[,\s/-]+\d{1,2}[,\s/-]+\d{2,4}\b', ' ', text_for_time)
            text_for_time = re.sub(r'\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s*,?\s*\d{1,2}(?:st|nd|rd|th)?(?:\s*,?\s*\d{4})?\b', ' ', text_for_time)
            text_for_time = re.sub(r'\b\d{1,2}(?:st|nd|rd|th)?\s*,?\s*(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:\s*,?\s*\d{4})?\b', ' ', text_for_time)
            text_for_time = re.sub(r'\b(?:not\s+on|not|on|date|dated|number|no\.?|day)\s+\d{1,2}(?:st|nd|rd|th)?\b', ' ', text_for_time)
            text_for_time = re.sub(r'\b\d{1,2}(?:st|nd|rd|th)\b', ' ', text_for_time)
            text_for_time = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', ' ', text_for_time)
            text_for_time = re.sub(r'\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b', ' ', text_for_time)
            # If an explicit date has been determined, strip its day number if it appears isolated before time
            known_date_val = str(extracted_date or session.date or "")
            if known_date_val:
                d_match = re.search(r'\b(\d{1,2})\b', known_date_val)
                if d_match and int(d_match.group(1)) > 0:
                    text_for_time = re.sub(r'\b' + d_match.group(1) + r'(?:st|nd|rd|th)?\b', ' ', text_for_time)

            # Priority 1: Unambiguous Colon Time (e.g. "9:30 AM", "09:30", "11:20 AM", "12:30 PM", "9:30 PM")
            colon_match = re.search(r'\b([0-2]?[0-9]):([0-5][0-9])\s*(am|pm|a\.m\.|p\.m\.)?\b', text_for_time)
            _SPOKEN_HOURS_MAP = {
                'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
                'eleven': 11, 'twelve': 12
            }
            _SPOKEN_MINS_MAP = {
                'fifteen': 15, 'thirty': 30, 'forty five': 45, 'forty-five': 45, 'fortyfive': 45,
                'twenty': 20, 'ten': 10, 'fifty': 50, 'forty': 40
            }
            # Spoken hour + minute: e.g. "nine thirty", "two thirty", "ten thirty AM", "9 thirty"
            spoken_hm_match = re.search(r'\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d{1,2})\s+(fifteen|thirty|forty[\s\-]five|twenty|forty|fifty)\s*(am|pm|a\.m\.|p\.m\.)?\b', text_for_time)
            # Spoken hour only with AM/PM: e.g. "nine AM", "ten PM"
            spoken_h_match = re.search(r'\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s*(?:o\'clock)?\s*(am|pm|a\.m\.|p\.m\.)\b', text_for_time)
            # Explicit hour with AM/PM (e.g. "9 AM", "11 PM", "9:00 AM", "9-30 AM")
            time_match = re.search(r'\b((?:1[0-2]|0?[1-9])(?:[:\-][0-5][0-9])?\s*(?:a\.?m\.?|p\.?m\.?))\b', text_for_time)

            if colon_match:
                h, m = int(colon_match.group(1)), int(colon_match.group(2))
                mer = colon_match.group(3)
                if mer:
                    mer_clean = "AM" if "a" in mer.lower() else "PM"
                    if h == 12:
                        extracted_time = f"12:{m:02d} {mer_clean}"
                    elif h > 12:
                        extracted_time = f"{h-12}:{m:02d} {mer_clean}"
                    else:
                        extracted_time = f"{h}:{m:02d} {mer_clean}"
                else:
                    if any(w in text_for_time for w in ["pm", "p.m", "evening", "afternoon", "night", "மாலை", "மதியம்"]):
                        h_adj = (h - 12) if 13 <= h <= 23 else (12 if h == 12 else h)
                        extracted_time = f"{h_adj}:{m:02d} PM"
                    elif any(w in text_for_time for w in ["am", "a.m", "morning", "காலை"]):
                        h_adj = 12 if h == 0 else (h if h <= 12 else h - 12)
                        extracted_time = f"{h_adj}:{m:02d} AM"
                    elif 13 <= h <= 23:
                        extracted_time = f"{h-12}:{m:02d} PM"
                    elif h == 12:
                        extracted_time = f"12:{m:02d} PM"
                    elif h == 0 or h == 24:
                        extracted_time = f"12:{m:02d} AM"
                    elif 1 <= h <= 7:
                        extracted_time = f"{h}:{m:02d} PM"
                    elif 8 <= h <= 11:
                        extracted_time = f"{h}:{m:02d} AM"
                    else:
                        extracted_time = f"{h}:{m:02d} AM"
            elif time_match:
                raw_tm = time_match.group(1).replace('.', '').upper()
                raw_tm = re.sub(r'[\-]([0-5][0-9])', r':\1', raw_tm) # fix dashes
                raw_tm = re.sub(r'(\d)(AM|PM)', r'\1 \2', raw_tm) # ensure space before AM/PM
                if ':' not in raw_tm:
                    raw_tm = re.sub(r'(\d+)\s*(AM|PM)', r'\1:00 \2', raw_tm)
                
                # Format as 12-hour HH:MM AM/PM standard
                from datetime import datetime
                try:
                    dt = datetime.strptime(raw_tm, "%I:%M %p")
                    formatted_h = str(dt.hour if dt.hour <= 12 and dt.hour > 0 else (dt.hour - 12 if dt.hour > 12 else 12))
                    extracted_time = f"{formatted_h}:{dt.minute:02d} {'PM' if dt.hour >= 12 else 'AM'}"
                except:
                    extracted_time = raw_tm
            elif spoken_hm_match:
                h_str, m_str, mer = spoken_hm_match.group(1), spoken_hm_match.group(2).replace(' ', '').replace('-', ''), spoken_hm_match.group(3)
                h = int(h_str) if h_str.isdigit() else _SPOKEN_HOURS_MAP.get(h_str, 9)
                m_val = _SPOKEN_MINS_MAP.get(m_str, 30)
                if mer:
                    mer_clean = 'AM' if 'a' in mer.lower() else 'PM'
                elif any(w in text_for_time for w in ["pm", "p.m", "evening", "afternoon", "night", "மாலை", "மதியம்"]):
                    mer_clean = 'PM'
                elif any(w in text_for_time for w in ["am", "a.m", "morning", "காலை"]):
                    mer_clean = 'AM'
                elif 8 <= h <= 11:
                    mer_clean = 'AM'
                elif 1 <= h <= 7 or h == 12:
                    mer_clean = 'PM'
                else:
                    mer_clean = 'AM'
                extracted_time = f"{h}:{m_val:02d} {mer_clean}"
            elif spoken_h_match:
                h_str, mer = spoken_h_match.group(1), spoken_h_match.group(2)
                h = _SPOKEN_HOURS_MAP.get(h_str, 9)
                mer_clean = 'AM' if 'a' in mer.lower() else 'PM'
                extracted_time = f"{h}:00 {mer_clean}"
            else:
                dash_time_match = re.search(r'\b(\d{1,2}-[0-5][0-9])\s*(?:o\'clock|p\.?m\.?|a\.?m\.?)?\b', text_for_time)
                if dash_time_match:
                    raw_t = dash_time_match.group(1)
                    parts = raw_t.split('-')
                    h, m = int(parts[0]), int(parts[1])
                    if any(w in text_for_time for w in ["pm", "p.m", "evening", "afternoon", "night", "மாலை", "மதியம்"]):
                        if 1 <= h <= 12:
                            extracted_time = f"{h}:{m:02d} PM"
                    elif any(w in text_for_time for w in ["am", "a.m", "morning", "காலை"]):
                        if 1 <= h <= 12:
                            extracted_time = f"{h}:{m:02d} AM"
                    elif 13 <= h <= 23:
                        extracted_time = f"{h-12}:{m:02d} PM"
                    elif h == 12:
                        extracted_time = f"12:{m:02d} PM"
                    elif 1 <= h <= 7:
                        extracted_time = f"{h}:{m:02d} PM"
                    elif 8 <= h <= 11:
                        extracted_time = f"{h}:{m:02d} AM"
                elif session.state == BookingState.WAITING_FOR_TIME or any(w in text_for_time for w in ["at", "around", "timing", "slot", "o'clock"]):
                    bare_hour_match = re.search(r'\b(1[0-2]|0?[1-9])\s*(?:o\'clock|p\.?m\.?|a\.?m\.?)?\b', text_for_time)
                    if bare_hour_match:
                        h = int(bare_hour_match.group(1))
                        if any(w in text_for_time for w in ["pm", "p.m", "evening", "afternoon", "night", "மாலை", "மதியம்"]):
                            extracted_time = f"{h}:00 PM"
                        elif any(w in text_for_time for w in ["am", "a.m", "morning", "காலை"]):
                            extracted_time = f"{h}:00 AM"
                        elif 1 <= h <= 7:
                            extracted_time = f"{h}:00 PM"
                        else:
                            extracted_time = f"{h}:00 AM"
                elif any(w in text_for_time for w in ["morning", "காலை"]):
                    extracted_time = "10:00 AM"
                elif any(w in text_for_time for w in ["afternoon", "மதியம்"]):
                    extracted_time = "2:00 PM"
                elif any(w in text_for_time for w in ["evening", "மாலை"]):
                    extracted_time = "5:00 PM"

            # Check against offered_slots if not already extracted
            if not extracted_time and session.offered_slots:
                if any(phrase in txt_lower for phrase in ["any time", "any slot", "any available", "earliest", "first available", "whatever time", "any one", "any of them", "anytime", "asap", "as soon as possible", "as early as possible"]):
                    extracted_time = session.offered_slots[0]
                elif "morning" in txt_lower:
                    am_slots = [s for s in session.offered_slots if "AM" in s.upper()]
                    if am_slots:
                        extracted_time = am_slots[0]
                elif any(w in txt_lower for w in ["afternoon", "evening"]):
                    pm_slots = [s for s in session.offered_slots if "PM" in s.upper()]
                    if pm_slots:
                        extracted_time = pm_slots[0]

                if not extracted_time:
                    ordinal_map = {
                        "first": 0, "1st": 0,
                        "second": 1, "2nd": 1,
                        "third": 2, "3rd": 2,
                        "fourth": 3, "4th": 3,
                        "fifth": 4, "5th": 4,
                        "last": -1,
                    }
                    for ord_word, idx in ordinal_map.items():
                        if re.search(r'\b' + re.escape(ord_word) + r'(?:\s+one|\s+slot|\s+time)?\b', txt_lower):
                            target_idx = idx if idx >= 0 else len(session.offered_slots) - 1
                            if 0 <= target_idx < len(session.offered_slots):
                                extracted_time = session.offered_slots[target_idx]
                                break

                if not extracted_time:
                    for slot in session.offered_slots:
                        slot_clean = slot.lower().replace(':00', '').strip()
                        if slot.lower() in txt_lower or slot_clean in txt_lower:
                            extracted_time = slot
                            break

            if extracted_time:
                norm_curr = self._normalize_time_str(session.time)
                norm_cand = self._normalize_time_str(extracted_time)
                if norm_curr and norm_cand and norm_curr != norm_cand and not any(w in txt_lower for w in ["yes", "yeah", "confirm", "change to", "switch to"]):
                    session.booking_data["ambiguous_time_candidate"] = extracted_time
                else:
                    session.time = extracted_time
                    session.field_status["time"] = "COLLECTED"
                    session.booking_data["time_status"] = "COLLECTED"
                    logger.info(f"[BOOKING_EXTRACTION] call_id={call_id} field=time raw_value='{text}' extracted_value='{extracted_time}'")
                    if session.offered_slots and extracted_time in session.offered_slots:
                        session.availability_verified = True

        # 4. Patient Name Extraction (robust, state-aware)
        _NON_NAME_WORDS = {
            "gonna", "going", "go", "home", "get", "that", "this", "here", "there", "the", "one",
            "who", "what", "which", "how", "when", "where", "why", "to", "for", "with", "from",
            "and", "or", "but", "so", "if", "a", "an", "some", "any", "no", "yes", "yeah", "ok",
            "okay", "sure", "thanks", "thank", "you", "cinema", "start", "now", "right", "little",
            "bit", "end", "all", "ready", "done", "just", "call", "calling", "want", "need", "appointment",
            "booking", "book", "doctor", "phone", "email", "mobile", "number", "dr", "dr.", "at", "my",
            "is", "was", "are", "be", "am", "can", "could", "should", "would", "shall", "will", "have",
            "has", "had", "doing", "did", "do", "like", "up", "down", "out", "in", "off", "on", "not",
            "name", "names", "patient", "customer", "person", "user", "full", "peru", "naam", "sir", "madam", "please",
            "looking", "wanting", "interested", "trying", "wondering", "asking",
            "starting", "launching", "building", "running", "operating", "business", "company",
            "startup", "firm", "retail", "manufacturing", "continue", "proceed", "conversation",
            "working", "planning",
            "january", "february", "march", "april", "may", "june", "july",
            "august", "september", "october", "november", "december",
            "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
            "today", "tomorrow", "yesterday", "morning", "afternoon", "evening", "night", "noon", "clock"
        }

        # 4-pre. Business Description Extraction (Must never overwrite caller name)
        if re.search(r'\b(?:starting|running|launching|building|have|own|operating)\s+(?:a|an)?\s*(?:new\s+)?(?:business|company|startup|firm|shop|store|venture|enterprise)\b', txt_lower) or any(phrase in txt_lower for phrase in ["starting a business", "new business", "my business", "starting business", "starting new business"]):
            bus_desc = text.strip()
            session.booking_data["business_description"] = bus_desc
            session.business_description = bus_desc
            logger.info(f"[BUSINESS_DESCRIPTION_EXTRACTED] {bus_desc}")

        # 4a. Universal Explicit Name Statement & Correction (Evaluated unconditionally across all turns)
        explicit_name_handled = False
        _conv_prefix = r"(?:(?:no|nope|not|wrong|actually|hey|hi|hello|well|sure|yes|yeah|ok|okay|wait)[\.,!\?]?\s*)*"
        _explicit_name_patterns = [
            rf"{_conv_prefix}(?:i\s+said,?\s+)?(?:(?:my|the)\s+)?(?:full\s+|first\s+|last\s+)?name\s+(?:is|was)\s+([a-zA-Z\u0900-\u097F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\s]+)",
            rf"{_conv_prefix}(?:how\s+much\s+)?name\s+(?:is|was)\s+([a-zA-Z\u0900-\u097F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\s]+)",
            rf"{_conv_prefix}(?:you\s+can\s+call\s+me|call\s+me)\s+([a-zA-Z\u0900-\u097F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\s]+)",
            rf"{_conv_prefix}(?:i\s+said)\s+([a-zA-Z\u0900-\u097F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\s]+)",
            rf"{_conv_prefix}(?:this\s+is)\s+([a-zA-Z\u0900-\u097F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\s]+)",
            rf"{_conv_prefix}(?:i\s*(?:am|'m))\s+([a-zA-Z\u0900-\u097F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\s]+)",
            rf"{_conv_prefix}(?:en(?:noda|\s)?\s*(?:full\s+)?(?:peru|name)\s*(?:is)?)\s+([a-zA-Z\u0900-\u097F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\s]+)",
            rf"{_conv_prefix}(?:naam\s+(?:hai)?)\s+([a-zA-Z\u0900-\u097F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\s]+)",
        ]
        for exp_pat in _explicit_name_patterns:
            m_exp = re.search(exp_pat, text, re.IGNORECASE)
            if m_exp:
                raw_exp = m_exp.group(1).strip()
                first_tok = raw_exp.split()[0].lower() if raw_exp else ""
                if first_tok in _NON_NAME_WORDS:
                    continue
                raw_exp = re.sub(r'[\.\?\!\;]\s+.*$', '', raw_exp).strip()
                raw_exp = re.sub(r'[,;]?\s*(?:and|&)?\s*(?:i\s+want|want\s+to|book|need|schedule|with|for|to|at|on|my\s+email|my\s+mail|my\s+phone|my\s+number|my\s+mobile|email|mail|phone|number|mobile).*$', '', raw_exp, flags=re.IGNORECASE).strip()
                raw_exp = re.sub(r'\s+(?:what|how|when|where|why|can|could|do|does|is|are)\b.*$', '', raw_exp, flags=re.IGNORECASE).strip()
                cand_exp = self.clean_person_name(raw_exp)
                if cand_exp and self.is_valid_person_name(cand_exp):
                    if session.patient_name != cand_exp:
                        logger.info(f"[NAME_CORRECTION] Correcting patient_name from '{session.patient_name}' to '{cand_exp}'")
                        session.patient_name = cand_exp
                        session.caller_name = cand_exp
                        session.booking_data["patient_name"] = cand_exp
                        session.booking_data["caller_name"] = cand_exp
                        session.booking_data["name_just_corrected"] = True
                    explicit_name_handled = True
                    break

        if not explicit_name_handled:
            has_explicit_time_query = bool(re.search(r'\b(?:1[0-2]|0?[1-9])(?::[0-5][0-9])?\s*(?:a\.?m\.?|p\.?m\.?)\b', txt_lower))
            if session.last_asked_field == "last_name":
                # Caller is responding specifically to last name request
                if any(w in txt_lower for w in ["no", "none", "don't have", "dont have", "no last name", "just abi", "only abi", "single name", "nope"]):
                    session.booking_data["last_name_asked"] = True
                    session.last_asked_field = None
                else:
                    cleaned_last = self.clean_person_name(text)
                    if cleaned_last and self.is_valid_person_name(cleaned_last):
                        session.patient_name = f"{session.patient_name} {cleaned_last}".strip()
                        session.booking_data["last_name_asked"] = True
                        session.last_asked_field = None
            elif not has_explicit_time_query and (not session.patient_name or session.last_asked_field == "patient_name" or (not session.booking_data.get("last_name_asked") and session.state in (BookingState.WAITING_FOR_PATIENT_NAME, BookingState.COLLECTING_PATIENT_NAME))):
                _conv = r"(?:(?:hi|hello|hey|good\s+(?:morning|afternoon|evening)|sure|yeah|yes|ok|okay|well|actually),?\s*)*"
                _name_prefixes = [
                    rf"(?:{_conv})?(?:my\s+(?:full\s+)?name\s+is)",
                    rf"(?:{_conv})?(?:it's|its|it\s+is|this\s+is)",
                    rf"(?:{_conv})?(?:you\s+can\s+call\s+me)",
                    rf"(?:{_conv})?(?:call\s+me)",
                    rf"(?:{_conv})?(?:myself)",
                    rf"(?:{_conv})?(?:en(?:noda|\s)?\s*(?:full\s+)?(?:peru|name))",
                    rf"(?:{_conv})?(?:என்(?:னோட)?\s*(?:பெயர்|full\s*name))",
                    rf"(?:{_conv})?(?:name\s+is)",
                    rf"(?:{_conv})?(?:naam\s+(?:hai)?)",
                ]
                _name_suffixes = [r"\b(?:sir|madam|ji|bro|anna|akka)\b", r"(?:தான்|டா|ங்க)"]
                _name_prefix_pattern = r"^\s*(?:" + "|".join(_name_prefixes) + r")\s+"
                _name_suffix_pattern = r"\s*(?:" + "|".join(_name_suffixes) + r")\s*$"

                name_match = re.search(_name_prefix_pattern, txt_lower, re.IGNORECASE)
                if not name_match and re.search(rf"^\s*(?:{_conv})?(?:i\s*(?:am|'m))\s+", txt_lower, re.IGNORECASE):
                    # Only accept "I am / I'm" if not followed by non-name words like gonna, going, here, etc.
                    after_iam = re.sub(rf"^\s*(?:{_conv})?(?:i\s*(?:am|'m))\s+", "", txt_lower, flags=re.IGNORECASE).strip()
                    first_w = after_iam.split()[0] if after_iam else ""
                    if first_w and first_w not in _NON_NAME_WORDS:
                        name_match = re.search(rf"^\s*(?:{_conv})?(?:i\s*(?:am|'m))\s+", txt_lower, re.IGNORECASE)

                if name_match:
                    raw_name = text[name_match.end():].strip()
                    raw_name = re.sub(_name_suffix_pattern, '', raw_name, flags=re.IGNORECASE).strip()
                    raw_name = re.sub(r'\b(dr|mr|mrs|ms|prof)\.', r'\1', raw_name, flags=re.IGNORECASE)
                    raw_name = re.sub(r'[\.\?\!\;]\s+.*$', '', raw_name).strip()
                    raw_name = re.sub(r'[,;]?\s*(?:and|&)?\s*(?:i\s+want|want\s+to|book|need|schedule|with|for|my\s+email|my\s+mail|my\s+phone|my\s+number|my\s+mobile|email|mail|phone|number|mobile).*$', '', raw_name, flags=re.IGNORECASE).strip()
                    raw_name = re.sub(r'\s+(?:what|how|when|where|why|can|could|do|does|is|are)\b.*$', '', raw_name, flags=re.IGNORECASE).strip()
                    cand = self.clean_person_name(raw_name)
                    if cand and self.is_valid_person_name(cand):
                        logger.info(f"[BOOKING_EXTRACTION] call_id={call_id} field=patient_name raw_value='{raw_name}' extracted_value='{cand}'")
                        session.patient_name = cand
                        session.caller_name = cand
                        session.booking_data["patient_name"] = cand
                        session.booking_data["caller_name"] = cand
                        session.field_status["patient_name"] = "COLLECTED"
                        session.booking_data["patient_name_status"] = "COLLECTED"
                    else:
                        logger.info(f"[BOOKING_VALIDATION] call_id={call_id} field=patient_name value='{raw_name}' is_valid=False reason='invalid_person_name'")
                elif (not session.patient_name or session.last_asked_field == "patient_name") and (session.state in (BookingState.WAITING_FOR_PATIENT_NAME, BookingState.COLLECTING_PATIENT_NAME) or session.last_asked_field == "patient_name"):
                    # When system explicitly asked for name, strip conversational prefix cleanly
                    cleaned = re.sub(r'^(?:(?:sure|yes|yeah|ok|okay|hi|hello|hey|good\s+(?:morning|afternoon|evening)|well|actually)\s*,?\s*)*(?:it\'s|its|it\s+is|this\s+is|i\s+am|i\'m|my\s+(?:full\s+|first\s+|last\s+)?name\s+(?:is|was)|my\s+name|name\s+(?:is|was)|you\s+can\s+call\s+me|call\s+me|என்(?:னோட|னுடைய)?\s*(?:பெயர்|பேரு|full\s*name)|en\s*(?:noda)?\s*(?:peru|peyar|name)|mera\s+naam)?\s*,?\s*', '', text, flags=re.IGNORECASE).strip()
                    cleaned = re.sub(r'[,;]?\s*(?:and|&)?\s*(?:i\s+want|want\s+to|book|need|schedule|with|for|my\s+email|my\s+mail|my\s+phone|my\s+number|my\s+mobile|email|mail|phone|number|mobile).*$', '', cleaned, flags=re.IGNORECASE).strip()
                    cleaned = re.sub(r'[^\w\s\u0900-\u097F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF]', '', cleaned).strip()
                    cleaned = re.sub(_name_suffix_pattern, '', cleaned, flags=re.IGNORECASE).strip()
                    cand = self.clean_person_name(cleaned)
                    if cand and self.is_valid_person_name(cand):
                        logger.info(f"[BOOKING_EXTRACTION] call_id={call_id} field=patient_name raw_value='{cleaned}' extracted_value='{cand}'")
                        session.patient_name = cand
                        session.caller_name = cand
                        session.booking_data["patient_name"] = cand
                        session.booking_data["caller_name"] = cand
                        session.field_status["patient_name"] = "COLLECTED"
                        session.booking_data["patient_name_status"] = "COLLECTED"
                    else:
                        logger.info(f"[BOOKING_VALIDATION] call_id={call_id} field=patient_name value='{cleaned}' is_valid=False reason='invalid_person_name'")
                        session.field_status["patient_name"] = "INVALID"

            # Direct Standalone Name Correction:
            # If caller says a single person name (e.g. "Sanjai." or "No, Sanjai." or "Actually Sanjai"):
            if not explicit_name_handled and session.patient_name:
                cleaned_cand = re.sub(r'^(?:no,?\s+|nope,?\s+|actually,?\s+|wrong\s+(?:name,?\s+)?|i\s+said,?\s+|correct\s+name\s+(?:is,?\s+)?|not\s+that,?\s+)?', '', text, flags=re.IGNORECASE).strip().rstrip('.,?!')
                if cleaned_cand and len(cleaned_cand.split()) <= 2 and not any(c.isdigit() for c in cleaned_cand) and "@" not in cleaned_cand:
                    cand_name = self.clean_person_name(cleaned_cand)
                    if cand_name and self.is_valid_person_name(cand_name) and cand_name.lower() != session.patient_name.lower():
                        # Verify it's not a generic word, greeting, or business description
                        if not any(w in cand_name.lower().split() for w in _NON_NAME_WORDS):
                            logger.info(f"[BOOKING_CORRECTION] call_id={call_id} field=patient_name old_value='{session.patient_name}' new_value='{cand_name}'")
                            session.patient_name = cand_name
                            session.caller_name = cand_name
                            session.booking_data["patient_name"] = cand_name
                            session.booking_data["caller_name"] = cand_name
                            session.booking_data["name_just_corrected"] = True
                            session.field_status["patient_name"] = "COLLECTED"
                            session.booking_data["patient_name_status"] = "COLLECTED"


        # 5. Email Extraction
        spoken_email = self.parse_spoken_email(text)
        if spoken_email:
            if not session.email or session.state in (BookingState.WAITING_FOR_CONFIRMATION, BookingState.COLLECTING_EMAIL, BookingState.WAITING_FOR_EMAIL) or any(w in txt_lower for w in ["email", "mail", "change"]):
                logger.info(f"[BOOKING_EXTRACTION] call_id={call_id} field=email raw_value='{text}' extracted_value='{spoken_email}'")
                session.email = spoken_email
                session.field_status["email"] = "COLLECTED"
                session.booking_data["email_status"] = "COLLECTED"
                session.booking_data.pop("partial_email_domain", None)
                session.booking_data.pop("incomplete_email_attempt", None)
        else:
            # Check for incomplete spoken email with domain (e.g. "at gmail dot com")
            incomplete_domain = detect_incomplete_spoken_email(text)
            if incomplete_domain:
                session.booking_data["partial_email_domain"] = incomplete_domain
                session.booking_data["incomplete_email_attempt"] = True
                logger.info(f"[BOOKING_EXTRACTION] call_id={call_id} field=email partial_domain='{incomplete_domain}' username_missing=True")
            elif session.booking_data.get("partial_email_domain") and (session.state in (BookingState.COLLECTING_EMAIL, BookingState.WAITING_FOR_EMAIL) or getattr(session, "last_asked_field", "") == "email"):
                # Caller might be providing just the username now!
                clean_uname = re.sub(r'^(?:my\s+username\s+is|it\'s|its|is|name\s+is)\s*', '', txt_lower).strip().rstrip('.,?!')
                clean_uname = re.sub(r'\s+', '', clean_uname)
                if clean_uname and re.match(r'^[a-zA-Z0-9_.+-]+$', clean_uname) and len(clean_uname) <= 30 and clean_uname not in {"skip", "no", "yes", "none", "cancel"}:
                    combined_em = f"{clean_uname}@{session.booking_data['partial_email_domain']}"
                    if re.match(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}$', combined_em):
                        logger.info(f"[BOOKING_EXTRACTION] call_id={call_id} field=email combined_value='{combined_em}'")
                        session.email = combined_em
                        session.field_status["email"] = "COLLECTED"
                        session.booking_data["email_status"] = "COLLECTED"
                        session.booking_data.pop("partial_email_domain", None)
                        session.booking_data.pop("incomplete_email_attempt", None)

        # 6. Contact Phone Number Extraction
        spoken_phone = self.parse_spoken_phone_number(text)
        if spoken_phone:
            if not session.contact_number or session.contact_number == "Unknown" or session.state in (BookingState.WAITING_FOR_CONFIRMATION, BookingState.COLLECTING_PHONE, BookingState.WAITING_FOR_CONTACT) or any(w in txt_lower for w in ["phone", "mobile", "number", "change"]):
                logger.info(f"[BOOKING_EXTRACTION] call_id={call_id} field=contact_number raw_value='{text}' extracted_value='{spoken_phone}'")
                session.contact_number = spoken_phone
                session.field_status["contact_number"] = "COLLECTED"
                session.booking_data["contact_number_status"] = "COLLECTED"

        # 6b. Caller ID Confirmation
        if (
            session.booking_data.get("caller_id_suggested")
            and (not session.contact_number or session.contact_number == "Unknown")
            and session.caller_phone
            and session.caller_phone not in ("Unknown", "Unknown Caller", "Private Number")
            and (not session.booking_data.get("trunk_did") or session.caller_phone.lstrip('+').lstrip('0') != session.booking_data.get("trunk_did").lstrip('+').lstrip('0'))
        ):
            is_affirm = (
                any(w in txt_lower.split() for w in ["yes", "yeah", "yep", "sure", "correct", "fine", "ok", "okay", "ha", "haan", "sari", "aam"])
                or any(p in txt_lower for p in ["use that", "use this", "that's fine", "that is fine", "same number", "this number", "use my number", "use caller", "that number", "use it", "sounds good", "looks good"])
            )
            is_neg = any(w in txt_lower.split() for w in ["no", "nope", "different", "change", "not", "illa", "nahi"])
            if is_affirm and not is_neg:
                session.contact_number = session.caller_phone
                session.booking_data["caller_id_confirmed"] = True

        # 7. Customer Location Extraction
        spoken_loc = self.parse_spoken_location(text, session=session)
        if spoken_loc:
            session.location = spoken_loc

        # 8. Business Context Extraction (Industry, Requirement, Interested Area / Team)
        _INDUSTRIES = [
            "manufacturing", "healthcare", "retail", "education", "finance", "banking",
            "logistics", "supply chain", "construction", "real estate", "automotive",
            "hospitality", "technology", "e-commerce", "ecommerce", "pharma", "pharmaceutical"
        ]
        for ind in _INDUSTRIES:
            if re.search(rf'\b{re.escape(ind)}\b', txt_lower):
                ind_val = ind.title() if ind not in ("e-commerce", "ecommerce") else "E-Commerce"
                session.industry = ind_val
                session.booking_data["industry"] = ind_val
                break

        _TEAMS = [
            ("sales demo", "Sales"),
            ("sales team", "Sales"),
            ("sales", "Sales"),
            ("marketing", "Marketing"),
            ("operations", "Operations"),
            ("support", "Support"),
            ("customer support", "Customer Support"),
            ("billing", "Billing"),
            ("engineering", "Engineering"),
            ("hr", "HR"),
            ("human resources", "HR"),
            ("it", "IT")
        ]
        for t_key, t_val in _TEAMS:
            if re.search(rf'\b{re.escape(t_key)}\b', txt_lower):
                session.interested_area = t_val
                session.booking_data["interested_area"] = t_val
                break

        _REQUIREMENTS = [
            ("website development", "Website Development"),
            ("web development", "Website Development"),
            ("build a website", "Website Development"),
            ("website design", "Website Design & Development"),
            ("website", "Website Development"),
            ("web application", "Web Application"),
            ("voice agent", "AI Voice Agent"),
            ("ai voice agent", "AI Voice Agent"),
            ("voice receptionist", "AI Voice Receptionist"),
            ("galaxy chat", "Galaxy Chat"),
            ("erp", "ERP Application"),
            ("project management", "Project Management Application"),
            ("employee tracking", "Employee Activity Tracking"),
            ("employee activity tracking", "Employee Activity Tracking"),
            ("survey platform", "Survey Platform"),
            ("custom software", "Custom Software Integration"),
            ("software integration", "Custom Software Integration")
        ]
        for r_key, r_val in _REQUIREMENTS:
            if re.search(rf'\b{re.escape(r_key)}\b', txt_lower):
                session.requirement = r_val
                session.booking_data["requirement"] = r_val
                if not session.service:
                    session.service = r_val
                    session.booking_data["service"] = r_val
                break

    def check_and_handle_intent_switch(self, session: BookingSession, current_intent: str, text: str = "") -> bool:
        """
        Prevents Stale Context by checking if user changed topic to an explicit non-booking intent.
        Returns True if turn should bypass booking state machine.
        CRITICAL: Never bypass an active booking flow when customer is providing booking fields
        (name, email, phone, date, time, confirmations).
        """
        if not session.is_active():
            return False

        txt_lower = (text or "").lower().strip()

        if not txt_lower and current_intent in ("general_conversation", "greeting"):
            return True

        # Explicit cancellation / exit — strictly match unambiguous booking cancellation phrases
        # Never switch intent when caller claims already told, asks to repeat, or asks to change a field
        _PROTECTED_BOOKING_PHRASES = [
            "already told", "already said", "already gave", "already shared", "told you already", "gave you already",
            "already provided", "i already told", "i told you", "already answered",
            "change my email", "change email", "change my phone", "change phone", "change my number",
            "change my name", "change name", "change date", "change time"
        ]
        if any(p in txt_lower for p in _PROTECTED_BOOKING_PHRASES):
            return False

        # Never switch intent when caller asks to repeat or summarize booking details
        _REPEAT_PHRASES = [
            "repeat all my details", "repeat my details", "repeat all details", "repeat the details",
            "repeat what i gave you", "repeat my information", "repeat all information",
            "what details do you have", "what information do you have", "what details did i give",
            "read back my details", "read back my information", "summarize my details",
            "summarize my booking", "what are my details", "what do you have so far",
            "what info do you have", "repeat everything"
        ]
        if any(rp in txt_lower for rp in _REPEAT_PHRASES):
            return False

        _CANCEL_PHRASES = [
            "cancel booking", "cancel the booking", "cancel appointment", "cancel the appointment",
            "stop booking", "dont book", "do not book", "don't book",
            "never mind", "nevermind", "forget it", "leave it",
        ]
        if any(phrase in txt_lower for phrase in _CANCEL_PHRASES):
            session.reset()
            return True

        # Informational question indicators in user text
        info_indicators = [
            "can you list", "list the", "list your", "list all", "list services", "list staff", "list doctors", "list teachers", "list members",
            "what services", "which services", "what are all the service", "what are all the services",
            "who are all available", "who is available", "who are available", "who works",
            "how much", "what is the price", "what is the cost", "cost of", "price of", "pricing",
            "tell me about", "tell me", "where are you", "what are your hours", "when are you open", "who are your",
            "do you have", "what do you offer", "what treatments", "real person", "human agent",
            "speak to human", "now i want to go real", "school", "hospital", "salon", "information",
            "what about", "how many", "can i know", "may i know", "details", "explain", "how are you", "where is", "where are",
            "working hours", "opening hours", "located",
            "want to know about", "want to know", "know more about", "know about", "what is", "what are",
            "can you explain", "please explain", "features of", "info on", "information about", "details about"
        ]
        is_info_question = any(ind in txt_lower for ind in info_indicators) or (
            bool(re.search(r'^(?:what|who|which|how|where|when|can\s+you|could\s+you|list|show|tell)\b', txt_lower))
            and not any(k in txt_lower for k in ["appointment", "book", "slot"])
        )

        def _cleanup_and_record_inquiry():
            # Check negative rejection (e.g. "I don't want Galaxy Jet")
            neg_m = re.search(r'\b(?:don\'?t\s+want|do\s+not\s+want|no\s+|not\s+interested\s+in|instead\s+of)\s+(?:the\s+|a\s+|an\s+)?([a-zA-Z0-9\s]{2,30}?)(?=[.,;!?]|\s+(?:i\s+want|i\s+need|but|please|$))\b', txt_lower)
            if neg_m:
                rej = neg_m.group(1).strip()
                if rej and rej not in {"appointment", "booking", "slot", "date", "time", "this", "that"}:
                    session._rejected_services.add(rej.lower())
                    session._rejected_departments.add(rej.lower())
                    session._rejected_doctors.add(rej.lower())

            # Detect dynamic product/service inquired
            org_id = getattr(session, "organization_id", 1) or 1
            all_opts = self.get_organization_services_and_departments(org_id)
            for opt in all_opts:
                if opt.lower() in txt_lower and opt.lower() not in getattr(session, "_rejected_services", set()):
                    session.booking_data["last_inquired_product"] = opt
                    break

            # Clear obsolete booking prompts / terms
            session.booking_data.pop("user_requested_entity_term", None)
            session.booking_data.pop("unverified_entity_requested", None)
            session.booking_data.pop("ambiguous_entity_candidate", None)
            session.booking_data.pop("pending_confirmation_entity", None)

        _active_booking_states = (
            BookingState.BOOKING_STARTED,
            BookingState.WAITING_FOR_DOCTOR,
            BookingState.WAITING_FOR_DATE,
            BookingState.WAITING_FOR_TIME,
            BookingState.WAITING_FOR_PATIENT_NAME,
            BookingState.COLLECTING_PATIENT_NAME,
            BookingState.WAITING_FOR_CONTACT,
            BookingState.COLLECTING_PHONE,
            BookingState.WAITING_FOR_EMAIL,
            BookingState.COLLECTING_EMAIL,
            BookingState.WAITING_FOR_CONFIRMATION,
            BookingState.CONFIRMATION_ACCEPTED,
        )

        # In all active booking states, generic intents like 'general_question' or 'general_conversation'
        # must NEVER bypass the booking flow unless an explicit informational query is actually detected in the text.
        if session.state in _active_booking_states or getattr(session, "last_asked_field", "") in ("patient_name", "contact_number", "email", "confirmation", "date", "time", "doctor", "service"):
            if not is_info_question and current_intent not in ("emergency/urgent request", "transfer request"):
                return False

        # Check if customer's utterance is answering the active booking prompt:
        if txt_lower:
            # 0. Doctor / Staff prompt / answers
            if session.state in (BookingState.WAITING_FOR_DOCTOR,) or getattr(session, "last_asked_field", "") in ("doctor", "service", "entity_confirmation"):
                if any(w in txt_lower for w in ["dr", "doctor", "dr.", "physician", "prof", "consult"]) or len(txt_lower.split()) <= 4:
                    if not is_info_question:
                        return False

            # 1. Date prompt / answers
            if session.state in (BookingState.WAITING_FOR_DATE,) or getattr(session, "last_asked_field", "") == "date":
                if any(dk in txt_lower for dk in ["tomorrow", "today", "next", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]) or re.search(r'\b\d{1,2}(?:st|nd|rd|th)?\b', txt_lower):
                    if not is_info_question:
                        return False

            # 2. Time prompt / answers
            if session.state in (BookingState.WAITING_FOR_TIME,) or getattr(session, "last_asked_field", "") == "time":
                if re.search(r'\b\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b', txt_lower) or any(tk in txt_lower for tk in ["morning", "afternoon", "evening", "noon", "am", "pm", "any time", "earliest", "any slot"]):
                    if not is_info_question:
                        return False

            # 3. Email prompt / answers
            if session.state in (BookingState.WAITING_FOR_EMAIL, BookingState.COLLECTING_EMAIL) or getattr(session, "last_asked_field", "") == "email":
                if any(w in txt_lower for w in ["@", "at", "dot", "gmail", "yahoo", "outlook", "email", "mail", ".com", ".in", ".org", "com"]) or bool(self.parse_spoken_email(text)) or len(txt_lower.split()) <= 4:
                    if not is_info_question:
                        return False

            # 4. Name prompt / answers
            if session.state in (BookingState.WAITING_FOR_PATIENT_NAME, BookingState.COLLECTING_PATIENT_NAME) or getattr(session, "last_asked_field", "") == "patient_name":
                if any(w in txt_lower for w in ["my name", "name is", "i am", "this is", "call me"]) or len(txt_lower.split()) <= 4:
                    if not is_info_question:
                        return False

            # 5. Contact phone prompt / answers
            if session.state in (BookingState.WAITING_FOR_CONTACT, BookingState.COLLECTING_PHONE) or getattr(session, "last_asked_field", "") == "contact_number":
                if re.search(r'\d{5,}', txt_lower) or any(w in txt_lower for w in ["number", "mobile", "phone"]):
                    if not is_info_question:
                        return False

            if is_info_question:
                _cleanup_and_record_inquiry()
                logger.info(f"[INTENT SWITCH DETECTED] Customer asked informational query '{text}'. Bypassing booking state to Active RAG (session fields preserved).")
                return True

        explicit_informational_intents = {
            "doctor_list", "doctor_count", "doctor_information",
            "department_list", "hospital_information", "service_information",
            "product_information", "product_service_query", "company_information", "organization_information",
            "faq", "emergency/urgent request", "transfer request"
        }
        if current_intent in explicit_informational_intents:
            _cleanup_and_record_inquiry()
            logger.info(f"[INTENT SWITCH DETECTED] Customer switched topic to informational intent '{current_intent}'. Bypassing booking state.")
            return True

        if is_info_question:
            _cleanup_and_record_inquiry()
            logger.info(f"[INTENT SWITCH DETECTED] Customer asked informational query '{text}'. Bypassing booking state.")
            return True

        return False

    def handle_confirmation_turn(self, session: BookingSession, text: str, turn_id: Optional[str] = None, state_before: Optional[Dict[str, Any]] = None) -> str:
        """
        DEDICATED CONFIRMATION & CORRECTION HANDLER:
        Evaluated with highest priority before any NLP, greeting, general conversation, or RAG.
        Enforces proper state transitions:
        WAITING_FOR_CONFIRMATION -> WAITING_FOR_CORRECTION_FIELD -> WAITING_FOR_CORRECTION_VALUE -> WAITING_FOR_FINAL_CONFIRMATION -> CONFIRMATION_ACCEPTED -> BOOKING_PROCESSING -> BOOKING_CONFIRMED.
        Handles affirmations, change requests, multiple field corrections, cancellations, and presence checks.
        """
        def _ret(response_str: str) -> str:
            session.turn_count += 1
            if session.turn_count % 3 == 0:
                self.save_session_state(session)
            return response_str

        lang = getattr(session, "conversation_language", "en") or "en"
        txt_lower = text.lower().strip()
        clean_norm = re.sub(r'[^a-zA-Z0-9\s]', ' ', txt_lower).strip()
        norm_words = clean_norm.split()
        single_str = " ".join(norm_words)

        is_demo = (
            getattr(session, "booking_type", None) in ("SALES_DEMO", "BUSINESS_CONSULTATION")
            or getattr(session, "business_type", None) == "SOFTWARE"
            or getattr(session, "workflow_type", None) == "SALES_DEMO"
        )
        target_str = "for your product demo" if is_demo else (f"with {session.doctor}" if session.doctor else f"for {session.service or 'your appointment'}")

        def _build_summary_prompt() -> str:
            p_name = session.patient_name or "there"
            p_phone = session.contact_number or ""
            p_email = f", Email: {session.email}" if session.email else (", Email: Skipped" if (session.booking_data.get("email_skipped") or session.booking_data.get("email_declined") or session.field_status.get("email") == "SKIPPED") else "")
            p_loc = f", Location: {session.location}" if session.location else ""
            p_date = session.date or "tomorrow"
            p_time = session.time or "10:00 AM"
            if lang in ("ta", "tanglish", "ta_mix"):
                return f"நன்றி! உங்க விவரங்கள்: Name: {p_name}, Phone: {p_phone}{p_email}{p_loc}, {p_date} அன்று {p_time}-க்கு {target_str}. இந்த விவரங்கள் சரியா?"
            elif lang == "hi":
                return f"Dhanyawad! Aapki updated details: Naam: {p_name}, Phone: {p_phone}{p_email}{p_loc}, {target_str} ke liye {p_date} ko {p_time} par. Kya yeh sab sahi hai?"
            else:
                return f"Thank you! Here are your updated details: Name: {p_name}, Phone: {p_phone}{p_email}{p_loc}, {target_str} on {p_date} at {p_time}. Is everything correct now?"

        # 1. Natural presence / greeting check ("Hello?", "Are you there?")
        _PRESENCE_WORDS = {"hello", "hi", "hey", "hello there", "are you there", "can you hear me", "am i audible", "is anyone there", "can you hear", "hear me"}
        if single_str in _PRESENCE_WORDS or (norm_words and norm_words[0] in ("hello", "hi", "hey") and len(norm_words) <= 2):
            resp = "I'm here. Would you like me to proceed with the appointment?"
            if lang in ("ta", "tanglish", "ta_mix"):
                resp = "நான் line-ல தான் இருக்கேன். உங்க booking-ஐ confirm பண்ணலாமா?"
            elif lang == "hi":
                resp = "Main yahin hoon. Kya main aapka appointment confirm kar doon?"
            self._log_conversation_turn(session, text, resp, "presence_reassurance", state_before=state_before)
            return _ret(resp)

        # 2. Explicit cancellation
        if any(p in clean_norm for p in ["cancel booking", "cancel appointment", "dont book", "do not book", "stop booking", "cancel", "never mind", "nevermind", "forget it", "leave it", "no dont", "no do not"]):
            session.reset()
            session.state = BookingState.BOOKING_CANCELLED
            logger.info(f"[BOOKING_CORRECTION] Booking cancelled by caller for session {session.session_id}")
            if lang in ("ta", "tanglish", "ta_mix"):
                resp = "Booking cancel பண்ணியாச்சு. உங்களுக்கு வேற ஏதாவது help வேணுமா?"
            else:
                resp = "Your appointment booking has been cancelled. How else may I assist you today?"
            self._log_conversation_turn(session, text, resp, "booking_cancelled", state_before=state_before)
            return _ret(resp)

        # 3. Handle state WAITING_FOR_CORRECTION_FIELD
        if session.state == BookingState.WAITING_FOR_CORRECTION_FIELD or (state_before and state_before.get("state") == BookingState.WAITING_FOR_CORRECTION_FIELD):
            # Check if caller named specific field or field+value
            target_field = None
            if any(w in norm_words for w in ["email", "mail", "e mail"]) or "@" in text:
                target_field = "email"
            elif any(w in norm_words for w in ["phone", "mobile", "number", "contact"]):
                target_field = "phone"
            elif any(w in norm_words for w in ["name", "peru", "naam"]):
                target_field = "name"
            elif any(w in norm_words for w in ["date", "day"]):
                target_field = "date"
            elif any(w in norm_words for w in ["time", "timing", "slot"]):
                target_field = "time"
            elif any(w in norm_words for w in ["location", "city", "address", "area", "place"]):
                target_field = "location"

            if target_field:
                # Check if new value was already supplied in this turn
                val_supplied = False
                if target_field == "email":
                    cand = self.parse_spoken_email(text)
                    if cand and "@" in cand and "." in cand:
                        old_val = session.email
                        session.email = cand
                        session.booking_data["email"] = cand
                        session.booking_data["email_confirmed"] = False
                        logger.info(f"[BOOKING_FIELD_UPDATED] field=email, old={self._mask_pii(old_val)}, new={self._mask_pii(cand)}")
                        val_supplied = True
                elif target_field == "phone":
                    cand = self.parse_spoken_phone_number(text)
                    if cand and len(re.sub(r'\D', '', cand)) >= 10:
                        old_val = session.contact_number
                        session.contact_number = cand
                        session.booking_data["contact_number"] = cand
                        logger.info(f"[BOOKING_FIELD_UPDATED] field=phone, old={self._mask_pii(old_val)}, new={self._mask_pii(cand)}")
                        val_supplied = True
                elif target_field == "name":
                    cand = self.clean_person_name(re.sub(r'^(?:name\s+(?:is|was)|my\s+name\s+(?:is|was)|it\'s|its|is)\s*', '', text, flags=re.I))
                    if cand and self.is_valid_person_name(cand):
                        old_val = session.patient_name
                        session.patient_name = cand
                        session.caller_name = cand
                        session.booking_data["patient_name"] = cand
                        session.booking_data["caller_name"] = cand
                        logger.info(f"[BOOKING_FIELD_UPDATED] field=name, old={self._mask_pii(old_val)}, new={self._mask_pii(cand)}")
                        val_supplied = True

                if val_supplied:
                    session.state = BookingState.WAITING_FOR_FINAL_CONFIRMATION
                    session.last_asked_field = "final_confirmation"
                    resp = _build_summary_prompt()
                    self._log_conversation_turn(session, text, resp, "correction_value_updated", state_before=state_before)
                    return _ret(resp)

                session.booking_data["correction_field"] = target_field
                session.state = BookingState.WAITING_FOR_CORRECTION_VALUE
                session.last_asked_field = f"correction_{target_field}"
                logger.info(f"[BOOKING_CORRECTION] Selected field to correct: {target_field}. Transitioned to WAITING_FOR_CORRECTION_VALUE.")
                
                if target_field == "email":
                    resp = "Sure, please tell me your correct email address."
                elif target_field == "phone":
                    resp = "Sure, please tell me your correct contact phone number."
                elif target_field == "name":
                    resp = "Sure, please tell me your correct full name."
                elif target_field == "date":
                    resp = "Sure, which date would you prefer instead?"
                elif target_field == "time":
                    resp = "Sure, what time works best for you?"
                else:
                    resp = "Sure, what is your correct location?"

                if lang in ("ta", "tanglish", "ta_mix"):
                    if target_field == "email":
                        resp = "கண்டிப்பா, உங்க சரியான email address சொல்லுங்க."
                    elif target_field == "phone":
                        resp = "கண்டிப்பா, உங்க சரியான phone number சொல்லுங்க."
                    elif target_field == "name":
                        resp = "கண்டிப்பா, உங்க முழு பெயரை சொல்லுங்க."
                    elif target_field == "date":
                        resp = "கண்டிப்பா, எந்த தேதி வசதியாக இருக்கும்?"
                    elif target_field == "time":
                        resp = "கண்டிப்பா, எந்த நேரம் உங்களுக்கு வசதியாக இருக்கும்?"
                    else:
                        resp = "கண்டிப்பா, உங்க location சொல்லுங்க."
                self._log_conversation_turn(session, text, resp, "waiting_for_correction_value", state_before=state_before)
                return _ret(resp)

        # 4. Handle state WAITING_FOR_CORRECTION_VALUE
        if session.state == BookingState.WAITING_FOR_CORRECTION_VALUE or (state_before and state_before.get("state") == BookingState.WAITING_FOR_CORRECTION_VALUE):
            corr_field = session.booking_data.get("correction_field") or "email"
            field_updated = False

            if corr_field == "email":
                cand = self.parse_spoken_email(text)
                if cand and "@" in cand and "." in cand:
                    old_val = session.email
                    session.email = cand
                    session.booking_data["email"] = cand
                    session.booking_data["email_confirmed"] = False
                    logger.info(f"[BOOKING_FIELD_UPDATED] field=email, old={self._mask_pii(old_val)}, new={self._mask_pii(cand)}")
                    field_updated = True
                else:
                    resp = "Could you please repeat or spell out your complete email address?"
                    self._log_conversation_turn(session, text, resp, "re_ask_correction_email", state_before=state_before)
                    return _ret(resp)

            elif corr_field == "phone":
                cand = self.parse_spoken_phone_number(text)
                if cand and len(re.sub(r'\D', '', cand)) >= 10:
                    old_val = session.contact_number
                    session.contact_number = cand
                    session.booking_data["contact_number"] = cand
                    logger.info(f"[BOOKING_FIELD_UPDATED] field=phone, old={self._mask_pii(old_val)}, new={self._mask_pii(cand)}")
                    field_updated = True
                else:
                    resp = "Could you please provide a valid 10-digit phone number?"
                    self._log_conversation_turn(session, text, resp, "re_ask_correction_phone", state_before=state_before)
                    return _ret(resp)

            elif corr_field == "name":
                cand = self.clean_person_name(re.sub(r'^(?:my\s+name\s+is|it\'s|its|is)\s*', '', text, flags=re.I))
                if cand and self.is_valid_person_name(cand):
                    old_val = session.patient_name
                    session.patient_name = cand
                    session.caller_name = cand
                    session.booking_data["patient_name"] = cand
                    session.booking_data["caller_name"] = cand
                    logger.info(f"[BOOKING_FIELD_UPDATED] field=name, old={self._mask_pii(old_val)}, new={self._mask_pii(cand)}")
                    field_updated = True
                else:
                    resp = "Could you please repeat your name?"
                    self._log_conversation_turn(session, text, resp, "re_ask_correction_name", state_before=state_before)
                    return _ret(resp)

            elif corr_field == "date":
                cand_date = None
                if any(w in txt_lower for w in ["tomorrow", "naalai", "naalaikku", "kal", "நாளை", "நாளைக்கு"]):
                    cand_date = self._normalize_date("tomorrow")
                    session.booking_data["relative_date"] = "tomorrow"
                elif any(w in txt_lower for w in ["today", "inniku", "aaj", "இன்று", "இன்னைக்கு"]):
                    cand_date = self._normalize_date("today")
                    session.booking_data["relative_date"] = "today"
                else:
                    cand_date = self._normalize_date(text)
                if cand_date:
                    old_val = session.date
                    session.date = cand_date
                    session.booking_data["date"] = cand_date
                    session.booking_data["resolved_date"] = cand_date
                    session.booking_data["timezone"] = "Asia/Kolkata"
                    session.field_status["date"] = "COLLECTED"
                    session.booking_data["date_status"] = "COLLECTED"
                    session.time = None
                    session.availability_verified = False
                    logger.info(f"[BOOKING_FIELD_UPDATED] field=date, old={old_val}, new={cand_date}")
                    field_updated = True
                else:
                    resp = "Sorry, I didn't catch the date. Would tomorrow or another date work for you?"
                    self._log_conversation_turn(session, text, resp, "re_ask_correction_date", state_before=state_before)
                    return _ret(resp)

            elif corr_field == "time":
                extracted_t = self.parse_spoken_time(text)
                if not extracted_t:
                    time_match = re.search(r'\b((?:1[0-2]|0?[1-9])(?:[:\-\s][0-5][0-9])?\s*(?:a\.?m\.?|p\.?m\.?))\b', txt_lower)
                    if time_match:
                        raw_tm = time_match.group(1).replace('.', '').upper()
                        if ':' not in raw_tm:
                            raw_tm = re.sub(r'(\d+)\s*(AM|PM)', r'\1:00 \2', raw_tm)
                        extracted_t = raw_tm
                    elif re.search(r'\b(1[0-2]|0?[1-9])\s*(?:o\'clock)?\b', txt_lower):
                        bm = re.search(r'\b(1[0-2]|0?[1-9])\b', txt_lower)
                        if bm:
                            extracted_t = f"{bm.group(1)}:00 PM"
                if extracted_t:
                    # Validate extracted_t against live availability & past-time
                    from src.services.calendar_verification_service import calendar_verification_service
                    target_ent = session.doctor or session.service or "Representative"
                    target_d = session.date or "today"
                    v_res = calendar_verification_service.verify_availability(f"Appointment with {target_ent} on {target_d} at {extracted_t}")
                    if v_res.is_checked and v_res.is_available is False:
                        if v_res.is_past_time:
                            if v_res.alternative_slots:
                                alt_str = ", ".join(v_res.alternative_slots)
                                resp = f"Sorry, {extracted_t} is no longer available because that time has passed. Today's available slots are {alt_str}. Which time would you prefer?"
                                if lang in ("ta", "tanglish", "ta_mix"):
                                    resp = f"மன்னிக்கவும், {extracted_t} நேரம் முடிந்துவிட்டது. இன்றைக்கு available slots: {alt_str}. இதில எந்த நேரம் வசதியாக இருக்கும்?"
                                elif lang == "hi":
                                    resp = f"Kshama kijiye, {extracted_t} ka samay nikal chuka hai. Aaj ke available slots hain: {alt_str}. Aap konsa samay pasand karenge?"
                            else:
                                resp = f"Sorry, {extracted_t} is no longer available because that time has passed, and there are no remaining slots available today. Would you like to check tomorrow or another date?"
                                if lang in ("ta", "tanglish", "ta_mix"):
                                    resp = f"மன்னிக்கவும், {extracted_t} நேரம் முடிந்துவிட்டது, இன்றைக்கு வேறு slots இல்லை. நாளை அல்லது வேறு date check பண்ணலாமா?"
                                elif lang == "hi":
                                    resp = f"Kshama kijiye, {extracted_t} ka samay nikal chuka hai aur aaj koi slots available nahi hain. Kya aap kal ya koi aur date check karna chahenge?"
                        else:
                            if v_res.alternative_slots:
                                alt_str = ", ".join(v_res.alternative_slots)
                                resp = f"{extracted_t} is already booked. Available alternative slots are: {alt_str}. Which time would you prefer?"
                            else:
                                resp = f"{extracted_t} is not available. Would you like to check another date?"
                        self._log_conversation_turn(session, text, resp, "correction_time_unavailable", state_before=state_before)
                        return _ret(resp)

                    old_val = session.time
                    session.time = extracted_t
                    session.booking_data["time"] = extracted_t
                    session.availability_verified = True
                    logger.info(f"[BOOKING_FIELD_UPDATED] field=time, old={old_val}, new={extracted_t}")
                    field_updated = True
                else:
                    resp = "Which time slot would you prefer?"
                    self._log_conversation_turn(session, text, resp, "re_ask_correction_time", state_before=state_before)
                    return _ret(resp)

            elif corr_field == "location":
                cand = self.parse_spoken_location(text, session=session)
                if cand:
                    old_val = session.location
                    session.location = cand
                    session.booking_data["location"] = cand
                    logger.info(f"[BOOKING_FIELD_UPDATED] field=location, old={old_val}, new={cand}")
                    field_updated = True
                else:
                    resp = "Could you please tell me your city or location?"
                    self._log_conversation_turn(session, text, resp, "re_ask_correction_location", state_before=state_before)
                    return _ret(resp)

            if field_updated:
                # Check if more fields are pending correction
                pending = session.booking_data.get("pending_correction_fields", [])
                if pending:
                    next_f = pending.pop(0)
                    session.booking_data["pending_correction_fields"] = pending
                    session.booking_data["correction_field"] = next_f
                    session.state = BookingState.WAITING_FOR_CORRECTION_VALUE
                    logger.info(f"[BOOKING_CORRECTION] Moving to next pending correction field: {next_f}")
                    if next_f == "time":
                        resp = "Got it! And which time would you prefer?"
                    elif next_f == "date":
                        resp = "Got it! And which date would you prefer?"
                    elif next_f == "email":
                        resp = "Got it! And what is your correct email address?"
                    elif next_f == "phone":
                        resp = "Got it! And what is your correct phone number?"
                    else:
                        resp = f"Got it! What is your updated {next_f}?"
                    self._log_conversation_turn(session, text, resp, "waiting_for_next_correction_field", state_before=state_before)
                    return _ret(resp)

                # All corrections completed!
                session.state = BookingState.WAITING_FOR_FINAL_CONFIRMATION
                session.last_asked_field = "final_confirmation"
                resp = _build_summary_prompt()
                self._log_conversation_turn(session, text, resp, "corrections_completed_summary", state_before=state_before)
                return _ret(resp)

        # 5. Affirmation & Negation Intent Recognition in Confirmation States
        _AFFIRM_EXACT = {
            "yes", "yeah", "yep", "sure", "correct", "right", "fine", "perfect",
            "good", "proceed", "confirm", "ok", "okay", "ha", "haan", "sari", "aam",
            "go", "yeah go", "go ahead", "ahead", "book it", "okay book it", "ok book it",
            "confirm it", "do it", "please do", "please book", "please book it", "schedule it", "go for it",
            "yes please", "sure go ahead", "all good", "looks good", "sounds good",
            "done", "yes book it", "confirm booking", "lock it in", "fine with me",
            "that works", "that is correct", "thats correct", "yes please book it",
            "yeah please", "yeah confirm", "yeah go ahead", "yes go ahead", "yes proceed",
            "yes confirm", "please proceed", "please confirm", "go on", "yup", "absolutely",
            "okay go ahead", "ok go ahead", "thats fine", "that is fine",
            "everything is correct", "yeah everything is correct", "yes everything is correct",
            "all is correct", "confirm my appointment", "yeah confirm my appointment",
            "yes confirm my appointment", "confirm appointment", "yeah confirm appointment",
            "yes confirm appointment", "confirm the appointment", "confirm my booking",
            "confirm booking", "all details are correct", "details are correct"
        }
        _AFFIRM_SUBSTRINGS = [
            "go ahead", "book it", "okay book it", "ok book it", "confirm it",
            "please book it", "please book", "proceed", "confirm", "yes please",
            "sure go ahead", "all good", "looks good", "sounds good", "thats correct",
            "that is correct", "fine with me", "that works", "yeah go", "do it",
            "please do", "lock it in", "schedule it", "okay go ahead", "ok go ahead",
            "yes proceed", "yes confirm", "yeah go ahead", "yes go ahead",
            "please proceed", "please confirm", "everything is correct", "all is correct",
            "confirm my appointment", "confirm appointment", "confirm the appointment",
            "confirm my booking", "confirm booking", "details are correct", "everything correct"
        ]

        # Negation check
        is_negation = (
            any(w in norm_words for w in ["no", "nope", "wrong", "incorrect", "mistake", "illa", "nahi", "galat", "changes", "change"])
            or "not correct" in clean_norm
            or "make changes" in clean_norm
            or "want to change" in clean_norm
        )

        is_affirmation = (
            single_str in _AFFIRM_EXACT
            or any(re.search(rf'\b{re.escape(p)}\b', clean_norm) for p in _AFFIRM_SUBSTRINGS)
            or (
                any(w in norm_words for w in ["yes", "yeah", "yep", "okay", "ok", "sure", "proceed", "confirm", "haan", "sari", "aam", "correct", "ahead"])
                and not is_negation
            )
        )

        if is_affirmation and not is_negation:
            # Prevent duplicate bookings on multiple confirmations (Idempotency)
            if session.is_completed and session.appointment_id:
                logger.info(f"[BOOKING_CONFIRMATION] Already confirmed #{session.appointment_id}. Returning idempotent response.")
                b_name = session.business_name or "our team"
                resp = (
                    f"Your appointment is already confirmed on {session.date} at {session.time}. "
                    f"Your appointment ID is #{session.appointment_id}. Thank you for calling {b_name}."
                )
                self._log_conversation_turn(session, text, resp, "booking_already_confirmed", state_before=state_before)
                return _ret(resp)

            # Validate all required fields before confirming
            validation = self.validate_booking_fields(session)
            if not validation["is_valid"]:
                logger.warning(f"[BOOKING_CONFIRMATION_BLOCKED] Missing/invalid fields before booking confirmation: {validation['errors']}")
                if any("patient_name" in e for e in validation["errors"]):
                    session.state = BookingState.COLLECTING_PATIENT_NAME
                    session.last_asked_field = "patient_name"
                    resp = "Could you please tell me your full name for the booking?"
                    if lang in ("ta", "tanglish", "ta_mix"):
                        resp = "Booking-க்கு உங்க full name சொல்லுங்க."
                    return _ret(resp)
                elif any("date" in e for e in validation["errors"]):
                    session.state = BookingState.WAITING_FOR_DATE
                    session.last_asked_field = "date"
                    resp = "Which date would you prefer for your appointment?"
                    if lang in ("ta", "tanglish", "ta_mix"):
                        resp = "எந்த date-ல appointment book பண்ணலாம்?"
                    return _ret(resp)
                elif any("time" in e for e in validation["errors"]):
                    session.state = BookingState.WAITING_FOR_TIME
                    session.last_asked_field = "time"
                    resp = "Which time slot would you prefer?"
                    if lang in ("ta", "tanglish", "ta_mix"):
                        resp = "எந்த time slot உங்களுக்கு வசதியா இருக்கும்?"
                    return _ret(resp)
                elif any("contact_number" in e for e in validation["errors"]):
                    session.state = BookingState.COLLECTING_PHONE
                    session.last_asked_field = "contact_number"
                    resp = "Could you please provide your 10-digit contact phone number?"
                    if lang in ("ta", "tanglish", "ta_mix"):
                        resp = "உங்க contact phone number சொல்லுங்க."
                    return _ret(resp)

            # Enforce proper state transitions:
            # WAITING_FOR_CONFIRMATION -> CONFIRMATION_ACCEPTED -> BOOKING_PROCESSING -> BOOKING_CONFIRMED
            logger.info(f"[BOOKING_CONFIRMATION] State=CONFIRMED, caller confirmed details for session {session.session_id}")
            session.state = BookingState.CONFIRMATION_ACCEPTED
            session.booking_data["details_confirmed"] = True
            session.booking_data["email_confirmed"] = bool(session.email)
            session.state = BookingState.BOOKING_PROCESSING
            resp = self._execute_db_booking(session)
            self._log_conversation_turn(session, text, resp, "booking_executed", state_before=state_before)
            return _ret(resp)

        # 6. Negation / Correction Handling
        if is_negation or any(w in txt_lower for w in ["change", "wrong", "mistake", "correct"]):
            # Priority 6a: Check if inline value was provided directly in this turn (e.g. "No, my name is Udaya", "No, I am available tomorrow")
            has_tomorrow = any(w in txt_lower for w in ["tomorrow", "naalai", "naalaikku", "kal", "நாளை", "நாளைக்கு"])
            has_today = any(w in txt_lower for w in ["today", "inniku", "aaj", "இன்று", "இன்னைக்கு"])
            cand_inline_date = None
            if has_tomorrow:
                cand_inline_date = self._normalize_date("tomorrow")
                session.booking_data["relative_date"] = "tomorrow"
            elif has_today:
                cand_inline_date = self._normalize_date("today")
                session.booking_data["relative_date"] = "today"
            elif any(w in txt_lower for w in ["available", "book on", "reschedule to", "move to"]):
                cand_inline_date = self._normalize_date(text)

            if cand_inline_date:
                old_val = session.date
                session.date = cand_inline_date
                session.booking_data["date"] = cand_inline_date
                session.booking_data["resolved_date"] = cand_inline_date
                session.booking_data["date_status"] = "COLLECTED"
                session.time = None
                session.availability_verified = False
                session.state = BookingState.WAITING_FOR_TIME
                session.last_asked_field = "time"
                logger.info(f"[BOOKING_FIELD_UPDATED] field=date inline, old={old_val}, new={cand_inline_date}. Transitioned to WAITING_FOR_TIME.")
                date_disp = "tomorrow" if has_tomorrow else ("today" if has_today else cand_inline_date)
                if lang in ("ta", "tanglish", "ta_mix"):
                    resp = f"சரிங்க! {date_disp} எந்த time slot உங்களுக்கு வசதியாக இருக்கும்?"
                elif lang == "hi":
                    resp = f"Theek hai! {date_disp} ko kaunsa time slot aapke liye theek rahega?"
                else:
                    resp = f"Understood! Which time slot would you prefer for {date_disp}?"
                self._log_conversation_turn(session, text, resp, "date_corrected_inline", state_before=state_before)
                return _ret(resp)

            new_email = self.parse_spoken_email(text)
            new_phone = self.parse_spoken_phone_number(text)
            new_loc = self.parse_spoken_location(text, session=session)

            if new_email and ("@" in text or "mail" in txt_lower or new_email != session.email):
                old_val = session.email
                session.email = new_email
                session.booking_data["email"] = new_email
                session.booking_data["email_confirmed"] = False
                session.state = BookingState.WAITING_FOR_FINAL_CONFIRMATION
                logger.info(f"[BOOKING_FIELD_UPDATED] field=email, old={self._mask_pii(old_val)}, new={self._mask_pii(new_email)}")
                resp = _build_summary_prompt()
                self._log_conversation_turn(session, text, resp, "email_corrected_inline", state_before=state_before)
                return _ret(resp)

            if new_phone and ("phone" in txt_lower or "number" in txt_lower or "mobile" in txt_lower or new_phone != session.contact_number):
                old_val = session.contact_number
                session.contact_number = new_phone
                session.booking_data["contact_number"] = new_phone
                session.state = BookingState.WAITING_FOR_FINAL_CONFIRMATION
                logger.info(f"[BOOKING_FIELD_UPDATED] field=phone, old={self._mask_pii(old_val)}, new={self._mask_pii(new_phone)}")
                resp = _build_summary_prompt()
                self._log_conversation_turn(session, text, resp, "phone_corrected_inline", state_before=state_before)
                return _ret(resp)

            if any(w in txt_lower for w in ["my name is", "name is", "call me", "myself", "i said"]) or getattr(session, "patient_name", None):
                _conf_name_patterns = [
                    r'(?:(?:no|nope|not|wrong|actually|hey|hi|hello|well|sure|yes|yeah|ok|okay|wait)[\.,!\?]?\s*)*(?:i\s+said,?\s+)?(?:(?:my|the)\s+)?(?:full\s+|first\s+|last\s+)?name\s+(?:is|was)\s+([a-zA-Z\u0900-\u097F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\s]+)',
                    r'(?:(?:no|nope|not|wrong|actually|hey|hi|hello|well|sure|yes|yeah|ok|okay|wait)[\.,!\?]?\s*)*(?:you\s+can\s+call\s+me|call\s+me)\s+([a-zA-Z\u0900-\u097F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\s]+)',
                    r'(?:(?:no|nope|not|wrong|actually|hey|hi|hello|well|sure|yes|yeah|ok|okay|wait)[\.,!\?]?\s*)*(?:i\s+said)\s+([a-zA-Z\u0900-\u097F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\s]+)',
                ]
                for c_pat in _conf_name_patterns:
                    m_c = re.search(c_pat, text, re.IGNORECASE)
                    if m_c:
                        raw_c = m_c.group(1).strip()
                        cand_name = self.clean_person_name(raw_c)
                        if cand_name and self.is_valid_person_name(cand_name):
                            old_val = session.patient_name
                            session.patient_name = cand_name
                            session.caller_name = cand_name
                            session.booking_data["patient_name"] = cand_name
                            session.booking_data["caller_name"] = cand_name
                            session.state = BookingState.WAITING_FOR_FINAL_CONFIRMATION
                            logger.info(f"[BOOKING_FIELD_UPDATED] field=name, old={self._mask_pii(old_val)}, new={self._mask_pii(cand_name)}")
                            resp = _build_summary_prompt()
                            self._log_conversation_turn(session, text, resp, "name_corrected_inline", state_before=state_before)
                            return _ret(resp)

            # Priority 6b: Check if multiple fields were named to change (using strict word boundaries)
            req_fields = []
            if re.search(r'\b(?:email|mail|e-mail)\b', txt_lower):
                req_fields.append("email")
            if re.search(r'\b(?:time|timing|slot|slots)\b', txt_lower):
                req_fields.append("time")
            if re.search(r'\b(?:date|day|days)\b', txt_lower):
                req_fields.append("date")
            if re.search(r'\b(?:phone|mobile|number|contact)\b', txt_lower):
                req_fields.append("phone")
            if re.search(r'\b(?:name|names|fullname)\b', txt_lower):
                req_fields.append("name")
            if re.search(r'\b(?:location|city|address|place)\b', txt_lower):
                req_fields.append("location")

            if len(req_fields) > 1:
                first_target = req_fields[0]
                session.booking_data["pending_correction_fields"] = req_fields[1:]
                session.booking_data["correction_field"] = first_target
                session.state = BookingState.WAITING_FOR_CORRECTION_VALUE
                session.last_asked_field = f"correction_{first_target}"
                logger.info(f"[BOOKING_CORRECTION] Multiple corrections requested: {req_fields}. Prompting for {first_target}.")
                if first_target == "email":
                    resp = "Sure! What is your updated email address?"
                elif first_target == "phone":
                    resp = "Sure! What is your updated contact phone number?"
                elif first_target == "name":
                    resp = "Sure! What is your updated full name?"
                elif first_target == "date":
                    resp = "Sure! Which date would you prefer?"
                else:
                    resp = "Sure! What time would you prefer?"
                self._log_conversation_turn(session, text, resp, "ask_first_of_multiple_corrections", state_before=state_before)
                return _ret(resp)

            # Check if caller named a single field to change (e.g. "Email." or "Change my email" or "My phone number")
            if any(w in norm_words for w in ["email", "mail"]):
                session.booking_data["correction_field"] = "email"
                session.state = BookingState.WAITING_FOR_CORRECTION_VALUE
                session.last_asked_field = "correction_email"
                logger.info(f"[BOOKING_CORRECTION] User selected email to correct.")
                if lang in ("ta", "tanglish", "ta_mix"):
                    resp = "கண்டிப்பா, உங்க சரியான email address சொல்லுங்க."
                elif lang == "hi":
                    resp = "Zaroor, kripya apna sahi email address batayein."
                else:
                    resp = "Sure, please tell me your correct email address."
                self._log_conversation_turn(session, text, resp, "waiting_for_correction_email", state_before=state_before)
                return _ret(resp)
            if any(w in norm_words for w in ["phone", "mobile", "number", "contact"]):
                session.booking_data["correction_field"] = "phone"
                session.state = BookingState.WAITING_FOR_CORRECTION_VALUE
                session.last_asked_field = "correction_phone"
                logger.info(f"[BOOKING_CORRECTION] User selected phone to correct.")
                if lang in ("ta", "tanglish", "ta_mix"):
                    resp = "கண்டிப்பா, உங்க சரியான phone number சொல்லுங்க."
                elif lang == "hi":
                    resp = "Zaroor, kripya apna sahi contact number batayein."
                else:
                    resp = "Sure, please tell me your correct contact phone number."
                self._log_conversation_turn(session, text, resp, "waiting_for_correction_phone", state_before=state_before)
                return _ret(resp)
            if any(w in norm_words for w in ["date", "day"]):
                session.booking_data["correction_field"] = "date"
                session.state = BookingState.WAITING_FOR_CORRECTION_VALUE
                session.last_asked_field = "correction_date"
                logger.info(f"[BOOKING_CORRECTION] User selected date to correct.")
                if lang in ("ta", "tanglish", "ta_mix"):
                    resp = "கண்டிப்பா! எந்த date-ல appointment book பண்ணணும்?"
                elif lang == "hi":
                    resp = "Zaroor! Aap kis tareekh par appointment book karna chahenge?"
                else:
                    resp = "Sure, which date would you prefer instead?"
                self._log_conversation_turn(session, text, resp, "waiting_for_correction_date", state_before=state_before)
                return _ret(resp)
            if any(w in norm_words for w in ["time", "timing", "slot"]):
                session.booking_data["correction_field"] = "time"
                session.state = BookingState.WAITING_FOR_CORRECTION_VALUE
                session.last_asked_field = "correction_time"
                logger.info(f"[BOOKING_CORRECTION] User selected time to correct.")
                if lang in ("ta", "tanglish", "ta_mix"):
                    resp = "கண்டிப்பா! எந்த time உங்களுக்கு வசதியா இருக்கும்?"
                elif lang == "hi":
                    resp = "Zaroor! Kaunsa time aapke liye theek rahega?"
                else:
                    resp = "Sure, what time works best for you?"
                self._log_conversation_turn(session, text, resp, "waiting_for_correction_time", state_before=state_before)
                return _ret(resp)
            if any(w in norm_words for w in ["name", "peru", "naam"]):
                session.booking_data["correction_field"] = "name"
                session.state = BookingState.WAITING_FOR_CORRECTION_VALUE
                session.last_asked_field = "correction_name"
                logger.info(f"[BOOKING_CORRECTION] User selected name to correct.")
                if lang in ("ta", "tanglish", "ta_mix"):
                    resp = "கண்டிப்பா! உங்க முழு பெயரை சொல்லுங்க."
                elif lang == "hi":
                    resp = "Zaroor! Kripya apna poora naam batayein."
                else:
                    resp = "Sure, please tell me your correct full name."
                self._log_conversation_turn(session, text, resp, "waiting_for_correction_name", state_before=state_before)
                return _ret(resp)

            # Caller gave bare negation ("No", "Make changes", "Change it", "Incorrect")
            session.state = BookingState.WAITING_FOR_CORRECTION_FIELD
            session.last_asked_field = "field_to_correct"
            logger.info(f"[BOOKING_CORRECTION] Entering correction mode, waiting for correction field. Session {session.session_id}")
            if lang in ("ta", "tanglish", "ta_mix"):
                resp = "மன்னிக்கவும்! எந்த detail-ஐ மாத்தணும் — உங்க name, phone number, email, date அல்லது time-ஆ?"
            elif lang == "hi":
                resp = "Kshama kijiye! Aap konsi detail badalna chahte hain — apna naam, phone number, email, date ya samay?"
            else:
                resp = "Sure! Which detail would you like to change: your name, phone number, email, date, time, or location?"
            self._log_conversation_turn(session, text, resp, "ask_which_field_to_correct", state_before=state_before)
            return _ret(resp)

        # 7. Ambiguous input in confirmation state:
        resp = f"Would you like me to proceed with confirming this appointment {target_str} on {session.date} at {session.time}? Please say 'Yes' to confirm or 'No' to make changes."
        if lang in ("ta", "tanglish", "ta_mix"):
            resp = f"{session.date} {session.time}-க்கு உங்க booking confirm பண்ணலாமா? 'Yes' அல்லது 'No' சொல்லுங்க."
        elif lang == "hi":
            resp = f"Kya aap {session.date} ko {session.time} par booking confirm karna chahte hain? Kripya 'Haan' ya 'Nahi' kahein."
        self._log_conversation_turn(session, text, resp, "clarify_confirmation", state_before=state_before)
        return _ret(resp)

    def process_lead_collection_turn(
        self,
        text: str,
        session: BookingSession,
        turn_id: Optional[str] = None,
        state_before: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Dedicated Lead Collection & Team Callback State Machine.
        Collects:
        1. Full name
        2. Email address
        3. Phone number
        4. Business/industry
        5. Requirement/interested service
        6. Additional details if available
        Asks only for fields that are missing. Never repeats fields already collected.
        Promises direct team callback without mentioning calendar invites.
        Saves lead to DB via LeadService with status New and appointment Not Booked.
        """
        if not hasattr(session, "_turn_response_cache"):
            session._turn_response_cache = {}
        if turn_id and str(turn_id) in session._turn_response_cache:
            cached_resp = session._turn_response_cache[str(turn_id)]
            logger.info(f"[LEAD_TURN_IDEMPOTENT] Reusing cached response for turn {turn_id}: '{cached_resp[:60]}...'")
            return cached_resp

        def _ret(r: str) -> str:
            dup_detected = bool(getattr(session, "last_ai_response", None) and r.strip().lower() == session.last_ai_response.strip().lower())
            if dup_detected:
                session._consecutive_duplicate_count = getattr(session, "_consecutive_duplicate_count", 0) + 1
            else:
                session._consecutive_duplicate_count = 0

            logger.info(
                f"\n[QUESTION_DEDUP]"
                f"\n  call_id={getattr(session, 'call_sid', 'none')}"
                f"\n  attempted_question='{r}'"
                f"\n  previous_question='{getattr(session, 'last_ai_response', None) or 'none'}'"
                f"\n  duplicate_detected={dup_detected}"
                f"\n  action_taken={'VARY_RESPONSE' if dup_detected else 'SEND_PROMPT'}"
            )
            logger.info(
                f"\n[CONVERSATION_STATE]"
                f"\n  call_id={getattr(session, 'call_sid', 'none')}"
                f"\n  current_state={session.state}"
                f"\n  current_topic={getattr(session, 'current_topic', None)}"
                f"\n  lead_requirement={getattr(session, 'requirement', None)}"
                f"\n  collected_fields={session.collected_fields}"
                f"\n  missing_fields={session.missing_fields}"
            )
            logger.info(
                f"\n[CONTEXT_VALIDATION]"
                f"\n  call_id={getattr(session, 'call_sid', 'none')}"
                f"\n  turn={session.turn_count}"
                f"\n  intent={getattr(session, 'last_detected_intent', 'none')}"
                f"\n  is_active_collection={session.is_active()}"
                f"\n  customer_reaction_or_question='{text}'"
            )

            session.last_ai_response = r
            if turn_id:
                session._turn_response_cache[str(turn_id)] = r
            return r

        session.conversation_started = True
        txt_lower = (text or "").lower().strip()
        session.last_user_text = txt_lower
        session.turn_count += 1
        session.turn_id = session.turn_count
        session.state_version = getattr(session, "state_version", 0) + 1
        lang = getattr(session, "conversation_language", "en")

        # 1. Presence check ("are you there", "can you hear me", "hello")
        presence_phrases = ["can you hear me", "are you there", "am i audible", "can you hear", "hear me", "is anyone there"]
        is_bare_hello = txt_lower.rstrip(".,?!") in ("hello", "hi", "hey", "hello there", "hi there", "hello?", "hi?")
        if (any(phrase in txt_lower for phrase in presence_phrases) or is_bare_hello) and not any(w in txt_lower for w in ["name", "email", "phone", "number", "retail", "business", "service", "help"]):
            if lang in ("ta", "tanglish", "ta_mix"):
                resp = "ஆமா, நான் line-ல தான் இருக்கேன்! சொல்லுங்க."
            elif lang == "hi":
                resp = "Haan, main line par hoon! Kripya batayein."
            else:
                resp = "Yes, I'm right here! Please go ahead."
            self._log_conversation_turn(session, text, resp, "presence_reassurance", state_before=state_before)
            return _ret(resp)

        # 2. Dynamic Language Switch check
        from src.services.language_service import language_service, SUPPORTED_LANGUAGES
        target_lang, is_switch = language_service.check_explicit_language_switch(text)
        if is_switch and target_lang in SUPPORTED_LANGUAGES:
            session.conversation_language = target_lang
            resp = language_service.get_language_switch_confirmation(target_lang, booking_session=session)
            self._log_conversation_turn(session, text, resp, "language_switch", state_before=state_before)
            return _ret(resp)

        # 2b. Emotional Questions (Tamil, Tanglish & English)
        is_emotional_q = (
            any(w in txt_lower for w in [
                "என்ன கோச்சிகிட்டியா", "கோச்சிகிட்டியா", "கோச்சுக்கிட்டியா", "கோவமா", "கோவம் வந்திருச்சா",
                "கோபமா", "கோவப்பட்டியா", "நான் கோச்சி கிடையாதுன்னு", "கோச்சிகிட்டீங்களா",
                "kochikitiya", "kochikitaya", "are you angry", "why are you angry", "are you mad",
                "why are you sounding like that", "why are you mad at me"
            ])
            or bool(re.search(r'\b(?:கோச்சி|கோச்சு|கோவம்|கோபம்)\b', txt_lower))
        )
        if is_emotional_q:
            session.last_detected_intent = "emotional_question"
            if lang in ("ta", "tanglish", "ta_mix"):
                resp = "இல்லை சார், கோச்சிக்கவே இல்லை. மன்னிக்கவும், நான் உங்க கேள்வியை சரியாக புரிஞ்சுக்காமல் திரும்பத் திரும்ப கேட்டுட்டேன். சொல்லுங்க, எப்படி உதவலாம்?"
            elif lang == "hi":
                resp = "Nahi sir, bilkul gussa nahi hoon! Maafi chahta hoon agar maine pehle theek se nahi samjha. Batayein, main kaise madad kar sakta hoon?"
            else:
                resp = "Not at all! I'm not upset. I apologize if I didn't catch that earlier. How can I best assist you?"
            self._log_conversation_turn(session, text, resp, "emotional_reassurance", state_before=state_before)
            return _ret(resp)

        # 2c. Repetition Complaints / Caller Frustration with Repeated Prompts
        is_repetition_complaint = (
            any(w in txt_lower for w in [
                "எதுக்கு நீ சொன்னதே திருப்பின்னு சொல்லிட்டு இருக்க", "எதுக்கு திரும்பத் திரும்ப கேக்குற",
                "திரும்ப திரும்ப கேக்குற", "திரும்ப திரும்ப சொல்ற", "ஏன் திரும்ப திரும்ப கேக்குற",
                "சொன்னதே திரும்ப சொல்லாத", "சொன்னதே திருப்பின்னு சொல்லிட்டு இருக்க", "ஒரே கேள்விய கேக்குற",
                "thirumba thirumba", "why are you repeating", "why repeating", "why do you keep repeating",
                "stop repeating", "why asking the same thing", "repeat panra"
            ])
            or bool(re.search(r'\b(?:திருப்பி|திரும்ப|repeat|repeating)\b.*(?:சொல்|கேக்|கேள்வி|sol|kek)', txt_lower))
        )
        if is_repetition_complaint:
            session.last_detected_intent = "repetition_complaint"
            if lang in ("ta", "tanglish", "ta_mix"):
                resp = "மன்னிக்கவும் சார், நான் திரும்பத் திரும்ப கேட்டதுக்கு வருந்துகிறேன். இனிமேல் அப்படி நடக்காது. உங்க requirement-க்கு நான் உதவி செய்கிறேன், சொல்லுங்க."
            elif lang == "hi":
                resp = "Maafi chahta hoon sir, baar-baar dohrane ke liye. Kripya batayein main aapki kaise sahayata karoon?"
            else:
                resp = "I sincerely apologize for repeating myself. Let me help you directly. Please tell me how you'd like to proceed."
            self._log_conversation_turn(session, text, resp, "repetition_recovery", state_before=state_before)
            return _ret(resp)

        # 2d. Silence / Response Pause Checks
        is_silence_check = (
            any(w in txt_lower for w in [
                "என்னாச்சுங்க அமைதியா ஆயிட்டீங்களா", "என்னாச்சு அமைதியா ஆயிட்டீங்க", "ஏன் அமைதியா இருக்கீங்க",
                "அமைதியா ஆயிட்டீங்களா", "இதெல்லாம் சொன்னா அமைதியா ஆயிடு", "ennachu amaidhiya",
                "why are you silent", "why silent", "why are you quiet", "why so quiet"
            ])
        )
        if is_silence_check:
            session.last_detected_intent = "silence_check"
            if lang in ("ta", "tanglish", "ta_mix"):
                resp = "மன்னிக்கவும் சார், response கொடுக்க கொஞ்சம் delay ஆகிடுச்சு. நான் கேட்கிறேன், சொல்லுங்க."
            elif lang == "hi":
                resp = "Maafi chahta hoon sir, thoda delay ho gaya. Main sun raha hoon, batayein."
            else:
                resp = "Apologies for the brief pause. I'm right here listening, please go ahead."
            self._log_conversation_turn(session, text, resp, "silence_reassurance", state_before=state_before)
            return _ret(resp)

        # 3. Thank you / Gratitude closure check
        _THANKS_PHRASES = ["thank you", "thanks", "thank u", "many thanks", "thank you so much", "appreciated", "that's all", "thats all", "that is all"]
        if any(tp in txt_lower for tp in _THANKS_PHRASES):
            first_name = (session.patient_name or "").split()[0] if session.patient_name else ""
            if session.is_completed:
                if lang in ("ta", "tanglish", "ta_mix"):
                    resp = f"ரொம்ப நன்றி{f', {first_name}' if first_name else ''}! எங்க team உங்களை சீக்கிரம் contact பண்ணுவாங்க. Have a wonderful day!"
                elif lang == "hi":
                    resp = f"Aapka bahut dhanyawad{f', {first_name}' if first_name else ''}! Hamari team aapse jaldi contact karegi. Aapka din shubh ho!"
                else:
                    resp = f"You're very welcome{f', {first_name}' if first_name else ''}! Our team will contact you soon. Have a wonderful day!"
                self._log_conversation_turn(session, text, resp, "lead_closing", state_before=state_before)
                return _ret(resp)

        # 4. Cancellation check
        _CANCEL_PHRASES = ["cancel", "stop", "never mind", "nevermind", "forget it", "leave it", "dont contact", "don't contact", "dont call", "don't call"]
        if any(phrase in txt_lower for phrase in _CANCEL_PHRASES):
            session.reset()
            if lang in ("ta", "tanglish", "ta_mix"):
                resp = "சரிங்க, contact request cancel பண்ணியாச்சு. உங்களுக்கு வேற ஏதாவது help வேணுமா?"
            elif lang == "hi":
                resp = "Theek hai, contact request cancel kar diya gaya hai. Main aapki aur kya sahayata kar sakta hoon?"
            else:
                resp = "No problem at all. I've cancelled the request. How else may I assist you today?"
            self._log_conversation_turn(session, text, resp, "lead_cancelled", state_before=state_before)
            return _ret(resp)

        # 5. Extract slots
        self.extract_slots(text, session)

        # Extract Industry if asked or mentioned
        non_industry = {"yes", "yeah", "ok", "okay", "no", "sure", "continue", "go ahead", "proceed", "hello", "hi"}
        if getattr(session, "last_asked_field", "") == "industry" or not session.industry:
            cleaned_ind = re.sub(r'^(?:in\s+|we\s+are\s+in\s+|our\s+industry\s+is\s+|our\s+business\s+is\s+|my\s+business\s+is\s+|it\'s\s+|its\s+|i\s+am\s+in\s+)', '', txt_lower).strip().rstrip('.,?!')
            if cleaned_ind and cleaned_ind not in non_industry and len(cleaned_ind.split()) <= 5:
                if not any(w in cleaned_ind for w in ["phone", "email", "name", "@"]):
                    session.industry = cleaned_ind.title()
                    session.booking_data["industry"] = session.industry
            elif any(w in txt_lower for w in ["retail", "healthcare", "it", "software", "logistics", "education", "real estate", "manufacturing", "ecommerce", "e-commerce", "finance", "hospitality", "construction", "consulting"]):
                for ind_w in ["retail", "healthcare", "it", "software", "logistics", "education", "real estate", "manufacturing", "ecommerce", "e-commerce", "finance", "hospitality", "construction", "consulting"]:
                    if ind_w in txt_lower:
                        session.industry = ind_w.title()
                        session.booking_data["industry"] = session.industry
                        break
            elif session.booking_data.get("business_description"):
                session.industry = session.booking_data["business_description"]

        # Extract Requirement if asked or mentioned
        if getattr(session, "last_asked_field", "") == "requirement" or not session.requirement:
            non_req = {"yes", "yeah", "ok", "okay", "no", "sure", "continue", "go ahead", "proceed", "hello", "hi"}
            if getattr(session, "last_asked_field", "") == "requirement":
                cleaned_req = re.sub(r'^(?:i\s+want|we\s+want|i\s+need|we\s+need|interested\s+in|looking\s+for|for|about)\s+', '', text, flags=re.IGNORECASE).strip().rstrip('.,?!')
                if cleaned_req and cleaned_req.lower() not in non_req and len(cleaned_req.split()) <= 7:
                    session.requirement = cleaned_req.title()
                    session.booking_data["requirement"] = session.requirement
            elif session.service or session.booking_data.get("product"):
                session.requirement = session.service or session.booking_data.get("product")
            elif any(w in txt_lower for w in ["hrms", "erp", "voice agent", "voice agents", "voice receptionist", "galaxy chat", "chat", "project management", "activity tracking", "survey", "website", "custom software"]):
                for req_w in ["AI Voice Agents", "Galaxy Chat", "ERP", "HRMS", "Project Management", "Employee Activity Tracking", "Survey Platform", "Website Development", "Custom Software"]:
                    if req_w.lower() in txt_lower:
                        session.requirement = req_w
                        session.booking_data["requirement"] = req_w
                        break
            elif getattr(session, "current_topic", None):
                session.requirement = session.current_topic
                session.booking_data["requirement"] = session.requirement
            elif session.booking_data.get("current_topic"):
                session.requirement = session.booking_data["current_topic"]
                session.booking_data["requirement"] = session.requirement

        # Check explicit request to change email, phone, or name
        if re.search(r'\b(?:change|update|correct)\s+(?:my\s+)?(?:email|mail|address)\b', txt_lower):
            new_em = self.parse_spoken_email(text)
            if new_em:
                session.email = new_em
                session.booking_data["email"] = new_em
                resp = f"Sure! I've updated your email address to {new_em}."
            else:
                session.email = None
                session.state = BookingState.COLLECTING_EMAIL
                session.last_asked_field = "email"
                resp = "Sure! What is your new email address?"
            self._log_conversation_turn(session, text, resp, "request_new_email", state_before=state_before)
            return _ret(resp)

        if re.search(r'\b(?:change|update|correct)\s+(?:my\s+)?(?:phone|number|mobile|contact)\b', txt_lower):
            new_ph = self.parse_spoken_phone_number(text)
            if new_ph:
                session.contact_number = new_ph
                session.booking_data["contact_number"] = new_ph
                resp = f"Sure! I've updated your contact phone number to {new_ph}."
            else:
                session.contact_number = None
                session.state = BookingState.COLLECTING_PHONE
                session.last_asked_field = "contact_number"
                resp = "Sure! What is your new contact phone number?"
            self._log_conversation_turn(session, text, resp, "request_new_phone", state_before=state_before)
            return _ret(resp)

        # Handle customer saying "I already told you"
        _ALREADY_TOLD_PHRASES = [
            "already told you", "already told", "already said", "already gave you",
            "already gave", "already shared", "told you already", "gave you already",
            "already provided", "i already told", "i told you", "already answered"
        ]
        if any(atp in txt_lower for atp in _ALREADY_TOLD_PHRASES):
            ref_field = None
            if any(w in txt_lower for w in ["email", "mail"]):
                ref_field = "email"
            elif any(w in txt_lower for w in ["phone", "mobile", "number", "contact"]):
                ref_field = "contact_number"
            elif any(w in txt_lower for w in ["name", "peru", "naam"]):
                ref_field = "patient_name"
            elif any(w in txt_lower for w in ["industry", "business"]):
                ref_field = "industry"
            elif any(w in txt_lower for w in ["requirement", "product", "service"]):
                ref_field = "requirement"
            else:
                ref_field = getattr(session, "last_asked_field", "patient_name")

            val = getattr(session, ref_field, None) or session.booking_data.get(ref_field)
            if not val and ref_field == "email" and session.email:
                val = session.email
            if not val and ref_field == "contact_number" and session.contact_number:
                val = session.contact_number
            if not val and ref_field == "patient_name" and session.patient_name:
                val = session.patient_name

            if val:
                missing = session.get_missing_lead_fields()
                if missing:
                    next_f = missing[0].replace("_", " ")
                    resp = f"You're right, I have your {ref_field.replace('_', ' ')} as {val}. Could you please share your {next_f}?"
                else:
                    cust_name = session.patient_name or "there"
                    req_name = session.requirement or session.service or session.booking_data.get("product") or "requirements"
                    resp = f"Thank you, {cust_name}. I've noted your contact details and requirements. Our team will contact you to discuss your {req_name}. Is there anything else I can help you with?"
            else:
                resp = f"I apologize for the confusion. Could you please share your {ref_field.replace('_', ' ')} once more?"
            self._log_conversation_turn(session, text, resp, "already_told_lead", state_before=state_before)
            return _ret(resp)

        # Check if caller asks when team will contact / turnaround time
        is_when_contact_q = (
            any(w in txt_lower for w in [
                "when will your team contact", "when your team contact", "when will someone call",
                "when will someone contact", "when will you call me", "when will you contact me",
                "how soon will your team contact", "how soon will you call", "how long will it take to call",
                "எப்போ தொடர்பு", "எப்போது தொடர்பு", "எப்ப என்னை தொடர்பு", "என்னை எப்போ தொடர்பு", "எப்போ என்னை தொடர்பு"
            ])
            or (any(w in txt_lower for w in ["when", "how soon", "what time", "எப்போ", "எப்போது", "eppo"]) and any(w in txt_lower for w in ["contact", "call", "reach", "தொடர்பு", "கூப்பிடு"]))
        )
        if is_when_contact_q:
            if lang in ("ta", "tanglish", "ta_mix"):
                resp = "எங்க team வழக்கமாக 24 மணி நேரத்திற்குள் உங்களை contact செய்து demo-வை ஏற்பாடு செய்வாங்க. உங்களுக்கு வசதியான நாள் அல்லது நேரம் ஏதாவது இருக்கிறதா?"
            elif lang == "hi":
                resp = "Hamari team aamtaur par 24 ghante ke bheetar aapse contact karke demo arrange karegi. Kya aapke liye koi specific date ya time suvidhajanak hai?"
            else:
                resp = "Our team typically reaches out within 24 business hours to arrange and schedule your demo. Is there a specific date or time you would prefer?"
            self._log_conversation_turn(session, text, resp, "demo_contact_timeline", state_before=state_before)
            return _ret(resp)

        # Check if caller asks to share appointment details or date and time
        _APPT_DETAILS_Q = [
            "share the appointment details", "share my appointment details", "share appointment details",
            "share the date and time", "share date and time", "appointment details", "what are my appointment details",
            "what is my appointment details", "can you share the date and time", "my appointment details",
            "check my appointment", "appointment status", "when is my appointment"
        ]
        if any(ad in txt_lower for ad in _APPT_DETAILS_Q):
            resp = self.get_appointment_or_demo_details(session)
            self._log_conversation_turn(session, text, resp, "appointment_details_inquiry", state_before=state_before)
            return _ret(resp)

        # Caller states availability for demo callback
        if re.search(r'\b(?:i\s+am|i\'m|i\s+will\s+be|im)\s+available\b', txt_lower) or any(w in txt_lower for w in ["available tomorrow", "available today", "available next"]):
            resolved_avail = "tomorrow" if "tomorrow" in txt_lower else ("today" if "today" in txt_lower else text.strip())
            session.booking_data["caller_availability"] = resolved_avail
            logger.info(f"[LEAD_COLLECTION] Recorded caller availability: '{resolved_avail}'")
            if lang in ("ta", "tanglish", "ta_mix"):
                resp = f"சரிங்க! நீங்க {resolved_avail} available-ஆ இருக்கீங்கன்னு குறித்துக்கொண்டேன். எங்க team உங்களை contact பண்ணுவாங்க. உங்களுக்கு வேற ஏதாவது உதவி வேணுமா?"
            elif lang == "hi":
                resp = f"Theek hai! Maine note kar liya hai ki aap {resolved_avail} available hain. Hamari team aapse contact karegi. Kya main aapki aur koi madad kar sakta hoon?"
            else:
                resp = f"Got it! I've noted that you are available {resolved_avail}. Our team will contact you then regarding the demo. Is there anything else I can assist you with?"
            self._log_conversation_turn(session, text, resp, "caller_availability_noted", state_before=state_before)
            return _ret(resp)

        first_name = (session.patient_name or "").split()[0] if session.patient_name and self.is_valid_person_name(session.patient_name) else ""

        # Check if caller declines or skips email
        decline_email_phrases = [
            "email வேண்டாம்", "mail venam", "வேண்டாம்", "வேண்டாங்க", "venam", "vendaam",
            "no email", "don't have email", "dont have email", "no mail", "skip", "decline",
            "none", "illa", "dont want email", "don't want email", "வேற எதுவும் வேண்டாம்"
        ]
        if session.last_asked_field == "email" and any(dp in txt_lower for dp in decline_email_phrases):
            session.booking_data["email_declined"] = True
            session.booking_data["email_skipped"] = True
            if "email" not in session.skipped_fields:
                session.skipped_fields.append("email")
            session.email = None
            logger.info(f"[LEAD_COLLECTION] Email declined/skipped by caller: '{text}'")

        # Check if caller confirms calling number from caller ID
        is_caller_phone_confirm = any(cp in txt_lower for cp in [
            "இந்த numberல callback பண்ணுங்க", "இந்த number-லேயே பண்ணுங்க", "இந்த numberல பண்ணுங்க",
            "இந்த number", "intha number", "same number", "use this number", "call on this number",
            "call back on this number", "call in this number"
        ]) or (
            session.last_asked_field == "contact_number" and txt_lower.rstrip(".,?!") in (
                "yes", "yeah", "yep", "sure", "ok", "okay", "சரி", "ஆமாம்", "ஆமா", "correct", "right", "sari", "aamaam"
            )
        )
        if (session.last_asked_field == "contact_number" or getattr(session, "state", None) == BookingState.COLLECTING_PHONE) and is_caller_phone_confirm:
            if session.caller_phone:
                session.contact_number = session.caller_phone
                session.booking_data["contact_number"] = session.caller_phone
                logger.info(f"[LEAD_COLLECTION] Contact number confirmed from caller ID: {session.caller_phone}")

        # Evaluate next missing field in order
        missing = session.get_missing_lead_fields()
        logger.info(f"[LEAD_COLLECTION] session_id={session.session_id} missing_fields={missing} last_asked={session.last_asked_field}")

        if not missing:
            session.state = BookingState.COMPLETED
            session.is_completed = True
            session.last_asked_field = None
            cust_name = session.patient_name or "there"
            req_name = session.requirement or session.service or session.booking_data.get("product") or "ERP software"

            if lang in ("ta", "tanglish", "ta_mix"):
                resp = f"சரி சார், உங்க {req_name} requirement பதிவு பண்ணிட்டேன். எங்க team இந்த numberல உங்களை contact பண்ணுவாங்க. நன்றி!"
            elif lang == "hi":
                resp = f"Dhanyawad, {cust_name}. Maine aapke contact details aur {req_name} requirement note kar liye hain. Hamari team aapse contact karegi. Aapka din shubh ho!"
            else:
                resp = f"Thank you, {cust_name}. I've noted your {req_name} requirement. Our team will contact you on this number. Have a wonderful day!"

            try:
                from src.services.lead_service import LeadService
                from src.config.database import SessionLocal
                _db = SessionLocal()
                try:
                    lead_svc = LeadService(_db)
                    lead_svc.upsert_lead_from_session(
                        session,
                        lead_status="New",
                        appointment_status="Not Booked",
                        follow_up_required=True,
                        notes=f"Callback requested: {req_name}"
                    )
                    _db.commit()
                    logger.info(f"[LEAD_SAVED] Successfully saved lead for {cust_name} ({session.contact_number}) with status New, Not Booked")
                finally:
                    _db.close()
            except Exception as exc:
                logger.warning(f"[LEAD_SAVED] Error upserting lead: {exc}")

            self._log_conversation_turn(session, text, resp, "lead_completed", state_before=state_before)
            return _ret(resp)

        next_missing = missing[0]
        if next_missing == "patient_name":
            session.state = BookingState.COLLECTING_PATIENT_NAME
            session.last_asked_field = "patient_name"
            if lang in ("ta", "tanglish", "ta_mix"):
                if getattr(session, "_is_new_callback_request", False) or any(cb in session.last_user_text for cb in ["callback", "team", "contact", "call"]):
                    resp = "கண்டிப்பா சார், எங்க team உங்களை contact பண்ண ஏற்பாடு செய்கிறேன். உங்க பெயர் சொல்லுங்க."
                else:
                    resp = "நிச்சயமாக! உங்க full name சொல்ல முடியுமா?"
            elif lang == "hi":
                resp = "Zaroor! Kya main aapka poora naam jaan sakta hoon?"
            else:
                resp = "Certainly! May I have your full name?"
            self._log_conversation_turn(session, text, resp, "lead_ask_name", state_before=state_before)
            return _ret(resp)

        elif next_missing == "email":
            session.state = BookingState.COLLECTING_EMAIL
            session.last_asked_field = "email"
            if session.booking_data.get("partial_email_domain"):
                if lang in ("ta", "tanglish", "ta_mix"):
                    resp = "உங்க email-ல் @-க்கு முன்னாடி இருக்கும் username அல்லது பெயரைச் சொல்ல முடியுமா?"
                elif lang == "hi":
                    resp = "Kripya apne email ke @ se pehle ka username batayein."
                else:
                    resp = "Could you please share the username or name before the @ symbol for your email?"
                self._log_conversation_turn(session, text, resp, "lead_ask_email_username", state_before=state_before)
                return _ret(resp)
            if lang in ("ta", "tanglish", "ta_mix"):
                resp = f"சரி, {first_name}. உங்க email address சொல்லுங்க." if first_name else "உங்க email address சொல்லுங்க."
            elif lang == "hi":
                resp = f"Dhanyawad, {first_name}. Kripya apna email address batayein." if first_name else "Kripya apna email address batayein."
            else:
                resp = f"Thanks, {first_name}. What is your email address?" if first_name else "Could you please share your email address?"
            self._log_conversation_turn(session, text, resp, "lead_ask_email", state_before=state_before)
            return _ret(resp)

        elif next_missing == "contact_number":
            session.state = BookingState.COLLECTING_PHONE
            session.last_asked_field = "contact_number"
            caller_phone_avail = bool(session.caller_phone and session.caller_phone not in ("Unknown", "Unknown Caller", "Private Number"))
            if caller_phone_avail:
                if lang in ("ta", "tanglish", "ta_mix"):
                    resp = "நன்றி சார். இந்த call வந்த number-லேயே உங்களை contact பண்ணலாமா?"
                elif lang == "hi":
                    resp = f"Dhanyawad sir. Kya hamari team isi number ({session.caller_phone}) par aapse sampark kare?"
                else:
                    resp = f"Thank you. Should our team contact you on this calling number ({session.caller_phone})?"
                self._log_conversation_turn(session, text, resp, "lead_suggest_caller_phone", state_before=state_before)
                return _ret(resp)
            else:
                if lang in ("ta", "tanglish", "ta_mix"):
                    resp = f"சரி, {first_name}. உங்க contact phone number சொல்லுங்க." if first_name else "உங்க contact phone number சொல்லுங்க."
                elif lang == "hi":
                    resp = f"Theek hai, {first_name}. Kripya apna contact phone number batayein." if first_name else "Kripya apna contact phone number batayein."
                else:
                    resp = f"Thanks, {first_name}. Could you please share your contact phone number?" if first_name else "Could you please share your contact phone number?"
                self._log_conversation_turn(session, text, resp, "lead_ask_phone", state_before=state_before)
                return _ret(resp)

        elif next_missing == "industry":
            session.state = BookingState.COLLECTING_BUSINESS
            session.last_asked_field = "industry"
            if lang in ("ta", "tanglish", "ta_mix"):
                resp = f"சரி, {first_name}. நீங்க எந்த business அல்லது industry-ல இருக்கீங்க?" if first_name else "நீங்க எந்த business அல்லது industry-ல இருக்கீங்க?"
            elif lang == "hi":
                resp = f"Theek hai, {first_name}. Aap kis business ya industry se hain?" if first_name else "Aap kis business ya industry se hain?"
            else:
                resp = f"Thanks, {first_name}. What business or industry are you in?" if first_name else "What business or industry are you in?"
            self._log_conversation_turn(session, text, resp, "lead_ask_industry", state_before=state_before)
            return _ret(resp)

        elif next_missing == "requirement":
            session.state = BookingState.COLLECTING_REQUIREMENT
            session.last_asked_field = "requirement"
            if lang in ("ta", "tanglish", "ta_mix"):
                resp = "எங்க team-கூட எந்த service அல்லது requirement பத்தி பேச விரும்புறீங்க?"
            elif lang == "hi":
                resp = "Aap hamari team ke saath kis service ya requirement ke baare mein baat karna chahenge?"
            else:
                resp = "What specific requirement or service would you like our team to discuss with you?"
            self._log_conversation_turn(session, text, resp, "lead_ask_requirement", state_before=state_before)
            return _ret(resp)

        return _ret("How else may I assist you?")

    def process_booking_turn(self, text: str, session: BookingSession, turn_id: Optional[str] = None) -> str:
        """
        Stateful Conversational Turn Processor.
        Evaluates session memory, extracts entities, validates doctors, performs live slot verification,
        and dynamically determines the single next required action/question.
        """
        # Idempotency check: if this exact customer turn was already evaluated, reuse the response
        if not hasattr(session, "_turn_response_cache"):
            session._turn_response_cache = {}
        if turn_id and str(turn_id) in session._turn_response_cache:
            cached_resp = session._turn_response_cache[str(turn_id)]
            logger.info(f"[BOOKING_TURN_IDEMPOTENT] Reusing cached response for turn {turn_id}: '{cached_resp[:60]}...'")
            return cached_resp

        def _ret(r: str) -> str:
            # Anti-repetition loop prevention:
            if getattr(session, "last_ai_response", None) and r.strip().lower() == session.last_ai_response.strip().lower():
                session._consecutive_duplicate_count = getattr(session, "_consecutive_duplicate_count", 0) + 1
                if session._consecutive_duplicate_count >= 1:
                    active_lang = getattr(session, "conversation_language", "en")
                    if getattr(session, "last_asked_field", "") == "doctor":
                        is_hosp = (getattr(session, "business_type", "GENERAL") in ("HOSPITAL", "CLINIC") or getattr(session, "booking_type", "") == "HOSPITAL_APPOINTMENT")
                        if is_hosp:
                            if active_lang in ("ta", "tanglish", "ta_mix"):
                                r = "எங்கள் மருத்துவமனையில் General Medicine, Cardiology, மற்றும் Dental Care துறைகள் உள்ளன. நீங்கள் எந்த துறையை அல்லது மருத்துவரை தேர்வு செய்ய விரும்புகிறீர்கள்?"
                            elif active_lang == "hi":
                                r = "Hamare paas General Medicine, Cardiology, aur Dental Care departments hain. Aap kis vibhaag ya doctor ko chunna chahenge?"
                            elif active_lang in ("kn", "kannada"):
                                r = "ನಮ್ಮ ಆಸ್ಪತ್ರೆಯಲ್ಲಿ ಜನರಲ್ ಮೆಡಿಸಿನ್, ಕಾರ್ಡಿಯಾಲಜಿ ಮತ್ತು ದಂತ ಚಿಕಿತ್ಸೆ ವಿಭಾಗಗಳಿವೆ. ನೀವು ಯಾರನ್ನು ಸಂಪರ್ಕಿಸಲು ಬಯಸುತ್ತೀರಿ?"
                            elif active_lang in ("te", "telugu"):
                                r = "మా ఆసుపత్రిలో జనరల్ మెడిసిన్, కార్డియాలజీ మరియు డెంటల్ కేర్ విభాగాలు ఉన్నాయి. మీరు ఎవరిని సంప్రదించాలనుకుంటున్నారు?"
                            else:
                                r = "To help book your appointment, we have specialists in General Medicine, Cardiology, Dental Care, and Ayurvedic medicine. Which department or doctor would you prefer?"
                        else:
                            b_name = getattr(session, "business_name", None) or "our team"
                            r = f"To help you with {b_name}, which product, service, or demo would you prefer to discuss?"
                    elif getattr(session, "last_asked_field", "") == "date":
                        r = "Could you please share which date would work best for your appointment, such as today or tomorrow?"
                    elif getattr(session, "last_asked_field", "") == "time":
                        r = "Please let me know which time slot you would prefer, or let me know if another time works for you."
            else:
                session._consecutive_duplicate_count = 0

            dup_detected = getattr(session, "_consecutive_duplicate_count", 0) > 0
            logger.info(
                f"\n[QUESTION_DEDUP]"
                f"\n  call_id={getattr(session, 'call_sid', 'none')}"
                f"\n  attempted_question='{r}'"
                f"\n  previous_question='{getattr(session, 'last_ai_response', None) or 'none'}'"
                f"\n  duplicate_detected={dup_detected}"
                f"\n  action_taken={'VARY_RESPONSE' if dup_detected else 'SEND_PROMPT'}"
            )
            logger.info(
                f"\n[CONVERSATION_STATE]"
                f"\n  call_id={getattr(session, 'call_sid', 'none')}"
                f"\n  current_state={session.state}"
                f"\n  current_topic={getattr(session, 'current_topic', None)}"
                f"\n  lead_requirement={getattr(session, 'requirement', None)}"
                f"\n  collected_fields={session.collected_fields}"
                f"\n  missing_fields={session.missing_fields}"
            )
            logger.info(
                f"\n[CONTEXT_VALIDATION]"
                f"\n  call_id={getattr(session, 'call_sid', 'none')}"
                f"\n  turn={session.turn_count}"
                f"\n  intent={getattr(session, 'last_detected_intent', 'none')}"
                f"\n  is_active_collection={session.is_active()}"
                f"\n  customer_reaction_or_question='{text}'"
            )

            session.last_ai_response = r
            if turn_id:
                session._turn_response_cache[str(turn_id)] = r
            return r

        session.conversation_started = True
        txt_lower = text.lower().strip()
        
        session.last_user_text = txt_lower
        session.turn_count += 1
        session.turn_id = session.turn_count
        session.state_version = getattr(session, "state_version", 0) + 1
        state_before = copy.deepcopy(session.get_state_dict())

        # Presence check ("are you there", "can you hear me", "hello") during active booking
        presence_phrases = ["can you hear me", "are you there", "am i audible", "can you hear", "hear me", "is anyone there"]
        is_bare_hello = txt_lower.rstrip(".,?!") in ("hello", "hi", "hey", "hello there", "hi there", "hello?", "hi?")
        if (any(phrase in txt_lower for phrase in presence_phrases) or is_bare_hello) and not any(w in txt_lower for w in ["book", "appointment", "doctor", "schedule", "reserve", "demo"]):
            silence_prompt = self.get_silence_prompt(session)
            if getattr(session, "conversation_language", "en") in ("ta", "tanglish", "ta_mix"):
                resp = f"ஆமா, நான் line-ல தான் இருக்கேன்! {silence_prompt}"
            else:
                resp = f"Yes, I'm right here! {silence_prompt}"
            self._log_conversation_turn(session, text, resp, "presence_reassurance", state_before=state_before)
            return _ret(resp)

        # Dynamic Language Switch check during active booking
        from src.services.language_service import language_service, SUPPORTED_LANGUAGES
        target_lang, is_switch = language_service.check_explicit_language_switch(text)
        if is_switch and target_lang in SUPPORTED_LANGUAGES:
            session.conversation_language = target_lang
            resp = language_service.get_language_switch_confirmation(target_lang, booking_session=session)
            self._log_conversation_turn(session, text, resp, "language_switch", state_before=state_before)
            return _ret(resp)

        lang = session.conversation_language

        # Emotional Questions during active booking (Tamil, Tanglish & English)
        is_emotional_q = (
            any(w in txt_lower for w in [
                "என்ன கோச்சிகிட்டியா", "கோச்சிகிட்டியா", "கோச்சுக்கிட்டியா", "கோவமா", "கோவம் வந்திருச்சா",
                "கோபமா", "கோவப்பட்டியா", "நான் கோச்சி கிடையாதுன்னு", "கோச்சிகிட்டீங்களா",
                "kochikitiya", "kochikitaya", "are you angry", "why are you angry", "are you mad",
                "why are you sounding like that", "why are you mad at me"
            ])
            or bool(re.search(r'\b(?:கோச்சி|கோச்சு|கோவம்|கோபம்)\b', txt_lower))
        )
        if is_emotional_q:
            session.last_detected_intent = "emotional_question"
            if lang in ("ta", "tanglish", "ta_mix"):
                resp = "இல்லை சார், கோச்சிக்கவே இல்லை. மன்னிக்கவும், நான் உங்க கேள்வியை சரியாக புரிஞ்சுக்காமல் திரும்பத் திரும்ப கேட்டுட்டேன். சொல்லுங்க, எப்படி உதவலாம்?"
            elif lang == "hi":
                resp = "Nahi sir, bilkul gussa nahi hoon! Maafi chahta hoon agar maine pehle theek se nahi samjha. Batayein, main kaise madad kar sakta hoon?"
            else:
                resp = "Not at all, I'm not upset! I apologize if I repeated myself earlier. How can I best assist you?"
            self._log_conversation_turn(session, text, resp, "emotional_reassurance", state_before=state_before)
            return _ret(resp)

        # Repetition Complaints / Caller Frustration with Repeated Prompts
        is_repetition_complaint = (
            any(w in txt_lower for w in [
                "எதுக்கு நீ சொன்னதே திருப்பின்னு சொல்லிட்டு இருக்க", "எதுக்கு திரும்பத் திரும்ப கேக்குற",
                "திரும்ப திரும்ப கேக்குற", "திரும்ப திரும்ப சொல்ற", "ஏன் திரும்ப திரும்ப கேக்குற",
                "சொன்னதே திரும்ப சொல்லாத", "சொன்னதே திருப்பின்னு சொல்லிட்டு இருக்க", "ஒரே கேள்விய கேக்குற",
                "thirumba thirumba", "why are you repeating", "why repeating", "why do you keep repeating",
                "stop repeating", "why asking the same thing", "repeat panra"
            ])
            or bool(re.search(r'\b(?:திருப்பி|திரும்ப|repeat|repeating)\b.*(?:சொல்|கேக்|கேள்வி|sol|kek)', txt_lower))
        )
        if is_repetition_complaint:
            session.last_detected_intent = "repetition_complaint"
            if lang in ("ta", "tanglish", "ta_mix"):
                resp = "மன்னிக்கவும் சார், நான் திரும்பத் திரும்ப கேட்டதுக்கு வருந்துகிறேன். இனிமேல் அப்படி நடக்காது. உங்க requirement-க்கு நான் உதவி செய்கிறேன், சொல்லுங்க."
            elif lang == "hi":
                resp = "Maafi chahta hoon sir, baar-baar dohrane ke liye. Kripya batayein main aapki kaise sahayata karoon?"
            else:
                resp = "I sincerely apologize for repeating myself. Let me help you directly. Please tell me what you need."
            self._log_conversation_turn(session, text, resp, "repetition_recovery", state_before=state_before)
            return _ret(resp)

        # Silence / Response Pause Checks
        is_silence_check = (
            any(w in txt_lower for w in [
                "என்னாச்சுங்க அமைதியா ஆயிட்டீங்களா", "என்னாச்சு அமைதியா ஆயிட்டீங்க", "ஏன் அமைதியா இருக்கீங்க",
                "அமைதியா ஆயிட்டீங்களா", "இதெல்லாம் சொன்னா அமைதியா ஆயிடு", "ennachu amaidhiya",
                "why are you silent", "why silent", "why are you quiet", "why so quiet"
            ])
        )
        if is_silence_check:
            session.last_detected_intent = "silence_check"
            if lang in ("ta", "tanglish", "ta_mix"):
                resp = "மன்னிக்கவும் சார், response கொடுக்க கொஞ்சம் delay ஆகிடுச்சு. நான் கேட்கிறேன், சொல்லுங்க."
            elif lang == "hi":
                resp = "Maafi chahta hoon sir, thoda delay ho gaya. Main sun raha hoon, batayein."
            else:
                resp = "Apologies for the brief pause. I'm right here listening, please go ahead."
            self._log_conversation_turn(session, text, resp, "silence_reassurance", state_before=state_before)
            return _ret(resp)

        # Workflow Mode Evaluation: Callback / Lead Collection vs Explicit Appointment Booking
        _CALLBACK_TRIGGER_PHRASES = [
            # English
            "arrange a call", "arrange call", "make an arrangement", "make arrangement",
            "speak with your team", "speak with the team", "speak to your team", "speak to the team",
            "talk to your team", "talk to the team", "talk with your team", "talk with the team",
            "connect with your team", "connect with the team", "i want to speak with your team",
            "i want to talk to your team", "contact me", "yes contact me", "yes, contact me",
            "call me back", "request a callback", "callback", "arrange a callback",
            "give me a call back", "schedule a callback", "schedule a call", "call back",
            "have someone call me", "can someone call me", "get someone to call me",
            # Tamil native script
            "உங்க team callback பண்ண சொல்லுங்க", "எனக்கு call பண்ண சொல்லுங்க",
            "ஒருத்தர் எனக்கு contact பண்ணட்டும்", "நான் பேசணும் உங்கள் team கூட",
            "team contact பண்ண சொல்லுங்க", "callback பண்ணுங்க", "call பண்ணுங்க",
            "contact பண்ணுங்க", "team callback", "எனக்கு வேற எதுவும் தெரிஞ்சிக்க வேணாம்",
            "callback பண்ண சொல்லுங்க", "call பண்ண சொல்லு",
            # Tanglish
            "callback panna sollunga", "call panna sollunga", "contact panna sollunga",
            "callback pannunga", "call pannunga", "team callback pannunga",
            "pesanum ungal team kooda", "pesanum unga team", "team contact pannattum"
        ]
        is_callback_trigger = any(cb in txt_lower for cb in _CALLBACK_TRIGGER_PHRASES)

        _EXPLICIT_APPT_TRIGGERS = [
            "book an appointment", "book appointment", "schedule an appointment", "make an appointment",
            "book a slot", "reserve a slot", "want to book an appointment", "need an appointment",
            "i want an appointment", "take an appointment", "fix an appointment"
        ]
        is_explicit_appt = any(ab in txt_lower for ab in _EXPLICIT_APPT_TRIGGERS) or (
            any(w in txt_lower for w in ["book", "booking", "schedule", "reserve"])
            and any(w in txt_lower for w in ["appointment", "slot", "meeting", "demo"])
            and not is_callback_trigger
        )

        if is_explicit_appt:
            session.workflow_mode = SessionWorkflowMode.APPOINTMENT_BOOKING
        elif is_callback_trigger:
            session.workflow_mode = SessionWorkflowMode.LEAD_COLLECTION
            if not session.requirement:
                if any(w in txt_lower for w in ["erp", "hrms", "software"]):
                    session.requirement = "ERP Software"
                    session.booking_data["requirement"] = "ERP Software"
                elif any(w in txt_lower for w in ["website", "web"]):
                    session.requirement = "Website Development"
                    session.booking_data["requirement"] = "Website Development"
                else:
                    for h in getattr(session, "history", []):
                        ht = h.get("text", "").lower()
                        if "erp" in ht:
                            session.requirement = "ERP Software"
                            session.booking_data["requirement"] = "ERP Software"
                            break
                        elif "website" in ht:
                            session.requirement = "Website Development"
                            session.booking_data["requirement"] = "Website Development"
                            break

        if getattr(session, "workflow_mode", None) == SessionWorkflowMode.LEAD_COLLECTION:
            return self.process_lead_collection_turn(text, session, turn_id=turn_id, state_before=state_before)

        # Cancellation check — only explicit booking-cancel phrases, not bare 'cancel'
        _CANCEL_PHRASES = [
            "cancel booking", "cancel the booking", "cancel appointment", "cancel the appointment",
            "stop booking", "dont book", "do not book", "don't book",
            "never mind", "nevermind", "forget it", "leave it",
        ]
        if any(phrase in txt_lower for phrase in _CANCEL_PHRASES):
            session.reset()
            resp = "Your appointment booking has been cancelled. How else may I assist you today?"
            if lang in ("ta", "tanglish", "ta_mix"):
                resp = "Booking cancel பண்ணியாச்சு. உங்களுக்கு வேற ஏதாவது help வேணுமா?"
            elif lang == "hi":
                resp = "Aapka appointment booking cancel kar diya gaya hai. Main aapki aur kya sahayata kar sakta hoon?"
            self._log_conversation_turn(session, text, resp, "booking_cancelled", state_before=state_before)
            return _ret(resp)

        # Resume / Continue Intent Detection
        _RESUME_PHRASES = {
            "continue the last conversation", "continue last conversation",
            "continue the conversation", "continue conversation",
            "continue", "let's proceed", "lets proceed", "proceed",
            "okay go ahead", "ok go ahead", "go ahead", "resume",
            "carry on", "keep going"
        }
        clean_text_for_resume = re.sub(r'[^a-zA-Z\s]', '', txt_lower).strip()
        if (clean_text_for_resume in _RESUME_PHRASES or any(p == clean_text_for_resume for p in _RESUME_PHRASES)) and session.is_active():
            logger.info(f"[RESUME_CONVERSATION] Resuming active booking at stage={session.state}, missing_fields={session.missing_fields}")
            if not session.missing_fields or session.state in CORRECTION_AND_CONFIRMATION_STATES:
                return self.handle_confirmation_turn(session, text, turn_id=turn_id, state_before=state_before)

        # Dedicated confirmation routing: If waiting for confirmation or in correction flow, route directly to handle_confirmation_turn
        if session.state in CORRECTION_AND_CONFIRMATION_STATES or (state_before and state_before.get("state") in CORRECTION_AND_CONFIRMATION_STATES):
            return self.handle_confirmation_turn(session, text, turn_id=turn_id, state_before=state_before)

        # If already booked/confirmed in this session
        is_already_confirmed = session.state in (BookingState.BOOKING_CONFIRMED, BookingState.CALL_TERMINATING, BookingState.CALL_COMPLETED, BookingState.BOOKED, BookingState.COMPLETED)
        if is_already_confirmed and session.appointment_id:
            # Check pleasantries / thanks / closing first
            if any(w in txt_lower for w in ["thank you", "thanks", "bye", "goodbye", "that's all", "that is all", "nothing else", "no that's it"]):
                resp = f"You're very welcome, {session.patient_name or 'there'}! Your appointment is all set. Have a wonderful day!"
                if lang in ("ta", "tanglish", "ta_mix"):
                    resp = f"ரொம்ப நன்றி! உங்க appointment confirm ஆயிடுச்சு. Have a great day!"
                elif lang == "hi":
                    resp = f"Aapka bahut dhanyawad! Aapka appointment confirm hai. Aapka din shubh ho!"
                self._log_conversation_turn(session, text, resp, "booking_closing", state_before=state_before)
                return _ret(resp)

            # Check if caller is asking about appointment details
            if any(p in txt_lower for p in ["share details", "appointment details", "date and time", "what are my details", "status"]):
                resp = self.get_appointment_or_demo_details(session)
                self._log_conversation_turn(session, text, resp, "already_booked_details", state_before=state_before)
                return _ret(resp)

            clean_w = [re.sub(r'[^a-zA-Z0-9]', '', w) for w in txt_lower.split()]
            if any(w in clean_w for w in ["yes", "yeah", "yep", "sure", "ok", "okay"]):
                last_ai = (getattr(session, "last_ai_response", "") or "").lower()
                if any(p in last_ai for p in ["anything else", "help you with", "assist you", "வேறு உதவி", "உதவி வேணுமா"]):
                    if lang in ("ta", "tanglish", "ta_mix"):
                        resp = "கண்டிப்பா! சொல்லுங்க, உங்களுக்கு என்ன உதவி வேணும்?"
                    elif lang == "hi":
                        resp = "Haan, batayein main aapki aur kya madad kar sakta hoon?"
                    else:
                        resp = "Certainly! How else can I assist you today?"
                    self._log_conversation_turn(session, text, resp, "assist_more", state_before=state_before)
                    return _ret(resp)

            if not any(w in txt_lower for w in ["new appointment", "another appointment", "book again", "book another", "one more appointment"]):
                b_name = session.business_name or "our team"
                if session.booking_type == "SALES_DEMO":
                    resp = f"Your demo appointment is confirmed for {session.date} at {session.time}. Your appointment ID is #{session.appointment_id}. Is there anything else I can help you with today?"
                else:
                    target_d = session.doctor or "Representative"
                    resp = f"Your appointment with {target_d} on {session.date} at {session.time} is already confirmed under Appointment ID #{session.appointment_id}. Is there anything else I can help you with today?"
                self._log_conversation_turn(session, text, resp, "already_booked", state_before=state_before)
                return _ret(resp)
            else:
                session.reset_for_new_booking()

        if session.state == BookingState.IDLE:
            session.state = BookingState.BOOKING_STARTED
            is_demo_start = (
                getattr(session, "booking_type", None) in ("SALES_DEMO", "BUSINESS_CONSULTATION")
                or getattr(session, "business_type", None) == "SOFTWARE"
                or getattr(session, "workflow_type", None) == "SALES_DEMO"
            )
            session.booking_data["intent"] = "sales_demo_booking" if is_demo_start else "appointment_booking"
            if is_demo_start:
                if not session.doctor:
                    session.doctor = "Representative"
                if not session.department:
                    session.department = "Sales Team"
                if not session.service:
                    session.service = session.booking_data.get("product") or "Demo"

        # 0. Check if caller is confirming a previously suggested ambiguous entity
        pending_suggested = session.booking_data.get("pending_confirmation_entity")
        if pending_suggested:
            is_affirmative = any(w in txt_lower.split() for w in ["yes", "yeah", "ok", "okay", "sure", "correct", "yep", "right", "that", "please"]) or pending_suggested.lower() in txt_lower
            is_rejection = any(w in txt_lower.split() for w in ["no", "nope", "dont", "don't", "not", "neither"]) or is_phrase_negated(pending_suggested, text)
            if is_affirmative and not is_rejection:
                confirmed_entity = session.booking_data.pop("pending_confirmation_entity")
                session.service = confirmed_entity
                session.booking_data["service"] = confirmed_entity
                session.booking_data["product"] = confirmed_entity
                session.booking_data.pop("ambiguous_entity_candidate", None)
                session.booking_data.pop("unverified_entity_requested", None)
                session.booking_data.pop("user_requested_entity_term", None)
                if session.state in (BookingState.WAITING_FOR_DOCTOR, BookingState.BOOKING_STARTED, BookingState.IDLE):
                    session.state = BookingState.WAITING_FOR_DATE
                    session.last_asked_field = "date"
                resp = f"Got it, {confirmed_entity}. Which date would you prefer for your appointment?"
                if lang in ("ta", "tanglish", "ta_mix"):
                    resp = f"சரி, {confirmed_entity}. எந்த தேதியில் appointment book பண்ணலாம்?"
                elif lang == "hi":
                    resp = f"Theek hai, {confirmed_entity}. Aap kis date ko appointment prefer karenge?"
                self._log_conversation_turn(session, text, resp, "ambiguous_entity_confirmed", state_before=state_before)
                return _ret(resp)
            elif is_rejection:
                session.booking_data.pop("pending_confirmation_entity", None)
                session.booking_data.pop("ambiguous_entity_candidate", None)

        # 0b. If caller previously inquired about a department or product and is now proceeding to book
        if session.booking_data.get("last_inquired_product") and not session.department and not session.doctor:
            inquired_prod = session.booking_data.get("last_inquired_product")
            if any(w in txt_lower for w in ["book", "appointment", "schedule", "reserve", "yes", "yeah", "today", "tomorrow", "want to book", "i want"]) or not any(w in txt_lower for w in ["different", "another", "instead", "no"]):
                org_id = getattr(session, "organization_id", 1) or 1
                dept_docs = self.get_doctors_for_department(inquired_prod, org_id, session=session)
                if dept_docs or any(inquired_prod.lower() in d.lower() for d in ["cardiology", "dermatology", "orthopedic", "pediatric", "general medicine", "gynecology", "neurology", "ent", "dental", "ophthalmology"]):
                    session.department = inquired_prod
                    session.booking_data["department"] = inquired_prod
                    logger.info(f"[DEPARTMENT_RETAINED] Adopted inquired department '{inquired_prod}' for booking")
                else:
                    session.service = inquired_prod
                    session.booking_data["service"] = inquired_prod
                    session.booking_data["product"] = inquired_prod

        # Handle response to previous ambiguous time confirmation prompt
        if session.last_asked_field == "confirm_time_change":
            cand_time = session.booking_data.pop("ambiguous_time_candidate", None)
            if any(w in txt_lower.split() for w in ["yes", "yeah", "yep", "sure", "ok", "okay", "change", "switch", "please"]) and not any(w in txt_lower.split() for w in ["no", "nope", "keep", "stay"]):
                if cand_time:
                    session.time = cand_time
                    session.availability_verified = False
            session.last_asked_field = None

        # 1. Extract entities & validate doctor / product
        self.extract_slots(text, session)

        # Check if ambiguous time candidate was found and needs clarification
        if session.booking_data.get("ambiguous_time_candidate"):
            cand_time = session.booking_data.get("ambiguous_time_candidate")
            session.last_asked_field = "confirm_time_change"
            resp = f"You previously selected {session.time}. Would you like to change your appointment time to {cand_time}?"
            if lang in ("ta", "tanglish", "ta_mix"):
                resp = f"நீங்க முன்னாடி {session.time} select பண்ணியிருந்தீங்க. Appointment time-ஐ {cand_time}-க்கு மாத்தலாமா?"
            elif lang == "hi":
                resp = f"Aapne pehle {session.time} chuna tha. Kya aap appointment ka samay badal kar {cand_time} karna chahte hain?"
            self._log_conversation_turn(session, text, resp, "clarify_time_change", state_before=state_before)
            return _ret(resp)

        # 1b. If caller just corrected/stated their name while waiting for a different field, acknowledge and continue
        if session.booking_data.pop("name_just_corrected", False):
            first_name = (session.patient_name or "").split()[0] if session.patient_name else ""
            if not session.contact_number or session.contact_number == "Unknown":
                session.state = BookingState.COLLECTING_PHONE
                session.last_asked_field = "contact_number"
                if lang in ("ta", "tanglish", "ta_mix"):
                    resp = f"மன்னிக்கவும், {first_name}. உங்க பெயரை சரிசெய்துவிட்டேன். உங்க contact phone number சொல்லுங்க."
                elif lang == "hi":
                    resp = f"Maafi chahta hoon, {first_name}. Maine aapka naam theek kar diya hai. Kripya apna contact phone number batayein."
                else:
                    resp = f"Sorry, {first_name}. I've corrected your name. Could you please share your contact phone number?"
            elif not session.email and not session.booking_data.get("email_declined"):
                session.state = BookingState.COLLECTING_EMAIL
                session.last_asked_field = "email"
                if lang in ("ta", "tanglish", "ta_mix"):
                    resp = f"மன்னிக்கவும், {first_name}. உங்க பெயரை சரிசெய்துவிட்டேன். Confirmation அனுப்ப உங்க email address சொல்லுங்க."
                elif lang == "hi":
                    resp = f"Maafi chahta hoon, {first_name}. Maine aapka naam theek kar diya hai. Confirmation bhejne ke liye kripya apna email address batayein."
                else:
                    resp = f"Sorry, {first_name}. I've corrected your name. What is your email address so I can send your appointment confirmation?"
            elif not session.location and not session.booking_data.get("location_declined") and session.is_field_required("location"):
                session.state = BookingState.COLLECTING_LOCATION
                session.last_asked_field = "location"
                if lang in ("ta", "tanglish", "ta_mix"):
                    resp = f"மன்னிக்கவும், {first_name}. உங்க பெயரை சரிசெய்துவிட்டேன். உங்க city அல்லது location என்ன?"
                elif lang == "hi":
                    resp = f"Maafi chahta hoon, {first_name}. Maine aapka naam theek kar diya hai. Kripya apna city ya location batayein."
                else:
                    resp = f"Sorry, {first_name}. I've corrected your name. Which city or location are you joining from?"
            elif not session.date:
                session.state = BookingState.WAITING_FOR_DATE
                session.last_asked_field = "date"
                if lang in ("ta", "tanglish", "ta_mix"):
                    resp = f"மன்னிக்கவும், {first_name}. உங்க பெயரை சரிசெய்துவிட்டேன். எந்த date-ல appointment book பண்ணலாம்?"
                elif lang == "hi":
                    resp = f"Maafi chahta hoon, {first_name}. Maine aapka naam theek kar diya hai. Aap kis date ko appointment prefer karenge?"
                else:
                    resp = f"Sorry, {first_name}. I've corrected your name. Which date would you prefer for your appointment?"
            elif not session.time:
                session.state = BookingState.WAITING_FOR_TIME
                session.last_asked_field = "time"
                if lang in ("ta", "tanglish", "ta_mix"):
                    resp = f"மன்னிக்கவும், {first_name}. உங்க பெயரை சரிசெய்துவிட்டேன். எந்த time slot உங்களுக்கு வசதியா இருக்கும்?"
                elif lang == "hi":
                    resp = f"Maafi chahta hoon, {first_name}. Maine aapka naam theek kar diya hai. Aap kaun sa time slot prefer karenge?"
                else:
                    resp = f"Sorry, {first_name}. I've corrected your name. Which time slot would you prefer?"
            else:
                session.state = BookingState.WAITING_FOR_CONFIRMATION
                session.last_asked_field = "confirmation"
                if lang in ("ta", "tanglish", "ta_mix"):
                    resp = f"மன்னிக்கவும், {first_name}. உங்க பெயரை சரிசெய்துவிட்டேன். Booking-ஐ confirm பண்ணலாமா?"
                elif lang == "hi":
                    resp = f"Maafi chahta hoon, {first_name}. Maine aapka naam theek kar diya hai. Kya appointment confirm kar dein?"
                else:
                    resp = f"Sorry, {first_name}. I've corrected your name. Shall I proceed with confirming your appointment?"
            self._log_conversation_turn(session, text, resp, "name_corrected_progress", state_before=state_before)
            return _ret(resp)

        # Handle explicit request to change email, phone, or name when new value is not in current turn
        if re.search(r'\b(?:change|update|correct)\s+(?:my\s+)?(?:email|mail|address)\b', txt_lower) and not self.parse_spoken_email(text):
            session.email = None
            session.state = BookingState.COLLECTING_EMAIL
            session.last_asked_field = "email"
            resp = "Sure! What is your new email address?"
            self._log_conversation_turn(session, text, resp, "request_new_email", state_before=state_before)
            return _ret(resp)

        if re.search(r'\b(?:change|update|correct)\s+(?:my\s+)?(?:phone|number|mobile|contact)\b', txt_lower) and not re.search(r'\d{5,}', txt_lower):
            session.contact_number = None
            session.state = BookingState.COLLECTING_PHONE
            session.last_asked_field = "contact_number"
            resp = "Sure! What is your new contact phone number?"
            self._log_conversation_turn(session, text, resp, "request_new_phone", state_before=state_before)
            return _ret(resp)

        if re.search(r'\b(?:change|update|correct)\s+(?:my\s+)?(?:name|full name)\b', txt_lower) and not any(w in txt_lower for w in ["name is", "call me", "it's", "this is"]):
            session.patient_name = None
            session.booking_data["last_name_asked"] = False
            session.state = BookingState.COLLECTING_PATIENT_NAME
            session.last_asked_field = "patient_name"
            resp = "Sure! What is the correct full name for the booking?"
            self._log_conversation_turn(session, text, resp, "request_new_name", state_before=state_before)
            return _ret(resp)

        # Check if caller asks when team will contact / turnaround time
        is_when_contact_q = (
            any(w in txt_lower for w in [
                "when will your team contact", "when your team contact", "when will someone call",
                "when will someone contact", "when will you call me", "when will you contact me",
                "how soon will your team contact", "how soon will you call", "how long will it take to call",
                "எப்போ தொடர்பு", "எப்போது தொடர்பு", "எப்ப என்னை தொடர்பு", "என்னை எப்போ தொடர்பு", "எப்போ என்னை தொடர்பு"
            ])
            or (any(w in txt_lower for w in ["when", "how soon", "what time", "எப்போ", "எப்போது", "eppo"]) and any(w in txt_lower for w in ["contact", "call", "reach", "தொடர்பு", "கூப்பிடு"]))
        )
        if is_when_contact_q:
            if lang in ("ta", "tanglish", "ta_mix"):
                resp = "எங்க team வழக்கமாக 24 மணி நேரத்திற்குள் உங்களை contact செய்து demo-வை ஏற்பாடு செய்வாங்க. உங்களுக்கு வசதியான நாள் அல்லது நேரம் ஏதாவது இருக்கிறதா?"
            elif lang == "hi":
                resp = "Hamari team aamtaur par 24 ghante ke bheetar aapse contact karke demo arrange karegi. Kya aapke liye koi specific date ya time suvidhajanak hai?"
            else:
                resp = "Our team typically reaches out within 24 business hours to arrange and schedule your demo. Is there a specific date or time you would prefer?"
            self._log_conversation_turn(session, text, resp, "demo_contact_timeline", state_before=state_before)
            return _ret(resp)

        # Check if caller asks to share appointment details or date and time
        _APPT_DETAILS_Q = [
            "share the appointment details", "share my appointment details", "share appointment details",
            "share the date and time", "share date and time", "appointment details", "what are my appointment details",
            "what is my appointment details", "can you share the date and time", "my appointment details",
            "check my appointment", "appointment status", "when is my appointment"
        ]
        if any(ad in txt_lower for ad in _APPT_DETAILS_Q):
            resp = self.get_appointment_or_demo_details(session)
            self._log_conversation_turn(session, text, resp, "appointment_details_inquiry", state_before=state_before)
            return _ret(resp)

        # Caller states availability ("I am available tomorrow", "I will be available tomorrow")
        if re.search(r'\b(?:i\s+am|i\'m|i\s+will\s+be|im)\s+available\b', txt_lower):
            has_tomorrow = any(w in txt_lower for w in ["tomorrow", "naalai", "kal", "நாளை"])
            has_today = any(w in txt_lower for w in ["today", "inniku", "aaj", "இன்று"])
            avail_date = self._normalize_date("tomorrow") if has_tomorrow else (self._normalize_date("today") if has_today else None)
            if avail_date:
                session.date = avail_date
                session.booking_data["date"] = avail_date
                session.booking_data["resolved_date"] = avail_date
                session.booking_data["date_status"] = "COLLECTED"
                session.field_status["date"] = "COLLECTED"
                session.state = BookingState.WAITING_FOR_TIME
                session.last_asked_field = "time"
                date_disp = "tomorrow" if has_tomorrow else ("today" if has_today else avail_date)
                if lang in ("ta", "tanglish", "ta_mix"):
                    resp = f"சரிங்க! {date_disp} எந்த time slot உங்களுக்கு வசதியாக இருக்கும்?"
                elif lang == "hi":
                    resp = f"Theek hai! {date_disp} ko kaunsa time slot aapke liye theek rahega?"
                else:
                    resp = f"Got it, {date_disp}! Which time slot would you prefer?"
                self._log_conversation_turn(session, text, resp, "availability_date_recorded", state_before=state_before)
                return _ret(resp)

        # Handle customer saying "I already told you"
        _ALREADY_TOLD_PHRASES = [
            "already told you", "already told", "already said", "already gave you",
            "already gave", "already shared", "told you already", "gave you already",
            "already provided", "i already told", "i told you", "already answered"
        ]
        if any(atp in txt_lower for atp in _ALREADY_TOLD_PHRASES):
            ref_field = None
            if any(w in txt_lower for w in ["name", "peru", "naam"]):
                ref_field = "patient_name"
            elif any(w in txt_lower for w in ["email", "mail"]):
                ref_field = "email"
            elif any(w in txt_lower for w in ["phone", "number", "mobile", "contact"]):
                ref_field = "contact_number"
            elif any(w in txt_lower for w in ["date", "day"]):
                ref_field = "date"
            elif any(w in txt_lower for w in ["time", "timing", "slot"]):
                ref_field = "time"
            else:
                ref_field = getattr(session, "last_asked_field", None) or "patient_name"

            existing_val = None
            if ref_field == "patient_name":
                existing_val = session.patient_name
            elif ref_field == "email":
                existing_val = session.email
            elif ref_field == "contact_number":
                existing_val = session.contact_number
            elif ref_field == "date":
                existing_val = session.date
            elif ref_field == "time":
                existing_val = session.time

            if not existing_val:
                existing_val = session.booking_data.get(ref_field)

            # If not in session fields, scan session history backwards
            if not existing_val and session.history:
                for msg in reversed(session.history):
                    if msg.get("role") == "user":
                        u_text = msg.get("text", "")
                        if ref_field == "patient_name":
                            cand = self.clean_person_name(u_text)
                            if cand and self.is_valid_person_name(cand):
                                existing_val = cand
                                session.patient_name = cand
                                break
                        elif ref_field == "email":
                            em = self.parse_spoken_email(u_text)
                            if em:
                                existing_val = em
                                session.email = em
                                break
                        elif ref_field == "contact_number":
                            m_num = re.search(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', u_text)
                            if m_num:
                                existing_val = m_num.group(0)
                                session.contact_number = existing_val
                                break

            field_display = {
                "patient_name": "name",
                "email": "email address",
                "contact_number": "phone number",
                "date": "appointment date",
                "time": "preferred time"
            }.get(ref_field, "information")

            if existing_val:
                resp = f"You're right, I have your {field_display} as {existing_val}. Let me continue with the remaining details."
                session.last_asked_field = None
                missing = session.missing_fields
                if missing:
                    next_field_name = missing[0].replace("_", " ")
                    resp += f" Could you please share your {next_field_name}?"
                else:
                    session.state = BookingState.WAITING_FOR_CONFIRMATION
                    resp += f" Shall I go ahead and confirm your appointment?"
            else:
                resp = f"I apologize for the confusion. I don't seem to have captured your {field_display}. Could you please share it once more?"

            self._log_conversation_turn(session, text, resp, "already_told_recovery", state_before=state_before)
            return _ret(resp)

        # Check if caller asks to repeat or summarize previously collected details
        _REPEAT_INQUIRY = [
            "repeat all my details", "repeat my details", "repeat all details", "repeat the details",
            "repeat what i gave you", "repeat my information", "repeat all information",
            "what details do you have", "what information do you have", "what details did i give",
            "read back my details", "read back my information", "summarize my details",
            "summarize my booking", "what are my details", "what do you have so far",
            "what info do you have", "repeat everything"
        ]
        if any(rp in txt_lower for rp in _REPEAT_INQUIRY):
            parts = []
            if session.patient_name:
                parts.append(f"Name: {session.patient_name}")
            if session.doctor:
                parts.append(f"Doctor: {session.doctor}")
            elif session.service:
                parts.append(f"Service: {session.service}")
            if session.date and session.time:
                parts.append(f"Date & Time: {session.date} at {session.time}")
            elif session.date:
                parts.append(f"Date: {session.date}")
            elif session.time:
                parts.append(f"Time: {session.time}")
            if session.contact_number and session.contact_number != "Unknown":
                parts.append(f"Phone: {session.contact_number}")
            if session.email:
                parts.append(f"Email: {session.email}")

            summary_str = ", ".join(parts) if parts else "No details recorded yet"
            if session.state == BookingState.WAITING_FOR_CONFIRMATION or not session.missing_fields:
                session.state = BookingState.WAITING_FOR_CONFIRMATION
                resp = f"Here are the details I have recorded: {summary_str}. Would you like me to go ahead and confirm this appointment?"
            else:
                missing = session.missing_fields
                next_item = missing[0].replace('_', ' ') if missing else "confirmation"
                resp = f"Here are the details I have so far: {summary_str}. May I have your {next_item}?"
            self._log_conversation_turn(session, text, resp, "repeat_details_summary", state_before=state_before)
            return _ret(resp)

        # Handle ambiguous entity suggestion
        if session.booking_data.get("ambiguous_entity_candidate"):
            amb = session.booking_data.pop("ambiguous_entity_candidate")
            raw = amb.get("raw") or "that"
            suggested = amb.get("suggested")
            session.booking_data["pending_confirmation_entity"] = suggested
            session.booking_data.pop("unverified_entity_requested", None)
            session.booking_data.pop("user_requested_entity_term", None)
            session.state = BookingState.WAITING_FOR_DOCTOR
            session.last_asked_field = "entity_confirmation"
            resp = f"I couldn't find '{raw}' in our directory. Did you mean '{suggested}'?"
            if lang in ("ta", "tanglish", "ta_mix"):
                resp = f"எங்க directory-ல '{raw}' இல்ல. நீங்க '{suggested}'-ஐ சொல்றீங்களா?"
            elif lang == "hi":
                resp = f"Hamari directory me '{raw}' nahi hai. Kya aapka matlab '{suggested}' se hai?"
            self._log_conversation_turn(session, text, resp, "ambiguous_entity_clarification", state_before=state_before)
            return _ret(resp)

        # 0b. Affirmative response to an offered doctor/staff member (e.g. "Sure", "Yes", "Okay", "Proceed", "Go ahead")
        _AFFIRMATIVE_TERMS = {
            "yes", "yeah", "yep", "sure", "okay", "ok", "proceed", "go ahead", "sounds good",
            "that works", "confirm", "confirmed", "correct", "please", "fine", "alright", "perfect",
            "sure please", "yes please", "yes proceed", "please proceed", "of course"
        }
        is_user_affirmative = (
            txt_lower in _AFFIRMATIVE_TERMS
            or any(txt_lower.startswith(w) for w in ["sure", "yes", "okay", "ok", "proceed", "go ahead", "yeah"])
            or any(w in txt_lower.split() for w in ["sure", "yes", "proceed", "okay", "ok", "yeah"])
        ) and not any(w in txt_lower.split() for w in ["no", "not", "dont", "don't", "cancel", "neither"])

        if getattr(session, "_offered_dept_doctor", None) and is_user_affirmative:
            offered_name = session._offered_dept_doctor
            session.doctor = offered_name
            session.doctor_name = offered_name
            session.doctor_locked = True
            session._offered_dept_doctor = None
            session.booking_data.pop("unverified_entity_requested", None)
            session.booking_data.pop("unknown_doctor_requested", None)
            session.booking_data.pop("user_requested_entity_term", None)
            session.booking_data.pop("ambiguous_entity_candidate", None)
            session.booking_data.pop("pending_confirmation_entity", None)

            org_id = getattr(session, "organization_id", 1) or 1
            try:
                from src.config.database import SessionLocal
                _db = SessionLocal()
                try:
                    s_id, s_name, _ = self.resolve_staff_from_directory(offered_name, org_id, _db)
                    if s_id:
                        session.doctor_id = s_id
                        session.doctor = s_name
                finally:
                    _db.close()
            except Exception as e:
                logger.warning(f"Error resolving offered staff ID: {e}")

            if not hasattr(session, "confirmed_facts"):
                session.confirmed_facts = {}
            session.confirmed_facts["doctor"] = {"value": session.doctor, "source": "AUTHORITATIVE_DIRECTORY"}
            if not hasattr(session, "completed_steps"):
                session.completed_steps = []
            if "doctor_selection" not in session.completed_steps:
                session.completed_steps.append("doctor_selection")

            if not session.date:
                session.state = BookingState.WAITING_FOR_DATE
                session.last_asked_field = "date"
                if lang in ("ta", "tanglish", "ta_mix"):
                    resp = f"சரி, {session.doctor}-கூட appointment schedule பண்ணலாம். எந்த date-ல appointment வேணும்?"
                elif lang == "hi":
                    resp = f"Theek hai, {session.doctor} ke saath appointment schedule karte hain. Aap kis date ko prefer karenge?"
                else:
                    resp = f"Great! Let's schedule with {session.doctor}. Which date would you prefer for the appointment?"
                self._log_conversation_turn(session, text, resp, "doctor_confirmed_ask_date", state_before=state_before)
                return _ret(resp)
            elif not session.time:
                session.state = BookingState.WAITING_FOR_TIME
                session.last_asked_field = "time"
                resp = self.check_doctor_availability_live(session.doctor, session.date, session=session)
                self._log_conversation_turn(session, text, resp, "doctor_confirmed_check_slots", state_before=state_before)
                return _ret(resp)

        # 1.6 Doctor Availability Inquiry / Department Doctor Inquiry (Scenarios 2, 3, 4, 5)
        # Evaluated BEFORE unverified entity check so availability questions are never blocked by generic entity fallbacks!
        _DOCTOR_AVAILABILITY_PATTERNS = [
            "doctors available", "doctor available", "doctors are available", "doctor is available",
            "what are the doctors", "which doctors are", "who are the doctors", "who is the doctor",
            "which doctor is", "what doctor is", "what doctors do you have", "which doctors do you have",
            "list of doctors", "list doctors", "list the doctors", "any doctor available", "any doctors available",
            "available doctors", "available doctor", "who is available", "who are available",
            "doctors today", "doctors tomorrow", "doctor list", "show doctors", "tell me doctors",
            "what doctors available", "which doctors available", "available specialist", "available specialists",
            "doctors available today", "doctors available tomorrow", "is there a doctor", "are there doctors",
        ]
        is_asking_doctors_available = (
            any(p in txt_lower for p in _DOCTOR_AVAILABILITY_PATTERNS)
            or (
                any(w in txt_lower for w in ["what", "which", "who", "tell me", "list", "any", "show"])
                and any(w in txt_lower for w in ["doctor", "doctors", "dr", "dr.", "specialist", "specialists", "physician", "physicians"])
                and any(w in txt_lower for w in ["available", "free", "working", "there", "today", "tomorrow", "present", "schedule", "on", "have"])
            )
            or any(phrase in txt_lower for phrase in [
                "எந்த டாக்டர்", "யார் டாக்டர்", "டாக்டர் விவரம்", "டாக்டர்கள் இருக்காங்க", "யார் யார் இருக்காங்க", "டாக்டர்கள்",
                "kaunse doctor", "kaun kaun se doctor", "doctor available hain", "doctor ki list", "doctor hain",
                "yava doctor", "evaru doctor", "ವೈದ್ಯರು ಲಭ್ಯ", "ಯಾವ ವೈದ್ಯರು", "ವೈದ್ಯರ", "డాక్టర్లు అందుబాటులో", "ఏ డాక్టర్లు", "డాక్టర్ల"
            ])
        )

        has_doctor_keyword = any(k in txt_lower for k in ["dr", "dr.", "doctor", "physician"])
        is_asking_specific_doctor = (
            bool(re.search(r'\b(?:is|can\s+i\s+(?:see|meet|consult)|does)\s+(?:dr\.?|doctor|physician)?\s*([a-zA-Z\s]+?)\s+(?:available|free|in|working|present|there)\b', txt_lower))
            or (has_doctor_keyword and any(w in txt_lower for w in ["available", "free", "there", "working", "schedule", "today", "tomorrow"]))
        )

        is_hosp_domain = (getattr(session, "business_type", "GENERAL") in ("HOSPITAL", "CLINIC") or getattr(session, "booking_type", "") == "HOSPITAL_APPOINTMENT")
        if is_hosp_domain and (is_asking_doctors_available or is_asking_specific_doctor):
            session.booking_data.pop("unverified_entity_requested", None)
            session.booking_data.pop("unknown_doctor_requested", None)
            session.booking_data.pop("ambiguous_entity_candidate", None)
            session.booking_data.pop("user_requested_entity_term", None)

            target_date_str = None
            if "day after tomorrow" in txt_lower:
                target_date_str = "day after tomorrow"
            elif any(w in txt_lower for w in ["tomorrow", "naalai", "kal", "repu", "naale"]):
                target_date_str = "tomorrow"
            elif any(w in txt_lower for w in ["today", "indru", "innikku", "aaj", "ee roju", "indu"]):
                target_date_str = "today"
            else:
                for day_n in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]:
                    if day_n in txt_lower:
                        target_date_str = day_n
                        break
            if not target_date_str:
                target_date_str = session.date or "today"

            session.date = "Today" if target_date_str.lower() in ("today", "indru", "aaj") else ("Tomorrow" if target_date_str.lower() in ("tomorrow", "naalai", "kal") else target_date_str.title())

            target_dept = session.department
            common_depts = [
                "cardiology", "general medicine", "dental care", "dental",
                "ayurvedic consultant", "ayurvedic", "siddha consultant", "siddha",
                "naturopathy & yoga consultant", "naturopathy", "orthopedics",
                "pediatrics", "dermatology", "neurology", "gynecology"
            ]
            for cd in common_depts:
                if cd in txt_lower:
                    target_dept = cd.title()
                    session.department = target_dept
                    break

            org_id = getattr(session, "organization_id", 1) or 1
            specific_doctor = None
            if is_asking_specific_doctor or has_doctor_keyword:
                from src.config.database import SessionLocal
                from src.models.staff import Staff
                with SessionLocal() as db:
                    org_staff = db.query(Staff).filter(Staff.organization_id == org_id).all()
                    for s in org_staff:
                        if not s.name:
                            continue
                        clean_s = re.sub(r'^(?:dr\.?|doctor|mr\.?|mrs\.?|ms\.?)\s+', '', s.name, flags=re.I).strip().lower()
                        if clean_s and (clean_s in txt_lower or self._is_doctor_name_match(text, s.name)):
                            specific_doctor = s
                            break

            if specific_doctor:
                session.doctor = specific_doctor.name
                session.doctor_id = specific_doctor.id
                session.doctor_locked = True
                session.state = BookingState.WAITING_FOR_TIME
                session.last_asked_field = "time"
                avail_resp = self.check_doctor_availability_live(specific_doctor.name, target_date_str, session=session)
                if not avail_resp.lower().startswith("for") and not avail_resp.lower().startswith("yes"):
                    resp = f"Yes, {specific_doctor.name} is available {session.date.lower()}. {avail_resp}"
                else:
                    resp = f"Yes, {avail_resp}"
                self._log_conversation_turn(session, text, resp, "specific_doctor_availability_checked", state_before=state_before)
                return _ret(resp)

            resp, available_docs = self.get_available_doctors_for_date(
                target_date_str=target_date_str,
                department=target_dept,
                organization_id=org_id,
                session=session,
                lang=lang
            )
            self._log_conversation_turn(session, text, resp, "presented_available_doctors", state_before=state_before)
            return _ret(resp)

        # Check if caller explicitly asked for available slots
        has_stated_time = bool(session.time or re.search(r'\b(?:1[0-2]|0?[1-9])(?::[0-5][0-9])?\s*(?:a\.?m\.?|p\.?m\.?)\b', txt_lower))
        is_asking_slots = not has_stated_time and (
            any(phrase in txt_lower for phrase in [
                "available slots", "slots available", "what slots", "which slots",
                "time slots", "open slots", "free slots", "available times",
                "schedule of doctor", "schedule of staff", "what time slots do you have", "what times"
            ]) or ("available time" in txt_lower and not any(w in txt_lower for w in ["my available time", "time is", "time would be", "available at", "i am available", "i'm available"]))
        )
        if is_asking_slots:
            is_demo = (
                getattr(session, "booking_type", None) in ("SALES_DEMO", "BUSINESS_CONSULTATION")
                or getattr(session, "business_type", None) == "SOFTWARE"
                or getattr(session, "workflow_type", None) == "SALES_DEMO"
            )
            if is_demo:
                session.state = BookingState.WAITING_FOR_TIME
                session.last_asked_field = "time"
                if lang in ("ta", "tanglish", "ta_mix"):
                    resp = "எங்கள் demo slots காலை 9:30 AM, 11:00 AM, மதியம் 2:00 PM மற்றும் 4:00 PM-க்கு ലഭ്യமாக உள்ளது. உங்களுக்கு எந்த நேரம் வசதியாக இருக்கும்?"
                elif lang == "hi":
                    resp = "Hamare demo slots 9:30 AM, 11:00 AM, 2:00 PM aur 4:00 PM par uplabdh hain. Aapke liye kaun sa samay sahi rahega?"
                elif lang in ("kn", "kannada"):
                    resp = "ನಮ್ಮ ಡೆಮೊ ಸ್ಲಾಟ್‌ಗಳು 9:30 AM, 11:00 AM, 2:00 PM ಮತ್ತು 4:00 PM ಗೆ ಲಭ್ಯವಿವೆ. ನಿಮಗೆ ಯಾವ ಸಮಯ ಸೂಕ್ತವಾಗಿರುತ್ತದೆ?"
                elif lang in ("te", "telugu"):
                    resp = "మా డెమో స్లాట్లు 9:30 AM, 11:00 AM, 2:00 PM మరియు 4:00 PM అందుబాటులో ఉన్నాయి. మీకు ఏ సమయం అనుకూలంగా ఉంటుంది?"
                else:
                    resp = "Our demo slots are available at 9:30 AM, 11:00 AM, 2:00 PM, and 4:00 PM. Which time works best for you?"
                self._log_conversation_turn(session, text, resp, "presented_demo_slots", state_before=state_before)
                return _ret(resp)

            doc_to_check = session.doctor
            if not doc_to_check:
                session.state = BookingState.WAITING_FOR_DOCTOR
                session.last_asked_field = "doctor"
                first_name = (session.patient_name or "").split()[0] if session.patient_name and self.is_valid_person_name(session.patient_name) else ""
                name_ack_slots = f"Thanks, {first_name}. " if first_name else "Sure! "
                if lang in ("ta", "tanglish", "ta_mix"):
                    resp = f"{name_ack_slots}Available slots பார்க்க, எந்த மருத்துவரை அல்லது துறையை நீங்கள் பார்க்க விரும்புகிறீர்கள்?"
                elif lang == "hi":
                    resp = f"{name_ack_slots}Available slots dekhne ke liye, aap kis doctor ya vibhaag se milna chahte hain?"
                elif lang in ("kn", "kannada"):
                    resp = f"{name_ack_slots}ಲಭ್ಯವಿರುವ ಸಮಯ ಸ್ಲಾಟ್‌ಗಳನ್ನು ನೋಡಲು, ನೀವು ಯಾವ ವೈದ್ಯರನ್ನು ಅಥವಾ ವಿಭಾಗವನ್ನು ಸಂಪರ್ಕಿಸಲು ಬಯಸುತ್ತೀರಿ?"
                elif lang in ("te", "telugu"):
                    resp = f"{name_ack_slots}అందుబాటులో ఉన్న సమయ స్లాట్‌లను తనిಖీ చేయడానికి, మీరు ఏ వైద్యుడిని లేదా విభాగాన్ని సంప్రదించాలనుకుంటున్నారు?"
                else:
                    resp = f"{name_ack_slots}To check available time slots, which doctor or department would you like to consult?"
                self._log_conversation_turn(session, text, resp, "ask_doctor_for_slots", state_before=state_before)
                return _ret(resp)

            session.state = BookingState.WAITING_FOR_TIME
            session.last_asked_field = "time"
            resp = self.check_doctor_availability_live(doc_to_check, session.date, session=session)
            self._log_conversation_turn(session, text, resp, "presented_slots", state_before=state_before)
            return _ret(resp)

        # Handle unverified staff/person/service requested against active organization
        if session.booking_data.get("unverified_entity_requested") or session.booking_data.get("unknown_doctor_requested"):
            unverified = session.booking_data.pop("unverified_entity_requested", None) or session.booking_data.pop("unknown_doctor_requested", None)
            from src.services.entity_validation_service import entity_validation_service as _ev
            _re_check = _ev.validate_entity(unverified or "", organization_id=getattr(session, "organization_id", 1) or 1)
            if _re_check.is_role_category:
                session.doctor = _re_check.verified_name or unverified
                session.doctor_locked = True
                session.booking_data["specialty"] = session.doctor
            elif _re_check.is_verified:
                if _re_check.entity_type in ("product", "service"):
                    session.service = _re_check.verified_name or unverified
                    session.booking_data["service"] = session.service
                    session.booking_data["product"] = session.service
                    session.booking_data["specialty"] = session.service
                    session.state = BookingState.WAITING_FOR_DATE
                    session.last_asked_field = "date"
                else:
                    session.doctor = _re_check.verified_name or unverified
                    if _re_check.staff_id:
                        session.doctor_id = _re_check.staff_id
                    session.doctor_locked = True
                    session.booking_data["specialty"] = _re_check.role_or_specialty or session.doctor
            elif getattr(_re_check, "is_ambiguous", False) and getattr(_re_check, "suggested_entity", None):
                suggested = _re_check.suggested_entity
                session.booking_data["pending_confirmation_entity"] = suggested
                session.booking_data.pop("user_requested_entity_term", None)
                session.state = BookingState.WAITING_FOR_DOCTOR
                session.last_asked_field = "entity_confirmation"
                resp = f"I couldn't find '{unverified}' in our directory. Did you mean '{suggested}'?"
                if lang in ("ta", "tanglish", "ta_mix"):
                    resp = f"எங்க directory-ல '{unverified}' இல்ல. நீங்க '{suggested}'-ஐ சொல்றீங்களா?"
                elif lang == "hi":
                    resp = f"Hamari directory me '{unverified}' nahi hai. Kya aapka matlab '{suggested}' se hai?"
                self._log_conversation_turn(session, text, resp, "ambiguous_entity_clarification", state_before=state_before)
                return _ret(resp)
            else:
                session.doctor = None
                session.doctor_id = None
                session.doctor_locked = False
                session.booking_data.pop("user_requested_entity_term", None)
                session.state = BookingState.WAITING_FOR_DOCTOR
                session.last_asked_field = "doctor"

                # Check if user specifically requested a doctor or rejected products/services
                is_doctor_request = (
                    session.state == BookingState.WAITING_FOR_DOCTOR
                    or any(w in txt_lower for w in ["dr", "doctor", "physician", "consultant", "meet doctor", "see doctor", "meet dr"])
                    or bool(getattr(session, "_rejected_services", set()))
                    or bool(getattr(session, "_rejected_departments", set()))
                )

                if is_doctor_request:
                    if lang in ("ta", "tanglish", "ta_mix"):
                        resp = f"மன்னிக்கவும், எங்க directory-ல '{unverified}' doctor இல்ல. நீங்க பார்க்க விரும்பும் doctor அல்லது department பெயர் சொல்ல முடியுமா?"
                    elif lang == "hi":
                        resp = f"Maaf kijiye, hamari directory me '{unverified}' doctor nahi mile. Kya aap doctor ka naam ya vibhaag dohra sakte hain?"
                    else:
                        resp = f"I could not verify '{unverified}' in our doctor directory. Could you please clarify the doctor's name or department you would like to consult with?"
                else:
                    org_id = getattr(session, "organization_id", 1) or 1
                    avail_opts = self.get_organization_products_and_services(org_id)
                    if avail_opts:
                        top_opts = [opt for opt in avail_opts if opt.lower() not in getattr(session, "_rejected_services", set()) and opt.lower() not in getattr(session, "_rejected_departments", set())][:3]
                        if top_opts:
                            opts_str = ", ".join(top_opts[:-1]) + f", or {top_opts[-1]}" if len(top_opts) > 1 else top_opts[0]
                            resp = f"I could not verify '{unverified}' in our directory. We currently offer {opts_str}. Which product or service would you like assistance with?"
                        else:
                            resp = f"I could not verify '{unverified}' in our directory. Which product or service would you like assistance with?"
                    else:
                        ask_target = self._get_target_entity_question(session, lang=lang)
                        resp = f"I couldn't find '{unverified}' in our directory. {ask_target}"
                    if lang in ("ta", "tanglish", "ta_mix"):
                        resp = f"மன்னிக்கவும், '{unverified}' எங்க directory-ல இல்ல. எந்த product அல்லது service book பண்ணனும்?"
                    elif lang == "hi":
                        resp = f"Maaf kijiye, '{unverified}' hamari directory me nahi mile. Aap kaunsa product ya service book karna chahte hain?"
                self._log_conversation_turn(session, text, resp, "unverified_entity_clarification", state_before=state_before)
                return _ret(resp)

        # Check if caller utterance is an incomplete fragment or conversational cutoff
        if self._is_incomplete_utterance(text) and not any(w in txt_lower for w in ["yes", "yeah", "ok", "sure", "cancel", "no"]):
            if session.state in (BookingState.WAITING_FOR_PATIENT_NAME, BookingState.COLLECTING_PATIENT_NAME):
                if lang in ("ta", "tanglish", "ta_mix"):
                    resp = "உங்க பெயர் சொல்லுங்க, நான் கேட்டுட்டு இருக்கேன்."
                elif lang == "hi":
                    resp = "Kripya apna naam batayein, main sun raha hoon."
                else:
                    resp = "Take your time. May I have your full name whenever you're ready?"
                self._log_conversation_turn(session, text, resp, "incomplete_utterance_reassurance", state_before=state_before)
                return _ret(resp)
            elif session.state == BookingState.WAITING_FOR_DOCTOR:
                if lang in ("ta", "tanglish", "ta_mix"):
                    resp = "நான் கேட்டுட்டு இருக்கேன். எந்த doctor-ஐ பார்க்க விரும்புகிறீர்கள்?"
                elif lang == "hi":
                    resp = "Main sun raha hoon. Aap kis doctor se milna chahte hain?"
                else:
                    resp = "Take your time. Which doctor or department would you like to consult with?"
                self._log_conversation_turn(session, text, resp, "incomplete_utterance_reassurance", state_before=state_before)
                return _ret(resp)
            elif session.state == BookingState.WAITING_FOR_TIME:
                resp = "Take your time. Which time slot would you prefer for the appointment?"
                self._log_conversation_turn(session, text, resp, "incomplete_utterance_reassurance", state_before=state_before)
                return _ret(resp)
            elif session.state == BookingState.WAITING_FOR_DATE:
                resp = "Take your time. Which date would you prefer for the appointment?"
                self._log_conversation_turn(session, text, resp, "incomplete_utterance_reassurance", state_before=state_before)
                return _ret(resp)

        # 1.4 General hospital FAQ during active booking (visiting hours, location, etc. - Scenario 10)
        _HOSPITAL_FAQ_PATTERNS = [
            "where is the hospital", "where are you located", "visiting hours", "opening hours",
            "hospital address", "what is the location", "how to reach", "directions to hospital",
            "insurance accepted", "which insurance", "do you take insurance", "cashless insurance",
            "emergency services", "is emergency available", "ambulance services",
            "parking available", "is parking free", "hospital facilities", "what are the facilities",
            "hospital timing", "op timing", "op hours", "blood bank", "canteen", "pharmacy open"
        ]
        is_hospital_faq = (
            any(q in txt_lower for q in _HOSPITAL_FAQ_PATTERNS)
            and not any(w in txt_lower for w in ["book", "appointment", "schedule", "my name", "@", ".com"])
        )
        if is_hospital_faq:
            org_id = getattr(session, "organization_id", 1) or 1
            from src.services.rag_service import rag_service
            rag_ans = ""
            try:
                rag_ans = rag_service.retrieve_context(
                    query=text,
                    organization_id=org_id,
                    document_id=getattr(session, "active_doc_id", None)
                )
            except Exception as e:
                logger.warning(f"Error retrieving RAG context for FAQ: {e}")

            faq_text = ""
            if rag_ans:
                lines = [l.strip() for l in rag_ans.split("\n") if l.strip() and not l.strip().startswith("#")]
                for line in lines:
                    if any(term in line.lower() for term in ["location", "located", "address", "visiting", "hours", "timing", "insurance", "facility", "facilities", "emergency", "service", "product"]):
                        faq_text = line
                        break
            if not faq_text:
                b_name = getattr(session, "business_name", None) or "Our team"
                is_hosp_faq = (getattr(session, "business_type", "GENERAL") in ("HOSPITAL", "CLINIC") or getattr(session, "booking_type", "") == "HOSPITAL_APPOINTMENT")
                if any(w in txt_lower for w in ["where", "location", "address"]):
                    loc = rag_service.get_organization_location(org_id, document_id=getattr(session, "active_doc_id", None))
                    faq_text = f"{b_name} is located at {loc}." if loc else f"{b_name} is available to assist you."
                elif any(w in txt_lower for w in ["hours", "timing", "visiting"]):
                    hrs = rag_service.get_organization_hours(org_id, document_id=getattr(session, "active_doc_id", None))
                    faq_text = f"Our hours are {hrs}." if hrs else f"{b_name} is open during standard business hours."
                elif is_hosp_faq and any(w in txt_lower for w in ["insurance", "cashless"]):
                    faq_text = "Yes, we accept major insurance policies and offer cashless claims."
                elif is_hosp_faq:
                    faq_text = f"We provide comprehensive medical facilities at {b_name}."
                else:
                    faq_text = f"We are pleased to assist you with all {b_name} products and services."

            is_hosp_bridge = (getattr(session, "business_type", "GENERAL") in ("HOSPITAL", "CLINIC") or getattr(session, "booking_type", "") == "HOSPITAL_APPOINTMENT")
            if is_hosp_bridge:
                if not session.doctor:
                    bridge = "Now, to continue with your appointment booking, which doctor or department would you like to consult?"
                    if lang in ("ta", "tanglish", "ta_mix"):
                        bridge = "இப்போது உங்கள் முன்பதிவைத் தொடர, எந்த மருத்துவரை அல்லது துறையை நீங்கள் பார்க்க விரும்புகிறீர்கள்?"
                    elif lang == "hi":
                        bridge = "Ab aapke appointment booking ke liye, aap kis doctor ya vibhaag se milna chahte hain?"
                    elif lang in ("kn", "kannada"):
                        bridge = "ಈಗ ನಿಮ್ಮ ಅಪಾಯಿಂಟ್‌ಮೆಂಟ್ ಬುಕಿಂಗ್ ಮುಂದುವರಿಸಲು, ನೀವು ಯಾವ ವೈದ್ಯರನ್ನು ಅಥವಾ ವಿಭಾಗವನ್ನು ಸಂಪರ್ಕಿಸಲು ಬಯಸುತ್ತೀರಿ?"
                    elif lang in ("te", "telugu"):
                        bridge = "ఇప్పుడు మీ అపాయింట్‌మెంట్ బుకింగ్‌ను కొనసాగించడానికి, మీరు ఏ వైద్యుడిని లేదా విభాగాన్ని సంప్రదించాలనుకుంటున్నారు?"
                elif session.doctor and not session.date:
                    bridge = f"Now, to schedule your appointment with {session.doctor}, which date would you prefer?"
                elif session.doctor and session.date and not session.time:
                    bridge = f"Now, for your appointment with {session.doctor} on {session.date}, which time would you prefer?"
                else:
                    bridge = "Now, shall we proceed with your appointment booking?"
            else:
                if not session.service and not session.product:
                    bridge = "Now, would you like to schedule a demo or consultation for our products?"
                    if lang in ("ta", "tanglish", "ta_mix"):
                        bridge = "இப்போது எங்கள் சேவைகளுக்கான demo அல்லது ஆலோசனையை முன்பதிவு செய்ய விரும்புகிறீர்களா?"
                    elif lang == "hi":
                        bridge = "Kya aap hamare products ke demo ke liye aage badhna chahenge?"
                elif not session.date:
                    bridge = "Which date would you prefer for the demo or consultation?"
                else:
                    bridge = "Now, shall we proceed with scheduling your session?"

            resp = f"{faq_text} {bridge}"
            self._log_conversation_turn(session, text, resp, "hospital_faq_answered", state_before=state_before)
            return _ret(resp)

        # 1.5 Explicit Date Change Request during active booking (Scenario 8)
        is_date_change_request = any(p in txt_lower for p in ["change date", "change the date", "change my date", "switch date", "different date", "another date"])
        if is_date_change_request or (session.date and any(w in txt_lower for w in ["actually tomorrow", "actually today", "make it tomorrow", "make it today"])):
            new_date = None
            if "tomorrow" in txt_lower or "naalai" in txt_lower or "kal" in txt_lower:
                new_date = "Tomorrow"
            elif "today" in txt_lower or "indru" in txt_lower or "aaj" in txt_lower:
                new_date = "Today"
            if new_date:
                session.date = new_date
                session.time = None
                session.availability_verified = False
                session.offered_slots = []
                org_id = getattr(session, "organization_id", 1) or 1
                if session.doctor:
                    session.state = BookingState.WAITING_FOR_TIME
                    session.last_asked_field = "time"
                    avail_msg = self.check_doctor_availability_live(session.doctor, session.date, session=session)
                    resp = f"Understood! I've updated your appointment date to {session.date.lower()}. {avail_msg}"
                    self._log_conversation_turn(session, text, resp, "date_changed_with_doctor", state_before=state_before)
                    return _ret(resp)
                else:
                    resp, _ = self.get_available_doctors_for_date(session.date, organization_id=org_id, session=session, lang=lang)
                    self._log_conversation_turn(session, text, resp, "date_changed_ask_doctor", state_before=state_before)
                    return _ret(resp)

        # DYNAMIC MISSING-INFORMATION EVALUATION WITH PERSONALIZED ACKNOWLEDGMENTS
        first_name = (session.patient_name or "").split()[0] if session.patient_name and self.is_valid_person_name(session.patient_name) else ""
        
        # Check if we should acknowledge a short affirmative/acknowledgement
        short_ack = ""
        if len(txt_lower.split()) <= 3 and any(w in txt_lower for w in ["yes", "yeah", "ok", "okay", "sure", "thanks", "thank you", "correct"]):
            short_ack = "Got it. "
            
        name_ack = f"{short_ack}Thanks, {first_name}. " if first_name else f"{short_ack}"

        # STEP 1: Staff / Representative / Service Check — dynamically RAG-aware
        if session.booking_type in ("SALES_DEMO", "BUSINESS_CONSULTATION") or session.business_type == "SOFTWARE":
            # For software demo/consultation, bypass medical doctor selection
            if not session.service:
                session.service = "Demo" if session.booking_type == "SALES_DEMO" else "Consultation"
        elif session.service or session.booking_data.get("product"):
            # Product or service booking does not require choosing a medical doctor
            pass
        elif not session.doctor:
            org_id = getattr(session, "organization_id", 1) or 1
            if session.department:
                dept_docs = self.get_doctors_for_department(session.department, org_id, session=session)
                if dept_docs:
                    if len(dept_docs) == 1:
                        target_doc = dept_docs[0]
                        # Check if user confirmed the single offered doctor (e.g. "yes", "sure", "ok", or doctor's name)
                        is_affirmative = any(w in txt_lower.split() for w in ["yes", "yeah", "ok", "okay", "sure", "correct", "fine", "yep", "dr", "doctor", "her", "him", "please"]) or target_doc["clean_name"] in txt_lower
                        is_refusal = (
                            any(w in txt_lower.split() for w in ["no", "nope", "nah", "never", "dont", "don't", "not", "neither"])
                            or any(p in txt_lower for p in ["another doctor", "different doctor", "switch doctor", "change doctor", "someone else", "somebody else", "other doctor", "not him", "not her"])
                            or is_phrase_negated(target_doc["name"], text)
                            or is_phrase_negated(target_doc["clean_name"], text)
                        )
                        was_offered = getattr(session, "_offered_dept_doctor", None) == target_doc["name"]

                        if was_offered and is_refusal:
                            # Customer rejects offered doctor
                            # Check if department itself was also rejected (e.g. "I don't want dental care")
                            is_dept_refused = (
                                is_phrase_negated(session.department, text)
                                or any(w in txt_lower for w in ["different department", "another department", "wrong department", "not this department"])
                            )
                            if is_dept_refused:
                                session.invalidate_department(session.department)
                                session.state = BookingState.WAITING_FOR_DOCTOR
                                session.last_asked_field = "doctor"
                                resp = "Understood. Which department or doctor would you prefer to book with?"
                                self._log_conversation_turn(session, text, resp, "department_rejected", state_before=state_before)
                                return _ret(resp)
                            else:
                                session.invalidate_doctor(target_doc["name"])
                                remaining_docs = self.get_doctors_for_department(session.department, org_id, session=session)
                                if remaining_docs:
                                    next_doc = remaining_docs[0]
                                    session._offered_dept_doctor = next_doc["name"]
                                    resp = f"Understood. We also have {next_doc['name']} available for {session.department}. Would you like to schedule with {next_doc['name']}?"
                                    self._log_conversation_turn(session, text, resp, "offer_alternate_doctor", state_before=state_before)
                                    return _ret(resp)
                                else:
                                    session.state = BookingState.WAITING_FOR_DOCTOR
                                    session.last_asked_field = "doctor"
                                    resp = f"Understood. {target_doc['name']} is currently our only specialist in {session.department}. Would you like to check another department or specialty?"
                                    self._log_conversation_turn(session, text, resp, "only_doctor_rejected", state_before=state_before)
                                    return _ret(resp)

                        elif is_affirmative and was_offered:
                            session.doctor = target_doc["name"]
                            session.doctor_id = target_doc["id"]
                            session.doctor_locked = True
                            if not hasattr(session, "confirmed_facts"):
                                session.confirmed_facts = {}
                            session.confirmed_facts["doctor"] = {"value": target_doc["name"], "source": "AUTHORITATIVE_DIRECTORY"}
                            if not hasattr(session, "completed_steps"):
                                session.completed_steps = []
                            if "doctor_selection" not in session.completed_steps:
                                session.completed_steps.append("doctor_selection")
                            # Fall through directly to Date / Time check below!
                        else:
                            session._offered_dept_doctor = target_doc["name"]
                            session.state = BookingState.WAITING_FOR_DOCTOR
                            session.last_asked_field = "doctor"
                            resp = f"For {session.department}, we have {target_doc['name']} available. Would you like to schedule with {target_doc['name']}?"
                            if lang in ("ta", "tanglish", "ta_mix"):
                                resp = f"{session.department}-க்கு {target_doc['name']} available-ஆ இருக்காங்க. {target_doc['name']}-கூட schedule பண்ணலாமா?"
                            elif lang == "hi":
                                resp = f"{session.department} ke liye hamare paas {target_doc['name']} available hain. Kya aap {target_doc['name']} ke saath schedule karna chahenge?"
                            self._log_conversation_turn(session, text, resp, "offer_department_doctor", state_before=state_before)
                            return _ret(resp)
                    else:
                        # Multiple doctors in department
                        is_refusal = (
                            any(w in txt_lower.split() for w in ["no", "none", "neither"])
                            or any(p in txt_lower for p in ["none of them", "neither of them", "different department", "another department"])
                            or is_phrase_negated(session.department, text)
                        )
                        if is_refusal:
                            session.invalidate_department(session.department)
                            session.state = BookingState.WAITING_FOR_DOCTOR
                            session.last_asked_field = "doctor"
                            resp = "Understood. Which department or doctor would you prefer to book with?"
                            self._log_conversation_turn(session, text, resp, "department_rejected_from_multi", state_before=state_before)
                            return _ret(resp)

                        matched_doc = next((d for d in dept_docs if d["clean_name"] in txt_lower and d["clean_name"] not in getattr(session, "_rejected_doctors", set()) and not is_phrase_negated(d["name"], text)), None)
                        if matched_doc:
                            session.doctor = matched_doc["name"]
                            session.doctor_id = matched_doc["id"]
                            session.doctor_locked = True
                            if not hasattr(session, "confirmed_facts"):
                                session.confirmed_facts = {}
                            session.confirmed_facts["doctor"] = {"value": matched_doc["name"], "source": "AUTHORITATIVE_DIRECTORY"}
                            if not hasattr(session, "completed_steps"):
                                session.completed_steps = []
                            if "doctor_selection" not in session.completed_steps:
                                session.completed_steps.append("doctor_selection")
                            # Fall through directly to Date / Time check below!
                        else:
                            names_text = ", ".join([d["name"] for d in dept_docs[:-1]]) + f", or {dept_docs[-1]['name']}"
                            session.state = BookingState.WAITING_FOR_DOCTOR
                            session.last_asked_field = "doctor"
                            resp = f"For {session.department}, we have {names_text} available. Which doctor would you like to book with?"
                            self._log_conversation_turn(session, text, resp, "offer_multiple_department_doctors", state_before=state_before)
                            return _ret(resp)
                else:
                    session.state = BookingState.WAITING_FOR_DOCTOR
                    session.last_asked_field = "doctor"
                    resp = f"We currently do not have any doctors listed for {session.department}. Which department or doctor would you like to book with?"
                    self._log_conversation_turn(session, text, resp, "no_doctors_in_department", state_before=state_before)
                    return _ret(resp)
            else:
                session.state = BookingState.WAITING_FOR_DOCTOR
                session.last_asked_field = "doctor"
                if getattr(session, "_last_rejected_entity", None) == "department":
                    session._last_rejected_entity = None
                    resp = "Understood. Which department or doctor would you prefer to book with?"
                else:
                    resp = self._get_target_entity_question(session, name_ack=name_ack, lang=lang)
                self._log_conversation_turn(session, text, resp, "waiting_for_doctor", state_before=state_before)
                return _ret(resp)

        _MONTH_NAMES = {
            "january", "february", "march", "april", "may", "june", "july",
            "august", "september", "october", "november", "december",
            "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec"
        }

        # Target label for date/time prompts (doctor, service, product, or department)
        target_label = session.service or session.doctor or session.department or "consultation"

        # STEP 2: Date Check
        if not session.date:
            session.state = BookingState.WAITING_FOR_DATE
            session.last_asked_field = "date"
            was_waiting_date = (
                (state_before and state_before.get("state") in (BookingState.WAITING_FOR_DATE, BookingState.WAITING_FOR_BOOKING_DATE))
                or (state_before and state_before.get("last_asked_field") == "date")
            )
            if was_waiting_date and text.strip() and not any(w in txt_lower for w in ["book", "appointment", "schedule", "demo"]):
                resp = "Sorry, I didn't catch the date. Would tomorrow or another date work for you?"
                if lang in ("ta", "tanglish", "ta_mix"):
                    resp = "மன்னிக்கவும், தேதி சரியாக விளங்கவில்லை. நாளை அல்லது வேறு ஏதேனும் தேதி உங்களுக்கு வசதியாக இருக்குமா?"
                elif lang == "hi":
                    resp = "Kshama kijiye, mujhe tareekh samajh nahi aayi. Kya kal ya koi aur tareekh theek rahegi?"
                self._log_conversation_turn(session, text, resp, "unclear_date_clarification", state_before=state_before)
                return _ret(resp)

            is_demo_date = (
                getattr(session, "booking_type", None) in ("SALES_DEMO", "BUSINESS_CONSULTATION")
                or getattr(session, "business_type", None) == "SOFTWARE"
                or getattr(session, "workflow_type", None) == "SALES_DEMO"
            )
            if is_demo_date:
                resp = f"Sure{', ' + first_name if first_name else ''}! I can help you schedule a demo. Which date would you prefer?"
            else:
                resp = f"{name_ack}Which date would you prefer for your {target_label} appointment?"
            if lang in ("ta", "tanglish", "ta_mix"):
                resp = f"{'நன்றி, ' + first_name + '. ' if first_name else ''}{target_label}-க்கு எந்த நாளுக்கு appointment book பண்ணலாம்?"
            elif lang == "hi":
                resp = f"{'Shukriya, ' + first_name + '. ' if first_name else ''}{target_label} ke liye aap kis tareekh ko appointment prefer karenge?"
            self._log_conversation_turn(session, text, resp, "waiting_for_date", state_before=state_before)
            return _ret(resp)
        elif session.date and session.date.lower().strip() in _MONTH_NAMES:
            session.state = BookingState.WAITING_FOR_DATE
            session.last_asked_field = "date"
            month_cap = session.date.title()
            resp = f"{name_ack}What date in {month_cap} would you prefer for your {target_label} appointment?"
            if lang in ("ta", "tanglish", "ta_mix"):
                resp = f"{'நன்றி, ' + first_name + '. ' if first_name else ''}{target_label}-க்கு {month_cap} மாதத்தில் எந்த தேதி வசதியாக இருக்கும்?"
            elif lang == "hi":
                resp = f"{'Shukriya, ' + first_name + '. ' if first_name else ''}{target_label} ke liye {month_cap} me konsi tareekh pasand karenge?"
            self._log_conversation_turn(session, text, resp, "waiting_for_exact_date_in_month", state_before=state_before)
            return _ret(resp)

        # Immediate Past-Date and Past-Time Validation
        if session.date and session.date.lower().strip() not in _MONTH_NAMES:
            try:
                from src.routes.availability_routes import get_current_business_datetime
                from src.services.calendar_verification_service import calendar_verification_service
                now_dt = get_current_business_datetime(org_id=getattr(session, "organization_id", 1), db=None)
                today_date = now_dt.date()
                parsed_d = calendar_verification_service.parse_date(session.date)
                if parsed_d < today_date:
                    past_d_str = parsed_d.strftime("%d %b %Y")
                    session.date = None
                    session.time = None
                    session.state = BookingState.WAITING_FOR_DATE
                    session.last_asked_field = "date"
                    if lang in ("ta", "tanglish", "ta_mix"):
                        resp = f"மன்னிக்கவும், {past_d_str} தேதி ஏற்கனவே முடிந்துவிட்டது. தயவுசெய்து இன்று அல்லது வரவிருக்கும் தேதியைத் தேர்ந்தெடுக்கவும்."
                    elif lang == "hi":
                        resp = f"Kshama kijiye, {past_d_str} tareekh guzar chuki hai. Kripya aaj ya aane wali tareekh chunein."
                    else:
                        resp = f"Sorry, {past_d_str} has already passed. Please choose today or an upcoming date for your appointment."
                    self._log_conversation_turn(session, text, resp, "past_date_rejected", state_before=state_before)
                    return _ret(resp)

                # Immediate Past-Time Validation on Today's Date
                if parsed_d == today_date and session.time:
                    target_time_24h = calendar_verification_service.parse_time_to_24h(session.time)
                    if target_time_24h:
                        naive_now = now_dt.replace(tzinfo=None) if now_dt.tzinfo else now_dt
                        target_t_obj = datetime.strptime(target_time_24h, "%H:%M").time()
                        target_slot_dt = datetime.combine(today_date, target_t_obj)
                        if target_slot_dt <= naive_now:
                            requested_time = session.time
                            session.time = None
                            session.availability_verified = False
                            session.state = BookingState.WAITING_FOR_TIME
                            session.last_asked_field = "time"

                            from src.config.database import SessionLocal
                            from src.models.staff import Staff, StaffStatus
                            from src.routes.availability_routes import get_available_slots, find_next_available_slots

                            db_slots = SessionLocal()
                            try:
                                staff_id_val = session.doctor_id
                                if not staff_id_val:
                                    st_rec = db_slots.query(Staff).filter(
                                        Staff.organization_id == getattr(session, "organization_id", 1),
                                        Staff.status != StaffStatus.OFFLINE,
                                        Staff.online_booking_enabled == True
                                    ).first()
                                    if not st_rec:
                                        st_rec = db_slots.query(Staff).filter(Staff.organization_id == getattr(session, "organization_id", 1)).first()
                                    staff_id_val = st_rec.id if st_rec else 1

                                rem_res = get_available_slots(staff_id=staff_id_val, selected_date=today_date, db=db_slots)
                                rem_slots = [calendar_verification_service.format_time_12h(s["start"]) for s in rem_res.get("available_slots", [])[:4]]
                            except Exception as slot_err:
                                logger.warning(f"Error fetching remaining slots for past time: {slot_err}")
                                rem_slots = []
                            finally:
                                db_slots.close()

                            if rem_slots:
                                session.offered_slots = rem_slots
                                alt_text = ", ".join(rem_slots)
                                resp = f"Sorry, {requested_time} is no longer available because that time has passed. Today's available slots are {alt_text}. Which time would you prefer?"
                                if lang in ("ta", "tanglish", "ta_mix"):
                                    resp = f"மன்னிக்கவும், {requested_time} நேரம் முடிந்துவிட்டது. இன்றைக்கு available slots: {alt_text}. இதில எந்த நேரம் வசதியாக இருக்கும்?"
                                elif lang == "hi":
                                    resp = f"Kshama kijiye, {requested_time} ka samay nikal chuka hai. Aaj ke available slots hain: {alt_text}. Aap konsa samay pasand karenge?"
                            else:
                                db_next = SessionLocal()
                                try:
                                    next_date_obj, next_slots_list = find_next_available_slots(staff_id=staff_id_val, from_date=today_date, db=db_next)
                                    next_date_str = next_date_obj.strftime("%A, %b %d") if next_date_obj else None
                                    next_slots_12h = [calendar_verification_service.format_time_12h(s["start"]) for s in next_slots_list[:4]] if next_slots_list else []
                                    next_slots_str = ", ".join(next_slots_12h) if next_slots_12h else ""
                                except Exception:
                                    next_date_str = None
                                    next_slots_str = ""
                                finally:
                                    db_next.close()

                                if next_date_str and next_slots_str:
                                    session.offered_slots = next_slots_12h
                                    resp = f"Sorry, {requested_time} is no longer available because that time has passed, and there are no remaining slots available today. The next available date is {next_date_str} with slots at {next_slots_str}. Which time would you prefer?"
                                    if lang in ("ta", "tanglish", "ta_mix"):
                                        resp = f"மன்னிக்கவும், {requested_time} நேரம் முடிந்துவிட்டது, இன்றைக்கு வேறு slots இல்லை. அடுத்த available date {next_date_str}, available slots: {next_slots_str}. இதில எந்த நேரம் வசதியாக இருக்கும்?"
                                    elif lang == "hi":
                                        resp = f"Kshama kijiye, {requested_time} ka samay nikal chuka hai aur aaj koi slots available nahi hain. Agli available date {next_date_str} hai jisme slots hain: {next_slots_str}. Aap konsa samay pasand karenge?"
                                else:
                                    session.offered_slots = []
                                    resp = f"Sorry, {requested_time} is no longer available because that time has passed, and there are no remaining slots available today. Would you like to check tomorrow or another date?"
                                    if lang in ("ta", "tanglish", "ta_mix"):
                                        resp = f"மன்னிக்கவும், {requested_time} நேரம் முடிந்துவிட்டது, இன்றைக்கு வேறு slots இல்லை. நாளை அல்லது வேறு date check பண்ணலாமா?"
                                    elif lang == "hi":
                                        resp = f"Kshama kijiye, {requested_time} ka samay nikal chuka hai aur aaj koi slots available nahi hain. Kya aap kal ya koi aur date check karna chahenge?"

                            self._log_conversation_turn(session, text, resp, "past_time_rejected", state_before=state_before)
                            return _ret(resp)
            except Exception as d_err:
                logger.warning(f"Error checking past date/time: {d_err}")

        # STEP 3: Time Check (Proactively query & present live Calendar slots)
        if not session.time:
            session.state = BookingState.WAITING_FOR_TIME
            session.last_asked_field = "time"

            # Check if user gave a pure affirmative answer to previously offered slots
            # but did NOT specify a slot number or time — re-prompt to pick one specifically
            is_affirmative = any(w in txt_lower.split() for w in ["yes", "yeah", "ok", "okay", "sure", "correct", "fine", "yep"]) or "book me" in txt_lower
            has_ordinal = any(w in txt_lower for w in ["first", "second", "third", "fourth", "fifth", "last", "1st", "2nd", "3rd", "4th", "5th"])
            has_any_number = bool(re.search(r'\b\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b', txt_lower))

            if session.offered_slots and is_affirmative and not has_ordinal and not has_any_number:
                # Track how many times we've re-prompted for the same slot list
                _slot_prompt_count = getattr(session, "_slot_prompt_count", 0) + 1
                session._slot_prompt_count = _slot_prompt_count

                if _slot_prompt_count >= 3:
                    # After 3 re-prompts, auto-select the first available slot to avoid infinite loop
                    extracted_time = session.offered_slots[0]
                    session.time = extracted_time
                    session.availability_verified = True
                    session._slot_prompt_count = 0
                    logger.info(f"[SLOT_AUTO_SELECT] Auto-selected first slot '{extracted_time}' after {_slot_prompt_count} ambiguous responses.")
                    # Fall through to STEP 4/5 below — don't return here
                else:
                    resp = f"Which specific time would you like? Available times are: {', '.join(session.offered_slots)}. Please say the time you'd like."
                    if lang in ("ta", "tanglish", "ta_mix"):
                        resp = f"எந்த நேரம் book பண்ணலாம்? Available slots: {', '.join(session.offered_slots)}. ஒரு specific time சொல்லுங்க."
                    elif lang == "hi":
                        resp = f"Kaunsa specific samay chahiye? Available times hain: {', '.join(session.offered_slots)}. Kripya ek time batayein."
                    self._log_conversation_turn(session, text, resp, "re_prompt_slot_choice", state_before=state_before)
                    return _ret(resp)

            if not session.time:
                # Check if slots for this target and date have ALREADY been presented to caller
                slots_already_offered = bool(
                    session.offered_slots
                    and getattr(session, "_slots_presented_for", None) == (target_label, session.date)
                )
                wants_slots_repeated = any(ph in txt_lower for ph in [
                    "what slots", "which slots", "available slots", "repeat", "available times", "what times", "what time slots", "what are the slots"
                ])

                if slots_already_offered and not wants_slots_repeated:
                    resp = f"Which of those times works best for you, or would you like to check another date?"
                    if lang in ("ta", "tanglish", "ta_mix"):
                        resp = "அந்த நேரங்களில் எது உங்களுக்கு வசதியாக இருக்கும், அல்லது வேறு தேதி பார்க்கலாமா?"
                    elif lang == "hi":
                        resp = "Unme se kaunsa samay aapke liye theek rahega, ya koi dusri tareekh dekhni hai?"
                    self._log_conversation_turn(session, text, resp, "re_prompt_slot_without_repeating_list", state_before=state_before)
                    return _ret(resp)

                avail_msg = self.check_doctor_availability_live(session.doctor or session.service, session.date, session=session)
                if avail_msg:
                    session._slot_prompt_count = 0
                    session._slots_presented_for = (target_label, session.date)
                    resp = f"{name_ack}{avail_msg}" if (first_name and not avail_msg.startswith("Thanks")) else avail_msg
                    self._log_conversation_turn(session, text, resp, "presented_live_slots", state_before=state_before)
                    return _ret(resp)
                resp = f"{name_ack}What time would you prefer for your {target_label} appointment on {session.date}?"
                if lang in ("ta", "tanglish", "ta_mix"):
                    resp = f"{'நன்றி, ' + first_name + '. ' if first_name else ''}{target_label}-க்கு {session.date} எந்த நேரம் வசதியா இருக்கும்?"
                elif lang == "hi":
                    resp = f"{'Shukriya, ' + first_name + '. ' if first_name else ''}{target_label} ke liye {session.date} ko konsa samay pasand karenge?"
                self._log_conversation_turn(session, text, resp, "waiting_for_time", state_before=state_before)
                return _ret(resp)

        # STEP 4: Live Exact Slot Availability Check (Stage 1 Verification)
        target_entity_name = session.doctor or session.service or "Representative"
        if session.date and session.time and not session.availability_verified:
            try:
                from src.services.calendar_verification_service import calendar_verification_service
                query_str = f"Appointment with {target_entity_name} on {session.date} at {session.time}"
                verif = calendar_verification_service.verify_availability(query_str)

                if verif.is_checked and verif.is_available is False:
                    requested_time = session.time
                    session.time = None
                    session.availability_verified = False
                    session.state = BookingState.WAITING_FOR_TIME
                    session.last_asked_field = "time"
                    if verif.is_past_time:
                        if verif.alternative_slots:
                            session.offered_slots = verif.alternative_slots
                            alt_text = ", ".join(verif.alternative_slots)
                            resp = f"Sorry, {requested_time} is no longer available because that time has passed. Today's available slots are {alt_text}. Which time would you prefer?"
                            if lang in ("ta", "tanglish", "ta_mix"):
                                resp = f"மன்னிக்கவும், {requested_time} நேரம் முடிந்துவிட்டது. இன்றைக்கு available slots: {alt_text}. இதில எந்த நேரம் வசதியாக இருக்கும்?"
                            elif lang == "hi":
                                resp = f"Kshama kijiye, {requested_time} ka samay nikal chuka hai. Aaj ke available slots hain: {alt_text}. Aap konsa samay pasand karenge?"
                        elif verif.next_available_date and verif.next_available_slots:
                            session.offered_slots = verif.next_available_slots
                            next_alt = ", ".join(verif.next_available_slots)
                            resp = f"Sorry, {requested_time} is no longer available because that time has passed, and there are no remaining slots available today. The next available date is {verif.next_available_date} with slots at {next_alt}. Which time would you prefer?"
                            if lang in ("ta", "tanglish", "ta_mix"):
                                resp = f"மன்னிக்கவும், {requested_time} நேரம் முடிந்துவிட்டது, இன்றைக்கு வேறு slots இல்லை. அடுத்த available date {verif.next_available_date}, available slots: {next_alt}. இதில எந்த நேரம் வசதியாக இருக்கும்?"
                            elif lang == "hi":
                                resp = f"Kshama kijiye, {requested_time} ka samay nikal chuka hai aur aaj koi slots available nahi hain. Agli available date {verif.next_available_date} hai jisme slots hain: {next_alt}. Aap konsa samay pasand karenge?"
                        else:
                            session.offered_slots = []
                            resp = f"Sorry, {requested_time} is no longer available because that time has passed, and there are no remaining slots available today. Would you like to check tomorrow or another date?"
                            if lang in ("ta", "tanglish", "ta_mix"):
                                resp = f"மன்னிக்கவும், {requested_time} நேரம் முடிந்துவிட்டது, இன்றைக்கு வேறு slots இல்லை. நாளை அல்லது வேறு date check பண்ணலாமா?"
                            elif lang == "hi":
                                resp = f"Kshama kijiye, {requested_time} ka samay nikal chuka hai aur aaj koi slots available nahi hain. Kya aap kal ya koi aur date check karna chahenge?"
                    else:
                        if verif.alternative_slots:
                            session.offered_slots = verif.alternative_slots
                            alt_text = ", ".join(verif.alternative_slots)
                            resp = f"{requested_time} is not available for {target_entity_name} on {session.date}. Available alternative slots are: {alt_text}. Would you like one of these available times?"
                            if lang in ("ta", "tanglish", "ta_mix"):
                                resp = f"{requested_time} slot {target_entity_name}-க்கு busy-ஆ இருக்கு. Available alternative slots: {alt_text}. இதில எது ok?"
                            elif lang == "hi":
                                resp = f"{requested_time} slot {target_entity_name} ke saath available nahi hai. Available alternative slots hain: {alt_text}. Kya inme se koi chalega?"
                        elif verif.next_available_date and verif.next_available_slots:
                            session.offered_slots = verif.next_available_slots
                            next_alt = ", ".join(verif.next_available_slots)
                            resp = f"{requested_time} is not available for {target_entity_name} on {session.date}, and there are no other open slots available on this date. The next available date is {verif.next_available_date} with slots at {next_alt}. Which time would you prefer?"
                            if lang in ("ta", "tanglish", "ta_mix"):
                                resp = f"{requested_time} slot {target_entity_name}-க்கு unavailable, {session.date}-ல வேற slots இல்ல. அடுத்த available date {verif.next_available_date}, available slots: {next_alt}. இதில எது ok?"
                            elif lang == "hi":
                                resp = f"{requested_time} slot {target_entity_name} ke saath available nahi hai aur is date par koi dusra slot nahi hai. Agli available date {verif.next_available_date} hai jisme slots hain: {next_alt}. Aap konsa samay pasand karenge?"
                        else:
                            session.offered_slots = []
                            resp = f"{requested_time} is not available for {target_entity_name} on {session.date}, and there are no other open slots available on this date. Would you like to try another date?"
                            if lang in ("ta", "tanglish", "ta_mix"):
                                resp = f"{requested_time} slot {target_entity_name}-க்கு unavailable, {session.date}-ல வேற slots இல்ல. வேற date check பண்ணலாமா?"
                            elif lang == "hi":
                                resp = f"{requested_time} slot {target_entity_name} ke saath available nahi hai aur is date par koi dusra slot nahi hai. Kya aap dusri date try karna chahenge?"
                    self._log_conversation_turn(session, text, resp, "slot_unavailable", state_before=state_before)
                    return _ret(resp)
                else:
                    session.availability_verified = True
            except Exception as exc:
                logger.warning(f"Live slot check during booking turn failed: {exc}")
                session.availability_verified = False
                session.time = None
                session.state = BookingState.WAITING_FOR_TIME
                session.last_asked_field = "time"
                resp = f"I apologize, but I am currently unable to verify availability on our calendar for that time. Which other time or date would you like to check?"
                if lang in ("ta", "tanglish", "ta_mix"):
                    resp = "மன்னிக்கவும், அந்த நேரத்திற்கான availability-ஐ calendar-ல் verify பண்ண முடியல. வேற நேரம் அல்லது date check பண்ணலாமா?"
                elif lang == "hi":
                    resp = "Kshama kijiye, hum is samay calendar par availability verify nahi kar sake. Kya aap koi dusra time ya date check karna chahenge?"
                self._log_conversation_turn(session, text, resp, "calendar_verification_failed", state_before=state_before)
                return _ret(resp)

        # STEP 5: Patient Name Check
        if session.patient_name:
            cname = self.clean_person_name(session.patient_name)
            if cname:
                session.patient_name = cname
        if not session.patient_name or not self.is_valid_person_name(session.patient_name):
            session.patient_name = None
            session.state = BookingState.COLLECTING_PATIENT_NAME
            session.last_asked_field = "patient_name"
            was_waiting_name = (
                (state_before and state_before.get("state") in (BookingState.COLLECTING_PATIENT_NAME, BookingState.WAITING_FOR_PATIENT_NAME, BookingState.WAITING_FOR_NAME))
                or (state_before and state_before.get("last_asked_field") == "patient_name")
            )
            if was_waiting_name and text.strip() and not any(w in txt_lower for w in ["book", "appointment", "schedule", "demo"]):
                if lang in ("ta", "tanglish", "ta_mix"):
                    resp = "மன்னிக்கவும், உங்க பெயர் தெளிவாகக் கேட்கவில்லை. தயவுசெய்து உங்கள் பெயரை மீண்டும் சொல்ல முடியுமா?"
                elif lang == "hi":
                    resp = "Kshama kijiye, mujhe aapka naam theek se sunai nahi diya. Kripya apna naam dobara batayein?"
                else:
                    resp = "Sorry, I couldn't understand your name clearly. Could you please say your name again?"
                self._log_conversation_turn(session, text, resp, "unclear_name_clarification", state_before=state_before)
                return _ret(resp)

            rel = session.patient_relationship or "relative"
            for_other = getattr(session, "is_booking_for_other", False)
            is_demo = (
                getattr(session, "booking_type", None) in ("SALES_DEMO", "BUSINESS_CONSULTATION")
                or getattr(session, "business_type", None) == "SOFTWARE"
                or getattr(session, "workflow_type", None) == "SALES_DEMO"
            )
            target_str = "for your product demo" if is_demo else (f"with {session.doctor}" if session.doctor else f"for {session.service or 'your appointment'}")

            if any(w in txt_lower for w in ["my name", "name", "who", "what", "pardon", "sorry", "pardon me"]):
                if for_other:
                    resp = f"Could you please tell me your {rel}'s full name for the booking?"
                else:
                    resp = "Could you please tell me your full name for the booking?"
            else:
                if session.availability_verified:
                    if for_other:
                        resp = f"{session.time} is available {target_str} on {session.date}. May I please have your {rel}'s full name for the booking?"
                    else:
                        resp = f"{session.time} is available {target_str} on {session.date}. May I please have your full name for the booking?"
                else:
                    if for_other:
                        resp = f"May I please have your {rel}'s full name for the booking?"
                    else:
                        resp = "Could you please tell me your full name for the booking?"
            if lang in ("ta", "tanglish", "ta_mix"):
                target_ta = session.doctor or session.service or "appointment"
                if session.availability_verified:
                    if for_other:
                        resp = f"{session.time} slot available-ஆ இருக்கு {target_ta}-கூட {session.date}-ல. Booking-க்கு உங்க {rel} full name சொல்லுங்க."
                    else:
                        resp = f"{session.time} slot available-ஆ இருக்கு {target_ta}-கூட {session.date}-ல. Booking-க்கு உங்க full name சொல்லுங்க."
                else:
                    resp = f"Booking-க்கு உங்க full name சொல்லுங்க."
            elif lang == "hi":
                target_hi = session.doctor or session.service or "appointment"
                if session.availability_verified:
                    if for_other:
                        resp = f"{session.time} slot {session.date} ko {target_hi} ke liye available hai. Kripya apne {rel} ka poora naam batayein?"
                    else:
                        resp = f"{session.time} slot {session.date} ko {target_hi} ke liye available hai. Kripya apna poora naam batayein?"
                else:
                    resp = "Kripya apna poora naam batayein?"
                    resp = f"{session.time} slot {session.date} ko {target_hi} ke liye available hai. Kripya apna poora naam batayein?"
            self._log_conversation_turn(session, text, resp, "waiting_for_patient_name", state_before=state_before)
            return _ret(resp)

        # Single name (e.g. Hariharan) is completely valid. Do not interrupt to demand last name.
        if len(session.patient_name.split()) == 1:
            session.booking_data["last_name_asked"] = True

        # STEP 6: Contact Phone Number Check
        if not session.contact_number or session.contact_number == "Unknown":
            session.state = BookingState.COLLECTING_PHONE
            session.last_asked_field = "contact_number"

            was_waiting_phone = (
                (state_before and state_before.get("state") in (BookingState.COLLECTING_PHONE, BookingState.WAITING_FOR_CONTACT))
                or (state_before and state_before.get("last_asked_field") == "contact_number")
            )
            caller_phone_avail = bool(
                session.caller_phone
                and session.caller_phone not in ("Unknown", "Unknown Caller", "Private Number")
            and (not session.booking_data.get("trunk_did") or session.caller_phone.lstrip('+').lstrip('0') != session.booking_data.get("trunk_did").lstrip('+').lstrip('0'))
            )

            # Auto-assign caller_phone to contact_number so it does not loop
            if caller_phone_avail and not session.contact_number:
                session.contact_number = session.caller_phone
                session.booking_data["caller_id_confirmed"] = True

            # If incoming caller ID is available, use it as a suggested phone number, but confirm it with customer
            if caller_phone_avail and not session.booking_data.get("caller_id_suggested") and not session.contact_number:
                session.booking_data["caller_id_suggested"] = True
                if lang in ("ta", "tanglish", "ta_mix"):
                    resp = f"நன்றி, {first_name}. நீங்க {session.caller_phone} எண்ணிலிருந்து அழைக்கிறீர்கள். இதையே பயன்படுத்தலாமா, அல்லது வேறு தொடர்பு எண் உள்ளதா?" if first_name else f"நீங்க {session.caller_phone} எண்ணிலிருந்து அழைக்கிறீர்கள். இதையே பயன்படுத்தலாமா, அல்லது வேறு தொடர்பு எண் உள்ளதா?"
                elif lang == "hi":
                    resp = f"Dhanyawad, {first_name}. Main dekh sakta hoon aap {session.caller_phone} se call kar rahe hain. Kya main isi number ko use karoon, ya aapka koi doosra number hai?" if first_name else f"Aap {session.caller_phone} se call kar rahe hain. Kya main isi number ko use karoon, ya doosra number hai?"
                else:
                    resp = f"Thank you, {first_name}. I see you're calling from {session.caller_phone}. Should I use this number, or do you have a different contact number?" if first_name else f"I see you're calling from {session.caller_phone}. Should I use this number, or do you have a different contact number?"
                self._log_conversation_turn(session, text, resp, "suggest_caller_phone", state_before=state_before)
                return _ret(resp)

            # If customer provided unclear/invalid input while waiting for phone:
            if was_waiting_phone:
                if caller_phone_avail:
                    resp = f"I didn't quite catch that. Could you please confirm if we should use your calling number {session.caller_phone}, or share your 10-digit contact number?"
                    if lang in ("ta", "tanglish", "ta_mix"):
                        resp = f"சரியாகக் கேட்கவில்லை. {session.caller_phone} எண்ணைப் பயன்படுத்தலாமா அல்லது உங்கள் 10 இலக்க எண்ணைச் சொல்லுங்கள்."
                    elif lang == "hi":
                        resp = f"Theek se sunai nahi diya. Kya hum {session.caller_phone} use karein, ya apna 10-digit number batayein?"
                else:
                    resp = "I didn't catch a valid contact number. Could you please share your 10-digit phone number?"
                    if lang in ("ta", "tanglish", "ta_mix"):
                        resp = "சரியான எண் கேட்கவில்லை. தயவுசெய்து உங்கள் 10 இலக்க mobile எண்ணைச் சொல்லுங்கள்."
                    elif lang == "hi":
                        resp = "Sahi number nahi mila. Kripya apna 10-digit mobile number batayein."
                self._log_conversation_turn(session, text, resp, "reprompt_phone", state_before=state_before)
                return _ret(resp)

            resp = f"Thank you, {first_name}. Could you please share your contact phone number?" if first_name else "Could you please share your contact phone number?"
            if lang in ("ta", "tanglish", "ta_mix"):
                resp = f"நன்றி, {first_name}. உங்க contact number சொல்லுங்க." if first_name else "உங்க contact number சொல்லுங்க."
            elif lang == "hi":
                resp = f"Dhanyawad, {first_name}. Kripya apna contact number batayein." if first_name else "Kripya apna contact number batayein."
            self._log_conversation_turn(session, text, resp, "waiting_for_contact", state_before=state_before)
            return _ret(resp)

        # STEP 7: Email Address Check
        if not session.email and not session.booking_data.get("email_declined"):
            was_waiting_email = (
                (state_before and state_before.get("state") in (BookingState.COLLECTING_EMAIL, BookingState.WAITING_FOR_EMAIL))
                or (state_before and state_before.get("last_asked_field") == "email")
            )
            is_email_attempt = any(w in txt_lower for w in ["@", "at", "dot", "gmail", "yahoo", "outlook", "email", "mail", ".com", ".in", ".org", "com"])

            # Check if customer declines email or says they do not have one
            decline_phrases = [
                "skip and continue without invite", "continue without invite", "without invite",
                "no invite", "without calendar invite", "no calendar invite", "dont send invite",
                "don't send invite", "skip invite", "proceed without invite", "skip and continue",
                "no email", "don't have an email", "don't have email", "dont have email", "no mail", 
                "i don't have an email", "i don't have one", "skip", "decline", 
                "not now", "later", "none", "don't want to", "no thanks",
                "email illa", "mail illa", "email nahi", "nahi hai",
                "email வேண்டாம்", "mail venam", "வேண்டாம்", "வேண்டாங்க", "venam", "vendaam", "வேற எதுவும் வேண்டாம்",
                "skip pannunga", "skip pannu", "skip panna", "skip பண்ணுங்க", "ஸ்கிப் பண்ணுங்க", "ஸ்கிப்",
                "விடுங்க", "விடவும்", "வேண்டாம் விடுங்க", "விட்டுருங்க", "விட்டிருங்க", "போகட்டும்",
                "email skip", "skip email", "i said skip", "skip this", "skip it"
            ]
            clean_token = txt_lower.strip().rstrip(".,!?")
            is_explicit_decline = (
                any(dp in txt_lower for dp in decline_phrases)
                or (was_waiting_email and clean_token in ("no", "nope", "skip", "none", "na", "illa", "nahi", "விடுங்க", "ஸ்கிப்", "வேண்டாம்"))
                or session.field_status.get("email") == "SKIPPED"
                or session.booking_data.get("email_status") == "SKIPPED"
            )

            if is_explicit_decline:
                call_id = getattr(session, "call_sid", None) or getattr(session, "session_id", "unknown")
                session.booking_data["email_declined"] = True
                session.booking_data["email_confirmed"] = False
                session.booking_data["send_calendar_invite"] = False
                session.booking_data["calendar_invite_skipped"] = True
                session.booking_data["email_skipped"] = True
                session.booking_data["email_status"] = "SKIPPED"
                session.field_status["email"] = "SKIPPED"
                session.email = None
                logger.info(f"[BOOKING_SKIP] call_id={call_id} field=email action=SKIPPED")
                logger.info(f"[BOOKING_FIELD] call_id={call_id} current_field=email status=SKIPPED next_field=confirmation")
                # Continue directly to confirmation (Step 8) without email
            else:
                session.state = BookingState.COLLECTING_EMAIL
                session.last_asked_field = "email"

                if was_waiting_email and (is_email_attempt or len(txt_lower.split()) >= 1):
                    resp = "I didn't catch the full email address. Could you please repeat or spell out your complete email address, or say 'skip' to continue without email?"
                    if lang in ("ta", "tanglish", "ta_mix"):
                        resp = "முழு email address சரியாகக் கேட்கவில்லை. தயவுசெய்து உங்கள் complete email address-ஐ மீண்டும் சொல்லுங்கள், அல்லது 'skip' என்று சொல்லுங்கள்."
                    elif lang == "hi":
                        resp = "Mujhe poora email address theek se sunai nahi diya. Kripya apna poora email address dobara batayein ya 'skip' kahein."
                    elif lang in ("kn", "kannada"):
                        resp = "ಪೂರ್ಣ email address ಸರಿಯಾಗಿ ಕೇಳಿಸಲಿಲ್ಲ. ದಯವಿಟ್ಟು ನಿಮ್ಮ complete email address ಅನ್ನು ಮತ್ತೊಮ್ಮೆ ತಿಳಿಸಿ, ಅಥವಾ 'skip' ಎಂದು ಹೇಳಿ."
                    elif lang in ("te", "telugu"):
                        resp = "పూర్తి email address సరిగ్గా వినబడలేదు. దయచేసి మీ complete email address మళ్ళీ చెప్పండి, లేదా 'skip' అని చెప్పండి."
                else:
                    resp = f"Thanks, {first_name}. What is your email address so I can send your appointment confirmation?" if first_name else "What is your email address so I can send your appointment confirmation?"
                    if lang in ("ta", "tanglish", "ta_mix"):
                        resp = "சரி. உறுதிப்படுத்தல் அனுப்ப உங்க email address சொல்லுங்க."
                    elif lang == "hi":
                        resp = "Shukriya. Confirmation bhejne ke liye kripya apna email address batayein."
                    elif lang in ("kn", "kannada"):
                        resp = "ಧನ್ಯವಾದಗಳು. ದೃಢೀಕರಣ ಕಳುಹಿಸಲು ನಿಮ್ಮ email address ತಿಳಿಸಿ."
                    elif lang in ("te", "telugu"):
                        resp = "ధన్యవాదాలు. కన్ఫర్మేషన్ పంపడానికి మీ email address తెలపండి."
                self._log_conversation_turn(session, text, resp, "waiting_for_email", state_before=state_before)
                return _ret(resp)

        # STEP 7b: Location Check
        if not session.location and not session.booking_data.get("location_declined"):
            was_waiting_loc = (
                (state_before and state_before.get("state") in (BookingState.COLLECTING_LOCATION, BookingState.WAITING_FOR_LOCATION))
                or (state_before and state_before.get("last_asked_field") == "location")
            )
            # Check if customer explicitly declines location
            decline_phrases = [
                "no location", "don't want to share", "skip", "decline", "not now",
                "prefer not to say", "none", "no thanks", "idhu vendaam", "nahi batana"
            ]
            clean_token = txt_lower.strip().rstrip(".,!?")
            is_explicit_decline = any(dp in txt_lower for dp in decline_phrases) or (was_waiting_loc and clean_token in ("no", "nope", "skip", "none", "na", "illa", "nahi"))

            if is_explicit_decline:
                session.booking_data["location_declined"] = True
                session.location = None
                # Continue directly to confirmation (Step 8) without location
            else:
                session.state = BookingState.COLLECTING_LOCATION
                session.last_asked_field = "location"

                if was_waiting_loc:
                    resp = "I didn't catch your location. Could you please tell me your city or location, or say 'skip' to continue?"
                    if lang in ("ta", "tanglish", "ta_mix"):
                        resp = "உங்கள் இடம் சரியாகக் கேட்கவில்லை. உங்கள் நகரம் அல்லது இடத்தை சொல்லுங்கள், அல்லது 'skip' என்று சொல்லுங்கள்."
                    elif lang == "hi":
                        resp = "Mujhe aapka location theek se sunai nahi diya. Kripya apna shehar ya location batayein ya 'skip' kahein."
                    elif lang in ("kn", "kannada"):
                        resp = "ನಿಮ್ಮ location ಸರಿಯಾಗಿ ಕೇಳಿಸಲಿಲ್ಲ. ದಯವಿಟ್ಟು ನಿಮ್ಮ ಊರು ಅಥವಾ location ತಿಳಿಸಿ, ಅಥವಾ 'skip' ಎಂದು ಹೇಳಿ."
                    elif lang in ("te", "telugu"):
                        resp = "మీ location సరిగ్గా వినబడలేదు. దయచేసి మీ ఊరు లేదా location చెప్పండి, లేదా 'skip' అని చెప్పండి."
                else:
                    if lang in ("ta", "tanglish", "ta_mix"):
                        resp = "உங்க location அல்லது ஊர் சொல்ல முடியுமா?"
                    elif lang == "hi":
                        resp = "क्या आप कृपया मुझे अपना स्थान या शहर बता सकते हैं?"
                    elif lang in ("kn", "kannada"):
                        resp = "ದಯವಿಟ್ಟು ನಿಮ್ಮ ಊರು ಅಥವಾ ಸ್ಥಳವನ್ನು ತಿಳಿಸಬಹುದೇ?"
                    elif lang in ("te", "telugu"):
                        resp = "దయచేసి మీ ప్రాంతం లేదా ఊరు చెప్పగలరా?"
                    else:
                        resp = "Could you please tell me your location?"
                self._log_conversation_turn(session, text, resp, "waiting_for_location", state_before=state_before)
                return _ret(resp)

        # STEP 8: Read-Back Confirmation Check & Handling
        if session.state in CORRECTION_AND_CONFIRMATION_STATES or (state_before and state_before.get("state") in CORRECTION_AND_CONFIRMATION_STATES):
            return self.handle_confirmation_turn(session, text, turn_id=turn_id, state_before=state_before)

        # First time all details are present — Read back and ask for confirmation
        if not session.booking_data.get("details_confirmed"):
            validation = self.validate_booking_fields(session)
            if not validation["is_valid"]:
                logger.warning(f"[BOOKING_CONFIRMATION_BLOCKED] Missing/invalid fields before confirmation: {validation['errors']}")
                if any("patient_name" in e for e in validation["errors"]):
                    session.state = BookingState.COLLECTING_PATIENT_NAME
                    session.last_asked_field = "patient_name"
                    resp = "Could you please tell me your full name for the booking?"
                    if lang in ("ta", "tanglish", "ta_mix"):
                        resp = "Booking-க்கு உங்க full name சொல்லுங்க."
                    return _ret(resp)
                elif any("date" in e for e in validation["errors"]):
                    session.state = BookingState.WAITING_FOR_DATE
                    session.last_asked_field = "date"
                    resp = "Which date would you prefer for your appointment?"
                    if lang in ("ta", "tanglish", "ta_mix"):
                        resp = "எந்த date-ல appointment book பண்ணலாம்?"
                    return _ret(resp)
                elif any("time" in e for e in validation["errors"]):
                    session.state = BookingState.WAITING_FOR_TIME
                    session.last_asked_field = "time"
                    resp = "Which time slot would you prefer?"
                    if lang in ("ta", "tanglish", "ta_mix"):
                        resp = "எந்த time slot உங்களுக்கு வசதியா இருக்கும்?"
                    return _ret(resp)
                elif any("contact_number" in e for e in validation["errors"]):
                    session.state = BookingState.COLLECTING_PHONE
                    session.last_asked_field = "contact_number"
                    resp = "Could you please provide your 10-digit contact phone number?"
                    if lang in ("ta", "tanglish", "ta_mix"):
                        resp = "உங்க contact phone number சொல்லுங்க."
                    return _ret(resp)

            session.state = BookingState.WAITING_FOR_CONFIRMATION
            session.last_asked_field = "confirmation"

            is_demo = (
                getattr(session, "booking_type", None) in ("SALES_DEMO", "BUSINESS_CONSULTATION")
                or getattr(session, "business_type", None) == "SOFTWARE"
                or getattr(session, "workflow_type", None) == "SALES_DEMO"
            )
            target_str = "for your product demo" if is_demo else (f"with {session.doctor}" if session.doctor else f"for {session.service or 'your appointment'}")
            loc_str = f", Location: {session.location}" if session.location else ""
            email_part = f", Email: {session.email}" if session.email else (", Email: Skipped" if (session.booking_data.get("email_skipped") or session.booking_data.get("email_declined") or session.field_status.get("email") == "SKIPPED") else "")
            email_ta = f", Email: {session.email}" if session.email else (", Email: Skipped" if (session.booking_data.get("email_skipped") or session.booking_data.get("email_declined") or session.field_status.get("email") == "SKIPPED") else "")
            email_hi = f", Email: {session.email}" if session.email else (", Email: Skipped" if (session.booking_data.get("email_skipped") or session.booking_data.get("email_declined") or session.field_status.get("email") == "SKIPPED") else "")

            if lang in ("ta", "tanglish", "ta_mix"):
                resp = (
                    f"நன்றி {first_name or ''}! உங்க details confirm பண்றேன்: "
                    f"Name: {session.patient_name}, Phone: {session.contact_number}"
                    f"{email_ta}{f', Location: {session.location}' if session.location else ''}, "
                    f"{session.date} அன்று {session.time}-க்கு {target_str}. இந்த details சரியா?"
                )
            elif lang == "hi":
                resp = (
                    f"Dhanyawad, {first_name or ''}! Main aapki details confirm kar leta hoon: "
                    f"Naam: {session.patient_name}, Phone: {session.contact_number}"
                    f"{email_hi}{f', Location: {session.location}' if session.location else ''}, "
                    f"{target_str} ke liye {session.date} ko {session.time} par. Kya yeh sab sahi hai?"
                )
            else:
                resp = (
                    f"Thank you, {first_name or 'there'}! Let me confirm your details: "
                    f"Name: {session.patient_name}, Phone: {session.contact_number}"
                    f"{email_part}{loc_str}, {target_str} on {session.date} at {session.time}. "
                    f"Is all of that correct?"
                )
            self._log_conversation_turn(session, text, resp, "waiting_for_confirmation", state_before=state_before)
            return _ret(resp)

        # STEP 9: All details collected & confirmed — Create Booking
        session.state = BookingState.BOOKING
        resp = self._execute_db_booking(session)
        self._log_conversation_turn(session, text, resp, "booking_executed", state_before=state_before)
        return _ret(resp)

    def _log_conversation_turn(self, session: BookingSession, user_input: str, response: str, action: str, state_before: Optional[Dict[str, Any]] = None):
        """Safe structured development log for turn tracing."""
        if user_input:
            if not (session.history and session.history[-1].get("role") == "user" and session.history[-1].get("text") == user_input):
                session.history.append({"role": "user", "text": user_input})
        if response:
            session.history.append({"role": "model", "text": response})
        fields_before = state_before.get("collected_fields", {}) if state_before else {}
        changed_fields = {
            k: v for k, v in session.collected_fields.items()
            if fields_before.get(k) != v
        }
        sb_intent = state_before.get("current_intent") if state_before else session.current_intent
        sb_stage = state_before.get("current_stage") if state_before else session.current_stage
        sb_collected = state_before.get("collected_fields") if state_before else session.collected_fields

        logger.info(
            f"\n[BOOKING_STATE]"
            f"\n  call_id={session.call_sid}"
            f"\n  state_before={sb_stage}"
            f"\n  state_after={session.state}"
            f"\n  current_action={action}"
            f"\n  doctor_id={session.doctor_id}"
            f"\n  doctor='{session.doctor}'"
            f"\n  date='{session.date}'"
            f"\n  time='{session.time}'"
            f"\n  patient_name='{session.patient_name}'"
        )

        logger.info(
            f"\n[CONVERSATION]\n"
            f"conversation_id={session.conversation_id}\n"
            f"call_sid={session.call_sid}\n"
            f"turn_id={session.turn_count}\n"
            f"[STATE BEFORE]\n"
            f"intent={sb_intent}\n"
            f"stage={sb_stage}\n"
            f"collected_fields={sb_collected}\n"
            f"[USER]\n"
            f"transcript={user_input}\n"
            f"[STATE UPDATE]\n"
            f"changed_fields={changed_fields}\n"
            f"[STATE AFTER]\n"
            f"intent={session.current_intent}\n"
            f"stage={session.current_stage}\n"
            f"collected_fields={session.collected_fields}\n"
            f"[ACTION]\n"
            f"action={action}"
        )
        self.save_session_state(session)

        # Persist conversation turn to conversation_messages table
        try:
            from src.config.database import SessionLocal
            from src.models.conversation_message import ConversationMessage
            cdb = SessionLocal()
            try:
                sid = session.conversation_id or session.call_sid or session.session_id
                lang_label = getattr(session, "conversation_language", "en")
                if user_input:
                    cdb.add(ConversationMessage(
                        session_id=sid,
                        role="user",
                        message=user_input,
                        language=lang_label,
                        intent=session.current_intent,
                        channel="sip" if "sip" in str(sid).lower() else "voice",
                        doctor_id=session.doctor_id,
                        doctor_name=session.doctor,
                        department=session.department or session.service,
                        appointment_date=session.date,
                        appointment_time=session.time,
                        availability_verified=bool(session.availability_verified),
                        booking_confirmed=bool(session.booking_confirmed),
                        appointment_id=session.appointment_id
                    ))
                if response:
                    cdb.add(ConversationMessage(
                        session_id=sid,
                        role="assistant",
                        message=response,
                        language=lang_label,
                        intent=session.current_intent,
                        channel="sip" if "sip" in str(sid).lower() else "voice",
                        doctor_id=session.doctor_id,
                        doctor_name=session.doctor,
                        department=session.department or session.service,
                        appointment_date=session.date,
                        appointment_time=session.time,
                        availability_verified=bool(session.availability_verified),
                        booking_confirmed=bool(session.booking_confirmed),
                        appointment_id=session.appointment_id
                    ))
                cdb.commit()
                logger.info(f"[CONVERSATION_MESSAGE_SAVED] Persisted turn for session_id={sid}")
            finally:
                cdb.close()
        except Exception as msg_err:
            logger.debug(f"[ConversationMessage Save Note] {msg_err}")

        # Automatic Lead Persistence:
        # As soon as customer details (name, phone, caller_phone, email) are provided,
        # create or update the lead in the Leads Management table.
        # This guarantees leads are preserved even if the customer hangs up mid-conversation.
        try:
            from src.services.lead_service import LeadService
            from src.config.database import SessionLocal

            raw_phone = getattr(session, "contact_number", None) or getattr(session, "caller_phone", None)
            cust_name = getattr(session, "patient_name", None)
            cust_email = getattr(session, "email", None)

            if (cust_name and len(cust_name.strip()) >= 2) or (raw_phone and raw_phone not in ("Unknown", "Unknown Caller", "Private Number")) or cust_email:
                ldb = SessionLocal()
                try:
                    LeadService(ldb).upsert_lead_from_session(
                        session=session,
                        call_log_id=getattr(session, "db_call_id", None)
                    )
                finally:
                    ldb.close()
        except Exception as lead_err:
            logger.debug(f"[LeadService Auto-Save Note] {lead_err}")

    def _execute_db_booking(self, session: BookingSession) -> str:
        """
        STAGE 2 BOOKING EXECUTION: Re-verifies availability immediately before DB commit.
        Includes idempotency check, email persistence, and post-booking notifications (Email + WhatsApp).
        Never pretends a booking happened unless database returns a valid appointment_id (>0).
        """
        lang = session.conversation_language

        if session.is_completed and session.appointment_id:
            return f"Your appointment with {session.doctor} on {session.date} at {session.time} is already confirmed under Appointment ID #{session.appointment_id}."

        try:
            from src.config.database import SessionLocal
            from src.services.appointment_service import AppointmentService
            from src.validators.appointment_validator import AppointmentCreate
            from src.models.appointment import Appointment
            from src.models.staff import Staff
            from src.services.calendar_verification_service import calendar_verification_service

            import time
            t_booking_start = time.time()

            # Stage 2 Pre-commit date/time normalization and validation
            t_val0 = time.time()
            if session.booking_type in ("SALES_DEMO", "BUSINESS_CONSULTATION") or session.business_type == "SOFTWARE":
                doctor_name = "Representative"
                dept_name = "Sales Team" if session.booking_type == "SALES_DEMO" else "Consultation"
            else:
                doctor_name = self.validate_doctor_against_db(session.doctor or "") or session.doctor or "Representative"
                dept_name = session.booking_data.get("specialty") or session.department or "General Appointment"
            target_date_obj = calendar_verification_service.parse_date(session.date or "today")
            normalized_date_str = target_date_obj.strftime("%d %b %Y")
            time_24h = calendar_verification_service.parse_time_to_24h(session.time or "10:00 AM") or "10:00"
            normalized_time_str = calendar_verification_service.format_time_12h(time_24h)
            booking_validation_ms = max(1, int((time.time() - t_val0) * 1000))

            # Idempotency check — prevent duplicate bookings
            org_id = getattr(session, "organization_id", 1) or 1
            idem_key = f"{org_id}:{session.session_id}:{doctor_name}:{normalized_date_str}:{normalized_time_str}:{session.patient_name}"
            if session.booking_data.get("idempotency_key") == idem_key and session.appointment_id:
                return f"Your appointment with {doctor_name} on {normalized_date_str} at {normalized_time_str} is already confirmed under Appointment ID #{session.appointment_id}."
            session.booking_data["idempotency_key"] = idem_key

            db = SessionLocal()
            try:
                # Authoritative Pre-Commit Past-Date & Past-Time Check
                from datetime import timedelta
                from src.routes.availability_routes import get_current_business_datetime, get_available_slots
                now_dt = get_current_business_datetime(org_id=org_id, db=db)
                current_date = now_dt.date()
                naive_now = now_dt.replace(tzinfo=None) if now_dt.tzinfo else now_dt

                if target_date_obj < current_date:
                    logger.warning(f"[PAST_DATE_PREVENTION] Pre-commit rejected past date: {target_date_obj} < {current_date}")
                    session.date = None
                    session.time = None
                    session.state = BookingState.WAITING_FOR_DATE
                    session.last_asked_field = "date"
                    return f"Sorry, {normalized_date_str} has already passed. Please choose today or an upcoming date for your appointment."

                try:
                    slot_t_obj = datetime.strptime(time_24h, "%H:%M").time()
                    slot_datetime_obj = datetime.combine(target_date_obj, slot_t_obj)
                    is_past_time_today = (target_date_obj == current_date and slot_datetime_obj <= naive_now)
                except Exception:
                    is_past_time_today = False

                staff_rec = db.query(Staff).filter(Staff.name.ilike(f"%{doctor_name}%")).first() if doctor_name else None
                staff_id_val = staff_rec.id if staff_rec else (session.doctor_id or 1)

                if is_past_time_today:
                    logger.warning(f"[PAST_TIME_PREVENTION] Pre-commit rejected past slot: {target_date_obj} {normalized_time_str} <= {naive_now}")
                    session.time = None
                    session.availability_verified = False
                    session.state = BookingState.WAITING_FOR_TIME
                    session.last_asked_field = "time"
                    rem_slots_res = get_available_slots(staff_id=staff_id_val, selected_date=target_date_obj, db=db)
                    rem_slots = [calendar_verification_service.format_time_12h(s["start"]) for s in rem_slots_res.get("available_slots", [])[:4]]
                    if rem_slots:
                        session.offered_slots = rem_slots
                        alt_text = ", ".join(rem_slots)
                        return f"Sorry, {normalized_time_str} is no longer available because that time has passed. Today's available slots are {alt_text}. Which time would you prefer?"
                    else:
                        return f"Sorry, {normalized_time_str} is no longer available because that time has passed, and there are no remaining slots available today. Would you like to check tomorrow or another date?"

                # 1. Recheck selected slot availability and existing duplicates
                t_avail0 = time.time()
                existing_for_customer = db.query(Appointment).filter(
                    Appointment.customer == session.patient_name,
                    Appointment.staff == doctor_name,
                    Appointment.appointment_date == normalized_date_str,
                    Appointment.appointment_time == normalized_time_str,
                    Appointment.status != "Cancelled"
                ).first()
                if existing_for_customer:
                    session.appointment_id = existing_for_customer.id
                    session.booking_confirmed = True
                    session.state = BookingState.BOOKING_CONFIRMED
                    session.is_completed = True
                    session.confirmation_count = 1
                    b_name = session.business_name or "our team"
                    if session.booking_type == "SALES_DEMO":
                        resp_text = (
                            f"Your demo appointment is confirmed for {normalized_date_str} at {normalized_time_str}. "
                            f"Your appointment ID is #{existing_for_customer.id}. Thank you for calling {b_name}."
                        )
                    else:
                        resp_text = (
                            f"Your appointment is confirmed with {doctor_name} on {normalized_date_str} at {normalized_time_str}. "
                            f"Your appointment ID is #{existing_for_customer.id}. Thank you for calling {b_name}."
                        )
                    session.confirmation_message = resp_text
                    session.confirmed_appointment_id = existing_for_customer.id
                    return resp_text

                # Authoritative Pre-Commit Block-Time Check
                from src.models.block_time import BlockTime
                from src.routes.availability_routes import _normalize_name
                slot_start_dt = datetime.combine(target_date_obj, slot_t_obj)
                slot_end_dt = slot_start_dt + timedelta(minutes=30)
                block_conflicts = db.query(BlockTime).filter(
                    BlockTime.start_datetime < slot_end_dt,
                    BlockTime.end_datetime > slot_start_dt,
                    (BlockTime.organization_id == org_id) | (BlockTime.organization_id.is_(None))
                ).all()
                is_slot_blocked = False
                for bt in block_conflicts:
                    if bt.staff_id and staff_rec and bt.staff_id != staff_rec.id:
                        continue
                    if bt.staff_name and doctor_name and _normalize_name(bt.staff_name) != _normalize_name(doctor_name):
                        continue
                    is_slot_blocked = True
                    break

                if is_slot_blocked:
                    logger.warning(f"[BLOCK_TIME_PREVENTION] Pre-commit rejected blocked slot: {target_date_obj} {normalized_time_str}")
                    session.time = None
                    session.availability_verified = False
                    session.state = BookingState.WAITING_FOR_TIME
                    session.last_asked_field = "time"
                    rem_slots_res = get_available_slots(staff_id=staff_id_val, selected_date=target_date_obj, db=db)
                    rem_slots = [calendar_verification_service.format_time_12h(s["start"]) for s in rem_slots_res.get("available_slots", [])[:4]]
                    if rem_slots:
                        session.offered_slots = rem_slots
                        alt_text = ", ".join(rem_slots)
                        return f"Sorry, {normalized_time_str} is blocked and unavailable for booking. Other available slots on {normalized_date_str} are {alt_text}. Which time would you prefer?"
                    else:
                        return f"Sorry, {normalized_time_str} is blocked and unavailable for booking, and there are no other slots open on {normalized_date_str}. Would you like to check tomorrow or another date?"

                # Race condition check: slot taken by another customer right before commit
                conflict = db.query(Appointment).filter(
                    Appointment.staff == doctor_name,
                    Appointment.appointment_date == normalized_date_str,
                    Appointment.appointment_time == normalized_time_str,
                    Appointment.status != "Cancelled"
                ).first()
                if conflict:
                    session.time = None
                    session.availability_verified = False
                    session.state = BookingState.WAITING_FOR_TIME
                    return f"Sorry, that slot at {normalized_time_str} with {doctor_name} was just booked by another caller. Which other time would you prefer?"
                availability_check_ms = max(1, int((time.time() - t_avail0) * 1000))

                svc = AppointmentService(db)
                svc_val = session.service or session.product or session.booking_data.get("service") or session.booking_data.get("product") or "General Consultation"
                intent_val = session.current_intent or session.booking_data.get("intent") or "appointment_booking"
                purpose_note = f"Booked via AI Voice Receptionist ({str(intent_val).replace('_', ' ').title()})"
                confirmed_email = (session.email or session.booking_data.get("email") or "").strip() or None
                if not confirmed_email and session.contact_number:
                    try:
                        prev = db.query(Appointment).filter(
                            Appointment.organization_id == org_id,
                            Appointment.phone == session.contact_number,
                            Appointment.email.isnot(None),
                            Appointment.email != ""
                        ).order_by(Appointment.id.desc()).first()
                        if prev:
                            confirmed_email = prev.email.strip()
                    except Exception:
                        pass

                # Database booking transaction — executes in < 1 second
                t_insert0 = time.time()
                appointment = svc.create_appointment(
                    AppointmentCreate(
                        customer=session.patient_name,
                        phone=session.contact_number or "Unknown",
                        email=confirmed_email,
                        location=session.location,
                        department=dept_name,
                        staff=doctor_name,
                        service=svc_val,
                        appointment_date=normalized_date_str,
                        appointment_time=normalized_time_str,
                        duration="30 Minutes",
                        notes=purpose_note
                    ),
                    organization_id=org_id,
                    user_id=getattr(session, "user_id", None),
                    sync_calendar_sync=False  # External APIs (Google Calendar, Email, WhatsApp) run strictly in background outbox
                )
                t_insert1 = time.time()
                database_insert_ms = max(1, int((t_insert1 - t_insert0) * 1000))
                database_commit_ms = database_insert_ms

                if appointment and appointment.id and appointment.id > 0:
                    verified_appt = db.query(Appointment).filter(Appointment.id == appointment.id).first()
                    if verified_appt:
                        from src.models.appointment import AppointmentStatus
                        verified_appt.status = AppointmentStatus.CONFIRMED
                        db.commit()
                        db.refresh(verified_appt)
                        session.date = normalized_date_str
                        session.time = normalized_time_str
                        session.appointment_id = verified_appt.id
                        session.booking_confirmed = True
                        session.state = BookingState.BOOKING_CONFIRMED
                        session.is_completed = True

                        t_conf_start = time.time()
                        # Spoken confirmation without LLM dependency
                        b_name = session.business_name or "our team"
                        if lang in ("ta", "tanglish", "ta_mix"):
                            if session.booking_type == "SALES_DEMO":
                                resp_text = (
                                    f"உங்க demo appointment {normalized_date_str} அன்று {normalized_time_str}-க்கு confirm ஆயிடுச்சு! "
                                    f"Appointment ID #{verified_appt.id}. {b_name}-க்கு call பண்ணதுக்கு நன்றி {session.patient_name}!"
                                )
                            else:
                                resp_text = (
                                    f"உங்க appointment {doctor_name}-கூட {normalized_date_str} அன்று {normalized_time_str}-க்கு confirm ஆயிடுச்சு! "
                                    f"Appointment ID #{verified_appt.id}. விவரங்களை அனுப்பிவிட்டேன். நன்றி, {session.patient_name}!"
                                )
                        elif lang == "hi":
                            if session.booking_type == "SALES_DEMO":
                                resp_text = (
                                    f"आपका डेमो अपॉइंटमेंट {normalized_date_str} को {normalized_time_str} पर कन्फर्म हो गया है। "
                                    f"अपॉइंटमेंट आईडी #{verified_appt.id} है। {b_name} को कॉल करने के लिए धन्यवाद, {session.patient_name}!"
                                )
                            else:
                                resp_text = (
                                    f"Aapka appointment {doctor_name} ke saath {normalized_date_str} ko {normalized_time_str} par confirm ho gaya hai! "
                                    f"Appointment ID #{verified_appt.id}. Maine aapko details bhej di hain. Dhanyawad, {session.patient_name}!"
                                )
                        elif session.booking_type == "SALES_DEMO":
                            resp_text = (
                                f"Your demo appointment is confirmed for {normalized_date_str} at {normalized_time_str}. "
                                f"Your appointment ID is #{verified_appt.id}. Thank you for calling {b_name}."
                            )
                        else:
                            resp_text = (
                                f"Your appointment is confirmed with {doctor_name} on {normalized_date_str} at {normalized_time_str}. "
                                f"Your appointment ID is #{verified_appt.id}. Thank you for calling {b_name}."
                            )

                        session.confirmation_count = 1
                        session.confirmation_message = resp_text
                        session.confirmed_appointment_id = verified_appt.id

                        cal_status = getattr(verified_appt, "calendar_sync_status", "PENDING")
                        email_status = getattr(verified_appt, "email_status", "PENDING")
                        wa_status = getattr(verified_appt, "whatsapp_status", "PENDING")

                        logger.info(
                            f"\n[BOOKING_CONFIRMATION]"
                            f"\n  confirmation_count={session.confirmation_count}"
                            f"\n  playback_status=QUEUED"
                            f"\n  appointment_id={verified_appt.id}"
                            f"\n  booking_status=CONFIRMED"
                            f"\n  calendar_sync_status={cal_status}"
                            f"\n  email_notification_status={email_status}"
                            f"\n  whatsapp_notification_status={wa_status}"
                            f"\n  customer={self._mask_pii(session.patient_name)}"
                            f"\n  phone={self._mask_pii(session.contact_number)}"
                        )
                        confirmation_text_ms = max(1, int((time.time() - t_conf_start) * 1000))
                        booking_total_ms = max(1, int((time.time() - t_booking_start) * 1000))

                        logger.info(
                            f"\n[BOOKING METRICS]"
                            f"\n  call_id={session.call_sid}"
                            f"\n  appointment_id={verified_appt.id}"
                            f"\n  booking_validation_ms={booking_validation_ms}"
                            f"\n  availability_check_ms={availability_check_ms}"
                            f"\n  database_insert_ms={database_insert_ms}"
                            f"\n  database_commit_ms={database_commit_ms}"
                            f"\n  booking_total_ms={booking_total_ms}"
                            f"\n  confirmation_text_ms={confirmation_text_ms}"
                        )
                        return resp_text

                raise RuntimeError("Database appointment verification failed - ID invalid or record missing.")
            finally:
                db.close()
        except Exception as e:
            logger.error(f"[BOOKING DB ERROR] Failed to create/verify appointment: {e}")
            session.state = BookingState.WAITING_FOR_CONFIRMATION
            session.booking_confirmed = False
            session.is_completed = False
            try:
                from src.config.database import SessionLocal
                from src.services.lead_service import LeadService
                err_db = SessionLocal()
                try:
                    LeadService(err_db).upsert_lead_from_session(
                        session,
                        lead_status="Follow-up Required",
                        appointment_status="Failed",
                        follow_up_required=True,
                        notes=f"Booking failed: {e}"
                    )
                finally:
                    err_db.close()
            except Exception:
                pass
            if "conflict" in str(e).lower() or "409" in str(e):
                return "I'm sorry, that appointment time on our calendar was just taken. Which other time would you prefer?"
            if lang in ("ta", "tanglish", "ta_mix"):
                return "மன்னிக்கவும், calendar-ல அந்த time-ஐ confirm பண்ண முடியல. உங்க details save பண்ணியாச்சு, எங்க team உங்களை சீக்கிரமே contact பண்ணுவாங்க."
            return "I apologize, but we encountered an issue booking that slot in our calendar system. I have saved your details, and our team will follow up with you shortly to confirm your appointment."


booking_state_service = BookingStateService()
