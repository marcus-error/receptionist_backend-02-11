import logging
import re
from typing import Optional, Dict, Any, List
from datetime import datetime
from sqlalchemy.orm import Session

from src.config.database import SessionLocal
from src.models.appointment import Appointment, AppointmentStatus
from src.services.appointment_service import AppointmentService

logger = logging.getLogger(__name__)


def _normalize_phone(p: Optional[str]) -> str:
    if not p:
        return ""
    digits = re.sub(r"\D", "", str(p))
    return digits[-10:] if len(digits) >= 10 else digits


def _normalize_name(name: Optional[str]) -> str:
    if not name:
        return ""
    cleaned = re.sub(r'^(?:dr\.?|doctor|mr\.?|mrs\.?|ms\.?)\s+', '', str(name).strip().lower())
    return re.sub(r'[^a-z0-9]', '', cleaned)


class CancellationService:
    """
    Centralized cancellation engine for AI Receptionist.
    Handles natural language cancellation requests, dynamically retrieves customer appointments,
    updates DB status to Cancelled, syncs with external calendars, and prevents duplicate cancellations.
    """

    def __init__(self, db: Optional[Session] = None):
        self._db_external = db is not None
        self.db = db or SessionLocal()

    def close(self):
        if not self._db_external and self.db:
            try:
                self.db.close()
            except Exception:
                pass

    def handle_cancellation(
        self,
        utterance: str,
        caller_phone: Optional[str] = None,
        caller_name: Optional[str] = None,
        session: Any = None,
        organization_id: int = 1,
        language: str = "en",
    ) -> Dict[str, Any]:
        """
        Process a cancellation request.
        """
        lang = str(language or "en").lower()
        is_ta = lang in ("ta", "tanglish", "ta_mix")
        txt_lower = (utterance or "").lower()

        # 1. Check if user is responding to a previous multiple-choice cancellation prompt
        candidates = getattr(session, "cancellation_candidates", None) if session else None
        if candidates and getattr(session, "state", "") == "WAITING_FOR_CANCELLATION_CHOICE":
            # Caller might say "the first one", "1", "first", "second", or mention a doctor / date / time
            chosen_id = None
            if any(w in txt_lower for w in ["first", "1st", "number 1", "one", "the first"]):
                chosen_id = candidates[0]
            elif len(candidates) > 1 and any(w in txt_lower for w in ["second", "2nd", "number 2", "two", "the second"]):
                chosen_id = candidates[1]
            elif len(candidates) > 2 and any(w in txt_lower for w in ["third", "3rd", "number 3", "three", "the third"]):
                chosen_id = candidates[2]
            else:
                # Check if caller specified an ID or doctor name in this response
                id_m = re.search(r'\b(?:id|#)?\s*(\d+)\b', utterance, re.IGNORECASE)
                if id_m and int(id_m.group(1)) in candidates:
                    chosen_id = int(id_m.group(1))

            if chosen_id:
                session.cancellation_candidates = None
                return self._cancel_single_appointment(chosen_id, organization_id, session, is_ta)

        # 2. Extract explicit appointment ID if mentioned in the caller utterance
        target_id = None
        id_match = re.search(r'\b(?:appointment\s*(?:id|#)?\s*|id\s*|#\s*)(\d+)\b', utterance, re.IGNORECASE)
        if id_match:
            target_id = int(id_match.group(1))
        elif session and getattr(session, "appointment_id", None):
            target_id = session.appointment_id

        if target_id:
            appt = self.db.query(Appointment).filter(
                Appointment.id == target_id,
                (Appointment.organization_id == organization_id) | (Appointment.organization_id.is_(None))
            ).first()
            if appt:
                if appt.status == AppointmentStatus.CANCELLED:
                    if is_ta:
                        resp = f"முன்பதிவு #{target_id} ஏற்கனவே ரத்து செய்யப்பட்டுள்ளது. உங்களுக்கு வேறு ஏதேனும் உதவி வேண்டுமா?"
                    else:
                        resp = f"Appointment #{target_id} has already been cancelled. Is there anything else I can assist you with?"
                    return {"response": resp, "cancelled": False, "already_cancelled": True, "appointment_id": target_id}
                return self._cancel_single_appointment(target_id, organization_id, session, is_ta)

        # 3. Dynamic lookup by caller's phone, name, or session details
        phone_norm = _normalize_phone(caller_phone or (session.contact_number if session else None))
        name_norm = _normalize_name(caller_name or (session.patient_name if session else None))

        active_query = self.db.query(Appointment).filter(
            Appointment.status != AppointmentStatus.CANCELLED,
            (Appointment.organization_id == organization_id) | (Appointment.organization_id.is_(None))
        )
        all_active = active_query.order_by(Appointment.id.asc()).all()

        matching_active: List[Appointment] = []
        for a in all_active:
            a_phone_norm = _normalize_phone(a.phone)
            a_name_norm = _normalize_name(a.customer)

            is_phone_match = bool(phone_norm and a_phone_norm and (phone_norm == a_phone_norm or phone_norm in a_phone_norm or a_phone_norm in phone_norm))
            is_name_match = bool(name_norm and a_name_norm and (name_norm == a_name_norm or name_norm in a_name_norm or a_name_norm in name_norm))

            if is_phone_match or is_name_match:
                matching_active.append(a)

        # If caller mentioned a specific date (e.g., "tomorrow", "Friday", or "15 Nov"), filter by date
        try:
            from src.services.calendar_verification_service import calendar_verification_service
            if any(w in txt_lower for w in ["tomorrow", "today", "yesterday", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]):
                parsed_d = calendar_verification_service.parse_date(utterance)
                if parsed_d:
                    fmt_d = parsed_d.strftime("%d %b %Y").lower()
                    date_filtered = [a for a in matching_active if (a.appointment_date or "").strip().lower() == fmt_d]
                    if date_filtered:
                        matching_active = date_filtered
        except Exception:
            pass

        # 4. Handle results
        if not matching_active:
            # Check if there is an already cancelled appointment
            cancelled_appts = self.db.query(Appointment).filter(
                Appointment.status == AppointmentStatus.CANCELLED,
                (Appointment.organization_id == organization_id) | (Appointment.organization_id.is_(None))
            ).order_by(Appointment.id.desc()).all()

            for ca in cancelled_appts:
                ca_phone_norm = _normalize_phone(ca.phone)
                ca_name_norm = _normalize_name(ca.customer)
                if (phone_norm and ca_phone_norm and phone_norm == ca_phone_norm) or (name_norm and ca_name_norm and name_norm == ca_name_norm):
                    staff_lbl = ca.staff or "our specialist"
                    d_lbl = ca.appointment_date or ""
                    t_lbl = ca.appointment_time or ""
                    if is_ta:
                        resp = f"{d_lbl} அன்று {t_lbl}-க்கு {staff_lbl}-உடன் இருந்த உங்கள் appointment ஏற்கனவே ரத்து செய்யப்பட்டுள்ளது. வேறு ஏதேனும் உதவி வேண்டுமா?"
                    else:
                        resp = f"Your appointment with {staff_lbl} on {d_lbl} at {t_lbl} has already been cancelled. Is there anything else I can assist you with?"
                    return {"response": resp, "cancelled": False, "already_cancelled": True, "appointment_id": ca.id}

            if is_ta:
                resp = "உங்கள் விவரங்களில் செயலில் உள்ள முன்பதிவுகள் எதுவும் காணப்படவில்லை. முன்பதிவை ரத்து செய்ய, தயவுசெய்து உங்கள் appointment ID அல்லது பதிவு செய்யப்பட்ட தொலைபேசி எண்ணைக் கூறவும்."
            else:
                resp = "I could not find any active appointments under your contact details. Could you please share your appointment ID or registered phone number so I can help you cancel?"
            return {"response": resp, "cancelled": False, "not_found": True}

        if len(matching_active) == 1:
            # Exactly one active appointment found — cancel it directly!
            target = matching_active[0]
            return self._cancel_single_appointment(target.id, organization_id, session, is_ta)

        # Multiple active appointments found — ask for clarification
        if session:
            session.cancellation_candidates = [a.id for a in matching_active]
            session.state = "WAITING_FOR_CANCELLATION_CHOICE"

        items = []
        for idx, a in enumerate(matching_active[:3], start=1):
            items.append(f"{idx}) With {a.staff or 'representative'} on {a.appointment_date} at {a.appointment_time}")
        items_str = ", ".join(items)

        if is_ta:
            resp = f"உங்களுக்கு {len(matching_active)} வரவிருக்கும் முன்பதிவுகள் உள்ளன: {items_str}. இதில் எந்த முன்பதிவை ரத்து செய்ய விரும்புகிறீர்கள்?"
        else:
            resp = f"You have {len(matching_active)} upcoming appointments: {items_str}. Which appointment would you like to cancel?"

        return {
            "response": resp,
            "cancelled": False,
            "multiple_found": True,
            "candidate_ids": [a.id for a in matching_active],
        }

    def _cancel_single_appointment(
        self,
        appointment_id: int,
        organization_id: int,
        session: Any,
        is_ta: bool
    ) -> Dict[str, Any]:
        """Cancel an appointment and sync with external calendar."""
        svc = AppointmentService(self.db)
        appt = self.db.query(Appointment).filter(Appointment.id == appointment_id).first()
        if not appt:
            resp = f"Appointment #{appointment_id} was not found."
            return {"response": resp, "cancelled": False, "not_found": True}

        staff_lbl = appt.staff or "your specialist"
        date_lbl = appt.appointment_date or ""
        time_lbl = appt.appointment_time or ""

        # Perform authoritative cancellation (soft-cancel + calendar sync + notification + webhook)
        try:
            svc.delete_appointment(appointment_id, organization_id=organization_id)
        except Exception as e:
            logger.error(f"Error cancelling appointment #{appointment_id}: {e}")
            # Fallback direct soft-cancel if needed
            appt.status = AppointmentStatus.CANCELLED
            self.db.commit()

        if session:
            session.cancelled_appointment_id = appointment_id
            session.appointment_id = None
            session.cancellation_candidates = None
            session.is_completed = True
            from src.services.booking_state_service import BookingState
            session.state = BookingState.IDLE

        logger.info(f"[CANCELLATION_SUCCESS] Cancelled Appointment #{appointment_id} for {appt.customer}")

        if is_ta:
            resp = f"{date_lbl} அன்று {time_lbl}-க்கு {staff_lbl}-உடன் இருந்த உங்கள் appointment (ID #{appointment_id}) வெற்றிகரமாக ரத்து செய்யப்பட்டது. உங்களுக்கு வேறு ஏதேனும் உதவி வேண்டுமா?"
        else:
            resp = f"Your appointment with {staff_lbl} on {date_lbl} at {time_lbl} (ID #{appointment_id}) has been successfully cancelled. Is there anything else I can assist you with?"

        return {
            "response": resp,
            "cancelled": True,
            "appointment_id": appointment_id,
            "customer": appt.customer,
            "staff": staff_lbl,
            "date": date_lbl,
            "time": time_lbl,
        }


cancellation_service = CancellationService()
