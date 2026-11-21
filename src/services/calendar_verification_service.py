import re
import logging
from datetime import date, datetime, timedelta
from typing import Dict, Any, Optional, List
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class CalendarVerificationResult(BaseModel):
    is_checked: bool = False
    doctor_name: Optional[str] = None
    requested_date: Optional[str] = None
    requested_time: Optional[str] = None
    is_available: Optional[bool] = None
    is_past_time: bool = False
    reason: str = ""
    alternative_slots: List[str] = []
    next_available_date: Optional[str] = None
    next_available_slots: List[str] = []
    prompt_summary: str = ""


class CalendarVerificationService:
    """
    Real-Time Calendar & Live DB Verification Service.

    Extracts doctor, date, and time entities from user text / session context,
    queries the live database & connected Google/Microsoft calendar events,
    and returns verified availability status & alternative free slots.
    """

    @staticmethod
    def parse_time_to_24h(time_str: str) -> Optional[str]:
        """Convert various time formats (9:30 PM, 9:30 p.m., 21:30, 9.30pm, 10am, 3 o'clock, after lunch) to HH:MM (24-hour)."""
        if not time_str:
            return None

        clean = time_str.strip().lower()
        clean = re.sub(r'\bp\.?m\.?\b', 'pm', clean)
        clean = re.sub(r'\ba\.?m\.?\b', 'am', clean)
        clean = re.sub(r"\bo\s*'?\s*clock\b", "", clean)
        clean = clean.replace(".", ":")

        # Check for natural relative phrases without explicit digits
        if any(ph in clean for ph in ["after lunch", "post lunch"]):
            return "14:00"
        if any(ph in clean for ph in ["this afternoon", "afternoon"]) and not re.search(r'\d', clean):
            return "14:00"
        if any(ph in clean for ph in ["this evening", "evening"]) and not re.search(r'\d', clean):
            return "17:00"
        if any(ph in clean for ph in ["this morning", "morning"]) and not re.search(r'\d', clean):
            return "10:00"
        if any(ph in clean for ph in ["at noon", "noon"]) and not re.search(r'\d', clean):
            return "12:00"

        # If natural time of day is combined with numbers (e.g. "3 in the afternoon", "4 evening", "10 in the morning")
        if any(w in clean for w in ["afternoon", "evening", "post lunch", "after lunch"]):
            if not re.search(r'\b(am|pm)\b', clean):
                clean += " pm"
        elif "morning" in clean:
            if not re.search(r'\b(am|pm)\b', clean):
                clean += " am"

        # Check for range expressions like "9 to 9:30 am", "9 - 9:30 am", "2 to 3 pm"
        range_match = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(?:to|-|–)\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', clean)
        if range_match:
            start_h = int(range_match.group(1))
            start_m = int(range_match.group(2)) if range_match.group(2) else 0
            ampm = range_match.group(5)
            hour = start_h
            minute = start_m
        else:
            match = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', clean)
            if not match:
                return None

            hour = int(match.group(1))
            minute = int(match.group(2)) if match.group(2) else 0
            ampm = match.group(3)

        if ampm:
            if ampm == "pm" and hour < 12:
                hour += 12
            elif ampm == "am" and hour == 12:
                hour = 0
        else:
            if 1 <= hour <= 7:
                hour += 12

        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return f"{hour:02d}:{minute:02d}"
        return None

    @staticmethod
    def format_time_12h(time_24h: str) -> str:
        """Convert HH:MM (24h) to 'h:mm AM/PM' for natural speech synthesis."""
        try:
            dt = datetime.strptime(time_24h, "%H:%M")
            return dt.strftime("%I:%M %p").lstrip("0")
        except Exception:
            return time_24h

    def parse_date(self, text: str, default: Any = "DEFAULT_TODAY", reference_now: Optional[datetime] = None) -> Optional[date]:
        """Parse date string into datetime.date object using timezone-aware real server clock."""
        txt = (text or "").lower().strip()
        from src.routes.availability_routes import get_current_business_datetime
        try:
            today = get_current_business_datetime(reference_now=reference_now).date()
        except Exception:
            from datetime import timezone
            today = datetime.now(timezone(timedelta(hours=5, minutes=30))).date()

        fallback_date = today if default == "DEFAULT_TODAY" else default

        if not txt:
            return fallback_date

        if "day after tomorrow" in txt:
            return today + timedelta(days=2)
        if any(w in txt for w in ["tomorrow", "naalai", "nalai", "kal"]):
            return today + timedelta(days=1)
        if any(w in txt for w in ["today", "indru", "inru", "aaj"]):
            return today
        if "next week" in txt:
            return today + timedelta(days=7)
        if "next month" in txt:
            return today + timedelta(days=30)

        # 1. Check standard date string formats directly
        for fmt in ("%d %b %Y", "%d %B %Y", "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d"):
            try:
                return datetime.strptime(txt.strip(), fmt).date()
            except ValueError:
                pass

        # 1b. Check numeric date patterns: DD-MM-YYYY, DD/MM/YYYY, DD-MM-YY, DD/MM/YY
        num_date_match = re.search(r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b', txt)
        if num_date_match:
            d_val, m_val, y_val = int(num_date_match.group(1)), int(num_date_match.group(2)), int(num_date_match.group(3))
            full_year = (2000 + y_val) if y_val < 100 else y_val
            for d_c, m_c in [(d_val, m_val), (m_val, d_val)]:
                if 1 <= m_c <= 12 and 1 <= d_c <= 31:
                    try:
                        return date(full_year, m_c, d_c)
                    except ValueError:
                        pass

        # 2. Check Day + Month format (e.g., "22 Sep 2026", "22nd of September", "22 September 2026")
        day_month_match = re.search(
            r'\b(\d{1,2})(?:st|nd|rd|th)?[,\s]+(?:of\s+)?(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:[,\s]+(\d{4}))?\b',
            txt
        )
        if day_month_match:
            try:
                d_num = int(day_month_match.group(1))
                m_str = day_month_match.group(2).capitalize()
                y_str = day_month_match.group(3)
                m_num = datetime.strptime(m_str[:3], "%b").month
                year = int(y_str) if y_str else today.year
                target = date(year, m_num, d_num)
                if not y_str and target < today:
                    target = date(year + 1, m_num, d_num)
                return target
            except Exception:
                pass

        # 3. Check Month + Day format (e.g., "Sep 19", "September 19", "Sep 19 2026", "November 15, 2026")
        month_day_match = re.search(
            r'\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)[,\s]+(\d{1,2})(?:st|nd|rd|th)?(?:[,\s]+(\d{4}))?\b',
            txt
        )
        if month_day_match:
            try:
                m_str = month_day_match.group(1).capitalize()
                d_num = int(month_day_match.group(2))
                y_str = month_day_match.group(3)
                m_num = datetime.strptime(m_str[:3], "%b").month
                year = int(y_str) if y_str else today.year
                target = date(year, m_num, d_num)
                if not y_str and target < today:
                    target = date(year + 1, m_num, d_num)
                return target
            except Exception:
                pass

        days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        for idx, day_name in enumerate(days):
            if re.search(r'\b' + day_name + r'\b', txt):
                days_ahead = idx - today.weekday()
                if days_ahead <= 0 or f"next {day_name}" in txt:
                    days_ahead += 7
                return today + timedelta(days=days_ahead)

        date_match = re.search(r'\b(\d{4}-\d{2}-\d{2})\b', txt)
        if date_match:
            try:
                return datetime.strptime(date_match.group(1), "%Y-%m-%d").date()
            except ValueError:
                pass

        return fallback_date

    def extract_doctor(self, text: str, known_staff: List[str] = None) -> Optional[str]:
        """Extract doctor/staff name dynamically from DB or text."""
        txt_lower = (text or "").lower()

        if not known_staff:
            try:
                from src.config.database import SessionLocal
                from src.models.staff import Staff
                db = SessionLocal()
                try:
                    staff_records = db.query(Staff.name).all()
                    known_staff = [s[0] for s in staff_records if s[0]]
                finally:
                    db.close()
            except Exception:
                known_staff = []

        if known_staff:
            for name in known_staff:
                tokens = [t.lower() for t in name.split() if len(t) > 2 and t.lower() not in {"dr", "dr.", "doctor"}]
                if tokens and any(t in txt_lower for t in tokens):
                    return name

        match = re.search(r'\b(dr\.?\s*[a-zA-Z]+(?:\s+[a-zA-Z]+)?|doctor\s*[a-zA-Z]+(?:\s+[a-zA-Z]+)?)\b', txt_lower)
        if match:
            extracted = match.group(1).title().strip()
            stop_words = {"appointment", "booking", "venum", "schedule", "tomorrow", "today", "slot", "kitta", "doctor"}
            if extracted.lower() not in stop_words:
                return extracted

        return None

    def verify_availability(self, text: str, session_history: List[dict] = None, intent: Optional[str] = None, reference_now: Optional[datetime] = None) -> CalendarVerificationResult:
        """
        Main entry point: Inspects user text and context for doctor/date/time requests,
        queries the live database via `get_available_slots`, and returns verified status.
        Only runs when the current user request is an appointment availability or booking query.
        Includes real-time past-time checking and dynamic next-available-date discovery.
        """
        if intent and intent not in ("appointment_availability", "appointment_booking", "BOOK_APPOINTMENT", "CHECK_AVAILABILITY", "appointment_cancellation", "appointment_rescheduling"):
            return CalendarVerificationResult(is_checked=False)

        current_txt_lower = (text or "").lower()
        is_availability_query = any(w in current_txt_lower for w in [
            "appointment", "available", "slot", "schedule", "book", "kitta", "timing", "free"
        ])

        if not is_availability_query and not (intent in ("appointment_availability", "appointment_booking", "BOOK_APPOINTMENT", "CHECK_AVAILABILITY", "appointment_cancellation", "appointment_rescheduling")):
            return CalendarVerificationResult(is_checked=False)

        combined_text = (text or "") + " " + " ".join([m.get("text", "") for m in (session_history or [])[-3:]])
        txt_lower = combined_text.lower()

        from src.config.database import SessionLocal
        from src.models.staff import Staff
        from src.routes.availability_routes import get_available_slots, get_current_business_datetime, find_next_available_slots

        db = SessionLocal()
        try:
            from src.models.staff import StaffStatus
            org_id = getattr(self, "organization_id", 1) or 1
            all_staff = db.query(Staff).filter(Staff.organization_id == org_id, Staff.status != StaffStatus.OFFLINE).all()
            staff_names = [s.name for s in all_staff]

            doctor_name = self.extract_doctor(combined_text, known_staff=staff_names)
            target_date = self.parse_date(combined_text, reference_now=reference_now)

            time_regex = r'\b(\d{1,2}(?::\d{2})?\s*(?:to|-|–)\s*\d{1,2}(?::\d{2})?\s*(?:[ap]\.?m\.?))\b|\b(\d{1,2}(?::\d{2})?\s*(?:[ap]\.?m\.?))\b|\b(\d{1,2}:\d{2})\b|\b(\d{1,2}\s*(?:[ap]\.?m\.?))\b|\b(after lunch|post lunch|this afternoon|afternoon|this evening|evening|this morning|morning|noon)\b'
            time_match = re.search(time_regex, text.lower())
            if not time_match and session_history:
                for m in reversed(session_history):
                    if m.get("role") == "user":
                        t_m = re.search(time_regex, m.get("text", "").lower())
                        if t_m:
                            time_match = t_m
                            break

            raw_time_str = (time_match.group(1) or time_match.group(2) or time_match.group(3) or (time_match.group(4) if len(time_match.groups()) >= 4 else None) or (time_match.group(5) if len(time_match.groups()) >= 5 else None)) if time_match else None
            if raw_time_str:
                raw_time_str = raw_time_str.upper()
            target_time_24h = self.parse_time_to_24h(raw_time_str) if raw_time_str else None

            is_generic_doc = bool(
                doctor_name and doctor_name.strip().lower() in (
                    "representative", "rep", "team", "sales team", "sales representative",
                    "demo", "consultation", "agent", "assistant", "ai agent", "general"
                )
            )

            staff = None
            if doctor_name and not is_generic_doc:
                clean_doc = re.sub(r'^(dr\.?|doctor)\s*', '', doctor_name, flags=re.IGNORECASE).strip()
                staff = db.query(Staff).filter(Staff.organization_id == org_id, Staff.name.ilike(f"%{clean_doc or doctor_name}%")).first()
                if not staff:
                    return CalendarVerificationResult(
                        is_checked=True,
                        is_available=False,
                        doctor_name=doctor_name,
                        reason=f"{doctor_name} is not listed in our active directory.",
                        prompt_summary=f"[LIVE CALENDAR VERIFICATION: '{doctor_name}' is not listed in our active directory.]"
                    )

            if not staff:
                # If booking is for a service/product, demo, representative, or general consultation, assign first active staff
                staff = db.query(Staff).filter(
                    Staff.organization_id == org_id,
                    Staff.status != StaffStatus.OFFLINE,
                    Staff.online_booking_enabled == True
                ).first()
                if not staff:
                    staff = db.query(Staff).filter(Staff.organization_id == org_id).first()
                if not staff:
                    # Defensive fallback: If no staff in DB, STILL enforce past-time validation
                    now_dt = get_current_business_datetime(org_id=org_id, db=db, reference_now=reference_now)
                    current_date = now_dt.date()
                    naive_now = now_dt.replace(tzinfo=None) if now_dt.tzinfo else now_dt
                    is_past = False
                    if target_time_24h:
                        try:
                            target_time_obj = datetime.strptime(target_time_24h, "%H:%M").time()
                            target_slot_dt = datetime.combine(target_date, target_time_obj)
                            if target_date < current_date or (target_date == current_date and target_slot_dt <= naive_now):
                                is_past = True
                        except Exception:
                            pass
                    if is_past:
                        freq_12h = self.format_time_12h(target_time_24h) if target_time_24h else (raw_time_str or "that time")
                        return CalendarVerificationResult(
                            is_checked=True,
                            is_available=False,
                            is_past_time=True,
                            doctor_name=doctor_name or "Representative",
                            requested_date=target_date.strftime("%A, %b %d, %Y"),
                            requested_time=freq_12h,
                            reason=f"Slot {freq_12h} is no longer available because that time has passed.",
                            prompt_summary=f"[LIVE CALENDAR VERIFICATION: Slot {freq_12h} on {target_date} has ALREADY PASSED.]"
                        )
                    return CalendarVerificationResult(
                        is_checked=True,
                        is_available=True,
                        doctor_name=doctor_name or "Representative",
                        requested_date=target_date.strftime("%A, %b %d, %Y"),
                        requested_time=raw_time_str or "11:30 AM",
                        prompt_summary=f"[LIVE CALENDAR VERIFICATION: Available for {doctor_name}]"
                    )

            raw_name = staff.name
            if raw_name:
                clean_raw = raw_name.title().strip()
                if clean_raw.lower().startswith("dr."):
                    matched_doctor_name = clean_raw
                elif clean_raw.lower().startswith("dr"):
                    matched_doctor_name = f"Dr. {clean_raw[2:].strip()}"
                else:
                    matched_doctor_name = f"Dr. {clean_raw}"
            else:
                matched_doctor_name = None

            staff_id = staff.id

            now_dt = get_current_business_datetime(
                org_id=org_id,
                db=db,
                reference_now=reference_now,
            )
            current_date = now_dt.date()
            naive_now = now_dt.replace(tzinfo=None) if now_dt.tzinfo else now_dt

            slots_res = get_available_slots(staff_id=staff_id, selected_date=target_date, db=db, reference_now=reference_now)
            available_slots_list = slots_res.get("available_slots", [])
            avail_times_24h = [s["start"] for s in available_slots_list]
            avail_times_12h = [self.format_time_12h(t) for t in avail_times_24h]

            formatted_date_str = target_date.strftime("%A, %b %d, %Y")
            entity_label = f"with {matched_doctor_name}" if matched_doctor_name else "for your request"

            if target_time_24h and raw_time_str:
                freq_12h = self.format_time_12h(target_time_24h)

                # Determine whether requested slot has already passed
                is_past = False
                try:
                    target_time_obj = datetime.strptime(target_time_24h, "%H:%M").time()
                    target_slot_dt = datetime.combine(target_date, target_time_obj)
                    if target_date < current_date or (target_date == current_date and target_slot_dt <= naive_now):
                        is_past = True
                except Exception:
                    pass

                if is_past:
                    if avail_times_12h:
                        alt_slots = avail_times_12h[:4]
                        alt_str = ", ".join(alt_slots)
                        reason = f"Slot {freq_12h} is no longer available because that time has passed. Today's available slots are {alt_str}."
                        summary = (
                            f"[LIVE CALENDAR VERIFICATION: Slot {freq_12h} {entity_label} on {formatted_date_str} has ALREADY PASSED. "
                            f"CRITICAL GROUNDING INSTRUCTION: You MUST inform the user that {freq_12h} is no longer available because that time has passed. "
                            f"Offer these verified available remaining slots for today: {alt_str}.]"
                        )
                        return CalendarVerificationResult(
                            is_checked=True,
                            doctor_name=matched_doctor_name,
                            requested_date=formatted_date_str,
                            requested_time=freq_12h,
                            is_available=False,
                            is_past_time=True,
                            reason=reason,
                            alternative_slots=alt_slots,
                            prompt_summary=summary
                        )
                    else:
                        # All slots for today have passed or are booked! Find next available date
                        next_date, next_slots = find_next_available_slots(staff_id=staff_id, from_date=target_date, db=db, reference_now=reference_now)
                        next_date_str = next_date.strftime("%A, %b %d") if next_date else None
                        next_slots_12h = [self.format_time_12h(s["start"]) for s in next_slots[:4]] if next_slots else []
                        next_slots_str = ", ".join(next_slots_12h) if next_slots_12h else ""
                        reason = f"Slot {freq_12h} is no longer available because that time has passed, and there are no remaining slots available today."
                        if next_date_str and next_slots_12h:
                            summary = (
                                f"[LIVE CALENDAR VERIFICATION: Slot {freq_12h} {entity_label} on {formatted_date_str} has ALREADY PASSED and there are no remaining slots today. "
                                f"The next available date is {next_date_str} with slots: {next_slots_str}.]"
                            )
                        else:
                            summary = f"[LIVE CALENDAR VERIFICATION: Slot {freq_12h} {entity_label} on {formatted_date_str} has ALREADY PASSED and there are no remaining slots today.]"
                        return CalendarVerificationResult(
                            is_checked=True,
                            doctor_name=matched_doctor_name,
                            requested_date=formatted_date_str,
                            requested_time=freq_12h,
                            is_available=False,
                            is_past_time=True,
                            reason=reason,
                            alternative_slots=[],
                            next_available_date=next_date_str,
                            next_available_slots=next_slots_12h,
                            prompt_summary=summary
                        )

                is_free = target_time_24h in avail_times_24h
                if is_free:
                    summary = (
                        f"[LIVE CALENDAR VERIFICATION: Slot {freq_12h} {entity_label} on {formatted_date_str} is AVAILABLE. "
                        f"Slot is free in the calendar.]"
                    )
                    return CalendarVerificationResult(
                        is_checked=True,
                        doctor_name=matched_doctor_name,
                        requested_date=formatted_date_str,
                        requested_time=freq_12h,
                        is_available=True,
                        is_past_time=False,
                        reason=f"Slot {freq_12h} is available.",
                        alternative_slots=avail_times_12h[:4],
                        prompt_summary=summary
                    )
                else:
                    alternatives = avail_times_12h[:4]
                    if alternatives:
                        alt_str = ", ".join(alternatives)
                        summary = (
                            f"[LIVE CALENDAR VERIFICATION: Slot {freq_12h} {entity_label} on {formatted_date_str} is NOT AVAILABLE (ALREADY BOOKED / CONFLICT). "
                            f"CRITICAL GROUNDING INSTRUCTION: You MUST inform the user that {freq_12h} is unavailable. "
                            f"Offer these verified available alternative slots: {alt_str}.]"
                        )
                        return CalendarVerificationResult(
                            is_checked=True,
                            doctor_name=matched_doctor_name,
                            requested_date=formatted_date_str,
                            requested_time=freq_12h,
                            is_available=False,
                            is_past_time=False,
                            reason=f"Slot {freq_12h} is already booked or unavailable.",
                            alternative_slots=alternatives,
                            prompt_summary=summary
                        )
                    else:
                        next_date, next_slots = find_next_available_slots(staff_id=staff_id, from_date=target_date, db=db, reference_now=reference_now)
                        next_date_str = next_date.strftime("%A, %b %d") if next_date else None
                        next_slots_12h = [self.format_time_12h(s["start"]) for s in next_slots[:4]] if next_slots else []
                        summary = (
                            f"[LIVE CALENDAR VERIFICATION: Slot {freq_12h} {entity_label} on {formatted_date_str} is NOT AVAILABLE (ALREADY BOOKED / CONFLICT) and there are no other slots on this date. "
                            f"The next available date is {next_date_str} with slots: {', '.join(next_slots_12h)}.]"
                            if next_date_str and next_slots_12h else
                            f"[LIVE CALENDAR VERIFICATION: Slot {freq_12h} {entity_label} on {formatted_date_str} is NOT AVAILABLE and no more slots on this date.]"
                        )
                        return CalendarVerificationResult(
                            is_checked=True,
                            doctor_name=matched_doctor_name,
                            requested_date=formatted_date_str,
                            requested_time=freq_12h,
                            is_available=False,
                            is_past_time=False,
                            reason=f"Slot {freq_12h} is already booked or unavailable, and no other slots remaining on {formatted_date_str}.",
                            alternative_slots=[],
                            next_available_date=next_date_str,
                            next_available_slots=next_slots_12h,
                            prompt_summary=summary
                        )

            alternatives = avail_times_12h[:4]
            if alternatives:
                alt_str = ", ".join(alternatives)
                summary = (
                    f"[LIVE CALENDAR VERIFICATION: Available slots {entity_label} on {formatted_date_str}: {alt_str}.]"
                )
                return CalendarVerificationResult(
                    is_checked=True,
                    doctor_name=matched_doctor_name,
                    requested_date=formatted_date_str,
                    requested_time=None,
                    is_available=len(available_slots_list) > 0,
                    is_past_time=False,
                    reason=f"Checked general availability.",
                    alternative_slots=alternatives,
                    prompt_summary=summary
                )
            else:
                next_date, next_slots = find_next_available_slots(staff_id=staff_id, from_date=target_date, db=db, reference_now=reference_now)
                next_date_str = next_date.strftime("%A, %b %d") if next_date else None
                next_slots_12h = [self.format_time_12h(s["start"]) for s in next_slots[:4]] if next_slots else []
                summary = (
                    f"[LIVE CALENDAR VERIFICATION: No open slots remaining {entity_label} on {formatted_date_str}. "
                    f"The next available date is {next_date_str} with slots: {', '.join(next_slots_12h)}.]"
                    if next_date_str and next_slots_12h else
                    f"[LIVE CALENDAR VERIFICATION: No open slots remaining {entity_label} on {formatted_date_str}.]"
                )
                return CalendarVerificationResult(
                    is_checked=True,
                    doctor_name=matched_doctor_name,
                    requested_date=formatted_date_str,
                    requested_time=None,
                    is_available=False,
                    is_past_time=False,
                    reason=f"No available slots on {formatted_date_str}.",
                    alternative_slots=[],
                    next_available_date=next_date_str,
                    next_available_slots=next_slots_12h,
                    prompt_summary=summary
                )

        except Exception as e:
            logger.error(f"Calendar verification error: {e}")
            return CalendarVerificationResult(
                is_checked=True,
                is_available=None,
                reason=f"Error querying calendar DB: {e}",
                prompt_summary="[LIVE CALENDAR VERIFICATION: System could not query live calendar database.]"
            )
        finally:
            db.close()


calendar_verification_service = CalendarVerificationService()
