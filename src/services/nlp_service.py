"""
Local Open-Source NLP & Intent Detection Service.

Classifies caller text into 14 supported intent categories with confidence scores.
Uses zero-latency rule pre-classifier combined with semantic similarity matching.
Loaded once at startup and reused.
"""

import asyncio
import logging
import os
import re
import time
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class NLPResult(BaseModel):
    intent: str = Field(description="The detected intent category")
    confidence: float = Field(default=0.90, description="Confidence score between 0.0 and 1.0")
    appointment_details: Optional[Dict[str, Any]] = None
    requested_language: Optional[str] = None
    requested_language_code: Optional[str] = None


# Supported Intents
INTENT_INITIAL_GREETING = "initial_greeting"
INTENT_CHANGE_LANGUAGE = "CHANGE_LANGUAGE"
INTENT_LANGUAGE_SWITCH = INTENT_CHANGE_LANGUAGE
INTENT_DOCTOR_LIST = "doctor_list"
INTENT_DOCTOR_COUNT = "doctor_count"
INTENT_DOCTOR_INFORMATION = "doctor_information"
INTENT_STAFF_LOOKUP = "staff_lookup"
INTENT_STAFF_AVAILABILITY = "staff_availability"
INTENT_DEPARTMENT_LIST = "department_list"
INTENT_ORGANIZATION_INFORMATION = "organization_information"
INTENT_COMPANY_INFORMATION = "company_information"
INTENT_BUSINESS_INFORMATION_QUERY = "company_information"
INTENT_HOSPITAL_INFORMATION = INTENT_ORGANIZATION_INFORMATION
INTENT_PRODUCT_INFORMATION = "product_information"
INTENT_SERVICE_INFORMATION = "service_information"
INTENT_PRODUCT_SERVICE_QUERY = "product_information"
INTENT_FAQ = "faq"
INTENT_APPOINTMENT_BOOKING = "appointment_booking"
INTENT_APPOINTMENT_REQUEST = "appointment_booking"
INTENT_APPOINTMENT_AVAILABILITY = "appointment_availability"
INTENT_APPOINTMENT_CANCELLATION = "appointment_cancellation"
INTENT_APPOINTMENT_RESCHEDULING = "appointment_rescheduling"
INTENT_GREETING = "greeting"
INTENT_CONNECTION_REASSURANCE = "connection_reassurance"
INTENT_GENERAL_CONVERSATION = "general_conversation"
INTENT_GOODBYE = "goodbye"
INTENT_THANKS = "thanks"
INTENT_GENERAL_QUESTION = "general_question"
INTENT_FOLLOW_UP_QUESTION = "follow_up_question"
INTENT_CALLBACK = "callback"
INTENT_SALES_REQUEST = "callback"
INTENT_UNKNOWN = "unknown"

# Semantic Conversation Intelligence Intents (Tamil / Tanglish / Multilingual)
INTENT_EMOTIONAL_QUESTION = "emotional_question"
INTENT_REPETITION_COMPLAINT = "repetition_complaint"
INTENT_SILENCE_CHECK = "silence_check"
INTENT_FRUSTRATION = "frustration"
INTENT_WEBSITE_INQUIRY = "website_inquiry"
INTENT_ERP_INQUIRY = "erp_inquiry"
INTENT_SERVICE_INQUIRY = "service_inquiry"
INTENT_PRODUCT_INQUIRY = "product_inquiry"
INTENT_PRICING_INQUIRY = "pricing_inquiry"
INTENT_CALLBACK_REQUEST = "callback"
INTENT_HUMAN_AGENT_REQUEST = "human_agent_request"
INTENT_CONTACT_INFORMATION_PROVIDED = "contact_information_provided"
INTENT_NAME_PROVIDED = "name_provided"
INTENT_EMAIL_PROVIDED = "email_provided"
INTENT_PHONE_PROVIDED = "phone_provided"
INTENT_CONFIRMATION = "confirmation"
INTENT_NEGATION = "negation"
INTENT_CLARIFICATION = "clarification"
INTENT_APPOINTMENT_DETAILS_INQUIRY = "appointment_details_inquiry"
INTENT_DEMO_CONTACT_INQUIRY = "demo_contact_inquiry"
INTENT_AVAILABILITY_PROVIDED = "availability_provided"
INTENT_CORRECTION = "correction"
INTENT_INTERRUPTION = "interruption"

# Additional safety & legacy compatibility intents
INTENT_EMERGENCY = "emergency/urgent request"
INTENT_TRANSFER = "transfer request"

# Backward compatibility aliases
INTENT_APPOINTMENT = INTENT_APPOINTMENT_BOOKING
INTENT_CANCELLATION = INTENT_APPOINTMENT_CANCELLATION
INTENT_RESCHEDULING = INTENT_APPOINTMENT_RESCHEDULING
INTENT_DOCTOR_ENQUIRY = INTENT_DOCTOR_LIST
INTENT_CLINIC_INFO = INTENT_HOSPITAL_INFORMATION
INTENT_GENERAL = INTENT_GENERAL_CONVERSATION


class NLPService:
    def __init__(self):
        self.encoder = None

    def initialize_model(self):
        """Pre-load local NLP model / sentence transformer for intent classification."""
        if "PYTEST_CURRENT_TEST" in os.environ or os.getenv("APP_ENV") == "test":
            self.encoder = "mock_nlp"
            return

        t0 = time.perf_counter()
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
            dur = (time.perf_counter() - t0) * 1000.0
            logger.info(f"Local NLP model (all-MiniLM-L6-v2) loaded successfully in {dur:.2f} ms")
        except Exception as exc:
            logger.warning(f"Could not load SentenceTransformer for NLP ({exc}). Using rule-based NLP classifier.")
            self.encoder = None

    def detect_intent(self, text: str) -> NLPResult:
        """
        Classify input text into one of the supported intents.
        Prioritizes exact caller intent over session context.
        """
        txt = (text or "").strip().lower()
        if not txt:
            return NLPResult(intent=INTENT_GENERAL_CONVERSATION, confidence=0.50)

        # 1. Emergency / Urgent Safety Rule
        if any(w in txt for w in ["chest pain", "bleeding", "unconscious", "emergency", "urgent", "ambulance", "heart attack", "stroke", "die", "dying"]):
            return NLPResult(intent=INTENT_EMERGENCY, confidence=0.99)

        # 1a. Language Switch Intent (Highest conversational priority)
        from src.services.language_service import language_service
        target_lang, is_switch = language_service.check_explicit_language_switch(text)
        if is_switch and target_lang:
            lang_name = language_service.get_language_name(target_lang)
            return NLPResult(
                intent=INTENT_CHANGE_LANGUAGE,
                confidence=0.99,
                requested_language=lang_name,
                requested_language_code=target_lang
            )

        # 1b. Emotional Questions (Tamil, Tanglish & English)
        is_emotional_q = (
            any(w in txt for w in [
                "என்ன கோச்சிகிட்டியா", "கோச்சிகிட்டியா", "கோச்சுக்கிட்டியா", "கோவமா", "கோவம் வந்திருச்சா",
                "கோபமா", "கோவப்பட்டியா", "நான் கோச்சி கிடையாதுன்னு", "கோச்சிகிட்டீங்களா",
                "kochikitiya", "kochikitaya", "are you angry", "why are you angry", "are you mad",
                "why are you sounding like that", "why are you mad at me"
            ])
            or bool(re.search(r'\b(?:கோச்சி|கோச்சு|கோவம்|கோபம்)\b', txt))
        )
        if is_emotional_q:
            return NLPResult(intent=INTENT_EMOTIONAL_QUESTION, confidence=0.98)

        # 1c. Repetition Complaints / Caller Frustration with Repeated Prompts
        is_repetition_complaint = (
            any(w in txt for w in [
                "எதுக்கு நீ சொன்னதே திருப்பின்னு சொல்லிட்டு இருக்க", "எதுக்கு திரும்பத் திரும்ப கேக்குற",
                "திரும்ப திரும்ப கேக்குற", "திரும்ப திரும்ப சொல்ற", "ஏன் திரும்ப திரும்ப கேக்குற",
                "சொன்னதே திரும்ப சொல்லாத", "சொன்னதே திருப்பின்னு சொல்லிட்டு இருக்க", "ஒரே கேள்விய கேக்குற",
                "thirumba thirumba", "why are you repeating", "why repeating", "why do you keep repeating",
                "stop repeating", "why asking the same thing", "repeat panra"
            ])
            or bool(re.search(r'\b(?:திருப்பி|திரும்ப|repeat|repeating)\b.*(?:சொல்|கேக்|கேள்வி|sol|kek)', txt))
        )
        if is_repetition_complaint:
            return NLPResult(intent=INTENT_REPETITION_COMPLAINT, confidence=0.98)

        # 1d. Silence / Response Pause Checks
        is_silence_check = (
            any(w in txt for w in [
                "என்னாச்சுங்க அமைதியா ஆயிட்டீங்களா", "என்னாச்சு அமைதியா ஆயிட்டீங்க", "ஏன் அமைதியா இருக்கீங்க",
                "அமைதியா ஆயிட்டீங்களா", "இதெல்லாம் சொன்னா அமைதியா ஆயிடு", "ennachu amaidhiya",
                "why are you silent", "why silent", "why are you quiet", "why so quiet"
            ])
        )
        if is_silence_check:
            return NLPResult(intent=INTENT_SILENCE_CHECK, confidence=0.98)

        # 1e. Confirmation / Caller ID Phone Reuse
        is_caller_id_confirm = (
            any(w in txt for w in [
                "இந்த numberல callback பண்ணுங்க", "இந்த number-லேயே பண்ணுங்க", "இந்த numberல பண்ணுங்க",
                "இந்த number", "intha number", "same number", "use this number",
                "call on this number", "call back on this number"
            ])
        )
        if is_caller_id_confirm:
            return NLPResult(intent=INTENT_CONFIRMATION, confidence=0.98)

        # 1f. Negation / Field Refusal (e.g. "Email வேண்டாம்", "skip", "skip pannunga")
        is_negation_req = (
            any(w in txt for w in [
                "email வேண்டாம்", "mail venam", "வேண்டாம்", "வேண்டாங்க", "venam", "vendaam",
                "வேற எதுவும் வேண்டாம்", "no email", "dont want email", "don't want email",
                "no mail", "skip email", "email skip", "skip pannunga", "skip pannu", "skip panna",
                "skip பண்ணுங்க", "ஸ்கிப் பண்ணுங்க", "ஸ்கிப்", "விடுங்க", "விடவும்", "வேண்டாம் விடுங்க",
                "i said skip", "skip this", "skip it"
            ])
            or txt in {"no", "nope", "skip", "none", "illa", "nahi", "வேண்டாம்", "venam", "விடுங்க", "ஸ்கிப்"}
        )
        if is_negation_req:
            return NLPResult(intent=INTENT_NEGATION, confidence=0.98)

        # 1f-2. Demo / Contact Turnaround Timeline Inquiry
        is_contact_turnaround_q = (
            any(w in txt for w in [
                "when will your team contact me", "when your team contact me", "when will someone call me",
                "when will someone contact me", "when will you call me", "when will you contact me",
                "how soon will your team contact me", "how soon will you call me", "how long will it take to call me",
                "எப்போ call பண்ணுவீங்க", "எப்போ contact பண்ணுவீங்க", "eppo call pannuvenga"
            ])
            or bool(re.search(r'\b(?:when|how\s+soon|what\s+time)\b.*\b(?:contact|call|reach)\b', txt))
        )
        if is_contact_turnaround_q:
            return NLPResult(intent=INTENT_DEMO_CONTACT_INQUIRY, confidence=0.98)

        # 1f-3. Stating Availability Preference ("I am available tomorrow", "I will be available tomorrow")
        is_avail_stated = bool(re.search(r'\b(?:i\s+am|i\'m|i\s+will\s+be|im)\s+available\b', txt))
        if is_avail_stated:
            return NLPResult(intent=INTENT_AVAILABILITY_PROVIDED, confidence=0.98)

        # 1f-4. Appointment Details Inquiry
        is_appt_details_q = any(p in txt for p in [
            "share the appointment details", "share my appointment details", "share appointment details",
            "share the date and time", "share date and time", "appointment details", "what are my appointment details",
            "what is my appointment details", "can you share the date and time", "my appointment details",
            "check my appointment", "appointment status", "when is my appointment",
            "appointment details சொல்லுங்க", "appointment விவரங்கள்"
        ])
        if is_appt_details_q:
            return NLPResult(intent=INTENT_APPOINTMENT_DETAILS_INQUIRY, confidence=0.98)

        # 1g. Callback / Team Contact Request (Tamil, Tanglish & English)
        is_callback_req = (
            any(w in txt for w in [
                "உங்க team callback பண்ண சொல்லுங்க", "எனக்கு call பண்ண சொல்லுங்க", "ஒருத்தர் எனக்கு contact பண்ணட்டும்",
                "நான் பேசணும் உங்கள் team கூட", "team contact பண்ண சொல்லுங்க", "callback பண்ணுங்க", "call பண்ணுங்க",
                "contact பண்ணுங்க", "team callback", "எனக்கு வேற எதுவும் தெரிஞ்சிக்க வேணாம்",
                "callback panna sollunga", "call panna sollunga", "contact panna sollunga", "callback pannunga",
                "call pannunga", "team callback pannunga", "pesanum ungal team kooda", "pesanum unga team",
                "team contact pannattum", "call me back", "request a callback", "callback", "arrange a callback",
                "give me a call back", "schedule a callback", "schedule a call", "call back",
                "have someone call me", "can someone call me", "arrange a call", "speak with your team",
                "speak with the team", "speak to your team", "speak to the team", "talk to your team",
                "connect with your team", "contact me"
            ])
        )
        if is_callback_req:
            return NLPResult(intent=INTENT_CALLBACK, confidence=0.98)

        # 1h. Service Inquiries: ERP & Website
        is_erp_req = any(w in txt for w in [
            "erp software வேணும்", "erp software ready", "erp software பற்றி",
            "erp software மட்டும் போதும்", "erp software", "custom erp", "erp", "enterprise resource planning"
        ])
        if is_erp_req:
            return NLPResult(intent=INTENT_ERP_INQUIRY, confidence=0.98)

        is_website_req = any(w in txt for w in [
            "website பற்றி சொல்லுங்க", "website பற்றி", "வலைத்தளம்", "website development",
            "web development", "website design", "website create", "about website"
        ])
        if is_website_req:
            return NLPResult(intent=INTENT_WEBSITE_INQUIRY, confidence=0.98)

        # 1i. Explicit Name Statement (Tamil, Tanglish & English)
        name_intro = re.search(r'\b(?:என்\s+பெயர்|என்\s+பேரு|என்னுடய\s+பெயர்|en\s+peyar|en\s+peru|my\s+name\s+is|this\s+is|myself)\s+([a-zA-Z\u0B80-\u0BFF\s]+)', txt)
        if name_intro:
            cand_name = name_intro.group(1).strip()
            if len(cand_name.split()) <= 4:
                return NLPResult(intent=INTENT_NAME_PROVIDED, confidence=0.98, appointment_details={"customer": cand_name.title()})

        # Business content terms to prevent compound queries like "Can you hear me? What are your products?"
        # from getting swallowed by pleasantry or connection check classifiers
        business_content_terms = {
            "product", "products", "service", "services", "pricing", "price", "cost", "fee",
            "feature", "features", "demo", "software", "solution", "solutions", "crm", "erp",
            "hrms", "payroll", "agent", "agents", "bot", "bots", "hospital", "doctor", "doctors",
            "clinic", "treatment", "treatments", "book", "booking", "schedule", "appointment"
        }
        words_in_txt = set(re.sub(r'[^\w\s]', ' ', txt).split())
        has_biz_query = bool(words_in_txt & business_content_terms) or any(
            p in txt for p in [
                "what products", "which products", "what services", "which services", "explain your services",
                "explain services", "tell me about your", "about your services", "about your products",
                "what do you do", "what can you do", "team call", "arrange a team call"
            ]
        )

        # 2a. Audio / Connection Reassurance (e.g. "can you hear me", "are you there", "am i audible", "hello are you there")
        # ONLY triggers if there is NO business inquiry in the utterance
        is_audio_check = any(phrase in txt for phrase in [
            "can you hear me", "are you there", "am i audible", "can you hear", "hear me",
            "is anyone there", "hello are you there", "hello are you listening", "are you listening"
        ])
        if is_audio_check and not has_biz_query:
            return NLPResult(intent=INTENT_CONNECTION_REASSURANCE, confidence=0.98)

        # 2b. General Conversation & Acknowledgements (e.g. "how are you", "I'm home", "I'm here", "yes", "okay", "alright")
        conversational_phrases = [
            "how are you", "how are you doing", "how do you do", "what's up", "what is up",
            "i am home", "i'm home", "im home", "i am here", "i'm here", "im here", "i am back", "i'm back",
            "i am ready", "i'm ready", "im ready", "i understand", "i see", "sounds good", "tell me more"
        ]
        if (any(phrase in txt for phrase in conversational_phrases) or txt in {"yes", "yeah", "yep", "sure", "ok", "okay", "alright", "fine", "cool", "great", "good"}) and not has_biz_query:
            return NLPResult(intent=INTENT_GENERAL_CONVERSATION, confidence=0.98)

        # 3. Greetings
        txt_clean_punc = re.sub(r'[^\w\s\u0b80-\u0bff\u0900-\u097f\u0c80-\u0cff\u0c00-\u0c7f]', '', txt).strip()
        if txt_clean_punc in {"hi", "hello", "hey", "good morning", "good afternoon", "good evening", "greetings", "vanakkam", "namaste", "hello there", "hi there", "hey there", "வணக்கம்", "நமஸ்காரம்", "नमस्ते", "నమస్కారం", "ನಮಸ್ಕಾರ"} and not has_biz_query:
            return NLPResult(intent=INTENT_GREETING, confidence=0.99)

        # 3b. Goodbye
        if any(w in txt for w in ["bye", "goodbye", "see you", "exit", "thank you bye", "thanks bye"]):
            return NLPResult(intent=INTENT_GOODBYE, confidence=0.99)

        # 3c. Thanks & Gratitude
        if (
            txt_clean_punc in {"thanks", "thank you", "thank u", "many thanks", "thank you so much", "thanks a lot", "appreciated", "much appreciated", "that is all", "thats all", "that will be all"}
            or any(p in txt for p in ["thank you", "thanks", "thank u", "appreciated", "that is all", "that's all", "that will be all"])
        ):
            return NLPResult(intent=INTENT_THANKS, confidence=0.99)

        # 4a. Doctor Count Request
        is_doctor_count = any(phrase in txt for phrase in [
            "how many doctors", "number of doctors", "doctor count", "count of doctors",
            "how many doctors available", "how many doctors do you have", "how many doctors are there", "total doctors"
        ])
        if is_doctor_count:
            return NLPResult(intent=INTENT_DOCTOR_COUNT, confidence=0.99)

        # 4b. Staff / Doctor / Teacher List Request
        is_doctor_or_staff_list = any(phrase in txt for phrase in [
            "list your doctors", "list the doctors", "list doctors", "who are your doctors", "who are the doctors",
            "tell me about your doctors", "what doctors do you have", "which doctors do you have", "doctor list",
            "show doctors", "list of doctors", "our doctors", "doctor names", "doctors available in your hospital",
            "doctors in the hospital", "show me your doctors", "show me the doctors",
            "list the staff", "list staff", "list staff members", "staff members", "who are all available",
            "who is available", "who are available", "staff list", "list of staff", "list teachers", "list the teachers"
        ]) or (
            any(w in txt for w in ["who", "what", "list", "tell me", "show", "irukka"])
            and any(w in txt for w in ["doctor", "doctors", "physician", "specialist", "staff", "teacher", "teachers", "stylist", "stylists", "member", "members"])
            and not any(w in txt for w in ["my name is", "i am", "call me", "book with dr", "book with stylist", "book with teacher"])
        )
        if is_doctor_or_staff_list:
            return NLPResult(intent=INTENT_DOCTOR_LIST, confidence=0.98)

        # 4c. Staff Lookup / Department Staff Mapping (e.g. "who is in HR?", "who handles sales?")
        is_staff_lookup = any(phrase in txt for phrase in [
            "who is in hr", "who handles sales", "who works in", "who is the ceo", "who is our ceo", "who handles",
            "who is in charge of", "who heads"
        ]) or (
            any(w in txt for w in ["who is in", "who works in", "who handles", "who manages"])
            and any(w in txt for w in ["hr", "sales", "marketing", "finance", "engineering", "support", "billing", "department"])
        )
        if is_staff_lookup:
            return NLPResult(intent=INTENT_STAFF_LOOKUP, confidence=0.98)

        # 4d. Staff Availability Check (e.g. "is Sanjay available?", "is Dr. Harini available?", "can I talk to Dr. Harini?")
        is_staff_avail = (
            any(phrase in txt for phrase in ["is available", "available today", "available tomorrow", "can i speak with", "can i talk to", "can i speak to", "is free", "free today", "free tomorrow"])
            and not any(w in txt for w in ["book", "appointment", "reserve", "slot"])
        ) or bool(re.search(r'\b(?:is\s+[a-zA-Z]+\s+available|can\s+i\s+(?:talk\s+to|speak\s+with|speak\s+to)\s+[a-zA-Z]+)\b', txt))
        if is_staff_avail:
            return NLPResult(intent=INTENT_STAFF_AVAILABILITY, confidence=0.96)

        # 5a. Department List Request
        is_dept_list = any(phrase in txt for phrase in [
            "what departments", "departments do you have", "list departments", "which departments",
            "show departments", "department list", "list of departments", "our departments"
        ]) or (
            any(w in txt for w in ["department", "departments", "branches", "specialties"])
            and any(w in txt for w in ["list", "what", "which", "show", "have", "irukka", "available", "tell me"])
            and not any(w in txt for w in ["my name is", "i am", "call me"])
        )
        if is_dept_list:
            return NLPResult(intent=INTENT_DEPARTMENT_LIST, confidence=0.98)

        # 5b. Product Information Request (RAG Only)
        is_product_info = any(phrase in txt for phrase in [
            "what products", "which products", "list products", "list the products", "product list", "list of products",
            "what are all the product", "what are all the products", "what products do you offer", "what products do you have",
            "what are your products", "our products", "features of", "tell me about hrms", "explain your hr", "product features",
            "about the product", "product details", "what does your company do", "tell me about your company"
        ]) or (
            any(w in txt for w in ["product", "products", "software", "platform", "tool", "tools", "hrms"])
            and any(w in txt for w in ["list", "what", "which", "show", "have", "offer", "provide", "feature", "features", "tell me", "explain", "detail", "details", "info", "about"])
            and not any(w in txt for w in ["my name is", "i am", "call me"])
        )
        if is_product_info:
            return NLPResult(intent=INTENT_PRODUCT_INFORMATION, confidence=0.98)

        # 5c. Service List Request (RAG Only)
        is_service_list = any(phrase in txt for phrase in [
            "what services", "which services", "list services", "list the services", "service list", "list of services",
            "what are all the service", "what are all the services", "what treatments", "list treatments", "our services",
            "what services do you offer", "what services do you provide", "explain your services", "explain the services",
            "can you explain your services", "explain services", "tell me about your services", "tell me about services",
            "about your services", "சேவைகள்", "சேவை பற்றி", "சேவைகள் பற்றி", "என்ன சேவைகள்", "sevai pathi", "services pathi"
        ]) or (
            any(w in txt for w in ["service", "services", "facilities", "treatments", "offer", "provide", "சேவை", "சேவைகள்", "சேவைகளை", "sevai", "sevaigal"])
            and any(w in txt for w in ["list", "what", "which", "show", "have", "irukka", "available", "tell me", "explain", "detail", "details", "info", "about", "பற்றி", "சொல்லுங்க", "சொல்லுங்கள்"])
            and not any(w in txt for w in ["my name is", "i am", "call me"])
        )
        if is_service_list:
            return NLPResult(intent=INTENT_SERVICE_INFORMATION, confidence=0.98)

        # 6. Staff / Doctor Information / Working Schedule Request (e.g. "tell me about Mr Rohit", "what does Dr. Harini do?", "what does Sanjay do?")
        is_doctor_info_or_schedule = (
            any(phrase in txt for phrase in [
                "tell me about", "who is", "specialization", "profile of", "details of",
                "schedule of", "working hours", "timing of", "hours of", "working schedule", "schedule", "what does"
            ]) and (
                any(w in txt for w in ["dr", "doctor", "physician", "specialist", "staff", "teacher", "stylist", "mr", "ms", "mrs", "mr.", "ms.", "mrs."])
                or bool(re.search(r'\bwhat\s+does\s+[a-zA-Z]+\s+do\b', txt))
                or bool(re.search(r'\btell\s+me\s+about\s+(?:dr\.?|mr\.?|ms\.?|mrs\.?)?\s*[a-zA-Z]+\b', txt))
            )
            and not any(w in txt for w in ["book", "reserve", "want to book", "book appointment", "take appointment", "slot", "product", "company", "service", "hrms"])
        ) or bool(re.search(r'\b(tell\s+me\s+about|schedule\s+of|working\s+hours\s+of|timing\s+of)\s+(dr|mr|ms|mrs)\.?\s*[a-zA-Z]+\b', txt))

        if is_doctor_info_or_schedule:
            return NLPResult(intent=INTENT_DOCTOR_INFORMATION, confidence=0.96)

        # 7. Cancellation Request
        cancellation_keywords = [
            "cancel", "cancelling", "cancellation", "cancel my appointment", "cancel my booking",
            "cancel the appointment", "cancel the booking", "cancel appointment", "cancel booking",
            "drop appointment", "drop booking", "drop my appointment", "drop the appointment",
            "dont want appointment", "don't want appointment", "dont want booking", "don't want booking",
            "want to cancel", "need to cancel", "like to cancel", "wish to cancel",
            "cancel tomorrow", "cancel today", "remove my appointment", "remove my booking",
            "delete appointment", "delete my appointment", "won't be able to make it",
            "cannot make it", "can't make it to my appointment", "can't make it",
            "appointment cancel", "booking cancel"
        ]
        is_cancellation = (
            any(w in txt for w in cancellation_keywords)
            or bool(re.search(r'\b(?:drop|cancel|delete|remove)\s+(?:the\s+|my\s+|our\s+)?(?:appointment|booking|visit|slot)\b', txt))
        )
        if is_cancellation:
            return NLPResult(intent=INTENT_APPOINTMENT_CANCELLATION, confidence=0.98)

        # 8. Rescheduling Request
        if any(w in txt for w in ["reschedule", "change time", "move appointment", "postpone", "change date"]):
            return NLPResult(intent=INTENT_APPOINTMENT_RESCHEDULING, confidence=0.95)

        # 8b. Appointment Details Inquiry
        if any(p in txt for p in [
            "share the appointment details", "share my appointment details", "share appointment details",
            "share the date and time", "share date and time", "appointment details", "what are my appointment details",
            "what is my appointment details", "can you share the date and time", "my appointment details",
            "check my appointment", "appointment status", "when is my appointment"
        ]):
            return NLPResult(intent=INTENT_APPOINTMENT_DETAILS_INQUIRY, confidence=0.98)

        # 8c. Stating Availability Preference ("I am available tomorrow", "I will be available tomorrow")
        if re.search(r'\b(?:i\s+am|i\'m|i\s+will\s+be|im)\s+available\b', txt):
            return NLPResult(intent=INTENT_AVAILABILITY_PROVIDED, confidence=0.98)

        # 8d. Demo / Contact Turnaround Timeline Inquiry
        if (
            any(w in txt for w in [
                "when will your team contact me", "when your team contact me", "when will someone call me",
                "when will someone contact me", "when will you call me", "when will you contact me",
                "how soon will your team contact me", "how soon will you call me", "how long will it take to call me"
            ])
            or bool(re.search(r'\b(?:when|how\s+soon|what\s+time)\b.*\b(?:contact|call|reach)\b', txt))
        ):
            return NLPResult(intent=INTENT_DEMO_CONTACT_INQUIRY, confidence=0.98)

        # 9. Appointment Availability Request (e.g. "what time slots do you have?", "is Dr. Arun available tomorrow?")
        if any(phrase in txt for phrase in [
            "what time slots", "time slots", "available slots", "slots available", "open slots", "free slots",
            "available times", "available time", "is dr", "available on", "available tomorrow"
        ]):
            return NLPResult(intent=INTENT_APPOINTMENT_AVAILABILITY, confidence=0.96)

        # 9b. Callback / Team Contact / Arrangement Request (Lead Collection)
        # Evaluated BEFORE appointment booking to ensure "arrange a call", "schedule a call", "speak with team"
        # are NEVER misrouted to calendar appointment booking!
        callback_phrases = [
            "arrange a team call", "arrange team call", "team call", "call with your team", "call with the team",
            "arrange a call with your team", "arrange a call with the team", "can i arrange a team call",
            "arrange a call", "arrange call", "make an arrangement", "make arrangement", "arrange an arrangement",
            "speak with your team", "speak with the team", "speak to your team", "speak to the team",
            "talk to your team", "talk to the team", "talk with your team", "talk with the team",
            "connect with your team", "connect with the team", "i want to speak with your team",
            "i want to talk to your team", "contact me", "yes contact me", "yes, contact me",
            "call me back", "request a callback", "callback", "arrange a callback",
            "give me a call back", "schedule a callback", "schedule a call", "call back",
            "have someone call me", "can someone call me", "ask someone to call me",
            "get someone to call me", "have your team call me", "can your team call me"
        ]
        if any(p in txt for p in callback_phrases):
            return NLPResult(intent=INTENT_CALLBACK, confidence=0.98)

        # 10. Appointment Booking Request
        if any(w in txt for w in ["book", "booking", "appointment", "schedule", "reserve", "slot", "slots", "want an appointment", "need an appointment", "make an appointment", "get an appointment", "take an appointment", "fix an appointment", "book doctor", "pannanum", "venum"]):
            customer = None
            date_str = None
            time_str = None
            staff = None

            # Extract customer name if present
            name_match = re.search(r'\b(?:my\s+(?:full\s+)?name\s+is|i\s*(?:am|\'m)|this\s+is|you\s+can\s+call\s+me|call\s+me|myself)\s+([a-zA-Z\s]+?)(?:\s+(?:and|&|want|with|for|to|at|on)|\s*$)', txt, re.IGNORECASE)
            if name_match:
                c_name = name_match.group(1).strip()
                if len(c_name.split()) <= 4:
                    customer = c_name.title()

            if "tomorrow" in txt:
                date_str = "Tomorrow"
            elif "today" in txt:
                date_str = "Today"

            time_match = re.search(r'\b(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b', txt)
            if time_match:
                time_str = time_match.group(1).upper()

            staff_match = re.search(r'\b(dr\.\s*[a-zA-Z]+|doctor\s*[a-zA-Z]+)\b', txt)
            if staff_match:
                staff = staff_match.group(1).title()

            return NLPResult(
                intent=INTENT_APPOINTMENT_BOOKING,
                confidence=0.95,
                appointment_details={
                    "customer": customer,
                    "appointment_date": date_str,
                    "appointment_time": time_str,
                    "staff": staff
                }
            )

        # 11. Organization / Company / Hospital Information / Location / Business Hours
        if any(phrase in txt for phrase in [
            "what is your hospital name", "which hospital is this", "where am i calling", "tell me the hospital name",
            "hospital name", "may i know your hospital name", "your hospital name", "hospital details",
            "what is your organization name", "what is your company name", "which company is this",
            "organization name", "company name", "clinic name", "apartment name", "hotel name"
        ]) or any(w in txt for w in ["location", "address", "where is", "hospital info", "org info", "hours", "open", "close", "timing", "when are you open"]):
            return NLPResult(intent=INTENT_HOSPITAL_INFORMATION, confidence=0.98)

        # 12. Service Information
        if any(w in txt for w in ["service", "services", "facilities", "treatments", "offer", "provide", "சேவை", "சேவைகள்", "சேவைகளை", "sevai", "sevaigal"]):
            return NLPResult(intent=INTENT_SERVICE_INFORMATION, confidence=0.92)

        # 13. FAQ / Pricing / Consultation Fees / Insurance
        if any(phrase in txt for phrase in ["how much", "how much is", "how much for", "what is the cost", "what is the price", "how much does"]) or any(w in txt for w in ["fee", "fees", "cost", "price", "pricing", "charge", "rate", "rates", "tariff", "amount", "insurance", "policy", "faq"]):
            return NLPResult(intent=INTENT_FAQ, confidence=0.95)

        # 14a. Callback Request
        if any(w in txt for w in ["call me back", "request a callback", "callback", "arrange a callback", "give me a call back", "schedule a callback", "call back"]):
            return NLPResult(intent=INTENT_CALLBACK, confidence=0.96)

        # 14b. Human Transfer Request
        # IMPORTANT: Only trigger on EXPLICIT requests to speak/talk TO a human.
        # Do NOT trigger on informational questions that mention "receptionist" or similar words.
        is_informational = any(phrase in txt for phrase in [
            "tell me about", "what is", "what are", "explain", "describe", "how does", "how do",
            "can you tell", "what do you", "give me information", "information about"
        ])
        if not is_informational and any(w in txt for w in [
            "transfer", "speak to human", "speak to a human", "speak with human", "speak with a human",
            "human representative", "speak to agent", "speak with agent", "operator",
            "real person", "human agent", "talk to a person", "talk to human", "talk to a human",
            "connect me to", "connect me with"
        ]):
            return NLPResult(intent=INTENT_TRANSFER, confidence=0.95)

        # 15. Spoken Phone Number / Contact Info
        raw_digits = re.sub(r'[^\d]', '', txt)
        if len(raw_digits) >= 5 and (len(txt.split()) <= 7 or any(w in txt for w in ["it's", "its", "number", "phone", "mobile", "contact"])):
            return NLPResult(intent=INTENT_GENERAL_CONVERSATION, confidence=0.95, appointment_details={"contact_number": raw_digits})

        # 16. Standalone Name / Entity statement (filter out common conversational adjectives & state words)
        NON_NAME_WORDS = {"home", "here", "there", "back", "ready", "fine", "good", "ok", "okay", "alright", "listening", "calling", "busy", "free", "available", "sorry", "done", "well", "happy", "sad", "sick", "speaking", "talking"}
        name_only_match = re.search(r'^\s*(?:my\s+(?:full\s+)?name\s+is|this\s+is|you\s+can\s+call\s+me|call\s+me|myself)\s+([a-zA-Z\s]+)\s*$', txt, re.IGNORECASE)
        if not name_only_match:
            i_am_match = re.search(r'^\s*(?:i\s*(?:am|\'m))\s+([a-zA-Z\s]+)\s*$', txt, re.IGNORECASE)
            if i_am_match and i_am_match.group(1).strip().lower() not in NON_NAME_WORDS:
                name_only_match = i_am_match

        if name_only_match:
            c_name = name_only_match.group(1).strip()
            if len(c_name.split()) <= 4 and c_name.lower() not in NON_NAME_WORDS:
                return NLPResult(
                    intent=INTENT_GENERAL_CONVERSATION,
                    confidence=0.90,
                    appointment_details={"customer": c_name.title()}
                )

        # Fallback to general_conversation or general_question based on query structure
        is_question = any(w in txt for w in ["who", "what", "where", "when", "why", "how", "which", "is", "are", "do", "does", "can", "could", "tell me"]) or "?" in text
        if is_question:
            return NLPResult(intent=INTENT_GENERAL_QUESTION, confidence=0.85)
        return NLPResult(intent=INTENT_GENERAL_CONVERSATION, confidence=0.85)

    async def detect_intent_async(self, text: str) -> NLPResult:
        """Asynchronous wrapper for intent detection."""
        return await asyncio.to_thread(self.detect_intent, text)


nlp_service = NLPService()
