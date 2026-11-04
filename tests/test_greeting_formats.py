import re

quote_chars = '"\'“”‘’«»„‟'

def is_valid_spoken_greeting(g: str, doc_title: str = None) -> bool:
    if not g or len(g.strip()) < 8:
        return False
    clean = g.strip()
    # Reject filename / extension
    if re.search(r'\.(pdf|docx|txt|doc|csv|wav|mp3)\b', clean, re.IGNORECASE):
        return False
    # Reject document title / filename stem
    if doc_title:
        dt_clean = re.sub(r'\.(pdf|docx|txt|doc|csv)$', '', str(doc_title), flags=re.IGNORECASE).strip().lower()
        if clean.lower() == dt_clean or clean.lower() == str(doc_title).lower().strip():
            return False
    # Reject underscore identifiers
    if "_" in clean and clean.count("_") >= 2:
        return False
    # Reject metadata or instruction headers
    bad_prefixes = [
        "retrieve", "check", "when", "the receptionist", "if a customer",
        "verify", "identify", "must", "note:", "step", "flow", "instructions:",
        "system:", "prompt:", "hospital name:", "company name:", "organization name:",
        "address:", "phone:", "timing:", "hours:", "department:", "services:"
    ]
    if any(clean.lower().startswith(bp) for bp in bad_prefixes):
        return False
    words = clean.split()
    if len(words) < 3:
        return False
    return True

def clean_spoken_text(text: str) -> str:
    """Strips quotes, stage directions (e.g. (warm tone)), speaker labels, and markdown."""
    s = text.strip()
    # Remove stage directions in parentheses or brackets e.g. (warmly), [in a friendly tone]
    s = re.sub(r'^\s*[\(\[][^\)\]]+[\)\]]\s*', '', s)
    s = re.sub(r'\s*[\(\[][^\)\]]+[\)\]]\s*$', '', s)
    # Remove leading speaker label e.g. "Receptionist:", "AI:", "Agent:"
    s = re.sub(r'^(?:AI\s+Receptionist|Receptionist|AI\s+Assistant|Assistant|Agent|AI)\s*:\s*', '', s, flags=re.IGNORECASE)
    # Remove instruction triggers like "Say:", "Says:"
    s = re.sub(r'^(?:Say|Says|Speaks|Spoken|Greet with)\s*:\s*', '', s, flags=re.IGNORECASE)
    # Strip quotes and markdown formatting
    s = s.strip(quote_chars).strip()
    s = s.replace("**", "").replace("*", "").replace("__", "").replace("`", "")
    s = re.sub(r'\s+', ' ', s).strip()
    return s.strip(quote_chars).strip()

def extract_greeting(text: str, doc_title: str = None) -> str:
    if not text or not text.strip():
        return ""
    clean_src = text.strip()

    # 1. Look for explicit GREETING section header
    # Supports headers like:
    # # 1. Greeting & Call Opening
    # ## Greeting
    # GREETING & CALL OPENING:
    # 1. Greeting:
    # [GREETING]
    # Voice Greeting:
    hdr_regex = re.compile(
        r'(?m)^[\s*\-•#]*(?:\d+[\.\)]\s*)?(?:\*\*|__)?'
        r'(?:Voice Greeting|Inbound Greeting|Opening Greeting|Receptionist Greeting|'
        r'Greeting\s*(?:&|and|/)\s*Call Opening|Call Opening\s*(?:&|and|/)\s*Greeting|'
        r'Call Opening|Welcome Greeting|Opening Message|Approved Greeting|Initial Greeting|'
        r'Standard Greeting|Spoken Greeting|Approved Spoken Greeting|Welcome Message|'
        r'Phone Greeting|First Response|Opening|Introduction|Greeting|வரவேற்பு|வணக்கம்)'
        r'(?:\*\*|__)?\s*[:\-\]]?\s*([^\n\r]*)',
        re.IGNORECASE
    )

    for m in hdr_regex.finditer(clean_src):
        same_line = m.group(1).strip()
        # Case A: Same line contains quotes
        q_match = re.search(r'[' + quote_chars + r']\s*(.+?)\s*[' + quote_chars + r']', same_line, re.DOTALL)
        if q_match:
            candidate = clean_spoken_text(q_match.group(1))
            if is_valid_spoken_greeting(candidate, doc_title):
                return candidate

        # Case B: Same line has spoken text after colon/dash
        if same_line:
            candidate = clean_spoken_text(same_line)
            if is_valid_spoken_greeting(candidate, doc_title):
                return candidate

        # Case C: Check lines immediately following the heading
        post_text = clean_src[m.end():]
        lines = post_text.splitlines()
        collected = []
        for line in lines[:10]:
            l_str = line.strip()
            if not l_str:
                if collected:
                    break
                continue
            # Stop if hitting another section header
            if re.match(r'^[\s*\-•#]*(?:[A-Z0-9\.\s]{3,}:|#|\d+\.)', l_str) and not any(l_str.lower().startswith(p) for p in ["receptionist", "ai", "agent", "say"]):
                break
            # Check for quote in this line
            q_line = re.search(r'[' + quote_chars + r']\s*(.+?)\s*[' + quote_chars + r']', l_str, re.DOTALL)
            if q_line:
                candidate = clean_spoken_text(q_line.group(1))
                if is_valid_spoken_greeting(candidate, doc_title):
                    return candidate
            collected.append(l_str)

        if collected:
            full_candidate = clean_spoken_text(" ".join(collected))
            if is_valid_spoken_greeting(full_candidate, doc_title):
                return full_candidate

    # 2. Check dialogue lines at start of script: Receptionist: "..." or Receptionist: ...
    diag_regex = re.compile(
        r'(?m)^[\s*\-•]*(?:AI\s+Receptionist|Receptionist|AI\s+Assistant|Assistant|Agent|AI)\s*:\s*([^\n\r]+)',
        re.IGNORECASE
    )
    for dm in diag_regex.finditer(clean_src[:3000]):
        cand_line = dm.group(1).strip()
        q_diag = re.search(r'[' + quote_chars + r']\s*(.+?)\s*[' + quote_chars + r']', cand_line)
        candidate = clean_spoken_text(q_diag.group(1) if q_diag else cand_line)
        if is_valid_spoken_greeting(candidate, doc_title):
            return candidate

    # 3. Check standalone direct quote at the very start of the script
    direct_q = re.search(r'^\s*[' + quote_chars + r']\s*(.+?)\s*[' + quote_chars + r']', clean_src, re.DOTALL)
    if direct_q:
        candidate = clean_spoken_text(direct_q.group(1))
        if is_valid_spoken_greeting(candidate, doc_title):
            return candidate

    # 4. Check if the first paragraph starts with standard greeting triggers
    paragraphs = [p.strip() for p in clean_src.split("\n\n") if p.strip()]
    triggers = ["good morning", "good afternoon", "good evening", "welcome to", "thank you for calling", "thanks for calling", "hello", "hi", "vanakkam", "namaste", "வணக்கம்"]
    for para in paragraphs[:3]:
        para_clean = clean_spoken_text(para)
        if any(para_clean.lower().startswith(t) for t in triggers):
            if is_valid_spoken_greeting(para_clean, doc_title):
                return para_clean

    return ""

# Test cases
test_cases = [
    (
        "Markdown header with quotes",
        """# SARANS HOSPITALS INBOUND SCRIPT
# 1. Greeting & Call Opening
"Good morning! Welcome to Sarans Hospitals. How may I help you today?"

# 2. Departments
- Cardiology
- Dental""",
        "SARANS_HOSPITALS_RAG_Inbound_Script.pdf",
        "Good morning! Welcome to Sarans Hospitals. How may I help you today?"
    ),
    (
        "Colon heading with quotes",
        """Voice Greeting: "Good afternoon, this is Priya from Apollo Clinic. How can I assist you?"
Doctor Directory:
- Dr. Kumar""",
        "apollo_script.txt",
        "Good afternoon, this is Priya from Apollo Clinic. How can I assist you?"
    ),
    (
        "Dialogue script with speaker label",
        """# Chennai Health Care Script
Receptionist: "Vanakkam! Welcome to Chennai Health Care. How can I help you today?"
Customer: I need an appointment.""",
        "chennai_health_care.docx",
        "Vanakkam! Welcome to Chennai Health Care. How can I help you today?"
    ),
    (
        "Tamil pure script",
        """# 1. GREETING
"வணக்கம்! அப்பல்லோ மருத்துவமனைக்கு உங்களை வரவேற்கிறோம். இன்று நான் உங்களுக்கு எவ்வாறு உதவ முடியும்?"

# 2. சிகிச்சைகள்""",
        "tamil_script.txt",
        "வணக்கம்! அப்பல்லோ மருத்துவமனைக்கு உங்களை வரவேற்கிறோம். இன்று நான் உங்களுக்கு எவ்வாறு உதவ முடியும்?"
    ),
    (
        "Thanglish script",
        """Call Opening: "Vanakkam! Sarans Dental Clinic-ku ungalai anbudan varaverkirom. Enna treatment venum?"
Services: Root Canal""",
        "sarans_thanglish.txt",
        "Vanakkam! Sarans Dental Clinic-ku ungalai anbudan varaverkirom. Enna treatment venum?"
    ),
    (
        "Instructions with 'The receptionist should say'",
        """# Greeting
The receptionist says: "Welcome to Apex Law Partners. How may I direct your call?"
# Practice Areas""",
        "apex_law.txt",
        "Welcome to Apex Law Partners. How may I direct your call?"
    ),
    (
        "Multi-line block without quotes",
        """[GREETING]
Good morning! Welcome to Ventura Care.
How may I help you today?

[SERVICES]
- ICU""",
        "ventura_script.txt",
        "Good morning! Welcome to Ventura Care. How may I help you today?"
    ),
    (
        "Script with NO greeting section (only doctors/info)",
        """# HOSPITAL INFORMATION
Hospital Name: ABC General Hospital
# DEPARTMENTS
- Cardiology
# DOCTOR DIRECTORY
Dr Priya""",
        "ABC_Hospital_Info.txt",
        ""
    ),
    (
        "Empty script",
        "",
        "empty.txt",
        ""
    )
]

print("Running greeting extractor tests...")
all_passed = True
for name, content, title, expected in test_cases:
    result = extract_greeting(content, title)
    if result == expected:
        print(f"  [PASS] {name} -> '{result}'")
    else:
        print(f"  [FAIL] {name}\n    Expected: '{expected}'\n    Got:      '{result}'")
        all_passed = False

if all_passed:
    print("\nALL 9 TEST CASES PASSED!")
else:
    print("\nSOME TESTS FAILED!")
