import re

path = 'receptionist_backend-02-11/src/services/booking_state_service.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. missing_fields doctor
old_doc = '''                    if s_rec:
                        self.doctor_id = s_rec.id
                        self.doctor = s_rec.name
                        has_valid_doctor = True
                    else:
                        self.doctor = None
                        self.doctor_id = None'''
new_doc = '''                    if s_rec:
                        self.doctor_id = s_rec.id
                        self.doctor = s_rec.name
                        has_valid_doctor = True
                    else:
                        has_valid_doctor = False'''
assert old_doc in content, 'old_doc not found'
content = content.replace(old_doc, new_doc, 1)

# 2. missing_fields department
old_dept = '''                    if d_exists:
                        has_valid_dept = True
                    else:
                        self.department = None'''
new_dept = '''                    if d_exists:
                        has_valid_dept = True
                    else:
                        has_valid_dept = False'''
assert old_dept in content, 'old_dept not found'
content = content.replace(old_dept, new_dept, 1)

# 3. clean_person_name
old_clean = '''    def clean_person_name(name: Optional[str]) -> Optional[str]:
        """
        Strips conversational prefixes, polite fillers, and non-name tokens from an extracted candidate name.
        E.g. 'Sure John' -> 'John', 'Yes My Name Is Karthik' -> 'Karthik', 'It Is Vignesh' -> 'Vignesh'.
        """
        if not name or not name.strip():
            return None
        clean = re.sub(r'[^a-zA-Z\\s]', '', name).strip()
        _fillers = {
            "sure", "yes", "yeah", "ok", "okay", "hi", "hello", "hey", "well",
            "thanks", "thank", "you", "sir", "madam", "please", "myself", "this",
            "it", "its", "is", "am", "im", "my", "name", "names", "call", "me",
            "dr", "doctor", "mr", "mrs", "ms", "miss", "prof",
            "before", "after", "during", "while", "say", "says", "saying", "said",
            "tell", "tells", "telling", "told", "speak", "talk", "listen", "hear"
        }
        words = [w for w in clean.split() if w.lower() not in _fillers and len(w) > 1]
        if words and len(words) <= 4:
            cand = " ".join(words).title()
            if BookingStateService.is_valid_person_name(cand):
                return cand
        return clean.title() if BookingStateService.is_valid_person_name(clean) else None'''

new_clean = '''    def clean_person_name(name: Optional[str]) -> Optional[str]:
        """
        Strips conversational prefixes, polite fillers, and non-name tokens from an extracted candidate name.
        E.g. 'Sure John' -> 'John', 'Yes My Name Is Karthik' -> 'Karthik', 'It Is Vignesh' -> 'Vignesh'.
        """
        if not name or not name.strip():
            return None
        clean = re.sub(r'[^a-zA-Z\\u0900-\\u097F\\u0B80-\\u0BFF\\u0C00-\\u0C7F\\u0C80-\\u0CFF\\s]', '', name).strip()
        _fillers = {
            "sure", "yes", "yeah", "ok", "okay", "hi", "hello", "hey", "well",
            "thanks", "thank", "you", "sir", "madam", "please", "myself", "this",
            "it", "its", "is", "am", "im", "my", "name", "names", "call", "me",
            "dr", "doctor", "mr", "mrs", "ms", "miss", "prof",
            "before", "after", "during", "while", "say", "says", "saying", "said",
            "tell", "tells", "telling", "told", "speak", "talk", "listen", "hear"
        }
        words = [w for w in clean.split() if w.lower() not in _fillers and len(w) > 1]
        if words and len(words) <= 4:
            cand = " ".join(words)
            if cand.isascii():
                cand = cand.title()
            if BookingStateService.is_valid_person_name(cand):
                return cand
        cand = clean.title() if clean.isascii() else clean
        return cand if BookingStateService.is_valid_person_name(clean) else None'''
assert old_clean in content, 'old_clean not found'
content = content.replace(old_clean, new_clean, 1)

# 4. is_valid_person_name
old_valid = '''    def is_valid_person_name(name: Optional[str]) -> bool:
        """
        Validates that an extracted string contains a genuine human name and not a standalone placeholder,
        dictionary word, question word, or conversational filler.
        """
        if not name or not name.strip():
            return False
        clean = re.sub(r'[^a-zA-Z\\s]', '', name).strip()'''

new_valid = '''    def is_valid_person_name(name: Optional[str]) -> bool:
        """
        Validates that an extracted string contains a genuine human name and not a standalone placeholder,
        dictionary word, question word, or conversational filler.
        """
        if not name or not name.strip():
            return False
        clean = re.sub(r'[^a-zA-Z\\u0900-\\u097F\\u0B80-\\u0BFF\\u0C00-\\u0C7F\\u0C80-\\u0CFF\\s]', '', name).strip()'''
assert old_valid in content, 'old_valid not found'
content = content.replace(old_valid, new_valid, 1)

# 5. line 3320
old_3320 = "cleaned = re.sub(r'[^\\w\\s]', '', cleaned).strip()"
new_3320 = "cleaned = re.sub(r'[^\\w\\s\\u0900-\\u097F\\u0B80-\\u0BFF\\u0C00-\\u0C7F\\u0C80-\\u0CFF]', '', cleaned).strip()"
assert old_3320 in content, 'old_3320 not found'
content = content.replace(old_3320, new_3320, 1)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print('SUCCESSFULLY APPLIED ALL UPDATES TO booking_state_service.py')

