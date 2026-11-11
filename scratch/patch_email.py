path = 'receptionist_backend-02-11/src/services/booking_state_service.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

target = r'''    # Insert @ before known email domains if omitted in spoken utterance
    cleaned = re.sub(r'(?<!@)\b(gmail|yahoo|hotmail|outlook)\.(com|org|in|net)\b', r'@\1.\2', cleaned, flags=re.IGNORECASE)

    # 4. Spelled-out local part & domain reconstruction'''

replacement = r'''    # Insert @ before known email domains if omitted in spoken utterance
    cleaned = re.sub(r'(?<!@)(?<!@\s)\b(gmail|yahoo|hotmail|outlook)\.(com|org|in|net)\b', r'@\1.\2', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*@[\s@]*', '@', cleaned)

    # 4. Spelled-out local part & domain reconstruction'''

assert target in content, 'target not found'
content = content.replace(target, replacement, 1)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print('SUCCESSFULLY PATCHED EMAIL PARSER')
