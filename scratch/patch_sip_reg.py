with open('src/services/sip_registrar.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = '    if hasattr(pyVoIP, "REGISTER_FAILURE_THRESHOLD"):\n        pyVoIP.REGISTER_FAILURE_THRESHOLD = 999999\n'
target_crlf = '    if hasattr(pyVoIP, "REGISTER_FAILURE_THRESHOLD"):\r\n        pyVoIP.REGISTER_FAILURE_THRESHOLD = 999999\r\n'

replacement = '    if hasattr(pyVoIP, "REGISTER_FAILURE_THRESHOLD"):\n        pyVoIP.REGISTER_FAILURE_THRESHOLD = 999999\n\n    # Set transmit delay reduction to 0.25 for low-latency RTP pacing\n    if hasattr(pyVoIP, "TRANSMIT_DELAY_REDUCTION"):\n        pyVoIP.TRANSMIT_DELAY_REDUCTION = 0.25\n'
replacement_crlf = '    if hasattr(pyVoIP, "REGISTER_FAILURE_THRESHOLD"):\r\n        pyVoIP.REGISTER_FAILURE_THRESHOLD = 999999\r\n\r\n    # Set transmit delay reduction to 0.25 for low-latency RTP pacing\r\n    if hasattr(pyVoIP, "TRANSMIT_DELAY_REDUCTION"):\r\n        pyVoIP.TRANSMIT_DELAY_REDUCTION = 0.25\r\n'

if target_crlf in text:
    text = text.replace(target_crlf, replacement_crlf, 1)
    with open('src/services/sip_registrar.py', 'w', encoding='utf-8', newline='') as f:
        f.write(text)
    print("SUCCESS CRLF")
elif target in text:
    text = text.replace(target, replacement, 1)
    with open('src/services/sip_registrar.py', 'w', encoding='utf-8', newline='') as f:
        f.write(text)
    print("SUCCESS LF")
else:
    print("TARGET NOT FOUND")

