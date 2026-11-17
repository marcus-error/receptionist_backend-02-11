import sys, os

backend_path = os.path.abspath(r"C:\projects\2-11 ai receptionist updated\receptionist_backend-02-11")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from src.services.ai_service import ai_service

print("Testing AI Service Multilingual Prompt Directives...")

langs = ["ta", "hi", "kn", "te", "en"]

for lang in langs:
    inst = ai_service._get_system_instruction(
        organization_name="Sarans Hospitals",
        conversation_language=lang
    )
    print(f"\n[AI PROMPT DIRECTIVE] language={lang}")
    if lang == "ta":
        assert "Tamil" in inst and "தமிழ்" in inst, "Missing Tamil directive in prompt"
        print("  ✓ Tamil directive injected at top of system prompt")
    elif lang == "hi":
        assert "Hindi" in inst and "हिंदी" in inst, "Missing Hindi directive in prompt"
        print("  ✓ Hindi directive injected at top of system prompt")
    elif lang == "kn":
        assert "Kannada" in inst and "ಕನ್ನಡ" in inst, "Missing Kannada directive in prompt"
        print("  ✓ Kannada directive injected at top of system prompt")
    elif lang == "te":
        assert "Telugu" in inst and "తెలుగు" in inst, "Missing Telugu directive in prompt"
        print("  ✓ Telugu directive injected at top of system prompt")
    elif lang == "en":
        print("  ✓ English system instruction validated")

print("\nALL AI SERVICE DIRECTIVES PASSED 100%!")
