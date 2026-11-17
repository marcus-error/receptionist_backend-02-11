import sys, os
sys.path.insert(0, r"C:\projects\2-11 ai receptionist updated\receptionist_backend-02-11")

from src.config.env import settings
print("DATABASE_URL:", settings.DATABASE_URL)
print("File exists:", os.path.exists(r"C:\projects\2-11 ai receptionist updated\receptionist_backend-02-11\receptionist.db"))

