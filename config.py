"""
Configuration management for the Agentic AI Testing Framework
"""
import os
import sys
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

class Config:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    
    GEMINI_FLASH_MODEL = "gemini-2.0-flash"
    GEMINI_PRO_MODEL = "gemini-1.5-pro"
    GROQ_MODEL = "llama-3.3-70b-versatile"
    
    TARGET_URL = os.getenv("TARGET_URL", "https://demo.example.com")
    HEADLESS = os.getenv("HEADLESS", "true").lower() == "true"
    TIMEOUT = int(os.getenv("TIMEOUT", "5000"))
    MAX_RETRIES = 3
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    LOGS_DIR = os.path.join(BASE_DIR, "logs")
    DATA_DIR = os.path.join(BASE_DIR, "data")
    SAMPLES_DIR = os.path.join(BASE_DIR, "samples")

    def __init__(self):
        os.makedirs(self.LOGS_DIR, exist_ok=True)
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.SAMPLES_DIR, exist_ok=True)

    @property
    def has_valid_google_key(self) -> bool:
        return bool(self.GOOGLE_API_KEY and not self.GOOGLE_API_KEY.startswith("your_"))

    @property
    def has_valid_groq_key(self) -> bool:
        return bool(self.GROQ_API_KEY and not self.GROQ_API_KEY.startswith("your_"))

config = Config()

if __name__ == "__main__":
    print("[OK] Configuration loaded successfully")
    print(f" Base Directory: {config.BASE_DIR}")
    print(f" Google API Key Present: {config.has_valid_google_key}")
    print(f" Groq API Key Present: {config.has_valid_groq_key}")
    print(f" Target URL: {config.TARGET_URL}")
    print(f" Headless Mode: {config.HEADLESS}")
