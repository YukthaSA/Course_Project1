"""
Configuration management for the Agentic AI Testing Framework
Includes full integration with the PURE (PUblic REquirements) Benchmark Dataset
"""
import os
import sys
import json
from typing import List, Dict, Any, Optional

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Zero-dependency .env loader with fallback to python-dotenv
def _load_env_file():
    try:
        from dotenv import load_dotenv
        load_dotenv()
        return
    except ImportError:
        pass

    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip("'\"")
                        if k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass

_load_env_file()

class Config:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Active default provider: "gemini", "groq", "ollama", "openai", "anthropic"
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").lower()

    GEMINI_FLASH_MODEL = os.getenv("GEMINI_FLASH_MODEL", "gemini-2.5-flash")
    GEMINI_PRO_MODEL = os.getenv("GEMINI_PRO_MODEL", "gemini-2.5-pro")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
    
    TARGET_URL = os.getenv("TARGET_URL", "http://127.0.0.1:8080")
    HEADLESS = os.getenv("HEADLESS", "true").lower() == "true"
    TIMEOUT = int(os.getenv("TIMEOUT", "5000"))
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    LOGS_DIR = os.path.join(BASE_DIR, "logs")
    DATA_DIR = os.path.join(BASE_DIR, "data")
    SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
    
    # PURE Dataset Directories
    PURE_DATA_DIR = os.path.join(DATA_DIR, "pure")
    PURE_XML_DIR = os.path.join(PURE_DATA_DIR, "xml")
    PURE_SAMPLES_DIR = os.path.join(SAMPLES_DIR, "pure")
    PURE_CATALOG_PATH = os.path.join(PURE_SAMPLES_DIR, "catalog.json")

    def __init__(self):
        os.makedirs(self.LOGS_DIR, exist_ok=True)
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.SAMPLES_DIR, exist_ok=True)
        os.makedirs(self.PURE_DATA_DIR, exist_ok=True)
        os.makedirs(self.PURE_SAMPLES_DIR, exist_ok=True)

    @property
    def has_valid_google_key(self) -> bool:
        return bool(self.GOOGLE_API_KEY and not self.GOOGLE_API_KEY.startswith("your_"))

    @property
    def has_valid_groq_key(self) -> bool:
        return bool(self.GROQ_API_KEY and not self.GROQ_API_KEY.startswith("your_"))

    @property
    def has_valid_openai_key(self) -> bool:
        return bool(self.OPENAI_API_KEY and not self.OPENAI_API_KEY.startswith("your_"))

    @property
    def has_valid_anthropic_key(self) -> bool:
        return bool(self.ANTHROPIC_API_KEY and not self.ANTHROPIC_API_KEY.startswith("your_"))

    @property
    def is_ollama_available(self) -> bool:
        try:
            import urllib.request
            req = urllib.request.Request(f"{self.OLLAMA_BASE_URL.rstrip('/')}/api/tags", headers={"User-Agent": "AgenticQA"})
            with urllib.request.urlopen(req, timeout=1.0) as res:
                return res.status == 200
        except Exception:
            return False

    def get_pure_catalog(self) -> List[Dict[str, Any]]:
        """Return cached PURE documents catalog or parse dynamically"""
        if os.path.exists(self.PURE_CATALOG_PATH):
            try:
                with open(self.PURE_CATALOG_PATH, "r", encoding="utf-8") as f:
                    catalog = json.load(f)
                # Catalog may hold another machine's absolute paths: re-point to local samples dir
                for doc in catalog:
                    fname = doc.get("filename") or os.path.basename(str(doc.get("path", "")).replace("\\", "/"))
                    local = os.path.join(self.PURE_SAMPLES_DIR, fname)
                    if fname and os.path.exists(local):
                        doc["path"] = local
                return catalog
            except Exception:
                pass
        try:
            from pure_loader import PUREDatasetLoader
            loader = PUREDatasetLoader(self.BASE_DIR)
            return loader.get_catalog()
        except Exception:
            return []

    def get_default_srs_path(self) -> str:
        """Return the default SRS document path, favoring PURE dataset"""
        for preferred in ["2008_keepass_srs.txt", "0000_cctns_srs.txt", "1998_themas_srs.txt"]:
            preferred_path = os.path.join(self.PURE_SAMPLES_DIR, preferred)
            if os.path.exists(preferred_path):
                return preferred_path
        
        catalog = self.get_pure_catalog()
        if catalog and os.path.exists(catalog[0].get("path", "")):
            return catalog[0]["path"]

        return os.path.join(self.SAMPLES_DIR, "sample_srs.txt")

    def load_pure_document(self, query: str) -> Optional[str]:
        """Load text content of a PURE document by ID or filename"""
        catalog = self.get_pure_catalog()
        q = query.lower().strip()
        for doc in catalog:
            if (q == doc["id"].lower() or 
                q in doc["filename"].lower() or 
                q in doc["title"].lower()):
                doc_path = doc.get("path") or os.path.join(self.PURE_SAMPLES_DIR, doc["filename"])
                if os.path.exists(doc_path):
                    with open(doc_path, "r", encoding="utf-8") as f:
                        return f.read()
        return None

config = Config()

if __name__ == "__main__":
    print("[OK] Configuration loaded successfully")
    print(f" Base Directory:     {config.BASE_DIR}")
    print(f" PURE Samples Dir:   {config.PURE_SAMPLES_DIR}")
    print(f" PURE Documents:     {len(config.get_pure_catalog())}")
    print(f" Default SRS Path:   {config.get_default_srs_path()}")
    print(f" Google API Key:     {config.has_valid_google_key}")
    print(f" Target URL:         {config.TARGET_URL}")
    print(f" Headless Mode:      {config.HEADLESS}")
