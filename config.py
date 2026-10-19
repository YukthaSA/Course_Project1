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

    GEMINI_FLASH_MODEL = os.getenv("GEMINI_FLASH_MODEL", "gemini-3.5-flash")
    GEMINI_PRO_MODEL = os.getenv("GEMINI_PRO_MODEL", "gemini-3.5-flash")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")

    # Local models on CPU are slow: generous read timeout, enough context for a requirement chunk
    OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))
    OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))

    # Per-stage model routing: stage name -> (provider, model).
    # Override any stage from .env with LLM_ROUTE_<STAGE>=provider:model, e.g.
    #   LLM_ROUTE_TEST_DESIGN=groq:openai/gpt-oss-120b
    # Defaults use only models that were verified to answer with the configured keys.
    STAGE_ROUTE_DEFAULTS = {
        "Requirement_Extraction": ("gemini", "gemini-3.1-flash-lite"),  # long SRS text -> big context, cheap and fast
        "Requirement_Validation": ("groq", "openai/gpt-oss-20b"),       # many short audits -> fastest inference
        "Risk_Coverage_Planning": ("groq", "openai/gpt-oss-120b"),      # judgement-heavy prioritisation -> largest open model
        "Test_Design": ("gemini", "gemini-3.5-flash"),                  # precise structured JSON, largest output
        "Review_Security": ("groq", "qwen/qwen3.8-27b"),                # different model family than the designer = independent review
        "Failure_Analysis": ("ollama", "llama3.2:3b"),                  # works on the client app's failure logs: stays on this machine
    }
    STAGE_ORDER = list(STAGE_ROUTE_DEFAULTS.keys())
    STAGE_ROUTES: Dict[str, Any] = {}

    # Models offered in the dashboard per provider (Ollama is listed live from the daemon)
    PROVIDER_MODEL_CHOICES = {
        "gemini": ["gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-3-flash-preview", "gemini-flash-latest"],
        "groq": ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"],
        "openai": ["gpt-4o", "gpt-4o-mini"],
        "anthropic": ["claude-3-5-sonnet-20241022"],
    }

    # Empty = built-in demo app (used by the CLI and CI). The dashboard always asks explicitly.
    TARGET_URL = os.getenv("TARGET_URL", "")
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
        self.STAGE_ROUTES = dict(self.STAGE_ROUTE_DEFAULTS)
        for stage in self.STAGE_ORDER:
            raw = os.getenv(f"LLM_ROUTE_{stage.upper()}", "").strip()
            if ":" in raw:
                provider, model = raw.split(":", 1)
                self.STAGE_ROUTES[stage] = (provider.strip().lower(), model.strip())

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
        return bool(self.list_ollama_models())

    def list_ollama_models(self) -> List[str]:
        """Models currently installed in the local Ollama daemon ([] if it is not running). Cached for 15s."""
        import time
        cached = getattr(self, "_ollama_cache", None)
        if cached and time.time() - cached[0] < 15:
            return cached[1]
        names: List[str] = []
        try:
            import urllib.request
            req = urllib.request.Request(f"{self.OLLAMA_BASE_URL.rstrip('/')}/api/tags", headers={"User-Agent": "AgenticQA"})
            with urllib.request.urlopen(req, timeout=1.5) as res:
                names = [m["name"] for m in json.loads(res.read().decode("utf-8")).get("models", [])]
        except Exception:
            names = []
        self._ollama_cache = (time.time(), names)
        return names

    def provider_available(self, provider: str) -> bool:
        return {
            "gemini": self.has_valid_google_key,
            "google": self.has_valid_google_key,
            "groq": self.has_valid_groq_key,
            "openai": self.has_valid_openai_key,
            "anthropic": self.has_valid_anthropic_key,
            "ollama": self.is_ollama_available,
        }.get(provider, False)

    def default_model(self, provider: str) -> str:
        return {
            "gemini": self.GEMINI_FLASH_MODEL, "google": self.GEMINI_FLASH_MODEL, "groq": self.GROQ_MODEL,
            "openai": self.OPENAI_MODEL, "anthropic": self.ANTHROPIC_MODEL, "ollama": self.OLLAMA_MODEL,
        }.get(provider, "")

    def get_stage_route(self, stage: str):
        """(provider, model) configured for a pipeline stage; stages without a route use the default model of the first available provider."""
        if stage in self.STAGE_ROUTES:
            return self.STAGE_ROUTES[stage]
        for provider in ("gemini", "groq", "openai", "anthropic", "ollama"):
            if self.provider_available(provider):
                return provider, self.default_model(provider)
        return "none", ""

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
