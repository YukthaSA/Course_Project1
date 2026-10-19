import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import config


@pytest.fixture(autouse=True)
def no_real_llm_calls(monkeypatch):
    monkeypatch.setattr(config, "provider_available", lambda provider: False)
