"""
Base agent class for all specialized agents in the testing framework
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable, Tuple
from state import AgentState
from config import config
from .llm_providers import PROVIDERS
import asyncio
import logging
import json
import time
import sys
import re

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("AgentFramework")

# Order in which other providers are tried when a stage's own provider is missing or fails.
# Ollama is last: it is the slowest and only worth waiting for when nothing else works.
FALLBACK_ORDER = ("gemini", "groq", "openai", "anthropic", "ollama")


class BaseAgent(ABC):
    def __init__(self, name: str, model_client: Optional[Any] = None):
        self.name = name
        self.model_client = model_client
        self.logger = logging.getLogger(f"Agent.{name}")
        self.config = config
        self.call_log: List[Dict[str, Any]] = []  # one record per call_llm: which model really answered

    @abstractmethod
    async def invoke(self, state: AgentState) -> AgentState:
        """Process input state and return updated AgentState"""
        pass

    def log_start(self, message: str):
        self.logger.info(f"[{self.name}] 🚀 {message}")

    def log_success(self, message: str):
        self.logger.info(f"[{self.name}] ✅ {message}")

    def log_error(self, message: str):
        self.logger.error(f"[{self.name}] ❌ {message}")

    def log_info(self, message: str):
        self.logger.info(f"[{self.name}] ℹ️ {message}")

    def _parse_json_response(self, response_text: str) -> Any:
        """Robustly extract and parse JSON object or array from LLM response"""
        if not response_text:
            return None

        # Strip reasoning model thinking tags (<think>...</think>)
        response_text = re.sub(r"<think>[\s\S]*?</think>", "", response_text).strip()

        # 1. Direct JSON parse
        try:
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            pass

        # 2. Extract code block ```json ... ``` or ``` ... ```
        code_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text, re.IGNORECASE)
        if code_block:
            try:
                return json.loads(code_block.group(1).strip())
            except json.JSONDecodeError:
                pass

        # 3. Extract JSON array [...]
        array_match = re.search(r"\[\s*\{[\s\S]*\}\s*\]", response_text, re.DOTALL)
        if array_match:
            try:
                return json.loads(array_match.group(0))
            except json.JSONDecodeError:
                pass

        # 4. Extract JSON object {...}
        object_match = re.search(r"\{[\s\S]*\}", response_text, re.DOTALL)
        if object_match:
            try:
                return json.loads(object_match.group(0))
            except json.JSONDecodeError:
                pass

        # 5. Fix common trailing commas
        cleaned = re.sub(r",\s*([\]}])", r"\1", response_text)
        try:
            return json.loads(cleaned.strip())
        except Exception:
            pass

        return None

    def _plan_attempts(self, provider: Optional[str], model_name: Optional[str]) -> List[Tuple[str, str]]:
        """Ordered (provider, model) attempts for one call.

        - explicit `provider` argument      -> exactly that provider, no cross-provider fallback
        - global engine forced in the UI    -> exactly that provider (so a benchmark is never silently
                                               answered by a different model)
        - "auto" (default)                  -> this stage's own route, then the other available providers
        """
        if provider:
            return [(provider.lower(), model_name or self.config.default_model(provider.lower()))]
        forced = (self.config.LLM_PROVIDER or "auto").lower()
        if forced != "auto":
            return [(forced, self.config.default_model(forced))]
        primary = self.config.get_stage_route(self.name)
        attempts = [primary]
        for other in FALLBACK_ORDER:
            if other != primary[0]:
                attempts.append((other, self.config.default_model(other)))
        return attempts

    async def call_llm(
        self,
        prompt: str,
        system_instruction: str = "",
        model_name: Optional[str] = None,
        mock_fallback: Optional[Callable[[], str]] = None,
        provider: Optional[str] = None
    ) -> str:
        """Call the LLM assigned to this pipeline stage (see Config.STAGE_ROUTES).

        Tries the stage's provider first, then the other available providers, and finally the
        deterministic `mock_fallback`, so a missing key or a dead local daemon never halts a run.
        Every call is recorded in `self.call_log` (provider, model, seconds, errors).
        """
        attempts = self._plan_attempts(provider, model_name)
        planned = f"{attempts[0][0]}/{attempts[0][1]}"
        errors: List[str] = []

        for index, (prov, model) in enumerate(attempts):
            adapter = PROVIDERS.get(prov)
            if adapter is None or not self.config.provider_available(prov):
                if index == 0:
                    errors.append(f"{prov}: not configured")
                continue
            started = time.time()
            try:
                text = await asyncio.to_thread(adapter, self.config, model, system_instruction, prompt)
            except Exception as exc:
                errors.append(f"{prov}/{model}: {exc}")
                self.log_error(f"{prov}/{model} call failed: {exc}")
                continue
            if not (text and text.strip()):
                errors.append(f"{prov}/{model}: empty response")
                self.log_error(f"{prov}/{model} returned an empty response")
                continue
            self.call_log.append({
                "stage": self.name, "planned": planned, "provider": prov, "model": model,
                "seconds": round(time.time() - started, 2), "fallback": f"{prov}/{model}" != planned,
                "errors": errors,
            })
            self.log_info(f"answered by {prov}/{model} in {time.time() - started:.1f}s")
            return text

        # Deterministic generator: always available, never hits the network
        if mock_fallback:
            self.log_info("Executing deterministic agent generator")
            self.call_log.append({
                "stage": self.name, "planned": planned, "provider": "deterministic", "model": "rule-based",
                "seconds": 0.0, "fallback": True, "errors": errors,
            })
            return mock_fallback()

        raise RuntimeError(f"No valid LLM provider succeeded for agent '{self.name}': {'; '.join(errors) or 'none configured'}")
