"""
Base agent class for all specialized agents in the testing framework
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
from state import AgentState
from config import config
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

class BaseAgent(ABC):
    def __init__(self, name: str, model_client: Optional[Any] = None):
        self.name = name
        self.model_client = model_client
        self.logger = logging.getLogger(f"Agent.{name}")
        self.config = config

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

    async def call_llm(
        self,
        prompt: str,
        system_instruction: str = "",
        model_name: Optional[str] = None,
        mock_fallback: Optional[Callable[[], str]] = None,
        provider: Optional[str] = None
    ) -> str:
        """Call LLM API across configured providers: Gemini, Groq, Ollama, OpenAI, Anthropic.
        Falls back seamlessly to smart deterministic agent implementation.
        """
        active_provider = (provider or self.config.LLM_PROVIDER or "auto").lower()
        full_prompt = f"{system_instruction}\n\n{prompt}" if system_instruction else prompt

        # 1. Groq Provider
        if active_provider == "groq" or (active_provider == "auto" and self.config.has_valid_groq_key):
            try:
                import urllib.request
                groq_model = model_name or self.config.GROQ_MODEL
                headers = {
                    "Authorization": f"Bearer {self.config.GROQ_API_KEY}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": groq_model,
                    "messages": [
                        *([{"role": "system", "content": system_instruction}] if system_instruction else []),
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.2
                }
                req = urllib.request.Request(
                    "https://api.groq.com/openai/v1/chat/completions",
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    text = data["choices"][0]["message"]["content"]
                    if text:
                        return text
            except Exception as e:
                self.log_error(f"Groq API call failed: {e}")

        # 2. Ollama Local Provider (via OpenAI-compatible endpoint)
        if active_provider == "ollama" or (active_provider == "auto" and not self.config.has_valid_google_key and self.config.is_ollama_available):
            try:
                import urllib.request
                ollama_model = model_name or self.config.OLLAMA_MODEL
                url = f"{self.config.OLLAMA_BASE_URL.rstrip('/')}/v1/chat/completions"
                payload = {
                    "model": ollama_model,
                    "messages": [
                        *([{"role": "system", "content": system_instruction}] if system_instruction else []),
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.2
                }
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=8) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    text = data["choices"][0]["message"]["content"]
                    if text:
                        return text
            except Exception as e:
                self.log_error(f"Ollama local call failed: {e}")

        # 3. Google Gemini Provider
        if active_provider in ("gemini", "google", "auto") and self.config.has_valid_google_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.config.GOOGLE_API_KEY)
                gem_model = model_name or self.config.GEMINI_FLASH_MODEL
                model = genai.GenerativeModel(gem_model)
                response = model.generate_content(full_prompt)
                if response and response.text:
                    return response.text
            except Exception as e:
                self.log_error(f"Google Gemini call failed: {e}")

        # 4. OpenAI Provider
        if (active_provider == "openai" or (active_provider == "auto" and self.config.has_valid_openai_key)) and self.config.has_valid_openai_key:
            try:
                import urllib.request
                headers = {
                    "Authorization": f"Bearer {self.config.OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": model_name or self.config.OPENAI_MODEL,
                    "messages": [
                        *([{"role": "system", "content": system_instruction}] if system_instruction else []),
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.2
                }
                req = urllib.request.Request(
                    "https://api.openai.com/v1/chat/completions",
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    text = data["choices"][0]["message"]["content"]
                    if text:
                        return text
            except Exception as e:
                self.log_error(f"OpenAI call failed: {e}")

        # 5. Anthropic Claude Provider
        if (active_provider == "anthropic" or (active_provider == "auto" and self.config.has_valid_anthropic_key)) and self.config.has_valid_anthropic_key:
            try:
                import urllib.request
                headers = {
                    "x-api-key": self.config.ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": model_name or self.config.ANTHROPIC_MODEL,
                    "max_tokens": 4096,
                    "messages": [{"role": "user", "content": full_prompt}],
                    "temperature": 0.2
                }
                req = urllib.request.Request(
                    "https://api.anthropic.com/v1/messages",
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    text = "".join(b.get("text", "") for b in data.get("content", []))
                    if text:
                        return text
            except Exception as e:
                self.log_error(f"Anthropic call failed: {e}")

        # Fallback to deterministic generator
        if mock_fallback:
            self.log_info("Executing deterministic agent generator")
            return mock_fallback()

        raise RuntimeError(f"No valid LLM provider succeeded for agent '{self.name}' and no fallback provided.")
