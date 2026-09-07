"""
Base agent class for all specialized agents in the testing framework
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
from state import AgentState
from config import config
import logging
import json
import sys
import re

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
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
        self.logger.info(f"[START] {message}")

    def log_success(self, message: str):
        self.logger.info(f"[SUCCESS] {message}")

    def log_error(self, message: str):
        self.logger.error(f"[ERROR] {message}")

    def log_info(self, message: str):
        self.logger.info(f"[INFO] {message}")

    def _parse_json_response(self, response_text: str) -> Any:
        """Robustly extract and parse JSON object or array from LLM response"""
        if not response_text:
            return None
        
        # 1. Direct JSON parse
        try:
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            pass

        # 2. Extract code block ```json ... ```
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

        return None

    async def call_llm(
        self,
        prompt: str,
        system_instruction: str = "",
        model_name: str = "gemini-2.0-flash",
        mock_fallback: Optional[Callable[[], str]] = None
    ) -> str:
        """Call LLM API if key is valid; otherwise fallback to mock implementation"""
        if self.config.has_valid_google_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.config.GOOGLE_API_KEY)
                model = genai.GenerativeModel(model_name)
                full_prompt = f"{system_instruction}\n\n{prompt}" if system_instruction else prompt
                response = model.generate_content(full_prompt)
                if response and response.text:
                    return response.text
            except Exception as e:
                self.log_error(f"Google Gemini call failed: {e}. Falling back to default handler.")

        if mock_fallback:
            self.log_info("Using fallback response generator")
            return mock_fallback()

        raise RuntimeError(f"No valid API keys available for agent '{self.name}' and no mock fallback provided.")
