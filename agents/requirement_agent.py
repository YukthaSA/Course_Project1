"""
Requirement Extraction Agent - Converts raw SRS text into structured requirements
"""
import json
from state import AgentState
from .base_agent import BaseAgent
from typing import List, Dict, Any

class RequirementExtractionAgent(BaseAgent):
    def __init__(self, model_client=None):
        super().__init__("Requirement_Extraction", model_client)

    async def invoke(self, state: AgentState) -> AgentState:
        self.log_start("Extracting individual requirements from SRS input...")
        srs_text = state.get("srs_input", "")
        if not srs_text:
            self.log_error("No SRS input provided in state")
            return state

        prompt = f"""Extract individual testable software requirements from the following System Requirement Specification (SRS).
For each requirement, provide:
- id: Requirement ID (e.g., "REQ-001", "REQ-002")
- description: Detailed statement of expected functional behavior
- priority: "High", "Medium", or "Low"
- risk_level: "Critical", "High", "Medium", or "Low"
- testable: boolean (true or false)
- category: "Functional", "Security", "UI", or "Performance"

Return ONLY a valid JSON array of objects.

SRS Document:
{srs_text}

JSON Array:"""

        def _mock_extraction() -> str:
            # Smart fallback extractor if LLM API key is not configured
            lines = [l.strip() for l in srs_text.split('\n') if l.strip()]
            extracted = []
            req_idx = 1
            for line in lines:
                if any(kw in line.lower() for kw in ["shall", "must", "should", "user can", "system", "req", "feature", "allow"]):
                    extracted.append({
                        "id": f"REQ-{req_idx:03d}",
                        "description": line,
                        "priority": "High" if req_idx <= 2 else "Medium",
                        "risk_level": "Critical" if "login" in line.lower() or "auth" in line.lower() else "Medium",
                        "testable": True,
                        "category": "Security" if "login" in line.lower() or "password" in line.lower() else "Functional"
                    })
                    req_idx += 1
            if not extracted:
                extracted = [{
                    "id": "REQ-001",
                    "description": "User authentication and login functional verification",
                    "priority": "High",
                    "risk_level": "Critical",
                    "testable": True,
                    "category": "Functional"
                }]
            return json.dumps(extracted, indent=2)

        try:
            response_text = await self.call_llm(
                prompt=prompt,
                system_instruction="You are an expert QA Requirements Engineer specializing in SRS analysis.",
                model_name=self.config.GEMINI_FLASH_MODEL,
                mock_fallback=_mock_extraction
            )

            requirements = self._parse_json_response(response_text)
            if not isinstance(requirements, list) or len(requirements) == 0:
                self.log_error("Failed to parse valid requirements JSON array from LLM response. Using fallback.")
                requirements = json.loads(_mock_extraction())

            state["requirements"] = requirements
            if "metrics" in state and isinstance(state["metrics"], dict):
                state["metrics"]["total_requirements"] = len(requirements)

            self.log_success(f"Successfully extracted {len(requirements)} requirements")
            for req in requirements:
                print(f" • [{req.get('id', 'REQ')}] ({req.get('priority')}/{req.get('risk_level')}): {req.get('description', '')[:60]}...")

            return state
        except Exception as e:
            self.log_error(f"Error during requirement extraction: {e}")
            return state
