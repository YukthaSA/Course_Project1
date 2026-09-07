"""
Requirement Validation Agent - Validates requirements for clarity, completeness, testability, consistency
"""
import json
from state import AgentState
from .base_agent import BaseAgent
from typing import List, Dict, Any

class RequirementValidationAgent(BaseAgent):
    def __init__(self, model_client=None):
        super().__init__("Requirement_Validation", model_client)

    async def invoke(self, state: AgentState) -> AgentState:
        self.log_start("Validating extracted requirements for testability and completeness...")
        requirements = state.get("requirements", [])
        if not requirements:
            self.log_error("No requirements found in state to validate")
            return state

        requirements_json = json.dumps(requirements, indent=2)
        prompt = f"""Analyze and validate the following list of software requirements.
Check each requirement for clarity, ambiguity, testability, completeness, and consistency.

Requirements:
{requirements_json}

For each requirement, output a JSON array of objects with the following fields:
- id: exact requirement ID
- is_valid: boolean (true if clear and testable, false if ambiguous or missing details)
- issues: array of strings describing any ambiguity or missing information (empty if valid)
- confidence: float score between 0.0 and 1.0 representing confidence in validation

Return ONLY valid JSON array:"""

        def _mock_validation() -> str:
            val_results = []
            for req in requirements:
                desc = req.get("description", "").lower()
                is_ambiguous = len(desc) < 15 or "etc" in desc or "stuff" in desc
                val_results.append({
                    "id": req.get("id"),
                    "is_valid": not is_ambiguous,
                    "issues": ["Requirement text is overly concise or vague"] if is_ambiguous else [],
                    "confidence": 0.95 if not is_ambiguous else 0.70
                })
            return json.dumps(val_results, indent=2)

        try:
            response_text = await self.call_llm(
                prompt=prompt,
                system_instruction="You are a Senior QA Auditor reviewing requirement specifications.",
                model_name=self.config.GEMINI_PRO_MODEL,
                mock_fallback=_mock_validation
            )

            val_list = self._parse_json_response(response_text)
            if not isinstance(val_list, list):
                val_list = json.loads(_mock_validation())

            validated_requirements = []
            for req in requirements:
                v_entry = next((item for item in val_list if item.get("id") == req.get("id")), None)
                if v_entry:
                    validated_requirements.append({
                        **req,
                        "is_valid": bool(v_entry.get("is_valid", True)),
                        "validation_issues": v_entry.get("issues", []),
                        "confidence": float(v_entry.get("confidence", 0.9))
                    })
                else:
                    validated_requirements.append({
                        **req,
                        "is_valid": True,
                        "validation_issues": [],
                        "confidence": 0.85
                    })

            state["validated_requirements"] = validated_requirements
            valid_count = sum(1 for r in validated_requirements if r.get("is_valid"))
            total_count = len(validated_requirements)
            
            self.log_success(f"Validation complete: {valid_count}/{total_count} requirements validated successfully")
            for vreq in validated_requirements:
                status_icon = "✅" if vreq.get("is_valid") else "⚠️"
                print(f" {status_icon} [{vreq.get('id')}] Valid: {vreq.get('is_valid')} | Issues: {len(vreq.get('validation_issues', []))}")

            return state
        except Exception as e:
            self.log_error(f"Error during requirement validation: {e}")
            return state
