"""
Requirement Validation Agent - Validates requirements for clarity, completeness, testability, consistency
"""
import json
import re
from state import AgentState
from .base_agent import BaseAgent
from typing import List, Dict, Any

AMBIGUITY_TERMS = [
    "user-friendly", "fast", "rapidly", "seamlessly", "easy to use", "robust", 
    "flexible", "efficient", "adequate", "sufficient", "as appropriate", 
    "state-of-the-art", "etc", "and so on", "high quality", "optimized", "normally"
]

class RequirementValidationAgent(BaseAgent):
    def __init__(self, model_client=None):
        super().__init__("Requirement_Validation", model_client)

    async def invoke(self, state: AgentState) -> AgentState:
        self.log_start("Validating extracted requirements for testability, ambiguity, and completeness...")
        requirements = state.get("requirements", [])
        if not requirements:
            self.log_error("No requirements found in state to validate")
            return state

        requirements_json = json.dumps(requirements[:25], indent=2)
        prompt = f"""Analyze and validate the following list of software requirements.
Check each requirement for clarity, ambiguity (vague qualifiers), testability, completeness, and consistency.

Requirements:
{requirements_json}

For each requirement, output a JSON array of objects with:
- id: exact requirement ID
- is_valid: boolean (true if clear and testable, false if ambiguous or untestable)
- issues: array of strings describing ambiguity or missing parameters (empty if valid)
- testability_score: integer from 0 to 100
- confidence: float score between 0.0 and 1.0

Return ONLY a valid JSON array:"""

        def _deterministic_validation() -> str:
            val_results = []
            for req in requirements:
                desc = req.get("description", "").lower()
                issues = []
                
                # Check for ambiguity keywords
                found_vague = [term for term in AMBIGUITY_TERMS if re.search(rf"\b{re.escape(term)}\b", desc)]
                if found_vague:
                    issues.append(f"Contains ambiguous qualifier(s): {', '.join(found_vague)}")

                if len(desc) < 20:
                    issues.append("Requirement is too concise or lacks context")

                if not any(kw in desc for kw in ["shall", "must", "should", "will", "can", "allow", "verify", "prevent", "display", "ensure"]):
                    issues.append("Missing definitive normative verb (shall/must/will)")

                testability = max(40, 100 - (len(issues) * 25))
                is_valid = len(issues) == 0

                val_results.append({
                    "id": req.get("id"),
                    "is_valid": is_valid,
                    "issues": issues,
                    "testability_score": testability,
                    "confidence": 0.95 if is_valid else 0.80
                })
            return json.dumps(val_results, indent=2)

        try:
            response_text = await self.call_llm(
                prompt=prompt,
                system_instruction="You are a Senior QA Auditor reviewing software requirement specifications.",
                model_name=self.config.GEMINI_PRO_MODEL,
                mock_fallback=_deterministic_validation
            )

            val_list = self._parse_json_response(response_text)
            if not isinstance(val_list, list) or len(val_list) == 0:
                val_list = json.loads(_deterministic_validation())

            validated_requirements = []
            ambiguous_terms_found = []
            scores = []

            for req in requirements:
                v_entry = next((item for item in val_list if item.get("id") == req.get("id")), None)
                if v_entry:
                    issues = v_entry.get("issues", [])
                    t_score = int(v_entry.get("testability_score", 85 if v_entry.get("is_valid") else 60))
                    scores.append(t_score)
                    if issues:
                        ambiguous_terms_found.extend(issues)
                    validated_requirements.append({
                        **req,
                        "is_valid": bool(v_entry.get("is_valid", True)),
                        "validation_issues": issues,
                        "testability_score": t_score,
                        "confidence": float(v_entry.get("confidence", 0.9))
                    })
                else:
                    scores.append(85)
                    validated_requirements.append({
                        **req,
                        "is_valid": True,
                        "validation_issues": [],
                        "testability_score": 85,
                        "confidence": 0.85
                    })

            state["validated_requirements"] = validated_requirements
            valid_count = sum(1 for r in validated_requirements if r.get("is_valid"))
            total_count = len(validated_requirements)
            avg_score = round(sum(scores) / len(scores)) if scores else 85

            state["quality_metrics"] = {
                "testability_score": avg_score,
                "ambiguous_terms_found": list(set(ambiguous_terms_found))[:10],
                "clarity_rating": "High" if avg_score >= 80 else ("Moderate" if avg_score >= 60 else "Needs Refinement"),
                "valid_count": valid_count,
                "total_count": total_count
            }

            state["validation_report"] = f"Validated {valid_count}/{total_count} requirements. Average testability index: {avg_score}%."
            self.log_success(f"Validation complete: {valid_count}/{total_count} valid ({avg_score}% testability index)")

            return state
        except Exception as e:
            self.log_error(f"Error during requirement validation: {e}")
            state["validated_requirements"] = json.loads(_deterministic_validation())
            return state
