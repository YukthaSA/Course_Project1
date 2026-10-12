"""
Requirement Extraction Agent - Converts raw SRS text and PURE specifications into structured requirements
"""
import json
import re
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

        # Prompt for LLM with domain awareness
        doc_meta = state.get("document_metadata", {})
        domain_hint = f"Domain: {doc_meta.get('domain', 'General Software')}" if doc_meta else ""

        prompt = f"""Extract testable software requirements from the following System Requirement Specification (SRS).
{domain_hint}

For each requirement, provide:
- id: Requirement ID (e.g., "REQ-001", "REQ-002")
- description: Clear functional or behavioral statement
- priority: "High", "Medium", or "Low"
- risk_level: "Critical", "High", "Medium", or "Low"
- testable: boolean (true or false)
- category: "Functional", "Security", "UI", "Performance", or "Safety"

Return ONLY a valid JSON array of objects.

SRS Document (Excerpt):
{srs_text[:4000]}

JSON Array:"""

        def _deterministic_extraction() -> str:
            extracted = []
            req_idx = 1
            
            # Check for PURE format: [REQ-xxx] [Category] Priority: High \n Requirement: ...
            pure_pattern = re.findall(
                r"\[(REQ-[^\]]+)\]\s*\[([^\]]+)\]\s*Priority:\s*([^\n]+)\s*\n\s*Requirement:\s*(.+?)(?=\n\s*\[REQ-|\Z)",
                srs_text,
                re.DOTALL | re.IGNORECASE
            )
            
            if pure_pattern:
                for match in pure_pattern[:25]:
                    rid, cat, prio, req_text = match
                    cat = cat.strip()
                    p = prio.strip()
                    r_text = re.sub(r"\s+", " ", req_text).strip()
                    
                    risk = "Critical" if cat in ["Security", "Safety"] or p == "High" else "Medium"
                    extracted.append({
                        "id": rid if rid.startswith("REQ-") else f"REQ-{req_idx:03d}",
                        "description": r_text,
                        "priority": p if p in ["High", "Medium", "Low"] else "High",
                        "risk_level": risk,
                        "testable": True,
                        "category": cat if cat in ["Functional", "Security", "UI", "Performance", "Safety"] else "Functional"
                    })
                    req_idx += 1

            # If not formatted as PURE blocks, scan lines
            if not extracted:
                lines = [l.strip() for l in srs_text.split("\n") if l.strip()]
                for line in lines:
                    line_lower = line.lower()
                    if len(line) < 25 or line.startswith("=") or line.startswith("-") or line.startswith("1. ") or line.startswith("2. "):
                        continue
                    if any(kw in line_lower for kw in ["shall", "must", "should", "will", "user can", "allows", "system verifies", "system authenticates"]):
                        # Extract existing ID if present
                        id_match = re.search(r"(REQ-[0-9A-Za-z_-]+|\b[0-9]+\.[0-9]+)\b", line)
                        rid = id_match.group(1) if id_match else f"REQ-{req_idx:03d}"
                        
                        clean_desc = re.sub(r"^(REQ-[0-9A-Za-z_-]+:?|[0-9]+\.[0-9]+:?)\s*", "", line).strip()
                        
                        cat = "Functional"
                        if any(w in line_lower for w in ["security", "auth", "password", "encrypt", "token", "permission"]):
                            cat = "Security"
                        elif any(w in line_lower for w in ["ui", "display", "screen", "button", "page", "modal"]):
                            cat = "UI"
                        elif any(w in line_lower for w in ["speed", "latency", "performance", "throughput", "timeout", "seconds"]):
                            cat = "Performance"
                        elif any(w in line_lower for w in ["safety", "hazard", "interlock", "fail-safe", "brake"]):
                            cat = "Safety"

                        prio = "High" if cat in ["Security", "Safety"] or "shall" in line_lower else "Medium"
                        risk = "Critical" if cat == "Security" or cat == "Safety" else "Medium"

                        extracted.append({
                            "id": rid,
                            "description": clean_desc,
                            "priority": prio,
                            "risk_level": risk,
                            "testable": True,
                            "category": cat
                        })
                        req_idx += 1
                        if len(extracted) >= 20:
                            break

            if not extracted:
                extracted = [{
                    "id": "REQ-001",
                    "description": "User authentication, credential verification, and session lifecycle management",
                    "priority": "High",
                    "risk_level": "Critical",
                    "testable": True,
                    "category": "Security"
                }]

            return json.dumps(extracted, indent=2)

        try:
            response_text = await self.call_llm(
                prompt=prompt,
                system_instruction="You are an expert QA Requirements Engineer specializing in IEEE/ISO SRS analysis.",
                model_name=self.config.GEMINI_FLASH_MODEL,
                mock_fallback=_deterministic_extraction
            )

            requirements = self._parse_json_response(response_text)
            if not isinstance(requirements, list) or len(requirements) == 0:
                self.log_info("Falling back to deterministic SRS requirement extractor.")
                requirements = json.loads(_deterministic_extraction())

            state["requirements"] = requirements
            if "metrics" in state and isinstance(state["metrics"], dict):
                state["metrics"]["total_requirements"] = len(requirements)

            self.log_success(f"Successfully extracted {len(requirements)} requirements")
            return state
        except Exception as e:
            self.log_error(f"Error during requirement extraction: {e}")
            state["requirements"] = json.loads(_deterministic_extraction())
            return state
