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

        prompt = f"""Extract ONLY individually testable software requirements from the following SRS document.
{domain_hint}

IMPORTANT RULES:
- SKIP introductory paragraphs, document overviews, tables of contents, and summary sections.
- SKIP any text that merely describes the document structure or lists section headings.
- Each requirement MUST describe a single, concrete, testable behavior or capability.
- If a block of text contains multiple requirements, split them into separate entries.

For each requirement, provide:
- id: Requirement ID (e.g., "REQ-001", "REQ-002")
- description: Clear functional or behavioral statement (one sentence, max 200 words)
- priority: "High", "Medium", or "Low"
- testable: boolean (true or false)
- category: "Functional", "Security", "UI", "Performance", or "Safety"

Do NOT assign risk_level — that is handled by a separate Risk Analysis agent.

Return ONLY a valid JSON array of objects.

SRS Document (Excerpt):
{srs_text[:4000]}

JSON Array:"""

        def _is_overview_text(text: str) -> bool:
            """Filter out introductory paragraphs, TOCs, and summaries that aren't individual requirements."""
            text_lower = text.lower().strip()
            word_count = len(text.split())
            # Too long — likely a preamble paragraph or TOC dump
            if word_count > 150:
                return True
            # Starts with "this document describes" or "the following requirements"
            if any(text_lower.startswith(p) for p in [
                "this document describes", "this document provides",
                "the following requirements describe", "the following high-level",
                "the purpose of this document",
            ]):
                return True
            # Contains multiple section references (e.g., "2.1 ... 2.2 ... 2.3 ...")
            section_refs = re.findall(r"\b\d+\.\d+\b", text)
            if len(section_refs) >= 4:
                return True
            return False

        def _correct_category(text: str, original_cat: str) -> str:
            """Re-classify category based on requirement content keywords."""
            t = text.lower()
            if any(w in t for w in ["security", "auth", "password", "encrypt", "token", "permission",
                                     "unauthorized", "sabotage", "protection", "privacy", "fraudulent"]):
                return "Security"
            if any(w in t for w in ["performance", "speed", "latency", "throughput", "timeout",
                                     "response time", "scalab"]):
                return "Performance"
            if any(w in t for w in ["safety", "hazard", "interlock", "fail-safe", "brake"]):
                return "Safety"
            if any(w in t for w in ["display", "screen", "button", "page", "modal", "user interface",
                                     "layout", "form field", "dropdown"]):
                return "UI"
            # Data capture, system behavior, integration → Functional (not UI)
            if any(w in t for w in ["capture data", "must have the capability", "data requirements",
                                     "exchange", "integration", "configur", "deploy", "synchroniz"]):
                return "Functional"
            if original_cat in ["Functional", "Security", "UI", "Performance", "Safety"]:
                return original_cat
            return "Functional"

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
                for match in pure_pattern[:40]:
                    rid, cat, prio, req_text = match
                    cat = cat.strip()
                    p = prio.strip()
                    r_text = re.sub(r"\s+", " ", req_text).strip()

                    # Skip overview/preamble paragraphs
                    if _is_overview_text(r_text):
                        continue

                    # Fix misclassified categories
                    cat = _correct_category(r_text, cat)

                    extracted.append({
                        "id": rid if rid.startswith("REQ-") else f"REQ-{req_idx:03d}",
                        "description": r_text,
                        "priority": p if p in ["High", "Medium", "Low"] else "Medium",
                        "testable": True,
                        "category": cat
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

                        # Skip overview text in line-scan mode too
                        if _is_overview_text(clean_desc):
                            continue
                        cat = _correct_category(clean_desc, cat)

                        extracted.append({
                            "id": rid,
                            "description": clean_desc,
                            "priority": prio,
                            "testable": True,
                            "category": cat
                        })
                        req_idx += 1
                        if len(extracted) >= 30:
                            break

            if not extracted:
                extracted = [{
                    "id": "REQ-001",
                    "description": "User authentication, credential verification, and session lifecycle management",
                    "priority": "High",
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