"""
Requirement Validation Agent - Validates requirements for clarity, completeness, testability, consistency
Enhanced: produces per-requirement issues, coverage gap analysis, conflict/duplicate detection, and rewrite suggestions.
"""
import json
import re
from state import AgentState
from .base_agent import BaseAgent
from typing import List, Dict, Any, Set, Tuple

AMBIGUITY_TERMS = [
    "user-friendly", "fast", "rapidly", "seamlessly", "easy to use", "robust",
    "flexible", "efficient", "adequate", "sufficient", "as appropriate",
    "state-of-the-art", "etc", "and so on", "high quality", "optimized", "normally",
    "intuitive", "reasonable", "appropriate", "timely", "quickly", "smoothly",
    "properly", "good", "nice", "better", "effective", "minimal", "significant",
    "various", "most", "some", "generally", "support", "handle", "process",
    "advanced", "next-generation", "smart", "modern", "real-time", "seamless",
    "roughly", "approximately", "frequently", "periodically", "as needed", "if possible",
    "scalable", "customizable", "reliable"
]

# Expected categories for a well-rounded SRS
EXPECTED_CATEGORIES: Set[str] = {
    "Security", "Performance", "Usability", "Functional", "Reliability",
    "Compatibility", "Data", "Error Handling",
}

NORMATIVE_VERBS = [
    "shall", "must", "should", "will", "can", "allow", "verify",
    "prevent", "display", "ensure", "enable", "restrict", "require",
    "support", "provide", "maintain", "generate", "validate", "authenticate",
]


class RequirementValidationAgent(BaseAgent):
    def __init__(self, model_client=None):
        super().__init__("Requirement_Validation", model_client)

    # ------------------------------------------------------------------
    # Deterministic helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_ambiguity(desc: str) -> List[str]:
        """Return list of ambiguous qualifiers found in a requirement description."""
        desc_lower = desc.lower()
        found = []
        for term in AMBIGUITY_TERMS:
            if re.search(rf"\b{re.escape(term)}\b", desc_lower):
                # For "real-time", check if threshold is missing
                if term == "real-time" and re.search(r"\b\d+\s*(ms|s|sec|second|minute|min)\b", desc_lower):
                    continue
                found.append(term)
        return found

    @staticmethod
    def _check_normative_verb(desc: str) -> bool:
        desc_lower = desc.lower()
        return any(kw in desc_lower for kw in NORMATIVE_VERBS)

    @classmethod
    def _analyze_single_requirement(cls, req: Dict[str, Any]) -> Tuple[List[str], int, bool]:
        """Run comprehensive heuristic checks against an individual requirement."""
        desc = req.get("description", "")
        desc_lower = desc.lower()
        cat = req.get("category", "Functional").lower()
        issues: List[str] = []

        # 1. Ambiguity qualifiers & vague verbs
        found_vague = cls._detect_ambiguity(desc)
        if found_vague:
            issues.append(f"Contains ambiguous qualifier(s): {', '.join(found_vague)}")

        # 2. Too short / concise
        if len(desc.strip()) < 25:
            issues.append("Requirement is too concise — lacks measurable acceptance criteria")

        # 3. Missing normative verb
        if not cls._check_normative_verb(desc):
            issues.append("Missing definitive normative verb (shall/must/will)")

        # 4. Performance / latency claim without measurable quantitative threshold
        is_perf_context = (
            cat in ["performance", "efficiency"] or
            any(w in desc_lower for w in ["real-time", "real time", "fast", "speed", "latency", "throughput", "response time"])
        )
        if is_perf_context and not re.search(r"\b\d+(\.\d+)?\s*(ms|s|sec|second|kb|mb|gb|req|tx|ops|%)", desc_lower):
            issues.append("Performance/timing claim lacks quantitative threshold (e.g., '< 2 seconds' or '< 200ms')")

        # 5. Passive voice construction
        if re.search(r"\b(is|are|was|were|be|been|being)\s+([a-z]+ed|to be)\b", desc_lower):
            issues.append("Uses passive voice — rewrite in active voice specifying the actor")

        # 6. Overly long / compound requirement
        if len(desc) > 180 or len(re.split(r",|\band\b|\bor\b", desc_lower)) > 4:
            issues.append("Overly long or compound requirement — consider splitting into atomic requirements")

        # 7. Missing actor / subject
        if not re.search(r"\b(system|user|admin|operator|application|server|client|service|database|browser|engine|keepass|software)\b", desc_lower):
            issues.append("Missing clear actor/subject — specify who or what initiates or performs this action")

        testability = max(30, 100 - (len(issues) * 15))
        is_valid = len(issues) == 0
        return issues, testability, is_valid

    @staticmethod
    def _detect_coverage_gaps(requirements: List[Dict]) -> List[str]:
        """Return missing requirement categories compared to EXPECTED_CATEGORIES."""
        present: Set[str] = set()
        for r in requirements:
            cat = r.get("category", "Functional")
            present.add(cat)
        missing = EXPECTED_CATEGORIES - present
        return [f"No requirements cover '{cat}' — consider adding {cat.lower()} specifications" for cat in sorted(missing)]

    @staticmethod
    def _detect_conflicts(requirements: List[Dict]) -> List[Dict[str, str]]:
        """Flag pairs where one requirement permits/allows and another denies/restricts overlapping nouns."""
        conflicts: List[Dict[str, str]] = []
        allow_reqs = []
        deny_reqs = []
        for r in requirements:
            desc_lower = r.get("description", "").lower()
            rid = r.get("id", "")
            if any(w in desc_lower for w in ["allow", "enable", "permit", "can", "accept"]):
                allow_reqs.append((rid, desc_lower))
            if any(w in desc_lower for w in ["prevent", "restrict", "deny", "block", "disable", "cannot", "never"]):
                deny_reqs.append((rid, desc_lower))

        for a_id, a_desc in allow_reqs:
            for d_id, d_desc in deny_reqs:
                if a_id == d_id:
                    continue
                a_words = set(w for w in re.findall(r"[a-z]{4,}", a_desc))
                d_words = set(w for w in re.findall(r"[a-z]{4,}", d_desc))
                shared = a_words & d_words - {"shall", "must", "will", "should", "system", "allow", "enable", "prevent", "restrict", "user", "users"}
                if len(shared) >= 2:
                    conflicts.append({
                        "req_a": a_id,
                        "req_b": d_id,
                        "reason": f"Potential contradiction — one allows and the other restricts around: {', '.join(list(shared)[:3])}"
                    })
        return conflicts[:5]

    @staticmethod
    def _detect_duplicates(requirements: List[Dict]) -> Dict[str, List[str]]:
        """Detect near-duplicate requirements by comparing word sets (Jaccard > 0.6)."""
        dup_pairs: Dict[str, List[str]] = {}
        descs = [(r.get("id", ""), set(re.findall(r"[a-z]{3,}", r.get("description", "").lower()))) for r in requirements]

        for i in range(len(descs)):
            for j in range(i + 1, len(descs)):
                id_a, words_a = descs[i]
                id_b, words_b = descs[j]
                if not words_a or not words_b:
                    continue
                intersection = words_a & words_b
                union = words_a | words_b
                jaccard = len(intersection) / len(union)
                if jaccard > 0.6:
                    dup_pairs.setdefault(id_a, []).append(id_b)
        return dup_pairs

    @staticmethod
    def _suggest_rewrites(req: Dict, issues: List[str]) -> str:
        """Return a cleaner, unambiguous rewrite of a requirement."""
        desc = req.get("description", "")
        rewrite = desc
        # Remove vague qualifiers
        for term in AMBIGUITY_TERMS:
            pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
            rewrite = pattern.sub("", rewrite)
        rewrite = re.sub(r"\s{2,}", " ", rewrite).strip()
        # Prepend standard actor + normative verb if missing
        if not any(kw in rewrite.lower() for kw in NORMATIVE_VERBS):
            rewrite = f"The system shall {rewrite[0].lower()}{rewrite[1:]}" if rewrite else rewrite
        cat = req.get("category", "").lower()
        if (cat == "performance" or "performance" in desc.lower()) and not re.search(r"\d", rewrite):
            rewrite += " within a response threshold of < 2.0 seconds"
        return rewrite

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
                issues, testability, is_valid = self._analyze_single_requirement(req)
                val_results.append({
                    "id": req.get("id"),
                    "is_valid": is_valid,
                    "issues": issues,
                    "testability_score": testability,
                    "confidence": 0.95 if is_valid else 0.75
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

            validated_requirements: List[Dict[str, Any]] = []
            all_issues: List[str] = []
            scores: List[int] = []
            suggested_rewrites: List[Dict[str, str]] = []

            for req in requirements:
                # Always run heuristic checks to guarantee deep findings
                h_issues, h_score, _ = self._analyze_single_requirement(req)
                v_entry = next((item for item in val_list if item.get("id") == req.get("id")), None)
                llm_issues = v_entry.get("issues", []) if v_entry else []

                # Combine issues without duplicates
                combined_issues = list(dict.fromkeys(h_issues + [i for i in llm_issues if i not in h_issues]))
                t_score = max(30, 100 - (len(combined_issues) * 15))
                is_valid = len(combined_issues) == 0
                scores.append(t_score)

                if combined_issues:
                    all_issues.extend(combined_issues)
                    rewrite = self._suggest_rewrites(req, combined_issues)
                    if rewrite != req.get("description", ""):
                        suggested_rewrites.append({
                            "id": req.get("id"),
                            "original": req.get("description", ""),
                            "rewrite": rewrite
                        })

                validated_requirements.append({
                    **req,
                    "is_valid": is_valid,
                    "validation_issues": combined_issues,
                    "testability_score": t_score,
                    "confidence": 0.95 if is_valid else 0.75
                })

            valid_count = sum(1 for r in validated_requirements if r.get("is_valid"))
            total_count = len(validated_requirements)
            avg_score = round(sum(scores) / len(scores)) if scores else 85

            coverage_gaps = self._detect_coverage_gaps(requirements)
            conflicts = self._detect_conflicts(requirements)
            duplicate_pairs = self._detect_duplicates(requirements)

            state["validated_requirements"] = validated_requirements
            state["quality_metrics"] = {
                "testability_score": avg_score,
                "ambiguous_terms_found": list(set(all_issues))[:20],
                "clarity_rating": "High" if avg_score >= 80 else ("Moderate" if avg_score >= 60 else "Needs Refinement"),
                "valid_count": valid_count,
                "total_count": total_count,
                "coverage_gaps": coverage_gaps,
                "conflicts": conflicts,
                "duplicate_pairs": duplicate_pairs,
                "suggested_rewrites": suggested_rewrites[:15],
            }
            state["validation_report"] = f"Validated {valid_count}/{total_count} requirements. Average testability index: {avg_score}%."
            self.log_success(f"Validation complete: {valid_count}/{total_count} valid ({avg_score}% testability index)")
            return state

        except Exception as e:
            self.log_error(f"Error during requirement validation: {e}")
            fallback_val = json.loads(_deterministic_validation())
            scores = []
            all_issues = []
            rewrites = []
            val_reqs = []

            for req in requirements:
                h_issues, h_score, is_valid = self._analyze_single_requirement(req)
                scores.append(h_score)
                if h_issues:
                    all_issues.extend(h_issues)
                    rewrites.append({
                        "id": req.get("id"),
                        "original": req.get("description", ""),
                        "rewrite": self._suggest_rewrites(req, h_issues)
                    })
                val_reqs.append({
                    **req,
                    "is_valid": is_valid,
                    "validation_issues": h_issues,
                    "testability_score": h_score,
                    "confidence": 0.95 if is_valid else 0.75
                })

            valid_count = sum(1 for r in val_reqs if r.get("is_valid"))
            avg_score = round(sum(scores) / len(scores)) if scores else 80
            state["validated_requirements"] = val_reqs
            state["quality_metrics"] = {
                "testability_score": avg_score,
                "ambiguous_terms_found": list(set(all_issues))[:20],
                "clarity_rating": "High" if avg_score >= 80 else ("Moderate" if avg_score >= 60 else "Needs Refinement"),
                "valid_count": valid_count,
                "total_count": len(val_reqs),
                "coverage_gaps": self._detect_coverage_gaps(requirements),
                "conflicts": self._detect_conflicts(requirements),
                "duplicate_pairs": self._detect_duplicates(requirements),
                "suggested_rewrites": rewrites[:15],
            }
            state["validation_report"] = f"Validated {valid_count}/{len(val_reqs)} requirements. Average testability index: {avg_score}%."
            return state
