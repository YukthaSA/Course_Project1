"""
Measured evaluation metrics for the requirement-to-test framework.

Everything here is computed from a run's actual output (test cases, results, failure analysis).
Nothing is estimated. Design quality is compared against baselines using the SAME metric
functions, so the comparison cannot favour the framework by definition:

  * Template baseline  - one generic positive (+ one negative for Security/Functional) test per
                         requirement, no test data, no assertions (what a fixed template produces).
  * Single-prompt LLM  - one plain "write test cases" prompt (needs an API key; optional).

Definitions
  coverage            requirements with >= 1 test case / all requirements
  with concrete data  non-positive cases whose test_data is a non-empty dict
  executable          cases with >= 1 machine-checkable assertion (can be judged pass/fail automatically)
"""
import json
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from agents.test_design_agent import ASSERTION_TYPES

LLM_BASELINE_MAX_REQS = 20


def _pct(part: int, whole: int) -> float:
    return round(100.0 * part / whole, 1) if whole else 0.0


def _is_executable(tc: Dict[str, Any]) -> bool:
    return any(isinstance(a, dict) and a.get("type") in ASSERTION_TYPES for a in (tc.get("assertions") or []))


def design_metrics(requirements: List[Dict[str, Any]], test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Measure the quality of a set of test cases against the requirements they should cover."""
    req_ids = {r.get("id") for r in requirements}
    covered = {tc.get("requirement_id") for tc in test_cases} & req_ids
    non_positive = [tc for tc in test_cases if tc.get("type") != "Positive"]
    with_data = [tc for tc in non_positive if isinstance(tc.get("test_data"), dict) and tc["test_data"]]
    return {
        "requirements": len(req_ids),
        "covered": len(covered),
        "coverage_pct": _pct(len(covered), len(req_ids)),
        "total_cases": len(test_cases),
        "cases_per_req": round(len(test_cases) / len(req_ids), 1) if req_ids else 0.0,
        "by_type": dict(Counter(tc.get("type") for tc in test_cases)),
        "boundary_security": sum(1 for tc in test_cases if tc.get("type") in ("Boundary", "Security")),
        "techniques": len({tc.get("technique") for tc in test_cases if tc.get("technique")}),
        "with_data_pct": _pct(len(with_data), len(non_positive)),
        "executable_pct": _pct(sum(1 for tc in test_cases if _is_executable(tc)), len(test_cases)),
    }


def template_baseline(requirements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fixed-template tests: one generic positive, plus a generic negative for Security/Functional."""
    cases = []
    for r in requirements:
        rid, desc = r.get("id"), r.get("description", "")
        base = {"requirement_id": rid, "expected_result": "System responds correctly", "steps": ["Open the application"]}
        cases.append({**base, "title": f"Verify {desc[:50]}", "type": "Positive"})
        if r.get("category") in ("Security", "Functional") or any(w in desc.lower() for w in ("login", "password")):
            cases.append({**base, "title": f"Reject invalid input for {rid}", "type": "Negative"})
    return cases


async def llm_baseline(agent, requirements: List[Dict[str, Any]]) -> Optional[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
    """One plain prompt, no technique guidance. Returns (cases, requirements_used) or None without an API key."""
    subset = requirements[:LLM_BASELINE_MAX_REQS]
    prompt = ("Write test cases for these software requirements:\n" + json.dumps(subset, indent=2) +
              '\nReturn ONLY a JSON array of objects with keys: id, title, type, requirement_id, steps, expected_result.')
    try:
        text = await agent.call_llm(prompt=prompt, system_instruction="You are a software test engineer.",
                                    model_name=agent.config.GEMINI_FLASH_MODEL, mock_fallback=None)
    except Exception:
        return None
    parsed = agent._parse_json_response(text)
    if isinstance(parsed, dict):
        parsed = parsed.get("test_cases", [])
    return (parsed, subset) if isinstance(parsed, list) else None


def execution_metrics(state: Dict[str, Any]) -> Dict[str, Any]:
    """Pass/fail/skip counts and defects by severity, taken from the run's results and failure analysis."""
    results = state.get("test_results") or []
    analysis = state.get("error_analysis") or {}
    failures = analysis.get("failures", [])
    defects = [f for f in failures if f.get("category") == "APPLICATION_DEFECT"]
    return {
        "executed": len(results),
        "passed": sum(1 for r in results if r.get("status") == "PASSED"),
        "failed": sum(1 for r in results if r.get("status") == "FAILED"),
        "skipped": sum(1 for r in results if r.get("status") == "SKIPPED"),
        "defects": len(defects),
        "defects_by_severity": dict(Counter(d.get("severity") for d in defects)),
        "script_issues": sum(1 for f in failures if f.get("category") == "SCRIPT_ISSUE"),
        "environment_issues": sum(1 for f in failures if f.get("category") == "ENVIRONMENT_ISSUE"),
    }


def comparison_markdown(framework: Dict[str, Any], baselines: Dict[str, Dict[str, Any]]) -> str:
    """Side-by-side table of design metrics: framework vs each baseline."""
    names = ["Proposed framework"] + list(baselines)
    cols = [framework] + list(baselines.values())
    rows = [
        ("Requirements covered", lambda m: f"{m['covered']}/{m['requirements']} ({m['coverage_pct']}%)"),
        ("Test cases (per requirement)", lambda m: f"{m['total_cases']} ({m['cases_per_req']})"),
        ("Boundary + Security cases", lambda m: str(m["boundary_security"])),
        ("Distinct design techniques used", lambda m: str(m["techniques"])),
        ("Non-positive cases with concrete test data", lambda m: f"{m['with_data_pct']}%"),
        ("Cases that can be auto-judged pass/fail", lambda m: f"{m['executable_pct']}%"),
    ]
    out = ["| Metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for label, fmt in rows:
        out.append(f"| {label} | " + " | ".join(fmt(m) for m in cols) + " |")
    return "\n".join(out)
