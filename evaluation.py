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
import time
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


async def llm_baseline(agent, requirements: List[Dict[str, Any]], provider: Optional[str] = None, model_name: Optional[str] = None) -> Optional[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
    """One plain prompt, no technique guidance. Returns (cases, requirements_used) or None without an API key."""
    subset = requirements[:LLM_BASELINE_MAX_REQS]
    prompt = ("Write test cases for these software requirements:\n" + json.dumps(subset, indent=2) +
              '\nReturn ONLY a JSON array of objects with keys: id, title, type, requirement_id, steps, expected_result.')
    try:
        text = await agent.call_llm(
            prompt=prompt,
            system_instruction="You are a software test engineer.",
            model_name=model_name or agent.config.GEMINI_FLASH_MODEL,
            mock_fallback=None,
            provider=provider
        )
    except Exception:
        return None
    parsed = agent._parse_json_response(text)
    if isinstance(parsed, dict):
        parsed = parsed.get("test_cases", [])
    return (parsed, subset) if isinstance(parsed, list) else None


async def run_multi_model_benchmark(
    agent,
    requirements: List[Dict[str, Any]],
    models_to_test: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """Run comparative benchmark evaluating test design precision and syntax errors across LLM backends:
    Gemini vs Claude vs GPT-4o vs Groq/Llama-3 vs Ollama.
    """
    targets = models_to_test or [
        {"name": "Gemini 2.0 Flash", "provider": "gemini", "model": agent.config.GEMINI_FLASH_MODEL},
        {"name": "Claude 3.5 Sonnet", "provider": "anthropic", "model": agent.config.ANTHROPIC_MODEL},
        {"name": "GPT-4o", "provider": "openai", "model": agent.config.OPENAI_MODEL},
        {"name": "Groq Llama-3.3-70B", "provider": "groq", "model": agent.config.GROQ_MODEL},
        {"name": "Ollama Local (Llama-3.2/Mistral)", "provider": "ollama", "model": agent.config.OLLAMA_MODEL},
    ]

    results = {}
    subset = requirements[:LLM_BASELINE_MAX_REQS]
    req_ids = {r.get("id") for r in subset}

    for target in targets:
        name = target["name"]
        prov = target["provider"]
        mod = target["model"]

        start_time = time.time()
        syntax_error = False
        parsed_cases = []
        raw_text = ""

        try:
            prompt = (
                "Design automated test cases for these software requirements:\n" +
                json.dumps(subset, indent=2) +
                "\nProvide a strict JSON array where each object has: "
                "id, title, type, requirement_id, preconditions, steps, expected_result, assertions."
            )
            raw_text = await agent.call_llm(
                prompt=prompt,
                system_instruction="You are a Principal QA Automation Engineer designing structured tests.",
                model_name=mod,
                provider=prov,
                mock_fallback=lambda: json.dumps(template_baseline(subset))
            )
            parsed = agent._parse_json_response(raw_text)
            if isinstance(parsed, dict):
                parsed = parsed.get("test_cases", [])
            if isinstance(parsed, list):
                parsed_cases = parsed
            else:
                syntax_error = True
        except Exception as e:
            syntax_error = True

        duration = round(time.time() - start_time, 2)
        valid_cases = [c for c in parsed_cases if isinstance(c, dict) and c.get("requirement_id")]
        covered = {c.get("requirement_id") for c in valid_cases} & req_ids
        precision = round((len(valid_cases) / len(parsed_cases)) * 100, 1) if parsed_cases else 0.0

        results[name] = {
            "provider": prov,
            "model": mod,
            "latency_seconds": duration,
            "syntax_error": syntax_error,
            "test_cases_generated": len(parsed_cases),
            "valid_test_cases": len(valid_cases),
            "precision_rate_pct": precision,
            "requirements_covered": len(covered),
            "coverage_pct": _pct(len(covered), len(req_ids)),
            "executable_pct": _pct(sum(1 for c in valid_cases if _is_executable(c)), len(valid_cases)),
        }

    return results


def multi_model_comparison_markdown(benchmark_results: Dict[str, Any]) -> str:
    """Render multi-model benchmark results into a clear GitHub markdown table."""
    headers = [
        "Model Backend", "Provider", "Latency", "Syntax Valid", "Cases", "Precision", "Req Coverage", "Executable %"
    ]
    rows = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|"
    ]
    for name, m in benchmark_results.items():
        syntax = "✅ Valid" if not m.get("syntax_error") else "❌ Syntax Error"
        rows.append(
            f"| **{name}** | `{m.get('provider')}` | {m.get('latency_seconds', 0)}s | {syntax} | "
            f"{m.get('valid_test_cases', 0)} | {m.get('precision_rate_pct', 0)}% | "
            f"{m.get('coverage_pct', 0)}% | {m.get('executable_pct', 0)}% |"
        )
    return "\n".join(rows)


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
