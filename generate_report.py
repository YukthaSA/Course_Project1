"""
Academic Evaluation & Capstone Report Generator
Runs comparative benchmarks on the PURE dataset and outputs comprehensive evaluation reports
"""
import asyncio
import os
import json
import time
import sys
import argparse
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from config import config
from pipeline import AgenticQAPipeline
from engine.demo_app import start_demo_server
from evaluation import (design_metrics, template_baseline, llm_baseline, execution_metrics,
                        comparison_markdown, run_multi_model_benchmark, multi_model_comparison_markdown)
from collections import Counter

def parse_args():
    parser = argparse.ArgumentParser(description="Academic Report Generator on PURE Benchmark")
    parser.add_argument("--doc", type=str, default="2008_keepass", help="PURE document ID to evaluate (e.g., 2008_keepass, 0000_cctns)")
    parser.add_argument("--port", type=int, default=8080, help="Local demo server port")
    return parser.parse_args()

async def generate_academic_report():
    args = parse_args()

    print("==================================================================")
    print("📊 GENERATING CAPSTONE EVALUATION REPORT & BENCHMARKS (PURE DATASET)")
    print("==================================================================")

    # 1. Start local demo server
    server, actual_port = start_demo_server(args.port)
    target_url = f"http://127.0.0.1:{actual_port}"
    time.sleep(0.5)

    # 2. Load PURE Document
    catalog = config.get_pure_catalog()
    selected_doc = None
    query = args.doc.lower()
    for doc in catalog:
        if query in doc["id"].lower() or query in doc["filename"].lower() or query in doc["title"].lower():
            selected_doc = doc
            break

    if not selected_doc:
        default_path = config.get_default_srs_path()
        selected_doc = {"title": "Default SRS", "domain": "Security & Systems", "path": default_path}

    doc_path = selected_doc.get("path") or os.path.join(config.PURE_SAMPLES_DIR, selected_doc.get("filename", ""))
    with open(doc_path, "r", encoding="utf-8") as f:
        srs_text = f.read()

    print(f"📄 Evaluating on PURE Document: {selected_doc.get('title')} ({selected_doc.get('domain')})")

    # 3. Run Pipeline against live local web server
    pipeline = AgenticQAPipeline()
    start_t = time.time()
    state = await pipeline.run(srs_text=srs_text, target_url=target_url, max_retries=2, metadata=selected_doc)
    duration = time.time() - start_t

    # 4. Extract measured metrics (nothing estimated)
    reqs = state.get("validated_requirements") or state.get("requirements", [])
    approved_tcs = state.get("approved_test_cases", [])
    retries = state.get("retry_history", [])
    exec_res = state.get("execution_result", {})
    framework = design_metrics(reqs, approved_tcs)
    execm = execution_metrics(state)

    baselines = {"Template baseline": design_metrics(reqs, template_baseline(reqs))}
    notes = ["Template baseline = one generic positive test (plus one generic negative for Security/Functional "
             "requirements) per requirement, with no test data or assertions."]
    llm = await llm_baseline(pipeline.test_designer, reqs)
    if llm:
        cases, subset = llm
        baselines[f"Single-prompt LLM (first {len(subset)} reqs)"] = design_metrics(subset, cases)
        notes.append(f"Single-prompt LLM baseline only covers the first {len(subset)} requirements, so its "
                     "coverage is measured against that subset.")
    else:
        notes.append("Single-prompt LLM baseline skipped (no valid API key or unparseable response).")

    by_tech = Counter(tc.get("technique") or "unspecified" for tc in approved_tcs)
    by_tier = Counter(f"Tier {tc.get('tier', '?')}" for tc in approved_tcs)
    analysis = state.get("error_analysis") or {}

    # 5. Multi-Model Benchmark Execution (Gemini vs Claude vs GPT-4o vs Groq/Llama-3 vs Ollama)
    print("🤖 Running Multi-Model Benchmark (Gemini vs Claude vs GPT-4o vs Groq/Llama-3 vs Ollama)...")
    mm_results = await run_multi_model_benchmark(pipeline.test_designer, reqs)
    mm_markdown = multi_model_comparison_markdown(mm_results)

    report_md = f"""# Comprehensive Evaluation Report & Multi-Model Benchmark
**Dataset:** PURE - {selected_doc.get('title')} ({selected_doc.get('domain')})  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Target:** {target_url} (DOM-Grounded Adaptive Sandbox)  
**Pipeline wall-clock time:** {duration:.1f} s

All figures below are measured from this run. No manual-effort or time-saving estimates are made.

## 1. Test Design Quality vs Baselines
{comparison_markdown(framework, baselines)}

""" + "\n".join(f"- {n}" for n in notes) + f"""

Cases by technique: {dict(by_tech)}  
Cases by tier: {dict(by_tier)}  
Cases by type: {framework['by_type']}

## 2. Multi-Model LLM Benchmark Comparison
Cross-model comparison across precision rate, JSON syntax validity, latency, and requirement coverage:

{mm_markdown}

## 3. Execution Results (Playwright, per test case)
| Executed | Passed | Failed | Skipped |
|---|---|---|---|
| {execm['executed']} | {execm['passed']} | {execm['failed']} | {execm['skipped']} |

Final run status: `{exec_res.get('status', 'UNKNOWN')}`

## 4. Failure Analysis & Self-Healing Telemetry
- Application defects: {execm['defects']} (by severity: {execm['defects_by_severity'] or 'none'})
- Script issues: {execm['script_issues']}
- Environment issues: {execm['environment_issues']}
- Root cause: {analysis.get('root_cause', 'n/a')}
- Automatic retries: {len(retries)}
- Escalated to human review: `{state.get('human_review_required', False)}` ({analysis.get('escalation_reason', 'n/a')})

## 5. Sample Approved Test Cases
"""
    for tc in approved_tcs[:6]:
        report_md += f"- **[{tc.get('id')}]** ({tc.get('type')}, {tc.get('technique', '-')}): {tc.get('title')}\n"

    report_path = os.path.join(config.DATA_DIR, "Evaluation_Report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    # Also output HTML version for CI/CD artifact publication
    html_report_path = os.path.join(config.DATA_DIR, "Evaluation_Report.html")
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Agentic QA Framework - Academic Evaluation Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; max-width: 1000px; margin: 40px auto; padding: 0 20px; color: #1e293b; background: #f8fafc; }}
        h1, h2, h3 {{ color: #0f172a; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; background: #ffffff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid #e2e8f0; }}
        th {{ background: #f1f5f9; font-weight: 600; color: #334155; }}
        tr:hover {{ background: #f8fafc; }}
        code {{ background: #e2e8f0; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; }}
        .badge {{ display: inline-block; padding: 4px 8px; border-radius: 4px; font-weight: 600; font-size: 0.85em; background: #e0e7ff; color: #3730a3; }}
    </style>
</head>
<body>
    <h1>🚀 Agentic QA Framework - Academic Evaluation Report</h1>
    <p><strong>PURE Benchmark Document:</strong> {selected_doc.get('title')} ({selected_doc.get('domain')})</p>
    <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <hr>
    <h2>Multi-Model LLM Benchmark</h2>
    <p>Empirical evaluation of test case generation precision across LLM backends:</p>
    <pre>{mm_markdown}</pre>
    <hr>
    <h2>Execution Summary</h2>
    <p><strong>Status:</strong> <code>{exec_res.get('status', 'UNKNOWN')}</code> | <strong>Executed:</strong> {execm['executed']} | <strong>Passed:</strong> {execm['passed']} | <strong>Failed:</strong> {execm['failed']}</p>
</body>
</html>"""
    with open(html_report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print()
    print("==================================================================")
    print(f"✅ CAPSTONE EVALUATION REPORT GENERATED AT:")
    print(f"   Markdown: {report_path}")
    print(f"   HTML:     {html_report_path}")
    print("==================================================================")

if __name__ == "__main__":
    asyncio.run(generate_academic_report())
