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
                        comparison_markdown)
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

    report_md = f"""# Evaluation Report
**Dataset:** PURE - {selected_doc.get('title')} ({selected_doc.get('domain')})  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Target:** {target_url} (built-in demo app)  
**Pipeline wall-clock time:** {duration:.1f} s

All figures below are measured from this run. No manual-effort or time-saving estimates are made.

## 1. Test design quality vs baselines
{comparison_markdown(framework, baselines)}

""" + "\n".join(f"- {n}" for n in notes) + f"""

Cases by technique: {dict(by_tech)}  
Cases by tier: {dict(by_tier)}  
Cases by type: {framework['by_type']}

## 2. Execution results (Playwright, per test case)
| Executed | Passed | Failed | Skipped |
|---|---|---|---|
| {execm['executed']} | {execm['passed']} | {execm['failed']} | {execm['skipped']} |

Final run status: `{exec_res.get('status', 'UNKNOWN')}`

## 3. Failure analysis
- Application defects: {execm['defects']} (by severity: {execm['defects_by_severity'] or 'none'})
- Script issues: {execm['script_issues']}
- Environment issues: {execm['environment_issues']}
- Root cause: {analysis.get('root_cause', 'n/a')}
- Automatic retries: {len(retries)}
- Escalated to human review: `{state.get('human_review_required', False)}` ({analysis.get('escalation_reason', 'n/a')})

## 4. Limitations
- Tests run against a built-in demo app, not the requirement document's real system.
- Baselines are stand-ins, not an independent expert-written suite; defect-detection comparison would need seeded faults.
- Keyword-based requirement detection may miss requirement types outside auth, email, password, numeric, text, upload, session.

## 5. Sample approved test cases
"""
    for tc in approved_tcs[:6]:
        report_md += f"- **[{tc.get('id')}]** ({tc.get('type')}, {tc.get('technique', '-')}): {tc.get('title')}\n"

    report_path = os.path.join(config.DATA_DIR, "Evaluation_Report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print()
    print("==================================================================")
    print(f"✅ CAPSTONE EVALUATION REPORT GENERATED AT:")
    print(f"   {report_path}")
    print("==================================================================")

if __name__ == "__main__":
    asyncio.run(generate_academic_report())
