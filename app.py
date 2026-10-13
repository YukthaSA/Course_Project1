"""
Streamlit Web Dashboard for Agentic AI Automated Requirement-to-Test Framework
Enhanced with PURE Benchmark Dataset Explorer & Multi-Stage Quality Engineering Analytics
"""
import streamlit as st
import asyncio
import os
import json
import time
from datetime import datetime

from config import config
from state import AgentState
from pipeline import AgenticQAPipeline

st.set_page_config(
    page_title="Agentic QA - PURE Benchmark Framework",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #555;
        margin-bottom: 15px;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid #1E88E5;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        margin-bottom: 10px;
    }
    .badge-pure {
        background-color: #e3f2fd;
        color: #0d47a1;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Fetch PURE catalog
pure_catalog = config.get_pure_catalog()

# Session state initialization
if "state" not in st.session_state:
    # Try loading existing run results from disk if available
    saved_results_path = os.path.join(config.DATA_DIR, "pipeline_results.json")
    if os.path.exists(saved_results_path):
        try:
            with open(saved_results_path, "r", encoding="utf-8") as f:
                st.session_state.state = json.load(f)
        except Exception:
            st.session_state.state = None
    else:
        st.session_state.state = None

if "srs_text" not in st.session_state:
    default_srs = config.get_default_srs_path()
    if os.path.exists(default_srs):
        with open(default_srs, "r", encoding="utf-8") as f:
            st.session_state.srs_text = f.read()
    else:
        st.session_state.srs_text = "The system shall allow users to log in with email and password."

if "selected_doc_id" not in st.session_state:
    st.session_state.selected_doc_id = "2008_keepass" if any(d["id"] == "2008_keepass" for d in pure_catalog) else (pure_catalog[0]["id"] if pure_catalog else "")

# ==============================================================================
# SIDEBAR: PURE DATASET EXPLORER & CONFIGURATION
# ==============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/robot.png", width=56)
    st.title("Framework Settings")

    st.subheader("📚 PURE Dataset Explorer")
    st.caption("Public Requirements Benchmark (IEEE RE '17)")

    # Domain filter
    domains = ["All Domains"] + sorted(list(set(d.get("domain", "General") for d in pure_catalog)))
    selected_domain = st.selectbox("Filter by Domain", domains)

    # Filter catalog documents
    if selected_domain == "All Domains":
        filtered_docs = pure_catalog
    else:
        filtered_docs = [d for d in pure_catalog if d.get("domain") == selected_domain]

    doc_options = {d["id"]: f"{d['title'][:32]}... ({d['requirements_count']} reqs)" for d in filtered_docs}

    current_selection_idx = 0
    doc_keys = list(doc_options.keys())
    if st.session_state.selected_doc_id in doc_keys:
        current_selection_idx = doc_keys.index(st.session_state.selected_doc_id)

    selected_doc_key = st.selectbox(
        "Select SRS Document",
        options=doc_keys,
        format_func=lambda k: doc_options[k],
        index=current_selection_idx if doc_keys else 0
    )

    selected_doc_meta = next((d for d in pure_catalog if d["id"] == selected_doc_key), None)

    if selected_doc_meta:
        st.markdown(f"""
        <div class="metric-card">
            <b>{selected_doc_meta['title']}</b><br>
            <span class="badge-pure">{selected_doc_meta['domain']}</span><br>
            <small style="color: #666;">{selected_doc_meta['description'][:110]}...</small><br>
            <b>Requirements:</b> {selected_doc_meta['requirements_count']}
        </div>
        """, unsafe_allow_html=True)

        if st.button("📥 Load Document into Pipeline", use_container_width=True, type="secondary"):
            doc_content = config.load_pure_document(selected_doc_key)
            if doc_content:
                st.session_state.srs_text = doc_content
                st.session_state.selected_doc_id = selected_doc_key
                st.success(f"Loaded '{selected_doc_meta['title']}'!")
                st.rerun()

    st.divider()

    st.subheader("API Status")
    google_status = "🟢 Active" if config.has_valid_google_key else "🟡 Deterministic Mode"
    groq_status = "🟢 Active" if config.has_valid_groq_key else "🟡 Deterministic Mode"
    st.write(f"**Google Gemini:** {google_status}")
    st.write(f"**Groq API:** {groq_status}")

    st.divider()
    st.subheader("Target Settings")
    target_url = st.text_input("Target URL", value=config.TARGET_URL)
    headless_mode = st.checkbox("Headless Browser", value=config.HEADLESS)
    max_retries = st.slider("Max Retries", min_value=1, max_value=4, value=config.MAX_RETRIES)

# ==============================================================================
# MAIN PAGE HEADER & SRS INPUT
# ==============================================================================
st.markdown('<p class="main-title">🤖 Agentic AI Quality Engineering Framework</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Automated Requirement Extraction ➔ Validation ➔ Risk Planning ➔ Test Design ➔ Security Review ➔ Playwright Execution</p>', unsafe_allow_html=True)

col_srs, col_run = st.columns([4, 1])
with col_srs:
    st.session_state.srs_text = st.text_area(
        "SRS Specification Document",
        value=st.session_state.srs_text,
        height=140,
        help="Paste custom specification text or select any of the 18 PURE dataset documents from the sidebar."
    )

with col_run:
    st.write("")
    st.write("")
    run_btn = st.button("🚀 Run Agent Pipeline", type="primary", use_container_width=True)

# ==============================================================================
# PIPELINE EXECUTION HANDLER
# ==============================================================================
if run_btn:
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_callback(msg, step, state):
        progress_bar.progress(int((step / 7) * 100))
        status_text.info(f"**{msg}**")

    meta = selected_doc_meta if selected_doc_meta else {"title": "Custom Specification", "domain": "User Input"}

    with st.spinner("Executing Multi-Agent Quality Engineering Pipeline..."):
        pipeline = AgenticQAPipeline()
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            final_state = loop.run_until_complete(
                pipeline.run(
                    srs_text=st.session_state.srs_text,
                    target_url=target_url,
                    max_retries=max_retries,
                    status_callback=update_callback,
                    metadata=meta
                )
            )
            st.session_state.state = final_state
            progress_bar.progress(100)
            status_text.success("✅ Agentic QA Pipeline Execution Completed Successfully!")
            time.sleep(1)
            st.rerun()
        except Exception as e:
            st.error(f"Error executing pipeline: {e}")

# ==============================================================================
# METRICS OVERVIEW CARDS
# ==============================================================================
state = st.session_state.state

if state:
    reqs = state.get("requirements", [])
    val_reqs = state.get("validated_requirements", [])
    tcs = state.get("test_cases", [])
    app_tcs = state.get("approved_test_cases", [])
    exec_res = state.get("execution_result", {})
    q_metrics = state.get("quality_metrics", {})
    sec_audit = state.get("security_audit", {})
    status_code = state.get("final_results", {}).get("status", "UNKNOWN")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Requirements", len(reqs), f"{sum(1 for r in val_reqs if r.get('is_valid'))} Valid")
    with c2:
        testability = q_metrics.get("testability_score", 90)
        st.metric("Testability Index", f"{testability}%", q_metrics.get("clarity_rating", "High"))
    with c3:
        st.metric("Test Cases", len(tcs), f"{len(app_tcs)} Approved")
    with c4:
        sec_score = sec_audit.get("compliance_score", 92)
        st.metric("Security Score", f"{sec_score}%", "OWASP Audited")
    with c5:
        exec_status = exec_res.get("status", "NOT_RUN")
        status_color = "🟢" if exec_status == "PASSED" else "🔴"
        st.metric("Sandbox Run", f"{status_color} {exec_status}", f"Retry #{state.get('retry_count', 0)}")

    st.divider()

    # ==============================================================================
    # MULTI-TAB DETAILED ANALYTICS
    # ==============================================================================
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📋 Requirements Matrix",
        "🔍 Quality & Testability",
        "⚠️ Risk Assessment",
        "🧪 Test Cases & BDD",
        "🛡️ Security Audit",
        "💻 Playwright Script",
        "⚡ Execution & Sandbox",
        "🔗 Traceability"
    ])

    with tab1:
        st.subheader("📋 Extracted System Requirements")
        if reqs:
            # Build a risk lookup from the risk matrix (assigned by Risk Agent, not extraction)
            risk_lookup = {}
            for rm in state.get("risk_matrix", []):
                rid = rm.get("requirement_id") or rm.get("id", "")
                risk_lookup[rid] = rm.get("risk_level", "—")

            table_data = []
            for r in reqs:
                table_data.append({
                    "ID": r.get("id"),
                    "Category": r.get("category", "Functional"),
                    "Priority": r.get("priority", "Medium"),
                    "Risk Level": risk_lookup.get(r.get("id"), "—"),
                    "Testable": "✅" if r.get("testable", True) else "❌",
                    "Description": r.get("description", "")
                })
            st.dataframe(table_data, use_container_width=True)
        else:
            st.info("No requirements extracted yet.")

    with tab2:
        st.subheader("🔍 Quality, Ambiguity & Testability Audit")
        col_q1, col_q2 = st.columns([1, 2])
        with col_q1:
            st.metric("Overall Testability Index", f"{q_metrics.get('testability_score', 85)}%")
            st.write(f"**Clarity Rating:** {q_metrics.get('clarity_rating', 'High')}")
            st.write(f"**Valid Requirements:** {q_metrics.get('valid_count', len(reqs))}/{len(reqs)}")

            # Coverage gap summary
            coverage_gaps = q_metrics.get("coverage_gaps", [])
            if coverage_gaps:
                st.write(f"**Coverage Gaps:** {len(coverage_gaps)}")
            conflicts = q_metrics.get("conflicts", [])
            if conflicts:
                st.write(f"**Conflicts Found:** {len(conflicts)}")
            dup_pairs = q_metrics.get("duplicate_pairs", {})
            if dup_pairs:
                st.write(f"**Duplicate Pairs:** {len(dup_pairs)}")

        with col_q2:
            st.write("**Ambiguity & Smell Findings:**")
            amb_terms = q_metrics.get("ambiguous_terms_found", [])
            if amb_terms:
                for term in amb_terms:
                    st.warning(f"⚠️ {term}")
            else:
                st.success("✅ No severe ambiguity or vague qualifiers detected in specifications.")

        # Coverage Gaps Section
        if coverage_gaps:
            st.divider()
            st.write("**🕳️ Missing Requirement Categories:**")
            for gap in coverage_gaps:
                st.error(f"🔻 {gap}")

        # Conflicts Section
        if conflicts:
            st.divider()
            st.write("**⚔️ Conflicting Requirements:**")
            for c in conflicts:
                st.warning(f"⚠️ **{c.get('req_a')}** ↔ **{c.get('req_b')}**: {c.get('reason', 'Potential contradiction')}")

        # Duplicates Section
        if dup_pairs:
            st.divider()
            st.write("**📋 Near-Duplicate Requirements:**")
            for req_id, similar_ids in dup_pairs.items():
                st.info(f"🔄 **{req_id}** is similar to: {', '.join(similar_ids)}")

        # Suggested Rewrites Section
        rewrites = q_metrics.get("suggested_rewrites", [])
        if rewrites:
            st.divider()
            st.write("**✏️ Suggested Rewrites for Ambiguous Requirements:**")
            for rw in rewrites:
                with st.expander(f"Rewrite for {rw.get('id', '?')}"):
                    st.code(rw.get("rewrite", ""), language=None)

    with tab3:
        st.subheader("⚠️ 2D Risk Assessment & Coverage Matrix")
        risk_matrix = state.get("risk_matrix", [])
        if risk_matrix:
            # Show a cleaner table with tier, requirement_id, severity, risk_score
            rm_table = []
            for entry in risk_matrix:
                rm_table.append({
                    "Tier": entry.get("tier", "—"),
                    "Req ID": entry.get("requirement_id", "—"),
                    "Area": entry.get("area", ""),
                    "Severity": entry.get("severity", ""),
                    "Likelihood": entry.get("likelihood", ""),
                    "Risk Score": entry.get("risk_score", ""),
                    "Mitigation": entry.get("mitigation", ""),
                })
            # Sort by tier
            rm_table.sort(key=lambda x: x.get("Tier", 9) if isinstance(x.get("Tier"), int) else 9)
            st.dataframe(rm_table, use_container_width=True)

        st.subheader("🎯 Prioritized Test Suites")
        priorities = state.get("testing_priorities", [])
        tier_colors = {
            "Tier 1": "🔴", "Tier 2": "🟡", "Tier 3": "🟢",
        }
        for p in priorities:
            tier_str = p.get("tier", "")
            icon = next((v for k, v in tier_colors.items() if k in tier_str), "⚪")
            st.markdown(f"{icon} **{tier_str}**: {p.get('focus')}")
            st.caption(f"Target Requirements: {', '.join(p.get('target_requirements', []))}")

    with tab4:
        st.subheader("🧪 Generated Test Cases & BDD (Gherkin) Scenarios")
        tier_icons = {1: "🔴 Tier 1 — Smoke / Blocker", 2: "🟡 Tier 2 — Core Business Logic", 3: "🟢 Tier 3 — Edge Cases & UI"}

        # Group test cases by tier
        tcs_by_tier: dict = {}
        for tc in tcs:
            t = tc.get("tier", 2)
            tcs_by_tier.setdefault(t, []).append(tc)

        for tier_num in sorted(tcs_by_tier.keys()):
            tier_label = tier_icons.get(tier_num, f"⚪ Tier {tier_num}")
            st.markdown(f"### {tier_label}  ({len(tcs_by_tier[tier_num])} tests)")
            for tc in tcs_by_tier[tier_num]:
                with st.expander(f"[{tc.get('id')}] {tc.get('title')} ({tc.get('type')})"):
                    st.write(f"**Mapped Requirement:** `{tc.get('requirement_id')}`  |  **Tier:** {tc.get('tier', '—')}")
                    st.write(f"**Preconditions:** {tc.get('preconditions')}")
                    st.write("**Steps:**")
                    for s in tc.get("steps", []):
                        st.write(f"- {s}")
                    st.write(f"**Expected Result:** {tc.get('expected_result')}")
                    if tc.get("gherkin"):
                        st.code(tc.get("gherkin"), language="gherkin")

    with tab5:
        st.subheader("🛡️ OWASP Security Review & Hardening")
        st.write(f"**Security Compliance Score:** {sec_audit.get('compliance_score', 90)}%")
        st.write("**Findings:**")
        for f in sec_audit.get("owasp_findings", []):
            st.info(f"🛡️ {f}")
        st.write("**Hardening Recommendations:**")
        for r in sec_audit.get("recommendations", []):
            st.write(f"- {r}")

    with tab6:
        st.subheader("💻 Executable Playwright Test Script")
        current_script = state.get("current_script", "# No script generated yet")
        st.code(current_script, language="python")
        st.download_button("💾 Download Script (test_suite.py)", current_script, file_name="test_suite.py", mime="text/x-python")

    with tab7:
        st.subheader("⚡ Sandboxed Playwright Live Execution")
        st.write(f"**Final Status:** {exec_res.get('status', 'N/A')}")
        st.write(f"**Duration:** {exec_res.get('duration_seconds', 0)} seconds")

        # Display execution screenshot if generated
        screenshot_path = os.path.join(config.LOGS_DIR, "execution_result.png")
        if os.path.exists(screenshot_path):
            st.image(screenshot_path, caption="📸 Playwright Execution Screenshot Proof", use_column_width=True)

        if exec_res.get("stdout"):
            with st.expander("Console Output (stdout)", expanded=True):
                st.text(exec_res.get("stdout"))
        if exec_res.get("stderr"):
            with st.expander("Error Logs (stderr)", expanded=False):
                st.text(exec_res.get("stderr"))

    with tab8:
        st.subheader("🔗 Requirement ↔ Test Case Traceability Matrix")
        traceability = state.get("traceability", [])
        test_results = state.get("test_results", [])

        if traceability:
            # Summary metrics
            total_reqs = len(set(t["requirement_id"] for t in traceability))
            verified = len(set(t["requirement_id"] for t in traceability if t["status"] == "VERIFIED"))
            failed = len(set(t["requirement_id"] for t in traceability if t["status"] == "FAILED"))
            not_covered = len(set(t["requirement_id"] for t in traceability if t["status"] == "NOT_COVERED"))

            tc1, tc2, tc3, tc4 = st.columns(4)
            with tc1:
                st.metric("Total Requirements", total_reqs)
            with tc2:
                st.metric("Verified", verified, delta=None)
            with tc3:
                st.metric("Failed", failed, delta=None)
            with tc4:
                st.metric("Not Covered", not_covered, delta=None)

            if total_reqs > 0:
                coverage_pct = round((verified / total_reqs) * 100, 1)
                st.progress(coverage_pct / 100, text=f"Requirement Coverage: {coverage_pct}%")

            st.divider()

            # Color-coded status display
            display_data = []
            for t in traceability:
                status = t["status"]
                if status == "VERIFIED":
                    badge = "🟢 VERIFIED"
                elif status == "FAILED":
                    badge = "🔴 FAILED"
                elif status == "NOT_COVERED":
                    badge = "⚪ NOT COVERED"
                else:
                    badge = "🟡 PARTIAL"
                display_data.append({
                    "Requirement": t["requirement_id"],
                    "Description": t["requirement_desc"],
                    "Test Case": t["test_case_id"],
                    "Test Title": t["test_title"],
                    "Status": badge
                })
            st.dataframe(display_data, use_container_width=True)
        elif test_results:
            st.subheader("Per-Test-Case Execution Results")
            result_data = []
            for r in test_results:
                status_icon = "🟢" if r.get("status") == "PASSED" else "🔴"
                result_data.append({
                    "Test Case": r.get("test_case_id", "?"),
                    "Requirement": r.get("requirement_id", "?"),
                    "Status": f"{status_icon} {r.get('status', 'UNKNOWN')}",
                    "Duration": f"{r.get('duration', 0)}s",
                    "Error": r.get("error") or "—"
                })
            st.dataframe(result_data, use_container_width=True)
        else:
            st.info("Run the pipeline to generate traceability data linking requirements to test execution results.")

else:
    st.info("👈 Select a PURE dataset document from the sidebar and click **'🚀 Run Agent Pipeline'** to execute the multi-agent QA workflow.")
