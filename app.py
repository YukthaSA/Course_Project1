"""
Streamlit Web Dashboard for Agentic AI Automated Requirement-to-Test Framework
Enhanced with PURE Benchmark Dataset Explorer & Multi-Stage Quality Engineering Analytics
"""
import streamlit as st
import asyncio
import os
import json
import re
import time
from datetime import datetime
import plotly.express as px

from config import config
from state import AgentState
from pipeline import AgenticQAPipeline, build_traceability

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

# Session state initialization - pristine canvas on startup
if "state" not in st.session_state:
    st.session_state.state = None

if "srs_text" not in st.session_state:
    st.session_state.srs_text = ""

if "selected_doc_id" not in st.session_state:
    st.session_state.selected_doc_id = ""

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
                st.session_state.state = None
                st.success(f"Loaded '{selected_doc_meta['title']}'.")
                st.rerun()

    st.divider()

    st.subheader("LLM Provider & Runtime")
    provider_choice = st.selectbox(
        "Active LLM Engine",
        options=["auto", "ollama", "groq", "gemini", "openai"],
        format_func=lambda x: {
            "auto": "⚡ Auto-Select (Smart Fallback)",
            "ollama": "🦙 Ollama Local (Offline / Free)",
            "groq": "⚡ Groq Cloud (Llama-3.3-70B)",
            "gemini": "✨ Google Gemini (2.0 Flash)",
            "openai": "🧠 OpenAI (GPT-4o)",
        }.get(x, x),
        index=0
    )
    config.LLM_PROVIDER = provider_choice

    if provider_choice == "ollama":
        ollama_models = ["llama3.2:3b", "qwen2.5:3b", "mistral:7b", "phi4-mini:latest", "gemma2:2b"]
        config.OLLAMA_MODEL = st.selectbox("Local Ollama Model", options=ollama_models, index=0)

    st.caption("**Active Engines Detected:**")
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.write(f"🦙 **Ollama:** {'🟢 Ready' if config.is_ollama_available else '🔴 Offline'}")
        st.write(f"🧠 **OpenAI:** {'🟢 Ready' if config.has_valid_openai_key else '⚪ None'}")
    with col_a2:
        st.write(f"⚡ **Groq:** {'🟢 Ready' if config.has_valid_groq_key else '⚪ None'}")
        st.write(f"✨ **Gemini:** {'🟢 Ready' if config.has_valid_google_key else '⚪ None'}")

    st.divider()
    st.subheader("Target Settings")
    target_url = st.text_input("Target URL (Live Website or App)", value=config.TARGET_URL, help="Enter your live app URL or external site (e.g., https://the-internet.herokuapp.com/login)")
    headless_mode = st.checkbox("Headless Browser", value=config.HEADLESS)
    max_retries = st.slider("Max Self-Healing Retries", min_value=1, max_value=4, value=config.MAX_RETRIES)

# ==============================================================================
# MAIN PAGE HEADER & SRS INPUT
# ==============================================================================
st.markdown('<p class="main-title">🤖 Agentic AI Quality Engineering Framework</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Automated Requirement Extraction ➔ Validation ➔ Risk Planning ➔ Test Design ➔ Security Review ➔ Playwright Execution</p>', unsafe_allow_html=True)

col_srs, col_btn1, col_btn2 = st.columns([3.8, 1.2, 0.8])
with col_srs:
    st.session_state.srs_text = st.text_area(
        "SRS Specification Document",
        value=st.session_state.srs_text,
        height=140,
        placeholder="Enter or paste SRS requirements here, or select a document from the sidebar and click 'Load Document into Pipeline'...",
        help="Paste custom specification text or select any of the 18 PURE dataset documents from the sidebar."
    )

with col_btn1:
    st.write("")
    st.write("")
    run_btn = st.button("Run Pipeline", type="primary", use_container_width=True)

with col_btn2:
    st.write("")
    st.write("")
    if st.button("Clear", use_container_width=True, help="Reset input and clear results"):
        st.session_state.srs_text = ""
        st.session_state.state = None
        st.rerun()

# ==============================================================================
# PIPELINE EXECUTION HANDLER
# ==============================================================================
if run_btn:
    if not st.session_state.srs_text or not st.session_state.srs_text.strip():
        st.warning("Please enter or load SRS requirements before running.")
    else:
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
            conflicts = q_metrics.get("conflicts", [])
            dup_pairs = q_metrics.get("duplicate_pairs", {})
            rewrites = q_metrics.get("suggested_rewrites", [])

            if coverage_gaps:
                st.write(f"**Coverage Gaps:** {len(coverage_gaps)}")
            if conflicts:
                st.write(f"**Conflicts Found:** {len(conflicts)}")
            if dup_pairs:
                st.write(f"**Duplicate Pairs:** {len(dup_pairs)}")
            if rewrites:
                st.write(f"**Suggested Rewrites:** {len(rewrites)}")

        with col_q2:
            st.write("**Ambiguity & Smell Findings:**")
            amb_terms = q_metrics.get("ambiguous_terms_found", [])
            if amb_terms:
                for term in amb_terms:
                    st.warning(f"⚠️ {term}")
            else:
                st.success("✅ No severe ambiguity or vague qualifiers detected in specifications.")

        # Horizontal bar chart of each requirement's testability_score
        if val_reqs:
            st.divider()
            st.write("**📊 Requirement Testability Score Distribution:**")
            req_ids = [r.get("id", f"REQ-{i+1}") for i, r in enumerate(val_reqs)]
            scores = [r.get("testability_score", 80) for r in val_reqs]
            colors = [
                "green" if s >= 80 else ("yellow" if s >= 60 else "red")
                for s in scores
            ]
            fig_qual = px.bar(
                x=scores,
                y=req_ids,
                orientation="h",
                labels={"x": "Testability Score (%)", "y": "Requirement ID"},
                title="Requirement Testability Scores"
            )
            fig_qual.update_traces(marker_color=colors)
            fig_qual.update_layout(height=max(320, len(val_reqs) * 22), yaxis={"autorange": "reversed"})
            st.plotly_chart(fig_qual, use_container_width=True)

        # Coverage Gaps Section
        if coverage_gaps:
            st.divider()
            st.write("**🕳️ Missing Requirement Categories (Coverage Gaps):**")
            for gap in coverage_gaps:
                st.error(f"🔻 {gap}")

        # Conflicts Section
        if conflicts:
            st.divider()
            st.write("**⚔️ Conflicting Requirements (Contradictions):**")
            for c in conflicts:
                st.warning(f"⚠️ **{c.get('req_a')}** ↔ **{c.get('req_b')}**: {c.get('reason', 'Potential contradiction')}")

        # Duplicates Section
        if dup_pairs:
            st.divider()
            st.write("**📋 Near-Duplicate Requirements:**")
            for req_id, similar_ids in dup_pairs.items():
                st.info(f"🔄 **{req_id}** is similar to: {', '.join(similar_ids)}")

        # Suggested Rewrites Section
        if rewrites:
            st.divider()
            st.write("**✏️ Suggested Rewrites for Ambiguous Requirements:**")
            for rw in rewrites:
                req_label = rw.get('id', '?')
                with st.expander(f"Rewrite for {req_label}"):
                    if rw.get("original"):
                        st.caption(f"**Original:** {rw.get('original')}")
                    st.code(rw.get("rewrite", ""), language=None)

    with tab3:
        st.subheader("⚠️ 2D Risk Assessment & Coverage Matrix")
        risk_matrix = state.get("risk_matrix", [])
        if risk_matrix:
            # Bar chart showing count of risk_matrix entries per severity level colored red/orange/yellow/green
            sev_levels = ["Critical", "High", "Medium", "Low"]
            sev_counts = {s: 0 for s in sev_levels}
            for entry in risk_matrix:
                s = str(entry.get("severity", "Medium")).strip().capitalize()
                if s in sev_counts:
                    sev_counts[s] += 1
                else:
                    sev_counts[s] = 1

            fig_risk = px.bar(
                x=list(sev_counts.keys()),
                y=list(sev_counts.values()),
                color=list(sev_counts.keys()),
                color_discrete_map={
                    "Critical": "red",
                    "High": "orange",
                    "Medium": "yellow",
                    "Low": "green"
                },
                labels={"x": "Severity Level", "y": "Count"},
                title="Risk Matrix Entries by Severity Level"
            )
            fig_risk.update_layout(showlegend=False, height=320)
            st.plotly_chart(fig_risk, use_container_width=True)

            # Build structured table with columns: Tier, Req ID, Area, Severity, Likelihood, Risk Score, Mitigation
            rm_table = []
            for idx, entry in enumerate(risk_matrix):
                rid = entry.get("requirement_id")
                if not rid or rid == "—":
                    area = entry.get("area", "")
                    m = re.search(r"REQ[-_]?\d+", area, re.IGNORECASE)
                    if m:
                        rid = m.group(0).upper()
                    else:
                        rid = area.split(" - ")[0].strip() if " - " in area else (
                            reqs[idx].get("id") if idx < len(reqs) else f"REQ-{idx+1:03d}"
                        )

                raw_tier = entry.get("tier")
                if raw_tier is None or raw_tier == "—":
                    sev = str(entry.get("severity", "Medium")).lower()
                    tier = 1 if sev == "critical" else (2 if sev == "high" else 3)
                else:
                    try:
                        tier = int(raw_tier)
                    except (ValueError, TypeError):
                        tier = 2

                rm_table.append({
                    "Tier": tier,
                    "Req ID": rid,
                    "Area": entry.get("area", ""),
                    "Severity": entry.get("severity", ""),
                    "Likelihood": entry.get("likelihood", ""),
                    "Risk Score": entry.get("risk_score", ""),
                    "Mitigation": entry.get("mitigation", ""),
                })

            # Sort by tier numerically
            rm_table.sort(key=lambda x: x.get("Tier", 9) if isinstance(x.get("Tier"), int) else 9)
            st.dataframe(rm_table, use_container_width=True)
        else:
            st.info("No risk matrix available. Run the pipeline to generate risk assessment.")

        st.divider()
        st.subheader("🎯 Prioritized Test Suites")
        priorities = state.get("testing_priorities", [])
        tier_colors = {
            "Tier 1": "🔴", "Tier 2": "🟡", "Tier 3": "🟢",
        }
        if priorities:
            for p in priorities:
                tier_str = p.get("tier", "")
                icon = next((v for k, v in tier_colors.items() if k in tier_str), "⚪")
                st.markdown(f"{icon} **{tier_str}**: {p.get('focus')}")
                st.caption(f"Target Requirements: {', '.join(p.get('target_requirements', []))}")
        else:
            st.info("No testing priorities defined.")

    with tab4:
        st.subheader("🧪 Generated Test Cases & BDD (Gherkin) Scenarios")

        if tcs:
            # Pie chart at the top showing test case count per tier colored red/gold/green
            tier_pie_counts = {"Tier 1": 0, "Tier 2": 0, "Tier 3": 0}
            for tc in tcs:
                raw_t = tc.get("tier", 2)
                try:
                    t_val = int(raw_t)
                except (ValueError, TypeError):
                    t_val = 2
                lbl = f"Tier {t_val}"
                tier_pie_counts[lbl] = tier_pie_counts.get(lbl, 0) + 1

            fig_tier = px.pie(
                names=list(tier_pie_counts.keys()),
                values=list(tier_pie_counts.values()),
                color=list(tier_pie_counts.keys()),
                color_discrete_map={
                    "Tier 1": "red",
                    "Tier 2": "gold",
                    "Tier 3": "green"
                },
                title="Test Case Breakdown by Tier"
            )
            fig_tier.update_layout(height=320)
            st.plotly_chart(fig_tier, use_container_width=True)

            # Group test cases by tier
            tcs_by_tier = {}
            for tc in tcs:
                t = tc.get("tier", 2)
                try:
                    t = int(t)
                except (ValueError, TypeError):
                    t = 2
                tcs_by_tier.setdefault(t, []).append(tc)

            tier_header_map = {
                1: "🔴 Tier 1 — Smoke / Blocker",
                2: "🟡 Tier 2 — Core Business Logic",
                3: "🟢 Tier 3 — Edge Cases & UI"
            }

            for tier_key in sorted(tcs_by_tier.keys()):
                header_prefix = tier_header_map.get(tier_key, f"⚪ Tier {tier_key}")
                count = len(tcs_by_tier[tier_key])
                st.markdown(f"### {header_prefix} ({count} tests)")

                for tc in tcs_by_tier[tier_key]:
                    tc_id = tc.get('id', 'TC')
                    req_id = tc.get('requirement_id', 'REQ')
                    title = tc.get('title', 'Scenario')
                    ttype = tc.get('type', 'Positive')
                    with st.expander(f"[{tc_id}] {title} ({ttype} | Tier {tier_key})"):
                        st.write(f"**Mapped Requirement:** `{req_id}`  |  **Tier:** Tier {tier_key}")
                        if tc.get("technique"):
                            st.write(f"**Technique:** {tc.get('technique')}")
                        st.write(f"**Preconditions:** {tc.get('preconditions', 'Application in ready state')}")
                        if tc.get("test_data"):
                            st.write("**Test Data:**")
                            st.json({k: (str(v)[:60] + "..." if len(str(v)) > 60 else v) for k, v in tc["test_data"].items()})
                        st.write("**Steps:**")
                        for s in tc.get("steps", []):
                            st.write(f"- {s}")
                        st.write(f"**Expected Result:** {tc.get('expected_result', 'N/A')}")
                        for a in tc.get("assertions", []):
                            st.caption(f"Assert: {a.get('type')} {a.get('value', '')}")
                        if tc.get("gherkin"):
                            st.code(tc.get("gherkin"), language="gherkin")
        else:
            st.info("No test cases generated yet.")

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

        # Donut chart showing passed vs failed test count if execution results exist
        exec_counts = {}
        if state.get("test_results"):
            t_res = state.get("test_results", [])
            p_count = sum(1 for r in t_res if r.get("status") == "PASSED")
            f_count = sum(1 for r in t_res if r.get("status") != "PASSED")
            if p_count > 0 or f_count > 0:
                exec_counts = {"Passed": p_count, "Failed": f_count}
        elif exec_res and exec_res.get("status") != "NOT_RUN":
            st_code = exec_res.get("status")
            if st_code == "PASSED":
                exec_counts = {"Passed": len(app_tcs) or 1, "Failed": 0}
            elif st_code in ("FAILED", "ERROR"):
                exec_counts = {"Passed": 0, "Failed": 1}

        if exec_counts:
            fig_donut = px.pie(
                names=list(exec_counts.keys()),
                values=list(exec_counts.values()),
                color=list(exec_counts.keys()),
                color_discrete_map={"Passed": "green", "Failed": "red"},
                hole=0.4,
                title="Execution Results (Pass vs Fail)"
            )
            fig_donut.update_layout(height=320)
            st.plotly_chart(fig_donut, use_container_width=True)

        # Display execution screenshots for all executed test cases
        st.markdown("#### 📸 Test Case Execution Evidence & Visual Proof")
        t_results = state.get("test_results") or []
        screenshots_rendered = 0

        if t_results:
            cols = st.columns(2)
            for idx, tr in enumerate(t_results):
                col = cols[idx % 2]
                with col:
                    tc_id = tr.get("test_case_id", f"TC-{idx+1:03d}")
                    tc_status = tr.get("status", "UNKNOWN")
                    tc_title = tr.get("title", "Test Case")
                    tc_dur = tr.get("duration", 0)
                    tc_err = tr.get("error")

                    status_icon = "🟢" if tc_status == "PASSED" else ("🔴" if tc_status == "FAILED" else "⚪")

                    with st.container(border=True):
                        st.markdown(f"**{status_icon} {tc_id}: {tc_title}**")
                        st.caption(f"Status: `{tc_status}` | Duration: {tc_dur}s")
                        if tc_err:
                            st.warning(f"Failure Detail: {tc_err}")

                        # Check for matching screenshots
                        candidate_files = [
                            tr.get("screenshot"),
                            f"{tc_id}_{tc_status.lower()}.png",
                            f"fail_{tc_id}.png"
                        ]
                        shot_found = False
                        for c_file in candidate_files:
                            if c_file:
                                full_p = os.path.join(config.LOGS_DIR, os.path.basename(c_file))
                                if os.path.exists(full_p):
                                    st.image(full_p, caption=f"Proof: {tc_id} ({tc_status})", use_container_width=True)
                                    shot_found = True
                                    screenshots_rendered += 1
                                    break
                        if not shot_found:
                            st.caption("No screenshot captured for this test case.")

        # Fallback if no per-test screenshots were rendered
        if screenshots_rendered == 0:
            screenshot_path = os.path.join(config.LOGS_DIR, "execution_result.png")
            if os.path.exists(screenshot_path):
                st.image(screenshot_path, caption="📸 Playwright Execution Screenshot Proof", use_container_width=True)
            else:
                st.info("No execution screenshots captured yet.")

        if exec_res.get("stdout"):
            with st.expander("Console Output (stdout)", expanded=True):
                st.text(exec_res.get("stdout"))
        if exec_res.get("stderr"):
            with st.expander("Error Logs (stderr)", expanded=False):
                st.text(exec_res.get("stderr"))
                
        # Failure analysis: application defects vs script/environment problems
        analysis = state.get("error_analysis") or {}
        if analysis.get("failures"):
            st.divider()
            st.subheader("🔬 Failure Analysis")
            st.write(f"**{analysis.get('root_cause')}** — {analysis.get('analysis_details')}")
            st.info(f"Recommended action: {analysis.get('corrective_action')}")
            st.dataframe([{
                "Test": f.get("test_case_id"),
                "Requirement": f.get("requirement_id"),
                "Tier": f.get("tier"),
                "Severity": f.get("severity"),
                "Category": f.get("category"),
                "Title": f.get("title"),
                "Assessment": f.get("likelihood"),
                "Suggested Fix": f.get("suggestion"),
            } for f in analysis["failures"]], use_container_width=True)

    with tab8:
        st.subheader("🔗 Requirement ↔ Test Case Traceability Matrix")
        traceability = state.get("traceability", [])
        test_results = state.get("test_results", [])

        # If traceability was not previously generated in cached state, compute on the fly
        if not traceability and (reqs or tcs):
            traceability = build_traceability(reqs, tcs, test_results)

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
                st.metric("Verified", verified)
            with tc3:
                st.metric("Failed", failed)
            with tc4:
                st.metric("Not Covered", not_covered)

            if total_reqs > 0:
                coverage_pct = round((verified / total_reqs) * 100, 1)
                st.progress(min(1.0, coverage_pct / 100.0), text=f"Requirement Coverage: {coverage_pct}%")

            st.divider()

            # Color-coded status display
            display_data = []
            for t in traceability:
                status = t.get("status", "NOT_COVERED")
                if status == "VERIFIED":
                    badge = "🟢 VERIFIED"
                elif status == "FAILED":
                    badge = "🔴 FAILED"
                elif status == "NOT_COVERED":
                    badge = "⚪ NOT COVERED"
                else:
                    badge = "🟡 PARTIAL"
                display_data.append({
                    "Requirement": t.get("requirement_id", "—"),
                    "Description": t.get("requirement_desc", ""),
                    "Test Case": t.get("test_case_id", "—"),
                    "Test Title": t.get("test_title", "—"),
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
    st.info("Enter or load an SRS document and click 'Run Pipeline' to execute.")
