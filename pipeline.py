"""
Agentic QA Pipeline Orchestration - Built on LangGraph StateGraph
Connects specialized agents, live site health probing, DOM element grounding,
Playwright sandboxed execution, failure analysis, and cyclical self-healing retries.
"""
import os
import re
import json
import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime

from langgraph.graph import StateGraph, START, END

from config import config
from state import AgentState, create_initial_state
from agents import (
    RequirementExtractionAgent,
    RequirementValidationAgent,
    RiskCoveragePlanningAgent,
    TestDesignAgent,
    ReviewSecurityAgent,
    ScriptGenerationAgent,
    FailureAnalysisAgent
)
from engine import PlaywrightRunner

logger = logging.getLogger("PipelineOrchestrator")


def build_traceability(requirements, test_cases, test_results):
    """Build a requirement → test case → execution status traceability table.

    Returns a list of dicts:
      [{requirement_id, requirement_desc, test_case_id, test_title, status}]
    where status is VERIFIED / FAILED / NOT_COVERED / PARTIAL
    """
    result_by_tc = {}
    for r in (test_results or []):
        tc_id = r.get("test_case_id", "")
        result_by_tc[tc_id] = r.get("status", "NOT_RUN")

    tc_by_req = {}
    for tc in (test_cases or []):
        req_id = tc.get("requirement_id", "")
        tc_by_req.setdefault(req_id, []).append(tc)

    table = []
    for req in (requirements or []):
        req_id = req.get("id", "")
        req_desc = req.get("description", "")[:80]
        linked_tcs = tc_by_req.get(req_id, [])

        if not linked_tcs:
            table.append({
                "requirement_id": req_id,
                "requirement_desc": req_desc,
                "test_case_id": "—",
                "test_title": "—",
                "status": "NOT_COVERED"
            })
        else:
            for tc in linked_tcs:
                tc_id = tc.get("id", "")
                tc_title = tc.get("title", "")
                exec_status = result_by_tc.get(tc_id, "NOT_RUN")

                if exec_status == "PASSED":
                    status = "VERIFIED"
                elif exec_status in ("FAILED", "ERROR"):
                    status = "FAILED"
                else:
                    status = "PARTIAL"

                table.append({
                    "requirement_id": req_id,
                    "requirement_desc": req_desc,
                    "test_case_id": tc_id,
                    "test_title": tc_title,
                    "status": status
                })

    return table


class AgenticQAPipeline:
    def __init__(self, model_client=None):
        self.req_extractor = RequirementExtractionAgent(model_client)
        self.req_validator = RequirementValidationAgent(model_client)
        self.risk_planner = RiskCoveragePlanningAgent(model_client)
        self.test_designer = TestDesignAgent(model_client)
        self.reviewer = ReviewSecurityAgent(model_client)
        self.script_generator = ScriptGenerationAgent(model_client)
        self.failure_analyzer = FailureAnalysisAgent(model_client)
        self.runner = PlaywrightRunner(timeout=config.TIMEOUT)
        self.status_callback = None
        self._compiled_graph = None

    def _notify(self, msg: str, step: int, state: AgentState):
        logger.info(f"[{step}/7] {msg}")
        if self.status_callback:
            try:
                self.status_callback(msg, step, state)
            except Exception:
                pass

    # =========================================================================
    # LANGGRAPH NODES
    # =========================================================================

    async def _node_site_probe(self, state: AgentState) -> Dict[str, Any]:
        """Stage 0: Probe target website/app health and extract DOM inventory."""
        self._notify("Stage 0/7: Opening the target in a real browser and checking its health...", 0, state)
        target_url = state.get("target_url", "")
        run_dir = state.get("run_dir", config.LOGS_DIR)
        health = await asyncio.to_thread(self.runner.probe_target, target_url, os.path.join(run_dir, "target"), config.HEADLESS)
        profile = health.get("inventory") or {}

        if not health.get("reachable"):
            reason = health.get("error") or "no response"
            logger.error(f"❌ Target {target_url} is not reachable: {reason}")
            return {
                "target_health": health,
                "target_profile": profile,
                "execution_result": {
                    "status": "TARGET_UNREACHABLE",
                    "error": f"{target_url} is not reachable: {reason}",
                    "stdout": "", "stderr": "", "duration_seconds": 0, "screenshots": []
                },
                "human_review_required": True,
                "human_review_context": {"reason": "target_unreachable", "last_error": reason, "total_attempts": 0}
            }

        logger.info(f"✅ Target is up: HTTP {health.get('status')}, {health.get('load_ms')} ms, "
                    f"health score {health.get('score')}/100, {len(profile.get('inputs', []))} input field(s) found")
        return {"target_health": health, "target_profile": profile}

    def _route_after_probe(self, state: AgentState) -> str:
        """Route to requirement extraction if target is reachable, else finalize immediately."""
        health = state.get("target_health") or {}
        if not health.get("reachable", True):
            return "finalize"
        return "extract_requirements"

    async def _node_extract_requirements(self, state: AgentState) -> Dict[str, Any]:
        """Stage 1: Extract testable requirements from SRS text."""
        self._notify("Stage 1/7: Extracting testable requirements from SRS...", 1, state)
        res = await self.req_extractor.invoke(state)
        return {"requirements": res.get("requirements", [])}

    async def _node_validate_requirements(self, state: AgentState) -> Dict[str, Any]:
        """Stage 2: Audit requirements for ambiguity, completeness, and testability."""
        self._notify("Stage 2/7: Auditing requirements for clarity, ambiguity, and testability...", 2, state)
        res = await self.req_validator.invoke(state)
        return {
            "validated_requirements": res.get("validated_requirements", []),
            "validation_report": res.get("validation_report"),
            "quality_metrics": res.get("quality_metrics")
        }

    async def _node_plan_risk_coverage(self, state: AgentState) -> Dict[str, Any]:
        """Stage 3: Assess risks and establish tier-based testing priorities."""
        self._notify("Stage 3/7: Analyzing risks and prioritizing automated test suites...", 3, state)
        res = await self.risk_planner.invoke(state)
        return {
            "test_plan": res.get("test_plan"),
            "high_risk_areas": res.get("high_risk_areas"),
            "testing_priorities": res.get("testing_priorities"),
            "risk_matrix": res.get("risk_matrix")
        }

    async def _node_design_tests(self, state: AgentState) -> Dict[str, Any]:
        """Stage 4: Design structured test cases with BDD/Gherkin and DOM element grounding."""
        self._notify("Stage 4/7: Designing structured test cases and BDD / Gherkin specifications...", 4, state)
        res = await self.test_designer.invoke(state)
        return {
            "test_cases": res.get("test_cases", []),
            "gherkin_scenarios": res.get("gherkin_scenarios")
        }

    async def _node_review_security(self, state: AgentState) -> Dict[str, Any]:
        """Stage 5: Review test cases against OWASP security rules and approve suite."""
        self._notify("Stage 5/7: Reviewing test cases against OWASP security compliance rules...", 5, state)
        res = await self.reviewer.invoke(state)
        return {
            "approved_test_cases": res.get("approved_test_cases", []),
            "review_results": res.get("review_results"),
            "security_audit": res.get("security_audit")
        }

    async def _node_generate_script(self, state: AgentState) -> Dict[str, Any]:
        """Stage 6: Generate Playwright automation script bound to target URL."""
        attempt = state.get("retry_count", 0)
        self._notify(f"Stage 6/7: Generating Playwright automation scripts (v{attempt + 1})...", 6, state)
        res = await self.script_generator.invoke(state)
        all_scripts = dict(state.get("all_scripts") or {})
        all_scripts[f"v{attempt + 1}"] = res.get("current_script", "")
        return {
            "current_script": res.get("current_script", ""),
            "all_scripts": all_scripts
        }

    async def _node_execute_playwright(self, state: AgentState) -> Dict[str, Any]:
        """Stage 7: Execute Playwright script in sandboxed browser against live target."""
        attempt = state.get("retry_count", 0)
        max_retries = state.get("max_retries", config.MAX_RETRIES)
        self._notify(f"Stage 7/7: Executing Playwright Script (Attempt #{attempt + 1}/{max_retries})...", 7, state)
        current_script = state.get("current_script", "")
        run_dir = state.get("run_dir", config.LOGS_DIR)

        exec_res = self.runner.execute_script(current_script, script_name=f"test_run_v{attempt + 1}.py",
                                              shot_dir=run_dir)
        test_results = exec_res.get("test_results") or []
        metrics = dict(state.get("metrics") or {})

        if exec_res.get("status") == "PASSED":
            logger.info(f"✅ EXECUTION SUCCESSFUL on Attempt #{attempt + 1}!")
            metrics["passed_scripts"] = metrics.get("passed_scripts", 0) + 1
            return {
                "execution_result": exec_res,
                "test_results": test_results,
                "metrics": metrics,
                "human_review_required": False
            }
        else:
            logger.warning(f"⚠️ Execution failed on Attempt #{attempt + 1}: {exec_res.get('error')}")
            metrics["failed_scripts"] = metrics.get("failed_scripts", 0) + 1
            retry_history = list(state.get("retry_history") or [])
            retry_history.append({
                "attempt": attempt + 1,
                "error": exec_res.get("error"),
                "duration": exec_res.get("duration_seconds")
            })
            return {
                "execution_result": exec_res,
                "test_results": test_results,
                "metrics": metrics,
                "last_error": exec_res.get("error"),
                "retry_history": retry_history
            }

    def _route_after_execution(self, state: AgentState) -> str:
        """Route to finalize if passed, or to analyze_failure if any test or script failed."""
        exec_res = state.get("execution_result") or {}
        if exec_res.get("status") == "PASSED":
            return "finalize"
        return "analyze_failure"

    async def _node_analyze_failure(self, state: AgentState) -> Dict[str, Any]:
        """Analyze failure logs, diagnose defect vs script issue, and manage retry cycle."""
        logger.info("⚡ Triggering Failure Analysis Agent...")
        res = await self.failure_analyzer.invoke(state)
        analysis = res.get("error_analysis") or {}
        retry_history = list(state.get("retry_history") or [])
        if retry_history:
            retry_history[-1]["root_cause"] = analysis.get("root_cause")

        attempt = state.get("retry_count", 0)
        max_retries = state.get("max_retries", config.MAX_RETRIES)
        can_recover = analysis.get("can_auto_recover", True) and (attempt < max_retries - 1)

        updates = {
            "error_analysis": analysis,
            "retry_history": retry_history
        }

        if can_recover:
            logger.info("🔧 Failures look script/locator related - retrying with an adjusted script.")
            updates["retry_count"] = attempt + 1
            updates["human_review_required"] = False
        else:
            if attempt < max_retries - 1:
                logger.info("🛑 Retrying cannot fix these failures - escalating without further attempts.")
            else:
                logger.error("❌ Max retries reached. Escalating persistent failure to Human Review.")
            exec_res = state.get("execution_result") or {}
            metrics = dict(state.get("metrics") or {})
            metrics["human_interventions"] = 1
            updates["metrics"] = metrics
            updates["human_review_required"] = True
            updates["human_review_context"] = {
                "reason": analysis.get("escalation_reason", "persistent_failure"),
                "last_error": exec_res.get("error"),
                "total_attempts": attempt + 1,
                "summary": analysis.get("summary", {}),
                "failures": analysis.get("failures", []),
                "script": state.get("current_script", ""),
                "stdout": exec_res.get("stdout"),
                "stderr": exec_res.get("stderr"),
                "screenshots": exec_res.get("screenshots", [])
            }

        return updates

    def _route_after_failure_analysis(self, state: AgentState) -> str:
        """Cyclical LangGraph Edge: route back to script generation if recoverable, else finalize."""
        attempt = state.get("retry_count", 0)
        max_retries = state.get("max_retries", config.MAX_RETRIES)
        analysis = state.get("error_analysis") or {}
        if analysis.get("can_auto_recover", True) and attempt < max_retries:
            if not state.get("human_review_required", False):
                return "generate_script"
        return "finalize"

    async def _node_finalize(self, state: AgentState) -> Dict[str, Any]:
        """Final stage: build requirement-test traceability matrix and save output artifact."""
        metadata = state.get("document_metadata") or {}
        traceability = build_traceability(
            state.get("requirements", []),
            state.get("test_cases", []),
            state.get("test_results", [])
        )
        llm_trace = self._llm_trace()
        if state.get("execution_result", {}).get("status") == "TARGET_UNREACHABLE":
            final_status = "TARGET_UNREACHABLE"
        elif state.get("human_review_required"):
            final_status = "REQUIRES_HUMAN_REVIEW"
        else:
            final_status = "PASSED"

        final_results = {
            "completed_at": datetime.now().isoformat(),
            "status": final_status,
            "passed": state.get("execution_result", {}).get("status") == "PASSED",
            "document_title": metadata.get("title") if metadata else "Custom SRS",
            "domain": metadata.get("domain") if metadata else "General",
            "total_requirements": len(state.get("requirements", [])),
            "testability_score": state.get("quality_metrics", {}).get("testability_score", 85),
            "total_test_cases": len(state.get("test_cases", [])),
            "approved_test_cases": len(state.get("approved_test_cases", []))
        }

        updates = {
            "traceability": traceability,
            "llm_trace": llm_trace,
            "final_results": final_results
        }
        merged_state = {**state, **updates}
        self.save_results(merged_state)
        logger.info("============================================================")
        logger.info(f"🏁 PIPELINE RUN COMPLETED. Final Status: {final_results['status']}")
        logger.info("============================================================")
        return updates

    # =========================================================================
    # GRAPH COMPILATION & EXECUTION
    # =========================================================================

    def build_graph(self):
        """Construct and compile the LangGraph StateGraph workflow."""
        workflow = StateGraph(AgentState)

        # Register specialized agent nodes
        workflow.add_node("site_probe", self._node_site_probe)
        workflow.add_node("extract_requirements", self._node_extract_requirements)
        workflow.add_node("validate_requirements", self._node_validate_requirements)
        workflow.add_node("plan_risk_coverage", self._node_plan_risk_coverage)
        workflow.add_node("design_tests", self._node_design_tests)
        workflow.add_node("review_security", self._node_review_security)
        workflow.add_node("generate_script", self._node_generate_script)
        workflow.add_node("execute_playwright", self._node_execute_playwright)
        workflow.add_node("analyze_failure", self._node_analyze_failure)
        workflow.add_node("finalize", self._node_finalize)

        # Connect edges and cyclical routes
        workflow.add_edge(START, "site_probe")
        workflow.add_conditional_edges("site_probe", self._route_after_probe, {
            "extract_requirements": "extract_requirements",
            "finalize": "finalize"
        })
        workflow.add_edge("extract_requirements", "validate_requirements")
        workflow.add_edge("validate_requirements", "plan_risk_coverage")
        workflow.add_edge("plan_risk_coverage", "design_tests")
        workflow.add_edge("design_tests", "review_security")
        workflow.add_edge("review_security", "generate_script")
        workflow.add_edge("generate_script", "execute_playwright")

        workflow.add_conditional_edges("execute_playwright", self._route_after_execution, {
            "finalize": "finalize",
            "analyze_failure": "analyze_failure"
        })

        # Cyclical self-healing edge: loops back to generate_script on recoverable script/locator issues
        workflow.add_conditional_edges("analyze_failure", self._route_after_failure_analysis, {
            "generate_script": "generate_script",
            "finalize": "finalize"
        })

        workflow.add_edge("finalize", END)
        return workflow.compile()

    def get_graph_mermaid(self) -> str:
        """Generate a Mermaid diagram string of the LangGraph workflow."""
        try:
            return self.build_graph().get_graph().draw_mermaid()
        except Exception:
            return ""

    async def run(
        self,
        srs_text: str,
        target_url: str = "",
        max_retries: int = config.MAX_RETRIES,
        status_callback=None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentState:
        """Execute the LangGraph Agentic QA Pipeline."""
        target_url = (target_url or config.TARGET_URL or "").strip()
        if not target_url:
            url_match = re.search(r"https?://[^\s,;\"'<>]+", srs_text)
            if url_match:
                target_url = url_match.group(0).rstrip(".,;")

        demo_server = None
        if target_url.lower() in ("", "demo", "builtin"):
            from engine.demo_app import start_demo_server
            demo_server, port = start_demo_server(0)
            target_url = f"http://127.0.0.1:{port}"
        elif "://" not in target_url:
            target_url = "https://" + target_url

        run_dir = os.path.join(config.LOGS_DIR, "runs", datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(run_dir, exist_ok=True)
        try:
            return await self._run(srs_text, target_url, max_retries, status_callback, metadata, run_dir,
                                   "demo" if demo_server else "external")
        finally:
            if demo_server:
                demo_server.shutdown()

    async def _run(self, srs_text, target_url, max_retries, status_callback, metadata, run_dir, target_mode) -> AgentState:
        logger.info("============================================================")
        logger.info("🚀 STARTING LANGGRAPH AGENTIC QA PIPELINE EXECUTION")
        if metadata:
            logger.info(f"📄 Document: {metadata.get('title', 'Unknown')} ({metadata.get('domain', 'General')})")
        logger.info(f"🎯 Target: {target_url} ({target_mode})")
        logger.info("============================================================")

        state = create_initial_state(srs_text, target_url, metadata)
        state["run_dir"] = run_dir
        state["target_mode"] = target_mode
        state["max_retries"] = max_retries

        self.status_callback = status_callback
        compiled_graph = self.build_graph()

        # Execute through the compiled LangGraph workflow
        final_state = await compiled_graph.ainvoke(state)
        return final_state

    def _llm_trace(self):
        """Per-stage summary of which LLM was assigned and which one actually answered."""
        agents = [self.req_extractor, self.req_validator, self.risk_planner, self.test_designer,
                  self.reviewer, self.failure_analyzer]
        rows = []
        for agent in agents:
            calls = agent.call_log
            if calls:
                planned = calls[0]["planned"]
            else:
                provider, model = config.get_stage_route(agent.name)
                planned = f"{provider}/{model}"
            used = {}
            for c in calls:
                key = f"{c['provider']}/{c['model']}"
                used[key] = used.get(key, 0) + 1
            errors = [e for c in calls for e in c.get("errors", [])]
            rows.append({
                "stage": agent.name,
                "assigned": planned,
                "answered_by": ", ".join(f"{k} x{n}" for k, n in used.items()) if used else "not needed",
                "calls": len(calls),
                "avg_seconds": round(sum(c["seconds"] for c in calls) / len(calls), 1) if calls else 0.0,
                "fell_back": any(c["fallback"] for c in calls),
                "first_error": errors[0][:140] if errors else "",
            })
        return rows

    def save_results(self, state: AgentState):
        """Persist execution run data to data/pipeline_results.json"""
        out_path = os.path.join(config.DATA_DIR, "pipeline_results.json")
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, default=str)
            logger.info(f"💾 Execution results saved to '{out_path}'")
        except Exception as e:
            logger.error(f"Failed to save pipeline results: {e}")


async def run_pipeline(srs_text: str, target_url: str = "", metadata: Optional[Dict[str, Any]] = None) -> AgentState:
    pipeline = AgenticQAPipeline()
    return await pipeline.run(srs_text, target_url, metadata=metadata)


if __name__ == "__main__":
    from config import config
    default_srs = config.get_default_srs_path()
    with open(default_srs, "r", encoding="utf-8") as f:
        content = f.read()
    asyncio.run(run_pipeline(content, metadata={"title": "Default SRS", "domain": "Security"}))