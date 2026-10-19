"""
Playwright Runner Engine - Sandboxed execution engine for generated Playwright automation scripts
Enhanced with adaptive subprocess timeouts and resilient error capture
"""
import os
import sys
import subprocess
import json
import time
import logging
from typing import Dict, Any, List
from config import config

logger = logging.getLogger("Engine.PlaywrightRunner")

class PlaywrightRunner:
    def __init__(self, timeout: int = 15000):
        self.timeout = max(10000, timeout)
        self.logs_dir = config.LOGS_DIR
        os.makedirs(self.logs_dir, exist_ok=True)

    def probe_target(self, url: str, out_dir: str, headless: bool = True) -> Dict[str, Any]:
        """Open the target in a real browser (engine/site_probe.py): reachability, health checks,
        desktop + mobile screenshots and a DOM inventory. Never raises."""
        os.makedirs(out_dir, exist_ok=True)
        probe_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "site_probe.py")
        cmd = [sys.executable, probe_script, url, out_dir]
        if not headless:
            cmd.append("--headed")
        failure = {"url": url, "reachable": False, "checks": [], "score": 0, "screenshots": {}, "inventory": {}}
        try:
            done = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                  timeout=150, cwd=config.BASE_DIR)
        except subprocess.TimeoutExpired:
            return {**failure, "error": "Target probe timed out after 150s"}
        except Exception as err:
            return {**failure, "error": f"Could not start target probe: {err}"}
        for line in done.stdout.splitlines():
            if line.startswith("PROBE_RESULT: "):
                try:
                    return json.loads(line[len("PROBE_RESULT: "):])
                except json.JSONDecodeError:
                    break
        return {**failure, "error": (done.stderr.strip().splitlines() or ["Target probe produced no result"])[-1][:200]}

    def execute_script(self, script_code: str, script_name: str = "temp_runner.py", shot_dir: str = "") -> Dict[str, Any]:
        """Execute Playwright Python script in isolated subprocess and return detailed execution report"""
        self.shot_dir = shot_dir or self.logs_dir
        script_path = os.path.join(self.logs_dir, script_name)
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_code)

        # Allow sufficient subprocess buffer time (startup + browser launch + execution)
        subproc_timeout = max(25.0, (self.timeout / 1000.0) + 15.0)
        # Per-test scripts need time proportional to the number of test cases (page load + up to 3 screenshots each)
        import re as _re
        _m = _re.search(r"Test cases:\s*(\d+)", script_code[:600])
        if _m:
            subproc_timeout = max(subproc_timeout, 30.0 + 6.0 * int(_m.group(1)))
        logger.info(f"▶️ Executing Playwright script at '{script_path}' (Subprocess Timeout: {subproc_timeout}s)...")
        start_time = time.time()

        try:
            process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=config.BASE_DIR
            )

            try:
                stdout, stderr = process.communicate(timeout=subproc_timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                duration = time.time() - start_time
                logger.error(f"❌ Script execution timed out after {duration:.2f} seconds")
                return {
                    "status": "FAILED",
                    "exit_code": -1,
                    "stdout": stdout or "",
                    "stderr": stderr or "Execution timed out",
                    "error": f"Script execution timed out after {subproc_timeout:.1f} seconds",
                    "duration_seconds": round(duration, 2),
                    "screenshots": self._gather_screenshots()
                }

            duration = time.time() - start_time
            exit_code = process.returncode

            # Parse JSON_RESULT from stdout if emitted by script
            json_result_found = False
            status = "FAILED"  # Default to FAILED — must be explicitly PASSED
            error_message = None
            test_results = []  # Per-test-case results list

            for line in stdout.splitlines():
                if "JSON_RESULT:" in line:
                    try:
                        json_str = line.split("JSON_RESULT:")[1].strip()
                        result_data = json.loads(json_str)
                        json_result_found = True

                        # Support both single result and per-test-case list
                        if isinstance(result_data, list):
                            test_results = result_data
                            # SKIPPED cases (not automatable / nothing to assert) are neutral:
                            # overall PASSED only if every executed test passed and at least one ran.
                            executed = [r for r in test_results if r.get("status") != "SKIPPED"]
                            if not test_results:
                                status = "FAILED_NO_TESTS"
                                error_message = "Script ran but produced zero test results"
                            elif not executed:
                                status = "FAILED_NO_TESTS"
                                error_message = "Every test case was skipped - no tests verified"
                            elif all(r.get("status") == "PASSED" for r in executed):
                                status = "PASSED"
                            else:
                                status = "FAILED"
                                failed = [r for r in executed if r.get("status") != "PASSED"]
                                error_message = f"{len(failed)}/{len(executed)} test cases failed"
                        else:
                            status = result_data.get("status", "FAILED")
                            error_message = result_data.get("error", None)
                    except Exception:
                        pass

            # Guard: if exit_code==0 but no JSON_RESULT was emitted, treat as inconclusive
            if not json_result_found:
                if exit_code == 0:
                    status = "FAILED_NO_TESTS"
                    error_message = "Script exited cleanly but did not emit JSON_RESULT — no tests verified"
                else:
                    error_message = stderr.strip() or f"Process exited with non-zero code {exit_code}"

            if exit_code != 0 and not error_message:
                error_message = stderr.strip() or f"Process exited with non-zero code {exit_code}"

            screenshots = self._gather_screenshots()

            if status == "PASSED":
                logger.info(f"✅ Script executed successfully in {duration:.2f}s")
            else:
                logger.error(f"❌ Script execution failed in {duration:.2f}s. Error: {error_message}")

            return {
                "status": status,
                "exit_code": exit_code,
                "stdout": stdout,
                "stderr": stderr,
                "error": error_message,
                "duration_seconds": round(duration, 2),
                "screenshots": screenshots,
                "test_results": test_results
            }

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"❌ Failed to launch script process: {e}")
            return {
                "status": "FAILED",
                "exit_code": -1,
                "stdout": "",
                "stderr": str(e),
                "error": f"Subprocess launch error: {str(e)}",
                "duration_seconds": round(duration, 2),
                "screenshots": []
            }

    def _gather_screenshots(self) -> List[str]:
        """Screenshot PNGs written by the script during this run"""
        folder = getattr(self, "shot_dir", self.logs_dir)
        shots = []
        if os.path.exists(folder):
            for fname in sorted(os.listdir(folder)):
                if fname.endswith(".png"):
                    shots.append(os.path.join(folder, fname))
        return shots