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

    def execute_script(self, script_code: str, script_name: str = "temp_runner.py") -> Dict[str, Any]:
        """Execute Playwright Python script in isolated subprocess and return detailed execution report"""
        script_path = os.path.join(self.logs_dir, script_name)
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_code)

        # Allow sufficient subprocess buffer time (startup + browser launch + execution)
        subproc_timeout = max(25.0, (self.timeout / 1000.0) + 15.0)
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
            status = "PASSED" if exit_code == 0 else "FAILED"
            error_message = None
            
            for line in stdout.splitlines():
                if "JSON_RESULT:" in line:
                    try:
                        json_str = line.split("JSON_RESULT:")[1].strip()
                        result_data = json.loads(json_str)
                        status = result_data.get("status", status)
                        error_message = result_data.get("error", None)
                    except Exception:
                        pass

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
                "screenshots": screenshots
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
        """Find any screenshot PNG files generated in logs directory"""
        shots = []
        if os.path.exists(self.logs_dir):
            for fname in os.listdir(self.logs_dir):
                if fname.endswith(".png"):
                    shots.append(os.path.join(self.logs_dir, fname))
        return shots
