"""Code execution service using subprocess for sandboxed Python execution."""

import subprocess
import sys
import re
from typing import Dict, Any, Optional
from pathlib import Path
from loguru import logger
from app.config import settings


class CodeExecutor:
    """Execute Python code in a restricted subprocess environment."""

    def __init__(self):
        self.timeout = settings.CODE_EXECUTION_TIMEOUT
        self.max_output_size = settings.CODE_EXECUTION_MAX_OUTPUT_SIZE
        self.allowed_modules = settings.allowed_modules

    async def execute(
        self,
        code: str,
        file_paths: Optional[Dict[str, str]] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute Python code in a subprocess.

        Args:
            code: Python code to execute
            file_paths: Dictionary mapping file_id to absolute file path
            session_id: Session ID for file access

        Returns:
            Dictionary with stdout, stderr, and status
        """
        if not settings.CODE_EXECUTION_ENABLED:
            return {
                "stdout": "",
                "stderr": "Code execution is disabled",
                "status": "error",
            }

        # Validate code for dangerous operations
        if not self._validate_code(code):
            return {
                "stdout": "",
                "stderr": "Code contains potentially dangerous operations",
                "status": "error",
            }

        # Prepare environment
        env_code = self._prepare_environment(file_paths)
        full_code = env_code + "\n\n" + code

        try:
            # Execute code in subprocess
            result = subprocess.run(
                [sys.executable, "-c", full_code],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )

            stdout = result.stdout[:self.max_output_size]
            stderr = result.stderr[:self.max_output_size]

            status = "ok" if result.returncode == 0 else "error"

            logger.info(
                f"Code execution completed: status={status}, "
                f"returncode={result.returncode}"
            )

            return {
                "stdout": stdout,
                "stderr": stderr,
                "status": status,
                "return_code": result.returncode,
            }

        except subprocess.TimeoutExpired:
            logger.warning(f"Code execution timeout after {self.timeout}s")
            return {
                "stdout": "",
                "stderr": f"Code execution timeout after {self.timeout} seconds",
                "status": "error",
            }
        except Exception as e:
            logger.error(f"Error executing code: {e}")
            return {
                "stdout": "",
                "stderr": str(e),
                "status": "error",
            }

    def _validate_code(self, code: str) -> bool:
        """
        Validate code for dangerous operations.

        Args:
            code: Code to validate

        Returns:
            True if code is safe
        """
        # List of dangerous patterns
        dangerous_patterns = [
            r"__import__\s*\(",
            r"exec\s*\(",
            r"eval\s*\(",
            r"compile\s*\(",
            r"open\s*\(",  # File operations should use provided file paths
            r"subprocess",
            r"os\.system",
            r"os\.popen",
            r"os\.spawn",
            r"os\.exec",
            r"multiprocessing",
            r"threading",
            r"socket",
            r"urllib",
            r"requests",
            r"http",
            r"ftplib",
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                logger.warning(f"Dangerous pattern detected: {pattern}")
                return False

        # Validate allowed imports
        import_pattern = r"^import\s+(\w+)|^from\s+(\w+)\s+import"
        for match in re.finditer(import_pattern, code, re.MULTILINE):
            module = match.group(1) or match.group(2)
            if module not in self.allowed_modules:
                logger.warning(f"Unauthorized module import: {module}")
                return False

        return True

    def _prepare_environment(self, file_paths: Optional[Dict[str, str]] = None) -> str:
        """
        Prepare code environment with file paths.

        Args:
            file_paths: Dictionary mapping file_id to path

        Returns:
            Initialization code
        """
        env_code = [
            "import sys",
            "import pandas as pd",
            "import numpy as np",
            "import json",
            "import csv",
            "import re",
            "import math",
            "from datetime import datetime, timedelta",
            "from collections import defaultdict, Counter",
            "",
            "# File paths provided by the system",
        ]

        if file_paths:
            env_code.append("files = {")
            for file_id, path in file_paths.items():
                env_code.append(f"    '{file_id}': r'{path}',")
            env_code.append("}")
        else:
            env_code.append("files = {}")

        env_code.append("")
        env_code.append("# Helper function to load file")
        env_code.append("def load_file(file_id):")
        env_code.append("    path = files.get(file_id)")
        env_code.append("    if not path:")
        env_code.append("        raise ValueError(f'File not found: {file_id}')")
        env_code.append("    ")
        env_code.append("    # Auto-detect file type and load")
        env_code.append("    if path.endswith('.csv'):")
        env_code.append("        return pd.read_csv(path)")
        env_code.append("    elif path.endswith('.json'):")
        env_code.append("        with open(path) as f:")
        env_code.append("            return json.load(f)")
        env_code.append("    elif path.endswith(('.xlsx', '.xls')):")
        env_code.append("        return pd.read_excel(path)")
        env_code.append("    else:")
        env_code.append("        with open(path) as f:")
        env_code.append("            return f.read()")
        env_code.append("")

        return "\n".join(env_code)

    async def validate_syntax(self, code: str) -> Dict[str, Any]:
        """
        Validate Python syntax without executing.

        Args:
            code: Python code to validate

        Returns:
            Dictionary with validation result
        """
        try:
            compile(code, "<string>", "exec")
            return {"valid": True, "error": None}
        except SyntaxError as e:
            return {
                "valid": False,
                "error": f"Syntax error at line {e.lineno}: {e.msg}",
            }
        except Exception as e:
            return {"valid": False, "error": str(e)}


# Global code executor instance
code_executor = CodeExecutor()
