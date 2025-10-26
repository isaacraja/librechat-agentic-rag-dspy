"""Code execution agent that generates and runs Python code."""

import dspy
from typing import List, Dict, Any
from pathlib import Path
from loguru import logger
from app.services.code_executor import code_executor
from app.services.llm import llm_service
from app.models.responses import SourceResult


class CodeGenerator(dspy.Signature):
    """Generate Python code to answer analytical queries."""

    query: str = dspy.InputField(desc="Analytical question to answer")
    file_info: str = dspy.InputField(desc="Information about available files")

    code: str = dspy.OutputField(desc="Python code to answer the query")
    explanation: str = dspy.OutputField(desc="Explanation of what the code does")


class CodeAgent:
    """Agent that generates and executes code to answer queries."""

    def __init__(self):
        self.code_generator = dspy.ChainOfThought(CodeGenerator)
        logger.info("CodeAgent initialized")

    async def execute_query(
        self,
        query: str,
        file_ids: List[str],
        file_metadata: List[Dict[str, Any]],
        file_paths: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Generate and execute code to answer query.

        Args:
            query: Analytical query
            file_ids: List of file IDs
            file_metadata: Metadata about files
            file_paths: Mapping of file_id to file path

        Returns:
            Dictionary with execution results
        """
        try:
            # Format file information
            file_info = self._format_file_info(file_metadata, file_ids)

            # Generate code using DSPy
            result = self.code_generator(
                query=query,
                file_info=file_info,
            )

            generated_code = result.code
            explanation = result.explanation

            logger.info(f"Generated code:\n{generated_code}")

            # Execute code
            execution_result = await code_executor.execute(
                code=generated_code,
                file_paths=file_paths,
            )

            # Format result as source
            source = SourceResult(
                type="code_execution",
                content=execution_result.get("stdout", ""),
                code=generated_code,
                metadata={
                    "explanation": explanation,
                    "stderr": execution_result.get("stderr", ""),
                    "status": execution_result.get("status"),
                    "return_code": execution_result.get("return_code", 0),
                },
            )

            success = execution_result.get("status") == "ok"

            return {
                "sources": [source],
                "tool": "code_execution",
                "success": success,
                "generated_code": generated_code,
                "explanation": explanation,
            }

        except Exception as e:
            logger.error(f"Error in code execution agent: {e}")
            return {
                "sources": [
                    SourceResult(
                        type="code_execution",
                        content="",
                        code="",
                        metadata={"error": str(e)},
                    )
                ],
                "tool": "code_execution",
                "success": False,
                "error": str(e),
            }

    def _format_file_info(
        self,
        file_metadata: List[Dict[str, Any]],
        file_ids: List[str],
    ) -> str:
        """Format file information for code generation."""
        lines = [
            "Available files (accessible via load_file(file_id)):",
            "",
        ]

        for file_id in file_ids:
            # Find metadata for this file
            metadata = next(
                (fm for fm in file_metadata if fm.get("id") == file_id),
                {}
            )

            filename = metadata.get("filename", "unknown")
            content_type = metadata.get("content_type", "unknown")

            lines.append(f"File ID: '{file_id}'")
            lines.append(f"  Filename: {filename}")
            lines.append(f"  Type: {content_type}")

            if ".csv" in filename.lower():
                lines.append("  Format: CSV (use pandas: df = load_file(file_id))")
            elif ".json" in filename.lower():
                lines.append("  Format: JSON (use: data = load_file(file_id))")
            elif ".xlsx" in filename.lower() or ".xls" in filename.lower():
                lines.append("  Format: Excel (use pandas: df = load_file(file_id))")
            else:
                lines.append("  Format: Text (use: text = load_file(file_id))")

            lines.append("")

        lines.append("Instructions:")
        lines.append("1. Use load_file(file_id) to access files")
        lines.append("2. Process data using pandas, numpy, etc.")
        lines.append("3. ALWAYS print() the final result")
        lines.append("4. Keep code concise and focused on answering the query")

        return "\n".join(lines)


# Global code agent instance
code_agent = CodeAgent()
