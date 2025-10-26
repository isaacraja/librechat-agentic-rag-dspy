"""DSPy-based query router for intelligent retrieval strategy selection."""

import dspy
from typing import Dict, Any, List
from loguru import logger


class QueryClassifier(dspy.Signature):
    """Classify user queries to determine optimal retrieval strategy."""

    query: str = dspy.InputField(desc="User's question about documents")
    file_metadata: str = dspy.InputField(desc="Information about available files")

    query_type: str = dspy.OutputField(
        desc="Query type: semantic, analytical, hybrid, or multi_hop"
    )
    reasoning: str = dspy.OutputField(desc="Explanation for this classification")
    tools_needed: str = dspy.OutputField(
        desc="Comma-separated list of tools: semantic_search, code_execution, table_extraction"
    )
    confidence: float = dspy.OutputField(desc="Confidence score between 0 and 1")


class QueryRouter:
    """Routes queries to appropriate retrieval strategies using DSPy."""

    def __init__(self):
        self.classifier = dspy.ChainOfThought(QueryClassifier)
        logger.info("QueryRouter initialized with DSPy")

    async def classify_query(
        self,
        query: str,
        file_metadata: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Classify query and determine retrieval strategy.

        Args:
            query: User query
            file_metadata: List of file metadata dicts

        Returns:
            Classification result with query_type, reasoning, and tools
        """
        # Format file metadata for the classifier
        metadata_str = self._format_metadata(file_metadata)

        try:
            # Use DSPy to classify
            result = self.classifier(
                query=query,
                file_metadata=metadata_str,
            )

            # Parse tools from comma-separated string
            tools = [t.strip() for t in result.tools_needed.split(",")]

            classification = {
                "query_type": result.query_type,
                "reasoning": result.reasoning,
                "tools": tools,
                "confidence": float(result.confidence) if hasattr(result, "confidence") else 0.8,
            }

            logger.info(
                f"Query classified as '{result.query_type}' "
                f"with tools: {tools}"
            )

            return classification

        except Exception as e:
            logger.error(f"Error classifying query: {e}")
            # Fallback to semantic search
            return {
                "query_type": "semantic",
                "reasoning": "Fallback due to classification error",
                "tools": ["semantic_search"],
                "confidence": 0.5,
            }

    def _format_metadata(self, file_metadata: List[Dict[str, Any]]) -> str:
        """Format file metadata for prompt."""
        if not file_metadata:
            return "No files available"

        lines = ["Available files:"]
        for fm in file_metadata:
            file_info = [
                f"- Filename: {fm.get('filename', 'unknown')}",
                f"  Type: {fm.get('content_type', 'unknown')}",
            ]

            if fm.get('pages'):
                file_info.append(f"  Pages: {fm['pages']}")

            if fm.get('has_tables'):
                file_info.append("  Contains: Tables")

            if fm.get('has_code'):
                file_info.append("  Contains: Code")

            if fm.get('chunk_count'):
                file_info.append(f"  Chunks: {fm['chunk_count']}")

            lines.extend(file_info)

        return "\n".join(lines)

    def determine_strategy(self, classification: Dict[str, Any]) -> str:
        """
        Determine execution strategy from classification.

        Args:
            classification: Classification result

        Returns:
            Strategy name
        """
        query_type = classification.get("query_type", "semantic")

        # Map query types to strategies
        strategy_map = {
            "semantic": "semantic_only",
            "analytical": "code_only",
            "hybrid": "hybrid",
            "multi_hop": "multi_hop",
        }

        return strategy_map.get(query_type, "semantic_only")


# Global router instance
query_router = QueryRouter()
