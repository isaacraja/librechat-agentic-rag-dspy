"""Hybrid retrieval agent combining semantic search and code execution."""

from typing import List, Dict, Any
from loguru import logger
from app.agents.semantic_retriever import semantic_retriever
from app.agents.code_agent import code_agent
from app.models.responses import SourceResult


class HybridRetriever:
    """Combines semantic search with code-based analysis."""

    async def retrieve(
        self,
        query: str,
        file_ids: List[str],
        user_id: str,
        file_metadata: List[Dict[str, Any]],
        file_paths: Dict[str, str],
        k: int = 5,
    ) -> Dict[str, Any]:
        """
        Perform hybrid retrieval using both semantic search and code execution.

        Args:
            query: User query
            file_ids: List of file IDs
            user_id: User ID
            file_metadata: File metadata
            file_paths: File paths mapping
            k: Number of semantic results

        Returns:
            Combined results from both methods
        """
        try:
            logger.info("Performing hybrid retrieval...")

            # Step 1: Semantic search to find relevant context
            semantic_result = await semantic_retriever.retrieve(
                query=query,
                file_ids=file_ids,
                user_id=user_id,
                k=k,
            )

            # Step 2: Extract context from semantic results for code generation
            context = self._extract_context(semantic_result.get("sources", []))

            # Step 3: Generate and execute code with context
            code_result = await code_agent.execute_query(
                query=query,
                file_ids=file_ids,
                file_metadata=file_metadata,
                file_paths=file_paths,
            )

            # Combine sources
            all_sources = semantic_result.get("sources", []) + code_result.get("sources", [])

            logger.info(
                f"Hybrid retrieval completed: "
                f"{len(semantic_result.get('sources', []))} semantic + "
                f"{len(code_result.get('sources', []))} code results"
            )

            return {
                "sources": all_sources,
                "tools": ["semantic_search", "code_execution"],
                "semantic_count": len(semantic_result.get("sources", [])),
                "code_success": code_result.get("success", False),
                "context_used": context,
            }

        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {e}")
            return {
                "sources": [],
                "tools": ["semantic_search", "code_execution"],
                "error": str(e),
            }

    def _extract_context(self, sources: List[SourceResult]) -> str:
        """Extract textual context from semantic search results."""
        if not sources:
            return ""

        context_parts = []
        for i, source in enumerate(sources[:3], 1):  # Top 3 results
            if isinstance(source, SourceResult):
                content = source.content
            else:
                content = source.get("content", "")

            if content:
                context_parts.append(f"Context {i}: {content[:200]}...")

        return "\n\n".join(context_parts)


# Global hybrid retriever instance
hybrid_retriever = HybridRetriever()
