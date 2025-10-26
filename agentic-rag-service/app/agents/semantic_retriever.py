"""Semantic retrieval agent using vector similarity search."""

from typing import List, Dict, Any
from loguru import logger
from app.services.vector_store import vector_store_service
from app.models.responses import SourceResult


class SemanticRetriever:
    """Performs semantic search using vector embeddings."""

    async def retrieve(
        self,
        query: str,
        file_ids: List[str],
        user_id: str,
        k: int = 5,
    ) -> Dict[str, Any]:
        """
        Retrieve documents using semantic search.

        Args:
            query: Search query
            file_ids: List of file IDs to search
            user_id: User ID for authorization
            k: Number of results

        Returns:
            Dictionary with sources and metadata
        """
        try:
            # Perform vector similarity search
            results = await vector_store_service.similarity_search(
                query=query,
                file_ids=file_ids,
                user_id=user_id,
                k=k,
            )

            # Format results
            sources = []
            for doc, score in results:
                source = SourceResult(
                    type="semantic_search",
                    content=doc.page_content,
                    metadata={
                        **doc.metadata,
                        "page": doc.metadata.get("page"),
                        "source": doc.metadata.get("source"),
                    },
                    score=float(score),
                )
                sources.append(source)

            logger.info(f"Semantic search returned {len(sources)} results")

            return {
                "sources": sources,
                "tool": "semantic_search",
                "result_count": len(sources),
            }

        except Exception as e:
            logger.error(f"Error in semantic retrieval: {e}")
            return {
                "sources": [],
                "tool": "semantic_search",
                "error": str(e),
            }


# Global semantic retriever instance
semantic_retriever = SemanticRetriever()
