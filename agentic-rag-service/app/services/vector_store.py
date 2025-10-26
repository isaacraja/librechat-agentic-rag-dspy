"""Vector store service using PGVector."""

from typing import List, Tuple, Optional, Dict, Any
from langchain_postgres import PGVector
from langchain_core.documents import Document
from loguru import logger
from app.config import settings
from app.services.embeddings import embeddings_service


class VectorStoreService:
    """Service for vector storage and retrieval using PGVector."""

    def __init__(self):
        self.vector_store: Optional[PGVector] = None
        self._initialize()

    def _initialize(self):
        """Initialize PGVector store."""
        logger.info("Initializing PGVector store...")

        try:
            self.vector_store = PGVector(
                collection_name=settings.COLLECTION_NAME,
                connection=settings.postgres_sync_dsn,
                embeddings=embeddings_service.embeddings,
                use_jsonb=True,
            )

            logger.info("PGVector store initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing PGVector: {e}")
            raise

    async def add_documents(
        self,
        documents: List[Document],
        file_id: str,
        user_id: str,
    ) -> List[str]:
        """
        Add documents to vector store.

        Args:
            documents: List of LangChain Document objects
            file_id: Associated file ID
            user_id: User ID for authorization

        Returns:
            List of document IDs
        """
        try:
            # Add file_id and user_id to metadata
            for doc in documents:
                doc.metadata["file_id"] = file_id
                doc.metadata["user_id"] = user_id

            # Add to vector store
            ids = self.vector_store.add_documents(documents)

            logger.info(f"Added {len(documents)} documents to vector store for file {file_id}")
            return ids

        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise

    async def similarity_search(
        self,
        query: str,
        file_ids: List[str],
        user_id: str,
        k: int = 5,
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search.

        Args:
            query: Query text
            file_ids: List of file IDs to search within
            user_id: User ID for authorization
            k: Number of results to return

        Returns:
            List of (Document, score) tuples
        """
        try:
            # Create filter for file_ids and user_id
            filter_dict = {
                "file_id": {"$in": file_ids},
                "user_id": user_id,
            }

            # Perform similarity search with scores
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter_dict,
            )

            logger.info(f"Found {len(results)} results for query in files {file_ids}")
            return results

        except Exception as e:
            logger.error(f"Error performing similarity search: {e}")
            raise

    async def delete_by_file_id(self, file_id: str, user_id: str) -> int:
        """
        Delete all documents associated with a file.

        Args:
            file_id: File ID to delete
            user_id: User ID for authorization

        Returns:
            Number of documents deleted
        """
        try:
            # This requires manual SQL execution
            # PGVector doesn't have a built-in delete by metadata
            from sqlalchemy import create_engine, text

            engine = create_engine(settings.postgres_sync_dsn)

            with engine.connect() as conn:
                result = conn.execute(
                    text("""
                        DELETE FROM langchain_pg_embedding
                        WHERE cmetadata->>'file_id' = :file_id
                        AND cmetadata->>'user_id' = :user_id
                    """),
                    {"file_id": file_id, "user_id": user_id}
                )
                deleted = result.rowcount

            logger.info(f"Deleted {deleted} embeddings for file {file_id}")
            return deleted

        except Exception as e:
            logger.error(f"Error deleting embeddings: {e}")
            raise

    def embed_query(self, query: str) -> List[float]:
        """Embed a query text."""
        return embeddings_service.embed_query(query)

    async def ensure_collection(self):
        """Ensure the collection exists (called on startup)."""
        try:
            # PGVector automatically creates tables
            # Just verify connection
            from sqlalchemy import create_engine, text

            engine = create_engine(settings.postgres_sync_dsn)
            with engine.connect() as conn:
                # Create vector extension if not exists
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()

            logger.info("PGVector collection ensured")
        except Exception as e:
            logger.error(f"Error ensuring collection: {e}")
            raise


# Global vector store service instance
vector_store_service = VectorStoreService()
