"""Embeddings service using AWS Bedrock."""

from typing import List
import boto3
from langchain_aws import BedrockEmbeddings
from loguru import logger
from app.config import settings


class EmbeddingsService:
    """Service for generating embeddings using AWS Bedrock."""

    def __init__(self):
        self.embeddings = None
        self._initialize()

    def _initialize(self):
        """Initialize Bedrock embeddings."""
        logger.info(f"Initializing Bedrock embeddings with model: {settings.EMBEDDINGS_MODEL}")

        # Create Bedrock client
        bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID if settings.AWS_ACCESS_KEY_ID else None,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY if settings.AWS_SECRET_ACCESS_KEY else None,
        )

        # Initialize LangChain Bedrock embeddings
        self.embeddings = BedrockEmbeddings(
            client=bedrock_client,
            model_id=settings.EMBEDDINGS_MODEL,
        )

        logger.info("Bedrock embeddings initialized successfully")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents.

        Args:
            texts: List of text documents to embed

        Returns:
            List of embedding vectors
        """
        try:
            embeddings = self.embeddings.embed_documents(texts)
            logger.debug(f"Generated embeddings for {len(texts)} documents")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating document embeddings: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        try:
            embedding = self.embeddings.embed_query(text)
            logger.debug(f"Generated embedding for query: {text[:50]}...")
            return embedding
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return settings.EMBEDDING_DIMENSION

    @property
    def model_name(self) -> str:
        """Get embedding model name."""
        return settings.EMBEDDINGS_MODEL


# Global embeddings service instance
embeddings_service = EmbeddingsService()
