"""Configuration management for Agentic RAG service."""

from pathlib import Path
from typing import Set
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    API_PREFIX: str = "/v1"
    LOG_LEVEL: str = "INFO"
    DEBUG: bool = False

    # Vector Database (PGVector)
    VECTOR_DB_TYPE: str = "pgvector"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "agentic_rag"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "password"
    COLLECTION_NAME: str = "agentic_documents"

    # AWS Configuration
    AWS_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_PROFILE: str = ""

    # Embeddings Configuration
    EMBEDDINGS_PROVIDER: str = "bedrock"
    EMBEDDINGS_MODEL: str = "amazon.titan-embed-text-v1"
    EMBEDDING_DIMENSION: int = 1536

    # LLM Configuration (for DSPy Agent)
    LLM_PROVIDER: str = "bedrock"
    LLM_MODEL: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 4096

    # Chunking Configuration
    CHUNK_SIZE: int = 1500
    CHUNK_OVERLAP: int = 100
    CHUNKING_STRATEGY: str = "recursive"  # recursive, semantic, markdown, code

    # File Storage
    FILE_STORAGE_TYPE: str = "local"
    FILE_UPLOAD_PATH: Path = Path("./uploads")
    FILE_MAX_SIZE_MB: int = 50
    FILE_ALLOWED_EXTENSIONS: str = "pdf,docx,txt,csv,xlsx,json,md,pptx,py,js,ts,java,cpp,c,html,xml"

    # Code Execution (Subprocess)
    CODE_EXECUTION_ENABLED: bool = True
    CODE_EXECUTION_TIMEOUT: int = 30
    CODE_EXECUTION_MAX_OUTPUT_SIZE: int = 1000000  # 1MB
    ALLOWED_PYTHON_MODULES: str = "pandas,numpy,json,csv,re,math,datetime,collections"

    # Security
    JWT_SECRET: str = "change-this-in-production"
    ENABLE_CORS: bool = True
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:3080"

    # Rate Limiting
    MAX_CONCURRENT_REQUESTS: int = 10
    RATE_LIMIT_UPLOADS: int = 10
    RATE_LIMIT_QUERIES: int = 30

    # Cleanup
    CLEANUP_ENABLED: bool = True
    CLEANUP_INTERVAL_HOURS: int = 24
    CLEANUP_FILE_MAX_AGE_HOURS: int = 168  # 7 days

    @property
    def postgres_dsn(self) -> str:
        """Get PostgreSQL connection string."""
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def postgres_sync_dsn(self) -> str:
        """Get synchronous PostgreSQL connection string."""
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def allowed_extensions(self) -> Set[str]:
        """Get set of allowed file extensions."""
        return set(ext.strip() for ext in self.FILE_ALLOWED_EXTENSIONS.split(","))

    @property
    def allowed_modules(self) -> Set[str]:
        """Get set of allowed Python modules for code execution."""
        return set(mod.strip() for mod in self.ALLOWED_PYTHON_MODULES.split(","))

    @property
    def cors_origins_list(self) -> list[str]:
        """Get list of CORS origins."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]

    @property
    def max_file_size_bytes(self) -> int:
        """Get maximum file size in bytes."""
        return self.FILE_MAX_SIZE_MB * 1024 * 1024


# Global settings instance
settings = Settings()

# Ensure upload directory exists
settings.FILE_UPLOAD_PATH.mkdir(parents=True, exist_ok=True)
