"""Response models for API endpoints."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class FileMetadata(BaseModel):
    """Metadata about an uploaded file."""

    filename: str
    size: int
    pages: Optional[int] = None
    has_tables: bool = False
    has_code: bool = False
    chunk_count: int = 0
    embedding_model: str
    content_type: str


class UploadResponse(BaseModel):
    """Response model for file upload."""

    file_id: str
    session_id: str
    metadata: FileMetadata
    processing_status: str = "completed"
    message: str = "File uploaded and processed successfully"


class SourceResult(BaseModel):
    """A single source result from retrieval."""

    type: str = Field(..., description="Type of source: semantic_search, code_execution, etc.")
    content: Optional[str] = Field(None, description="Content or result")
    code: Optional[str] = Field(None, description="Code that was executed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    score: Optional[float] = Field(None, description="Relevance score")


class QueryResponse(BaseModel):
    """Response model for query endpoint."""

    answer: str = Field(..., description="Generated answer to the query")
    sources: List[SourceResult] = Field(..., description="List of sources used")
    reasoning_trace: List[str] = Field(..., description="Step-by-step reasoning")
    tools_used: List[str] = Field(..., description="Tools invoked by the agent")
    session_id: str
    query_type: Optional[str] = Field(None, description="Detected query type")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str = "0.1.0"
    vector_db: str
    llm_provider: str
