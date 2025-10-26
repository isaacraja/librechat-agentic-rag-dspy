"""Request models for API endpoints."""

from typing import Optional, List
from pydantic import BaseModel, Field


class UploadOptions(BaseModel):
    """Options for file upload processing."""

    enable_rag: bool = True
    enable_code_analysis: bool = True
    enable_ocr: bool = False
    chunking_strategy: str = "recursive"  # recursive, semantic, markdown, code


class UploadRequest(BaseModel):
    """Request model for file upload (used with form data)."""

    file_id: Optional[str] = None
    user_id: str
    options: Optional[UploadOptions] = None


class QueryRequest(BaseModel):
    """Request model for querying documents."""

    query: str = Field(..., description="The question to answer")
    file_ids: List[str] = Field(..., description="List of file IDs to search")
    mode: str = Field(
        default="auto",
        description="Query mode: auto, semantic, analytical, hybrid",
    )
    k: int = Field(default=5, description="Number of results to return", ge=1, le=20)
    user_id: str = Field(..., description="User ID for authorization")
    session_id: Optional[str] = None
