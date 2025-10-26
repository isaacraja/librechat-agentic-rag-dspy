"""Upload endpoint for file processing."""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pathlib import Path
import aiofiles
from typing import Optional
import json
from loguru import logger

from app.models.requests import UploadOptions
from app.models.responses import UploadResponse, FileMetadata
from app.config import settings
from app.utils.id_generator import generate_file_id, generate_session_id
from app.services.database import db_manager
from app.services.document_processor import document_processor
from app.services.vector_store import vector_store_service
from app.services.embeddings import embeddings_service

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    file_id: Optional[str] = Form(None),
    options: Optional[str] = Form(None),
):
    """
    Upload and process a file for agentic RAG.

    Args:
        file: Uploaded file
        user_id: User ID
        file_id: Optional file ID (generated if not provided)
        options: JSON string of UploadOptions

    Returns:
        UploadResponse with file metadata
    """
    try:
        # Parse options
        upload_options = UploadOptions()
        if options:
            try:
                options_dict = json.loads(options)
                upload_options = UploadOptions(**options_dict)
            except Exception as e:
                logger.warning(f"Error parsing options: {e}, using defaults")

        # Validate file size
        file_size = 0
        content = await file.read()
        file_size = len(content)

        if file_size > settings.max_file_size_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File size ({file_size} bytes) exceeds maximum ({settings.max_file_size_bytes} bytes)"
            )

        # Validate file extension
        file_extension = Path(file.filename).suffix.lstrip(".")
        if file_extension not in settings.allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File extension .{file_extension} not allowed"
            )

        # Generate IDs
        if not file_id:
            file_id = generate_file_id()
        session_id = generate_session_id()

        # Create user directory
        user_dir = settings.FILE_UPLOAD_PATH / user_id
        user_dir.mkdir(parents=True, exist_ok=True)

        # Save file
        file_path = user_dir / f"{file_id}_{file.filename}"
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(content)

        logger.info(f"Saved file {file.filename} to {file_path}")

        # Process file
        documents, metadata = await document_processor.process_file(
            file_path=file_path,
            chunking_strategy=upload_options.chunking_strategy if upload_options.enable_rag else None,
        )

        # Add embeddings to vector store if RAG enabled
        chunk_count = 0
        if upload_options.enable_rag and documents:
            await vector_store_service.add_documents(
                documents=documents,
                file_id=file_id,
                user_id=user_id,
            )
            chunk_count = len(documents)
            logger.info(f"Added {chunk_count} chunks to vector store")

        # Store file metadata in database
        file_data = {
            "id": file_id,
            "user_id": user_id,
            "session_id": session_id,
            "filename": file.filename,
            "filepath": str(file_path),
            "size": file_size,
            "content_type": metadata.get("content_type", "application/octet-stream"),
            "pages": metadata.get("pages"),
            "has_tables": metadata.get("has_tables", False),
            "has_code": metadata.get("has_code", False),
            "chunk_count": chunk_count,
            "embedding_model": embeddings_service.model_name if upload_options.enable_rag else None,
        }

        await db_manager.add_file(file_data)

        # Prepare response
        response = UploadResponse(
            file_id=file_id,
            session_id=session_id,
            metadata=FileMetadata(
                filename=file.filename,
                size=file_size,
                pages=metadata.get("pages"),
                has_tables=metadata.get("has_tables", False),
                has_code=metadata.get("has_code", False),
                chunk_count=chunk_count,
                embedding_model=embeddings_service.model_name if upload_options.enable_rag else "none",
                content_type=metadata.get("content_type", "application/octet-stream"),
            ),
            processing_status="completed",
            message="File uploaded and processed successfully",
        )

        logger.info(f"Successfully processed file {file_id}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))
