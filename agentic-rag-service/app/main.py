"""Main FastAPI application for Agentic RAG service."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.config import settings
from app.utils.logger import setup_logging
from app.services.database import db_manager
from app.services.vector_store import vector_store_service
from app.services.llm import llm_service
from app.api import upload, query
from app.models.responses import HealthResponse


# Setup logging
setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management for FastAPI application."""
    logger.info("Starting Agentic RAG service...")

    # Initialize services
    try:
        # Initialize database
        await db_manager.initialize()

        # Ensure vector store collection exists
        await vector_store_service.ensure_collection()

        # Initialize LLM (already done in module import, but log it)
        logger.info(f"LLM service ready with model: {settings.LLM_MODEL}")

        logger.info("All services initialized successfully")

    except Exception as e:
        logger.error(f"Error during initialization: {e}")
        raise

    yield

    # Cleanup on shutdown
    logger.info("Shutting down Agentic RAG service...")
    await db_manager.close()
    logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Agentic RAG Service",
    description="Intelligent RAG service combining semantic search, code execution, and agentic routing",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
if settings.ENABLE_CORS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Include routers
app.include_router(upload.router, prefix=settings.API_PREFIX, tags=["upload"])
app.include_router(query.router, prefix=settings.API_PREFIX, tags=["query"])


@app.get(f"{settings.API_PREFIX}/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        vector_db=settings.VECTOR_DB_TYPE,
        llm_provider=settings.LLM_PROVIDER,
    )


@app.get("/", tags=["root"])
async def root():
    """Root endpoint."""
    return {
        "service": "Agentic RAG",
        "version": "0.1.0",
        "status": "running",
        "docs": f"{settings.API_PREFIX}/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
