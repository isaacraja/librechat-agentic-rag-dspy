"""Database service for file metadata management."""

import asyncpg
from typing import Optional, List, Dict, Any
from datetime import datetime
from loguru import logger
from app.config import settings


class DatabaseManager:
    """Manages file metadata in PostgreSQL."""

    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None

    async def initialize(self):
        """Initialize database connection pool and create tables."""
        logger.info("Initializing database connection pool...")

        self.pool = await asyncpg.create_pool(
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
            database=settings.POSTGRES_DB,
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
            min_size=2,
            max_size=10,
        )

        await self._create_tables()
        logger.info("Database initialized successfully")

    async def _create_tables(self):
        """Create required database tables."""
        async with self.pool.acquire() as conn:
            # Files table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS files (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    filepath TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    content_type TEXT NOT NULL,
                    pages INTEGER,
                    has_tables BOOLEAN DEFAULT FALSE,
                    has_code BOOLEAN DEFAULT FALSE,
                    chunk_count INTEGER DEFAULT 0,
                    embedding_model TEXT,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    last_modified TIMESTAMP NOT NULL DEFAULT NOW()
                );
            """)

            # Indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_files_user_id ON files(user_id);
                CREATE INDEX IF NOT EXISTS idx_files_session_id ON files(session_id);
                CREATE INDEX IF NOT EXISTS idx_files_created_at ON files(created_at);
            """)

            logger.info("Database tables created/verified")

    async def add_file(self, file_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add or update file metadata."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO files (
                    id, user_id, session_id, filename, filepath, size,
                    content_type, pages, has_tables, has_code, chunk_count,
                    embedding_model, created_at, last_modified
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                ON CONFLICT (id) DO UPDATE SET
                    last_modified = $14,
                    chunk_count = $11,
                    has_tables = $9,
                    has_code = $10
            """,
                file_data["id"],
                file_data["user_id"],
                file_data["session_id"],
                file_data["filename"],
                file_data["filepath"],
                file_data["size"],
                file_data["content_type"],
                file_data.get("pages"),
                file_data.get("has_tables", False),
                file_data.get("has_code", False),
                file_data.get("chunk_count", 0),
                file_data.get("embedding_model"),
                file_data.get("created_at", datetime.now()),
                datetime.now(),
            )

            return file_data

    async def get_file(self, file_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get file metadata by ID and user."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM files WHERE id = $1 AND user_id = $2
            """, file_id, user_id)

            if row:
                return dict(row)
            return None

    async def list_files(self, user_id: str) -> List[Dict[str, Any]]:
        """List all files for a user."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM files WHERE user_id = $1 ORDER BY created_at DESC
            """, user_id)

            return [dict(row) for row in rows]

    async def delete_file(self, file_id: str, user_id: str) -> bool:
        """Delete file metadata."""
        async with self.pool.acquire() as conn:
            result = await conn.execute("""
                DELETE FROM files WHERE id = $1 AND user_id = $2
            """, file_id, user_id)

            return result == "DELETE 1"

    async def cleanup_old_files(self, max_age_hours: int) -> List[Dict[str, Any]]:
        """Get files older than specified age for cleanup."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                DELETE FROM files
                WHERE created_at < NOW() - INTERVAL '%s hours'
                RETURNING *
            """, max_age_hours)

            return [dict(row) for row in rows]

    async def close(self):
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")


# Global database manager instance
db_manager = DatabaseManager()
