"""Document processing service for loading and chunking files."""

from typing import List, Tuple, Optional
from pathlib import Path
import magic
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    JSONLoader,
)
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
    PythonCodeTextSplitter,
)
from loguru import logger
from app.config import settings


class DocumentProcessor:
    """Process documents: load and chunk files."""

    # Mapping of content types to loaders
    LOADER_MAP = {
        "application/pdf": PyPDFLoader,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": Docx2txtLoader,
        "text/plain": TextLoader,
        "text/csv": CSVLoader,
        "text/markdown": UnstructuredMarkdownLoader,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": UnstructuredExcelLoader,
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": UnstructuredPowerPointLoader,
        "application/json": JSONLoader,
    }

    # Code file extensions
    CODE_EXTENSIONS = {".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".hpp", ".go", ".rs"}

    def __init__(self):
        self.text_splitters = self._initialize_splitters()

    def _initialize_splitters(self) -> dict:
        """Initialize text splitters for different strategies."""
        return {
            "recursive": RecursiveCharacterTextSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
                separators=["\n\n", "\n", ". ", " ", ""],
            ),
            "markdown": MarkdownTextSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
            ),
            "code": PythonCodeTextSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
            ),
        }

    def detect_content_type(self, file_path: Path) -> str:
        """
        Detect file content type using python-magic.

        Args:
            file_path: Path to file

        Returns:
            MIME type string
        """
        try:
            mime = magic.Magic(mime=True)
            content_type = mime.from_file(str(file_path))
            logger.debug(f"Detected content type: {content_type} for {file_path.name}")
            return content_type
        except Exception as e:
            logger.warning(f"Error detecting content type: {e}, using extension fallback")
            # Fallback to extension-based detection
            return self._content_type_from_extension(file_path)

    def _content_type_from_extension(self, file_path: Path) -> str:
        """Get content type from file extension."""
        extension_map = {
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".txt": "text/plain",
            ".csv": "text/csv",
            ".md": "text/markdown",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ".json": "application/json",
            ".py": "text/plain",
            ".js": "text/plain",
            ".ts": "text/plain",
        }
        return extension_map.get(file_path.suffix.lower(), "application/octet-stream")

    async def load_document(self, file_path: Path) -> Tuple[List[Document], dict]:
        """
        Load document using appropriate loader.

        Args:
            file_path: Path to file

        Returns:
            Tuple of (documents, metadata)
        """
        content_type = self.detect_content_type(file_path)
        metadata = {
            "content_type": content_type,
            "filename": file_path.name,
            "has_tables": False,
            "has_code": False,
            "pages": None,
        }

        try:
            # Get appropriate loader
            loader_class = self.LOADER_MAP.get(content_type, TextLoader)

            # Special handling for JSON (needs jq_schema)
            if content_type == "application/json":
                loader = loader_class(str(file_path), jq_schema=".", text_content=False)
            else:
                loader = loader_class(str(file_path))

            documents = loader.load()

            # Extract metadata
            if documents:
                # Check for pages (PDFs)
                if content_type == "application/pdf":
                    metadata["pages"] = len(documents)

                # Check for code
                if file_path.suffix in self.CODE_EXTENSIONS:
                    metadata["has_code"] = True

            logger.info(f"Loaded {len(documents)} documents from {file_path.name}")
            return documents, metadata

        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            # Fallback to text loader
            try:
                loader = TextLoader(str(file_path))
                documents = loader.load()
                logger.info(f"Loaded {len(documents)} documents using fallback TextLoader")
                return documents, metadata
            except Exception as fallback_error:
                logger.error(f"Fallback loader also failed: {fallback_error}")
                raise

    async def chunk_documents(
        self,
        documents: List[Document],
        strategy: Optional[str] = None,
    ) -> List[Document]:
        """
        Chunk documents using specified strategy.

        Args:
            documents: List of documents to chunk
            strategy: Chunking strategy (recursive, markdown, code, semantic)

        Returns:
            List of chunked documents
        """
        if not documents:
            return []

        # Auto-detect strategy if not provided
        if strategy is None:
            strategy = settings.CHUNKING_STRATEGY

        # Use appropriate splitter
        if strategy == "code":
            splitter = self.text_splitters["code"]
        elif strategy == "markdown":
            splitter = self.text_splitters["markdown"]
        else:
            splitter = self.text_splitters["recursive"]

        try:
            chunked_docs = splitter.split_documents(documents)
            logger.info(
                f"Chunked {len(documents)} documents into {len(chunked_docs)} chunks "
                f"using {strategy} strategy"
            )
            return chunked_docs
        except Exception as e:
            logger.error(f"Error chunking documents: {e}")
            raise

    async def process_file(
        self,
        file_path: Path,
        chunking_strategy: Optional[str] = None,
    ) -> Tuple[List[Document], dict]:
        """
        Complete file processing pipeline: load and chunk.

        Args:
            file_path: Path to file
            chunking_strategy: Optional chunking strategy

        Returns:
            Tuple of (chunked documents, metadata)
        """
        # Load document
        documents, metadata = await self.load_document(file_path)

        # Chunk documents
        chunked_docs = await self.chunk_documents(documents, chunking_strategy)

        # Update metadata
        metadata["chunk_count"] = len(chunked_docs)

        return chunked_docs, metadata


# Global document processor instance
document_processor = DocumentProcessor()
