# Agentic RAG Service

An intelligent document retrieval service that combines **semantic search**, **code execution**, and **agentic routing** to provide superior question-answering capabilities over documents.

## Features

🤖 **Agentic Query Routing** - DSPy-powered intelligent query classification
🔍 **Semantic Search** - Vector similarity search using PGVector
💻 **Code Execution** - Python subprocess execution for analytical queries
🎯 **Hybrid Retrieval** - Combines multiple strategies for complex queries
☁️ **AWS Bedrock Integration** - LLM and embeddings via AWS Bedrock
🔒 **Secure Sandboxing** - Restricted code execution with whitelisted modules
📊 **Multi-Format Support** - PDF, DOCX, CSV, XLSX, JSON, Markdown, and more

## Architecture

```
User Query → DSPy Agent Router → Strategy Selection:
                                  ├─ Semantic Search (Vector DB)
                                  ├─ Code Execution (Python)
                                  └─ Hybrid (Both)
                                        ↓
                                   LLM Answer Generation
```

## Quick Start

### Prerequisites

- Docker & Docker Compose
- AWS account with Bedrock access
- AWS credentials configured

### 1. Clone and Setup

```bash
# Copy environment file
cp .env.example .env

# Edit .env with your AWS credentials
nano .env
```

### 2. Configure AWS Credentials

Add to `.env`:
```bash
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key

# Choose your models
EMBEDDINGS_MODEL=amazon.titan-embed-text-v1
LLM_MODEL=anthropic.claude-3-5-sonnet-20241022-v2:0
```

### 3. Start Services

```bash
docker-compose up -d
```

### 4. Verify Health

```bash
curl http://localhost:8001/v1/health
```

## API Usage

### Upload a File

```bash
curl -X POST "http://localhost:8001/v1/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf" \
  -F "user_id=user123" \
  -F 'options={"enable_rag": true, "chunking_strategy": "recursive"}'
```

Response:
```json
{
  "file_id": "file_abc123xyz",
  "session_id": "sess_def456uvw",
  "metadata": {
    "filename": "document.pdf",
    "size": 102400,
    "pages": 10,
    "chunk_count": 45,
    "embedding_model": "amazon.titan-embed-text-v1"
  },
  "processing_status": "completed"
}
```

### Query Documents

```bash
curl -X POST "http://localhost:8001/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the average revenue in Q2?",
    "file_ids": ["file_abc123xyz"],
    "user_id": "user123",
    "mode": "auto",
    "k": 5
  }'
```

Response:
```json
{
  "answer": "The average revenue in Q2 was $1.2M according to the analysis.",
  "sources": [
    {
      "type": "semantic_search",
      "content": "Q2 revenue data shows...",
      "score": 0.89
    },
    {
      "type": "code_execution",
      "code": "df['Q2_Revenue'].mean()",
      "content": "1200000"
    }
  ],
  "reasoning_trace": [
    "Query classified as: hybrid",
    "Reasoning: Query requires both context retrieval and calculation",
    "Retrieved 3 semantic search results",
    "Generated and executed code"
  ],
  "tools_used": ["semantic_search", "code_execution"],
  "query_type": "hybrid"
}
```

## Query Modes

### Auto Mode (Recommended)
DSPy agent automatically determines the best strategy:

```json
{"mode": "auto"}
```

### Semantic Mode
Pure vector similarity search:

```json
{"mode": "semantic"}
```

### Analytical Mode
Code-based analysis only:

```json
{"mode": "analytical"}
```

### Hybrid Mode
Combines semantic search and code execution:

```json
{"mode": "hybrid"}
```

## Query Classification

The DSPy router classifies queries into:

| Query Type | Example | Strategy |
|------------|---------|----------|
| **Semantic** | "What does the document say about X?" | Vector search |
| **Analytical** | "What's the sum of column A?" | Code execution |
| **Hybrid** | "What was the average mentioned in section 3?" | Both |
| **Multi-hop** | "Compare Q1 and Q2, then rank by growth" | Sequential reasoning |

## Supported File Types

| Category | Extensions | Loader |
|----------|-----------|--------|
| Documents | `.pdf`, `.docx`, `.txt`, `.md` | PyPDF, Docx2txt, Text |
| Spreadsheets | `.csv`, `.xlsx`, `.xls` | CSV, Excel |
| Data | `.json`, `.xml` | JSON, XML |
| Presentations | `.pptx` | PowerPoint |
| Code | `.py`, `.js`, `.ts`, `.java`, `.cpp` | Text |

## Code Execution Security

### Whitelisted Python Modules
```python
pandas, numpy, json, csv, re, math, datetime, collections
```

### Blocked Operations
- Network access (requests, urllib, socket)
- File system operations (except provided files)
- Process spawning (subprocess, os.system)
- Dynamic code execution (eval, exec, compile)

### Resource Limits
- Timeout: 30 seconds
- Max output: 1MB

## Configuration

### Environment Variables

**Server:**
- `HOST`: Server host (default: `0.0.0.0`)
- `PORT`: Server port (default: `8001`)
- `LOG_LEVEL`: Logging level (default: `INFO`)

**Database:**
- `POSTGRES_HOST`: PostgreSQL host
- `POSTGRES_PORT`: PostgreSQL port
- `POSTGRES_DB`: Database name
- `POSTGRES_USER`: Database user
- `POSTGRES_PASSWORD`: Database password

**AWS Bedrock:**
- `AWS_REGION`: AWS region
- `AWS_ACCESS_KEY_ID`: AWS access key
- `AWS_SECRET_ACCESS_KEY`: AWS secret key
- `EMBEDDINGS_MODEL`: Bedrock embedding model
- `LLM_MODEL`: Bedrock LLM model

**Processing:**
- `CHUNK_SIZE`: Text chunk size (default: `1500`)
- `CHUNK_OVERLAP`: Chunk overlap (default: `100`)
- `CHUNKING_STRATEGY`: Chunking method (default: `recursive`)

**Code Execution:**
- `CODE_EXECUTION_ENABLED`: Enable code execution (default: `true`)
- `CODE_EXECUTION_TIMEOUT`: Execution timeout in seconds (default: `30`)

## Development

### Local Setup

```bash
# Install dependencies
pip install -e .

# Copy environment file
cp .env.example .env

# Start PostgreSQL
docker-compose up postgres -d

# Run application
python -m app.main
```

### API Documentation

Interactive API docs available at:
- Swagger UI: `http://localhost:8001/docs`
- ReDoc: `http://localhost:8001/redoc`

## Architecture Details

### Components

1. **Query Router** (`agents/router.py`)
   - DSPy-based query classification
   - Determines optimal retrieval strategy

2. **Semantic Retriever** (`agents/semantic_retriever.py`)
   - Vector similarity search
   - PGVector backend

3. **Code Agent** (`agents/code_agent.py`)
   - Generates Python code from queries
   - Executes in sandboxed subprocess

4. **Hybrid Retriever** (`agents/hybrid_retriever.py`)
   - Combines semantic + code
   - Context-aware code generation

5. **Document Processor** (`services/document_processor.py`)
   - Multi-format file loading
   - Smart chunking strategies

6. **Vector Store** (`services/vector_store.py`)
   - PGVector integration
   - Metadata filtering

### Data Flow

```
Upload:
  File → Document Processor → Chunks → Embeddings → PGVector
                                      ↓
                                  Metadata → PostgreSQL

Query:
  Query → Router → [Semantic|Code|Hybrid] → Sources → LLM → Answer
```

## Comparison with Existing Systems

| Feature | RAG API | Code Interpreter | **Agentic RAG** |
|---------|---------|------------------|-----------------|
| Semantic Search | ✅ | ❌ | ✅ |
| Code Execution | ❌ | ✅ | ✅ |
| Auto Strategy Selection | ❌ | ❌ | ✅ DSPy |
| Hybrid Queries | ❌ | ❌ | ✅ |
| Table Extraction | Manual | Manual | Automatic |
| Multi-hop Reasoning | ❌ | ❌ | ✅ |

## Performance

- **Semantic Search**: ~200ms for 5 results
- **Code Execution**: ~500ms-2s depending on complexity
- **Hybrid Retrieval**: ~1-3s total
- **Concurrent Requests**: Up to 10 (configurable)

## Troubleshooting

### "Connection to database failed"
- Ensure PostgreSQL is running
- Check credentials in `.env`
- Verify network connectivity

### "AWS Bedrock access denied"
- Verify AWS credentials
- Check IAM permissions for Bedrock
- Ensure model access is enabled in AWS console

### "Code execution timeout"
- Increase `CODE_EXECUTION_TIMEOUT`
- Optimize query/code complexity
- Check for infinite loops

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please open an issue or PR.

## Support

For issues and questions:
- GitHub Issues: [Create an issue](https://github.com/your-repo/issues)
- Documentation: See `/docs` folder

---

**Built with ❤️ using FastAPI, DSPy, LangChain, and AWS Bedrock**
