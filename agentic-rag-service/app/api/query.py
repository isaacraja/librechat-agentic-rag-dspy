"""Query endpoint for agentic RAG retrieval."""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from pathlib import Path
from loguru import logger

from app.models.requests import QueryRequest
from app.models.responses import QueryResponse, SourceResult
from app.services.database import db_manager
from app.agents.router import query_router
from app.agents.semantic_retriever import semantic_retriever
from app.agents.code_agent import code_agent
from app.agents.hybrid_retriever import hybrid_retriever
from app.services.llm import llm_service

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query documents using agentic RAG.

    Args:
        request: QueryRequest with query, file_ids, mode, etc.

    Returns:
        QueryResponse with answer and sources
    """
    try:
        logger.info(f"Processing query: {request.query[:100]}...")

        # Get file metadata
        file_metadata = []
        file_paths = {}

        for file_id in request.file_ids:
            file_info = await db_manager.get_file(file_id, request.user_id)
            if not file_info:
                raise HTTPException(
                    status_code=404,
                    detail=f"File not found or access denied: {file_id}"
                )
            file_metadata.append(file_info)
            file_paths[file_id] = file_info["filepath"]

        # Initialize response components
        sources: List[SourceResult] = []
        reasoning_trace: List[str] = []
        tools_used: List[str] = []
        query_type: str = request.mode

        # Route query based on mode
        if request.mode == "auto":
            # Use DSPy agent router to classify query
            classification = await query_router.classify_query(
                query=request.query,
                file_metadata=file_metadata,
            )

            query_type = classification["query_type"]
            reasoning_trace.append(f"Query classified as: {query_type}")
            reasoning_trace.append(f"Reasoning: {classification['reasoning']}")

            # Determine strategy
            strategy = query_router.determine_strategy(classification)

        elif request.mode == "semantic":
            strategy = "semantic_only"
            reasoning_trace.append("Using semantic search (user-specified)")

        elif request.mode == "analytical":
            strategy = "code_only"
            reasoning_trace.append("Using code execution (user-specified)")

        elif request.mode == "hybrid":
            strategy = "hybrid"
            reasoning_trace.append("Using hybrid retrieval (user-specified)")

        else:
            raise HTTPException(status_code=400, detail=f"Invalid mode: {request.mode}")

        # Execute retrieval strategy
        if strategy == "semantic_only":
            result = await semantic_retriever.retrieve(
                query=request.query,
                file_ids=request.file_ids,
                user_id=request.user_id,
                k=request.k,
            )
            sources = result.get("sources", [])
            tools_used.append("semantic_search")
            reasoning_trace.append(f"Retrieved {len(sources)} semantic search results")

        elif strategy == "code_only":
            result = await code_agent.execute_query(
                query=request.query,
                file_ids=request.file_ids,
                file_metadata=file_metadata,
                file_paths=file_paths,
            )
            sources = result.get("sources", [])
            tools_used.append("code_execution")
            reasoning_trace.append("Generated and executed code")
            if result.get("generated_code"):
                reasoning_trace.append(f"Code: {result['generated_code'][:100]}...")

        elif strategy == "hybrid":
            result = await hybrid_retriever.retrieve(
                query=request.query,
                file_ids=request.file_ids,
                user_id=request.user_id,
                file_metadata=file_metadata,
                file_paths=file_paths,
                k=request.k,
            )
            sources = result.get("sources", [])
            tools_used.extend(["semantic_search", "code_execution"])
            reasoning_trace.append(
                f"Hybrid retrieval: {result.get('semantic_count', 0)} semantic + code execution"
            )

        else:
            raise HTTPException(status_code=500, detail=f"Unknown strategy: {strategy}")

        # Generate final answer using LLM
        answer = await generate_answer(
            query=request.query,
            sources=sources,
            reasoning_trace=reasoning_trace,
        )

        reasoning_trace.append("Generated final answer using LLM")

        # Prepare response
        response = QueryResponse(
            answer=answer,
            sources=sources,
            reasoning_trace=reasoning_trace,
            tools_used=tools_used,
            session_id=request.session_id or "default",
            query_type=query_type,
        )

        logger.info(f"Query completed successfully with {len(sources)} sources")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def generate_answer(
    query: str,
    sources: List[SourceResult],
    reasoning_trace: List[str],
) -> str:
    """
    Generate final answer from sources using LLM.

    Args:
        query: User query
        sources: Retrieved sources
        reasoning_trace: Reasoning steps

    Returns:
        Generated answer
    """
    if not sources:
        return "I couldn't find any relevant information to answer your question."

    # Build context from sources
    context_parts = []

    for i, source in enumerate(sources, 1):
        if source.type == "semantic_search":
            context_parts.append(f"[Source {i} - Semantic Search]")
            context_parts.append(source.content)

            if source.metadata.get("page"):
                context_parts.append(f"(Page {source.metadata['page']})")

        elif source.type == "code_execution":
            context_parts.append(f"[Source {i} - Code Execution]")

            if source.code:
                context_parts.append(f"Code:\n{source.code}")

            if source.content:
                context_parts.append(f"Result:\n{source.content}")

        context_parts.append("")  # Blank line

    context = "\n".join(context_parts)

    # Create prompt for answer generation
    prompt = f"""Based on the following sources, answer the user's question.

Question: {query}

Sources:
{context}

Instructions:
1. Provide a clear, concise answer to the question
2. Cite specific sources when making claims
3. If the sources don't contain enough information, acknowledge it
4. If code was executed, explain the result in natural language

Answer:"""

    # Generate answer
    try:
        answer = llm_service.generate(prompt)
        return answer.strip()
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        # Fallback: return concatenated source content
        return "\n\n".join(
            source.content for source in sources[:3] if source.content
        )
