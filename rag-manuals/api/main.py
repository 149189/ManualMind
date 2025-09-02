# rag-manuals/api/main.py
import os
import logging
import tempfile
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import httpx

from ..ingestion.pdf_to_text import extract_pages, validate_pdf_file, get_pdf_info
from ..ingestion.chunker import chunk_page, validate_chunks
from ..ingestion.cleaner import clean_pages_batch
from ..embeddings.embedder import Embedder
from ..index.faiss_utils import FaissStore
from .prompt_templates import QUERY_SYSTEM, format_retrieved_snippets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment variables
LLM_BACKEND = os.environ.get("LLM_BACKEND", "local")
LLM_API_URL = os.environ.get("LLM_API_URL", "http://localhost:8080")
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "mistral-7b")
API_SECRET_KEY = os.environ.get("API_SECRET_KEY")
MAX_FILE_SIZE = int(os.environ.get("MAX_FILE_SIZE", 50 * 1024 * 1024))  # 50MB default
INDEX_PATH = os.environ.get("INDEX_PATH", "index/faiss.index")
META_PATH = os.environ.get("META_PATH", "index/meta.jsonl")

# Global variables for services
store: Optional[FaissStore] = None
embedder: Optional[Embedder] = None
llm_client: Optional[httpx.AsyncClient] = None

# Security setup
security = HTTPBearer(auto_error=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - startup and shutdown."""
    global store, embedder, llm_client
    
    try:
        # Startup
        logger.info("Starting ManualMind API...")
        
        # Initialize FAISS store
        store = FaissStore(index_path=INDEX_PATH, meta_path=META_PATH)
        logger.info(f"Initialized FAISS store with paths: {INDEX_PATH}, {META_PATH}")
        
        # Initialize embedder
        embedder = Embedder()
        logger.info("Initialized embedder")
        
        # Initialize HTTP client for LLM communication
        if LLM_BACKEND == "http":
            llm_client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
            )
            
            # Test LLM connection
            try:
                response = await llm_client.get(f"{LLM_API_URL}/health")
                if response.status_code == 200:
                    logger.info(f"Successfully connected to LLM at {LLM_API_URL}")
                else:
                    logger.warning(f"LLM health check returned {response.status_code}")
            except Exception as e:
                logger.error(f"Failed to connect to LLM: {e}")
        
        logger.info("ManualMind API started successfully")
        yield
        
    except Exception as e:
        logger.error(f"Failed to start ManualMind API: {e}")
        raise
    
    finally:
        # Shutdown
        logger.info("Shutting down ManualMind API...")
        if llm_client:
            await llm_client.aclose()
        logger.info("ManualMind API shutdown complete")


app = FastAPI(
    title="ManualMind API",
    description="RAG-powered assistant for product manuals",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class IngestResponse(BaseModel):
    """Response model for document ingestion."""
    ingested_chunks: int = Field(..., description="Number of chunks successfully ingested")
    message: str = Field(..., description="Status message")
    file_info: Optional[Dict[str, Any]] = Field(None, description="Information about the processed file")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    warnings: Optional[List[str]] = Field(None, description="Any warnings during processing")


class QueryRequest(BaseModel):
    """Request model for querying the knowledge base."""
    q: str = Field(..., min_length=1, max_length=1000, description="Query string")
    top_k: int = Field(5, ge=1, le=20, description="Number of top results to retrieve")
    min_score: Optional[float] = Field(0.0, ge=0.0, le=1.0, description="Minimum similarity score")
    
    @validator('q')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty or whitespace only')
        return v.strip()


class QueryResponse(BaseModel):
    """Response model for knowledge base queries."""
    answer: str = Field(..., description="Generated answer")
    citations: List[str] = Field(..., description="List of citation identifiers")
    confidence: float = Field(..., ge=0.0, le=100.0, description="Confidence score (0-100)")
    snippets: List[Dict[str, Any]] = Field(..., description="Retrieved text snippets")
    query_time: Optional[float] = Field(None, description="Query processing time in seconds")
    retrieval_stats: Optional[Dict[str, Any]] = Field(None, description="Retrieval statistics")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status")
    services: Dict[str, str] = Field(..., description="Status of individual services")
    version: str = Field(..., description="API version")


class IndexStats(BaseModel):
    """Response model for index statistics."""
    total_chunks: int = Field(..., description="Total number of chunks in index")
    total_documents: int = Field(..., description="Total number of documents")
    index_size: str = Field(..., description="Index file size")
    last_updated: Optional[str] = Field(None, description="Last update timestamp")


# Dependency functions
async def get_auth_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Validate API authentication if enabled."""
    if API_SECRET_KEY and (not credentials or credentials.credentials != API_SECRET_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials


async def validate_file_upload(file: UploadFile) -> UploadFile:
    """Validate uploaded file."""
    # Check file size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum allowed size of {MAX_FILE_SIZE} bytes"
        )
    
    # Reset file pointer
    await file.seek(0)
    
    # Check file type
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported"
        )
    
    return file


# API endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        services = {
            "api": "healthy",
            "embedder": "healthy" if embedder else "unavailable",
            "store": "healthy" if store else "unavailable",
        }
        
        # Check LLM service
        if LLM_BACKEND == "http" and llm_client:
            try:
                response = await llm_client.get(f"{LLM_API_URL}/health", timeout=5.0)
                services["llm"] = "healthy" if response.status_code == 200 else "degraded"
            except Exception:
                services["llm"] = "unavailable"
        else:
            services["llm"] = "local" if LLM_BACKEND == "local" else "not_configured"
        
        overall_status = "healthy" if all(
            status in ["healthy", "local"] for status in services.values()
        ) else "degraded"
        
        return HealthResponse(
            status=overall_status,
            services=services,
            version="1.0.0"
        )
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            services={"error": str(e)},
            version="1.0.0"
        )


@app.get("/stats", response_model=IndexStats)
async def get_index_stats(credentials = Depends(get_auth_token)):
    """Get index statistics."""
    try:
        if not store:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Index service not available"
            )
        
        # Load index if needed
        try:
            if store.index is None:
                store.load()
        except FileNotFoundError:
            return IndexStats(
                total_chunks=0,
                total_documents=0,
                index_size="0 bytes",
                last_updated=None
            )
        
        total_chunks = len(store.metadata)
        
        # Count unique documents
        unique_docs = set()
        for meta in store.metadata:
            if 'source' in meta:
                unique_docs.add(meta['source'])
        
        # Get index file size
        index_size = "unknown"
        try:
            index_path = Path(INDEX_PATH)
            if index_path.exists():
                size_bytes = index_path.stat().st_size
                index_size = format_file_size(size_bytes)
        except Exception as e:
            logger.warning(f"Could not get index file size: {e}")
        
        return IndexStats(
            total_chunks=total_chunks,
            total_documents=len(unique_docs),
            index_size=index_size,
            last_updated=None  # Could add timestamp tracking
        )
    
    except Exception as e:
        logger.error(f"Error getting index stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get index statistics: {str(e)}"
        )


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = Depends(validate_file_upload),
    credentials = Depends(get_auth_token)
):
    """Ingest a PDF document into the knowledge base."""
    import time
    start_time = time.time()
    
    warnings = []
    tmp_path = None
    
    try:
        # Validate services
        if not store or not embedder:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Required services not available"
            )
        
        logger.info(f"Starting ingestion of file: {file.filename}")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Validate PDF file
        if not validate_pdf_file(tmp_path):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or corrupted PDF file"
            )
        
        # Get file info
        file_info = get_pdf_info(tmp_path)
        if not file_info:
            warnings.append("Could not extract PDF metadata")
            file_info = {"filename": file.filename}
        
        # Extract pages
        try:
            pages = extract_pages(tmp_path, include_metadata=True)
            if not pages:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No text could be extracted from the PDF"
                )
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to extract text from PDF: {str(e)}"
            )
        
        # Clean pages (remove headers/footers)
        try:
            cleaned_pages = clean_pages_batch(pages, remove_headers=True)
            if len(cleaned_pages) != len(pages):
                warnings.append("Some pages could not be cleaned properly")
            pages = cleaned_pages
        except Exception as e:
            logger.warning(f"Page cleaning failed, using original pages: {e}")
            warnings.append("Page cleaning failed, using original text")
        
        # Chunk pages
        all_chunks = []
        for page in pages:
            try:
                page_chunks = chunk_page(page)
                all_chunks.extend(page_chunks)
            except Exception as e:
                logger.warning(f"Failed to chunk page {page.get('page', 'unknown')}: {e}")
                warnings.append(f"Failed to chunk page {page.get('page', 'unknown')}")
        
        if not all_chunks:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No valid chunks could be created from the document"
            )
        
        # Validate chunks
        valid_chunks = validate_chunks(all_chunks)
        if len(valid_chunks) != len(all_chunks):
            warnings.append(f"Filtered out {len(all_chunks) - len(valid_chunks)} invalid chunks")
        all_chunks = valid_chunks
        
        # Extract texts for embedding
        texts = [chunk["text"] for chunk in all_chunks]
        
        try:
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(texts)} chunks...")
            embeddings = embedder.embed(texts)
            
            # Build/update index
            logger.info("Building search index...")
            store.build(embeddings=embeddings, metadata=all_chunks)
            
        except Exception as e:
            logger.error(f"Embedding or indexing failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create embeddings or build index: {str(e)}"
            )
        
        processing_time = time.time() - start_time
        
        # Clean up temp file in background
        background_tasks.add_task(cleanup_temp_file, tmp_path)
        
        logger.info(f"Successfully ingested {len(all_chunks)} chunks in {processing_time:.2f}s")
        
        return IngestResponse(
            ingested_chunks=len(all_chunks),
            message=f"Successfully indexed {len(all_chunks)} chunks from {file.filename}",
            file_info=file_info,
            processing_time=processing_time,
            warnings=warnings if warnings else None
        )
    
    except HTTPException:
        # Clean up temp file on error
        if tmp_path:
            cleanup_temp_file(tmp_path)
        raise
    except Exception as e:
        # Clean up temp file on error
        if tmp_path:
            cleanup_temp_file(tmp_path)
        logger.error(f"Unexpected error during ingestion: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during ingestion: {str(e)}"
        )


@app.post("/query", response_model=QueryResponse)
async def query_knowledge_base(
    request: QueryRequest,
    credentials = Depends(get_auth_token)
):
    """Query the knowledge base."""
    import time
    start_time = time.time()
    
    try:
        # Validate services
        if not store or not embedder:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Required services not available"
            )
        
        logger.info(f"Processing query: {request.q[:100]}...")
        
        # Load index if needed
        try:
            if store.index is None:
                store.load()
        except FileNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No knowledge base found. Please ingest documents first."
            )
        
        # Generate query embedding
        try:
            query_embedding = embedder.embed([request.q])[0]
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process query"
            )
        
        # Search for relevant chunks
        try:
            results = store.search(query_embedding, top_k=request.top_k)
            
            # Filter by minimum score if specified
            if request.min_score and request.min_score > 0:
                results = [(score, meta) for score, meta in results if score >= request.min_score]
            
            if not results:
                return QueryResponse(
                    answer="I don't know based on the provided manuals. No relevant information found.",
                    citations=[],
                    confidence=0.0,
                    snippets=[],
                    query_time=time.time() - start_time,
                    retrieval_stats={
                        "total_retrieved": 0,
                        "min_score": 0.0,
                        "max_score": 0.0
                    }
                )
        
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Search failed"
            )
        
        # Format retrieved snippets
        formatted_snippets = format_retrieved_snippets(results)
        
        # Generate answer using LLM
        try:
            if LLM_BACKEND == "http":
                answer_data = await generate_answer_http(request.q, formatted_snippets)
            else:
                answer_data = generate_answer_local(request.q, formatted_snippets)
        
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            # Fallback response
            answer_data = {
                "answer": "I found relevant information but couldn't generate a proper response. Please check the snippets below.",
                "citations": [f"S{i+1}" for i in range(len(results))],
                "llm_confidence": 50
            }
        
        # Prepare response
        snippets_data = []
        citations = []
        
        for i, (score, meta) in enumerate(results, 1):
            snippet_id = f"S{i}"
            citations.append(snippet_id)
            snippets_data.append({
                "id": snippet_id,
                "text": meta.get("text", ""),
                "source": meta.get("source", "unknown"),
                "page": meta.get("page", "?"),
                "score": float(score),
                "chunk_id": meta.get("id", "")
            })
        
        query_time = time.time() - start_time
        
        # Calculate retrieval stats
        scores = [score for score, _ in results]
        retrieval_stats = {
            "total_retrieved": len(results),
            "min_score": float(min(scores)) if scores else 0.0,
            "max_score": float(max(scores)) if scores else 0.0,
            "avg_score": float(sum(scores) / len(scores)) if scores else 0.0
        }
        
        logger.info(f"Query processed in {query_time:.2f}s, retrieved {len(results)} chunks")
        
        return QueryResponse(
            answer=answer_data.get("answer", "No answer generated"),
            citations=answer_data.get("citations", citations),
            confidence=float(answer_data.get("llm_confidence", 50)),
            snippets=snippets_data,
            query_time=query_time,
            retrieval_stats=retrieval_stats
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during query processing"
        )


# Helper functions
def cleanup_temp_file(file_path: str) -> None:
    """Clean up temporary file."""
    try:
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)
            logger.debug(f"Cleaned up temp file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to clean up temp file {file_path}: {e}")


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    for unit in ['bytes', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


async def generate_answer_http(query: str, snippets: str) -> Dict[str, Any]:
    """Generate answer using HTTP LLM service."""
    if not llm_client:
        raise RuntimeError("HTTP client not initialized")
    
    prompt = f"{QUERY_SYSTEM}\n\nQuery: {query}\n\nSnippets:\n{snippets}"
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.1,
            "do_sample": True,
            "return_full_text": False
        }
    }
    
    response = await llm_client.post(
        f"{LLM_API_URL}/generate",
        json=payload,
        timeout=30.0
    )
    
    if response.status_code != 200:
        raise RuntimeError(f"LLM service returned {response.status_code}: {response.text}")
    
    result = response.json()
    generated_text = result.get("generated_text", "")
    
    # Try to parse JSON response from LLM
    try:
        return json.loads(generated_text)
    except json.JSONDecodeError:
        # Fallback to simple text response
        return {
            "answer": generated_text,
            "citations": [],
            "llm_confidence": 50
        }


def generate_answer_local(query: str, snippets: str) -> Dict[str, Any]:
    """Generate answer using local transformers model."""
    # Placeholder for local model implementation
    # This would use transformers library with a local model
    logger.warning("Local LLM generation not implemented, using fallback")
    
    return {
        "answer": "Local LLM generation is not yet implemented. Please use HTTP backend.",
        "citations": [],
        "llm_confidence": 0
    }


# Error handlers
@app.exception_handler(httpx.RequestError)
async def http_request_error_handler(request, exc):
    logger.error(f"HTTP request error: {exc}")
    return JSONResponse(
        status_code=503,
        content={"detail": "External service unavailable"}
    )


@app.exception_handler(httpx.TimeoutException)
async def http_timeout_error_handler(request, exc):
    logger.error(f"HTTP timeout error: {exc}")
    return JSONResponse(
        status_code=504,
        content={"detail": "Request timeout"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )