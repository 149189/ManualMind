# rag-manuals/api/main.py
import os
import logging
import tempfile
import json
import asyncio
import traceback
from pathlib import Path
from typing import List, Dict, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import httpx

from ingestion.pdf_to_text import extract_pages, validate_pdf_file, get_pdf_info
from ingestion.chunker import chunk_page, validate_chunks
from ingestion.cleaner import clean_pages_batch
from embeddings.embedder import Embedder
from index.faiss_utils import FaissStore
from api.prompt_templates import QUERY_SYSTEM, format_retrieved_snippets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/manualmind.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Configuration from environment variables
LLM_BACKEND = os.environ.get("LLM_BACKEND", "local")
LLM_API_URL = os.environ.get("LLM_API_URL", "http://localhost:8080")
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "mistral-7b")
MAX_FILE_SIZE = int(os.environ.get("MAX_FILE_SIZE", 50 * 1024 * 1024))  # 50MB default
INDEX_PATH = os.environ.get("INDEX_PATH", "index/faiss.index")
META_PATH = os.environ.get("META_PATH", "index/meta.jsonl")

# Global variables for services
store: Optional[FaissStore] = None
embedder: Optional[Embedder] = None
llm_client: Optional[httpx.AsyncClient] = None

# Custom exceptions
class ServiceUnavailableError(Exception):
    """Raised when a required service is unavailable."""
    pass

class ProcessingError(Exception):
    """Raised when document processing fails."""
    pass

class EmbeddingError(Exception):
    """Raised when embedding generation fails."""
    pass


@asynccontextmanager
async def lifespan(app: FastAPI):                                   
    """Manage application lifecycle - startup and shutdown."""
    global store, embedder, llm_client
    
    try:
        # Startup
        logger.info("Starting ManualMind API...")
        
        # Create necessary directories
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs("uploads", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        
        # Initialize FAISS store
        try:
            store = FaissStore(index_path=INDEX_PATH, meta_path=META_PATH)
            logger.info(f"Initialized FAISS store with paths: {INDEX_PATH}, {META_PATH}")
        except Exception as e:
            logger.error(f"Failed to initialize FAISS store: {e}")
            raise ServiceUnavailableError(f"FAISS store initialization failed: {e}")
        
        # Initialize embedder
        try:
            embedder = Embedder()
            logger.info("Initialized embedder")
        except Exception as e:
            logger.error(f"Failed to initialize embedder: {e}")
            raise ServiceUnavailableError(f"Embedder initialization failed: {e}")
        
        # Initialize HTTP client for LLM communication
        if LLM_BACKEND == "http":
            try:
                llm_client = httpx.AsyncClient(
                    timeout=httpx.Timeout(60.0),
                    limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
                )
                
                # Test LLM connection
                response = await llm_client.get(f"{LLM_API_URL}/health")
                if response.status_code == 200:
                    logger.info(f"Successfully connected to LLM at {LLM_API_URL}")
                else:
                    logger.warning(f"LLM health check returned {response.status_code}")
            except Exception as e:
                logger.warning(f"Failed to connect to LLM (will continue without): {e}")
                # Don't raise here - we can still work without LLM for ingestion
        
        logger.info("ManualMind API started successfully")
        yield
        
    except ServiceUnavailableError:
        logger.error("Critical service initialization failed")
        raise
    except Exception as e:
        logger.error(f"Unexpected startup error: {e}")
        logger.error(traceback.format_exc())
        raise
    
    finally:
        # Shutdown
        logger.info("Shutting down ManualMind API...")
        if llm_client:
            try:
                await llm_client.aclose()
                logger.info("HTTP client closed successfully")
            except Exception as e:
                logger.error(f"Error closing HTTP client: {e}")
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
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class IngestResponse(BaseModel):
    """Response model for document ingestion."""
    success: bool = Field(..., description="Whether ingestion was successful")
    ingested_chunks: int = Field(..., description="Number of chunks successfully ingested")
    message: str = Field(..., description="Status message")
    file_info: Optional[Dict[str, Any]] = Field(None, description="Information about the processed file")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    warnings: Optional[List[str]] = Field(None, description="Any warnings during processing")
    errors: Optional[List[str]] = Field(None, description="Any errors during processing")


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
    success: bool = Field(..., description="Whether query was successful")
    answer: str = Field(..., description="Generated answer")
    citations: List[str] = Field(..., description="List of citation identifiers")
    confidence: float = Field(..., ge=0.0, le=100.0, description="Confidence score (0-100)")
    snippets: List[Dict[str, Any]] = Field(..., description="Retrieved text snippets")
    query_time: Optional[float] = Field(None, description="Query processing time in seconds")
    retrieval_stats: Optional[Dict[str, Any]] = Field(None, description="Retrieval statistics")
    errors: Optional[List[str]] = Field(None, description="Any errors during processing")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status")
    services: Dict[str, str] = Field(..., description="Status of individual services")
    version: str = Field(..., description="API version")
    uptime: Optional[str] = Field(None, description="Service uptime")


class IndexStats(BaseModel):
    """Response model for index statistics."""
    total_chunks: int = Field(..., description="Total number of chunks in index")
    total_documents: int = Field(..., description="Total number of documents")
    index_size: str = Field(..., description="Index file size")
    last_updated: Optional[str] = Field(None, description="Last update timestamp")
    index_health: str = Field(..., description="Index health status")


# Dependency functions
async def validate_file_upload(file: UploadFile) -> UploadFile:
    """Validate uploaded file with comprehensive checks."""
    try:
        # Check if file exists
        if not file:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No file provided"
            )
        
        # Check filename
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Filename is required"
            )
        
        # Check file extension
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files are supported"
            )
        
        # Check file size
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size ({len(content)} bytes) exceeds maximum allowed size ({MAX_FILE_SIZE} bytes)"
            )
        
        if len(content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File is empty"
            )
        
        # Reset file pointer
        await file.seek(0)
        return file
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File validation error: {str(e)}"
        )


def validate_services() -> None:
    """Validate that required services are available."""
    if not store:
        raise ServiceUnavailableError("FAISS store not available")
    if not embedder:
        raise ServiceUnavailableError("Embedder not available")


# API endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check endpoint."""
    try:
        services = {}
        
        # Check API service
        services["api"] = "healthy"
        
        # Check embedder
        try:
            if embedder:
                # Test embedding generation
                test_embedding = embedder.embed(["test"])
                services["embedder"] = "healthy" if len(test_embedding) > 0 else "degraded"
            else:
                services["embedder"] = "unavailable"
        except Exception as e:
            logger.error(f"Embedder health check failed: {e}")
            services["embedder"] = "unhealthy"
        
        # Check FAISS store
        try:
            if store:
                stats = store.get_stats()
                services["store"] = "healthy" if stats else "degraded"
            else:
                services["store"] = "unavailable"
        except Exception as e:
            logger.error(f"Store health check failed: {e}")
            services["store"] = "unhealthy"
        
        # Check LLM service
        if LLM_BACKEND == "http" and llm_client:
            try:
                response = await llm_client.get(f"{LLM_API_URL}/health", timeout=5.0)
                services["llm"] = "healthy" if response.status_code == 200 else "degraded"
            except httpx.TimeoutException:
                services["llm"] = "timeout"
            except httpx.ConnectError:
                services["llm"] = "unreachable"
            except Exception as e:
                logger.error(f"LLM health check failed: {e}")
                services["llm"] = "unhealthy"
        else:
            services["llm"] = "local" if LLM_BACKEND == "local" else "not_configured"
        
        # Determine overall status
        critical_services = ["api", "embedder", "store"]
        overall_status = "healthy"
        
        for service in critical_services:
            if services.get(service) in ["unavailable", "unhealthy"]:
                overall_status = "unhealthy"
                break
            elif services.get(service) == "degraded":
                overall_status = "degraded"
        
        return HealthResponse(
            status=overall_status,
            services=services,
            version="1.0.0",
            uptime="runtime"  # Could implement actual uptime tracking
        )
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        logger.error(traceback.format_exc())
        return HealthResponse(
            status="unhealthy",
            services={"error": f"Health check failed: {str(e)}"},
            version="1.0.0"
        )


@app.get("/stats", response_model=IndexStats)
async def get_index_stats():
    """Get comprehensive index statistics."""
    try:
        validate_services()
        
        # Try to load index if needed
        try:
            if store.index is None:
                load_success = store.load()
                if not load_success:
                    return IndexStats(
                        total_chunks=0,
                        total_documents=0,
                        index_size="0 bytes",
                        last_updated=None,
                        index_health="empty"
                    )
        except FileNotFoundError:
            logger.info("No existing index found")
            return IndexStats(
                total_chunks=0,
                total_documents=0,
                index_size="0 bytes",
                last_updated=None,
                index_health="not_created"
            )
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return IndexStats(
                total_chunks=0,
                total_documents=0,
                index_size="0 bytes",
                last_updated=None,
                index_health="corrupted"
            )
        
        # Get index statistics
        stats = store.get_stats()
        total_chunks = stats.get("total_vectors", 0)
        
        # Count unique documents
        unique_docs = set()
        for meta in store.metadata:
            if 'source' in meta:
                unique_docs.add(meta['source'])
        
        # Get index file size
        index_size = "0 bytes"
        last_updated = None
        try:
            index_path = Path(INDEX_PATH)
            if index_path.exists():
                size_bytes = index_path.stat().st_size
                index_size = format_file_size(size_bytes)
                last_updated = index_path.stat().st_mtime
        except Exception as e:
            logger.warning(f"Could not get index file info: {e}")
        
        # Determine index health
        index_health = "healthy"
        if total_chunks == 0:
            index_health = "empty"
        elif stats.get("metadata_count", 0) != total_chunks:
            index_health = "inconsistent"
        
        return IndexStats(
            total_chunks=total_chunks,
            total_documents=len(unique_docs),
            index_size=index_size,
            last_updated=str(last_updated) if last_updated else None,
            index_health=index_health
        )
    
    except ServiceUnavailableError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error getting index stats: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get index statistics: {str(e)}"
        )


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Ingest a PDF document into the knowledge base with comprehensive error handling."""
    import time
    start_time = time.time()
    
    warnings = []
    errors = []
    tmp_path = None
    chunks_added = 0
    
    try:
        # Validate file upload
        try:
            file = await validate_file_upload(file)
        except HTTPException as e:
            return IngestResponse(
                success=False,
                ingested_chunks=0,
                message=f"File validation failed: {e.detail}",
                errors=[e.detail]
            )
        
        # Validate services
        try:
            validate_services()
        except ServiceUnavailableError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(e)
            )
        
        logger.info(f"Starting ingestion of file: {file.filename}")
        
        # Save uploaded file temporarily
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name
        except Exception as e:
            error_msg = f"Failed to save temporary file: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
            return IngestResponse(
                success=False,
                ingested_chunks=0,
                message="Failed to save uploaded file",
                errors=errors
            )
        
        # Validate PDF file
        try:
            if not validate_pdf_file(tmp_path):
                error_msg = "Invalid or corrupted PDF file"
                errors.append(error_msg)
                return IngestResponse(
                    success=False,
                    ingested_chunks=0,
                    message=error_msg,
                    errors=errors
                )
        except Exception as e:
            error_msg = f"PDF validation failed: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
            return IngestResponse(
                success=False,
                ingested_chunks=0,
                message="PDF validation failed",
                errors=errors
            )
        
        # Get file info
        file_info = None
        try:
            file_info = get_pdf_info(tmp_path)
            if not file_info:
                warnings.append("Could not extract PDF metadata")
                file_info = {"filename": file.filename, "page_count": 0}
        except Exception as e:
            warning_msg = f"Failed to extract PDF metadata: {str(e)}"
            warnings.append(warning_msg)
            logger.warning(warning_msg)
            file_info = {"filename": file.filename, "page_count": 0}
        
        # Extract pages
        pages = []
        try:
            pages = extract_pages(tmp_path, include_metadata=True)
            if not pages:
                error_msg = "No text could be extracted from the PDF"
                errors.append(error_msg)
                return IngestResponse(
                    success=False,
                    ingested_chunks=0,
                    message=error_msg,
                    file_info=file_info,
                    errors=errors
                )
            logger.info(f"Extracted {len(pages)} pages")
        except Exception as e:
            error_msg = f"Failed to extract text from PDF: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return IngestResponse(
                success=False,
                ingested_chunks=0,
                message="PDF text extraction failed",
                file_info=file_info,
                errors=errors
            )
        
        # Clean pages (remove headers/footers)
        try:
            cleaned_pages = clean_pages_batch(pages, remove_headers=True)
            if len(cleaned_pages) != len(pages):
                warnings.append("Some pages could not be cleaned properly")
            pages = cleaned_pages
            logger.info(f"Cleaned {len(cleaned_pages)} pages")
        except Exception as e:
            warning_msg = f"Page cleaning failed, using original pages: {str(e)}"
            warnings.append(warning_msg)
            logger.warning(warning_msg)
        
        # Chunk pages
        all_chunks = []
        failed_pages = 0
        try:
            for page in pages:
                try:
                    page_chunks = chunk_page(page)
                    all_chunks.extend(page_chunks)
                except Exception as e:
                    failed_pages += 1
                    warning_msg = f"Failed to chunk page {page.get('page', 'unknown')}: {str(e)}"
                    warnings.append(warning_msg)
                    logger.warning(warning_msg)
            
            if failed_pages > 0:
                warnings.append(f"Failed to chunk {failed_pages} pages")
            
            logger.info(f"Created {len(all_chunks)} chunks from {len(pages) - failed_pages} pages")
        except Exception as e:
            error_msg = f"Chunking process failed: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
            return IngestResponse(
                success=False,
                ingested_chunks=0,
                message="Document chunking failed",
                file_info=file_info,
                warnings=warnings,
                errors=errors
            )
        
        if not all_chunks:
            error_msg = "No valid chunks could be created from the document"
            errors.append(error_msg)
            return IngestResponse(
                success=False,
                ingested_chunks=0,
                message=error_msg,
                file_info=file_info,
                warnings=warnings,
                errors=errors
            )
        
        # Validate chunks
        try:
            valid_chunks = validate_chunks(all_chunks)
            if len(valid_chunks) != len(all_chunks):
                warning_msg = f"Filtered out {len(all_chunks) - len(valid_chunks)} invalid chunks"
                warnings.append(warning_msg)
                logger.warning(warning_msg)
            all_chunks = valid_chunks
        except Exception as e:
            error_msg = f"Chunk validation failed: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
            return IngestResponse(
                success=False,
                ingested_chunks=0,
                message="Chunk validation failed",
                file_info=file_info,
                warnings=warnings,
                errors=errors
            )
        
        if not all_chunks:
            error_msg = "No valid chunks remaining after validation"
            errors.append(error_msg)
            return IngestResponse(
                success=False,
                ingested_chunks=0,
                message=error_msg,
                file_info=file_info,
                warnings=warnings,
                errors=errors
            )
        
        # Extract texts for embedding
        texts = [chunk["text"] for chunk in all_chunks]
        
        # Generate embeddings and build index
        try:
            logger.info(f"Generating embeddings for {len(texts)} chunks...")
            embeddings = embedder.embed(texts)
            
            if embeddings.size == 0:
                raise EmbeddingError("No embeddings generated")
            
            logger.info("Building search index...")
            build_success = store.build(embeddings=embeddings, metadata=all_chunks)
            
            if not build_success:
                raise ProcessingError("Failed to build search index")
            
            chunks_added = len(all_chunks)
            
        except EmbeddingError as e:
            error_msg = f"Embedding generation failed: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
            return IngestResponse(
                success=False,
                ingested_chunks=0,
                message="Failed to generate embeddings",
                file_info=file_info,
                warnings=warnings,
                errors=errors
            )
        except ProcessingError as e:
            error_msg = f"Index building failed: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
            return IngestResponse(
                success=False,
                ingested_chunks=0,
                message="Failed to build search index",
                file_info=file_info,
                warnings=warnings,
                errors=errors
            )
        except Exception as e:
            error_msg = f"Unexpected error during embedding/indexing: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return IngestResponse(
                success=False,
                ingested_chunks=0,
                message="Embedding or indexing failed",
                file_info=file_info,
                warnings=warnings,
                errors=errors
            )
        
        processing_time = time.time() - start_time
        
        # Clean up temp file in background
        background_tasks.add_task(cleanup_temp_file, tmp_path)
        
        logger.info(f"Successfully ingested {chunks_added} chunks in {processing_time:.2f}s")
        
        return IngestResponse(
            success=True,
            ingested_chunks=chunks_added,
            message=f"Successfully indexed {chunks_added} chunks from {file.filename}",
            file_info=file_info,
            processing_time=processing_time,
            warnings=warnings if warnings else None,
            errors=None
        )
    
    except HTTPException:
        # Clean up temp file on HTTP error
        if tmp_path:
            background_tasks.add_task(cleanup_temp_file, tmp_path)
        raise
    except ServiceUnavailableError as e:
        # Clean up temp file on service error
        if tmp_path:
            background_tasks.add_task(cleanup_temp_file, tmp_path)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except Exception as e:
        # Clean up temp file on unexpected error
        if tmp_path:
            background_tasks.add_task(cleanup_temp_file, tmp_path)
        error_msg = f"Unexpected error during ingestion: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        return IngestResponse(
            success=False,
            ingested_chunks=0,
            message="Internal server error during ingestion",
            errors=[error_msg]
        )


@app.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """Query the knowledge base with comprehensive error handling."""
    import time
    start_time = time.time()
    errors = []
    
    try:
        # Validate services
        try:
            validate_services()
        except ServiceUnavailableError as e:
            return QueryResponse(
                success=False,
                answer="Service unavailable",
                citations=[],
                confidence=0.0,
                snippets=[],
                errors=[str(e)]
            )
        
        logger.info(f"Processing query: {request.q[:100]}...")
        
        # Load index if needed
        try:
            if store.index is None:
                load_success = store.load()
                if not load_success:
                    return QueryResponse(
                        success=False,
                        answer="No knowledge base found. Please ingest documents first.",
                        citations=[],
                        confidence=0.0,
                        snippets=[],
                        errors=["No index available"]
                    )
        except FileNotFoundError:
            return QueryResponse(
                success=False,
                answer="No knowledge base found. Please ingest documents first.",
                citations=[],
                confidence=0.0,
                snippets=[],
                errors=["Index file not found"]
            )
        except Exception as e:
            error_msg = f"Failed to load index: {str(e)}"
            logger.error(error_msg)
            return QueryResponse(
                success=False,
                answer="Failed to load knowledge base",
                citations=[],
                confidence=0.0,
                snippets=[],
                errors=[error_msg]
            )
        
        # Generate query embedding
        try:
            query_embedding = embedder.embed([request.q])[0]
        except Exception as e:
            error_msg = f"Failed to embed query: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
            return QueryResponse(
                success=False,
                answer="Failed to process query",
                citations=[],
                confidence=0.0,
                snippets=[],
                errors=errors
            )
        
        # Search for relevant chunks
        results = []
        try:
            results = store.search(query_embedding, top_k=request.top_k)
            
            # Filter by minimum score if specified
            if request.min_score and request.min_score > 0:
                original_count = len(results)
                results = [(score, meta) for score, meta in results if score >= request.min_score]
                if len(results) < original_count:
                    logger.info(f"Filtered {original_count - len(results)} results below min_score {request.min_score}")
            
            if not results:
                return QueryResponse(
                    success=True,
                    answer="I don't know based on the provided manuals. No relevant information found.",
                    citations=[],
                    confidence=0.0,
                    snippets=[],
                    query_time=time.time() - start_time,
                    retrieval_stats={
                        "total_retrieved": 0,
                        "min_score": 0.0,
                        "max_score": 0.0,
                        "avg_score": 0.0
                    }
                )
        
        except Exception as e:
            error_msg = f"Search failed: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return QueryResponse(
                success=False,
                answer="Search failed",
                citations=[],
                confidence=0.0,
                snippets=[],
                errors=errors
            )
        
        # Format retrieved snippets
        try:
            formatted_snippets = format_retrieved_snippets(results)
        except Exception as e:
            error_msg = f"Failed to format snippets: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
            # Continue with empty snippets
            formatted_snippets = ""
        
        # Generate answer using LLM
        answer_data = None
        try:
            if LLM_BACKEND == "http":
                answer_data = await generate_answer_http(request.q, formatted_snippets)
            else:
                answer_data = generate_answer_local(request.q, formatted_snippets)
        
        except Exception as e:
            error_msg = f"Answer generation failed: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            # Fallback response with error info
            answer_data = {
                "answer": "I found relevant information but couldn't generate a proper response due to an error. Please check the snippets below for relevant content.",
                "citations": [f"S{i+1}" for i in range(len(results))],
                "llm_confidence": 25
            }
            errors.append(error_msg)
        
        # Prepare response data
        snippets_data = []
        citations = []
        
        try:
            for i, (score, meta) in enumerate(results, 1):
                snippet_id = f"S{i}"
                citations.append(snippet_id)
                snippets_data.append({
                    "id": snippet_id,
                    "text": meta.get("text", ""),
                    "source": meta.get("source", "unknown"),
                    "page": meta.get("page", "?"),
                    "score": float(score),
                    "chunk_id": meta.get("id", ""),
                    "chunk_index": meta.get("chunk_index", "?")
                })
        except Exception as e:
            error_msg = f"Failed to format response data: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
        
        query_time = time.time() - start_time
        
        # Calculate retrieval stats
        retrieval_stats = {"total_retrieved": 0, "min_score": 0.0, "max_score": 0.0, "avg_score": 0.0}
        try:
            scores = [score for score, _ in results]
            retrieval_stats = {
                "total_retrieved": len(results),
                "min_score": float(min(scores)) if scores else 0.0,
                "max_score": float(max(scores)) if scores else 0.0,
                "avg_score": float(sum(scores) / len(scores)) if scores else 0.0
            }
        except Exception as e:
            logger.error(f"Failed to calculate retrieval stats: {e}")
        
        logger.info(f"Query processed in {query_time:.2f}s, retrieved {len(results)} chunks")
        
        return QueryResponse(
            success=True,
            answer=answer_data.get("answer", "No answer generated"),
            citations=answer_data.get("citations", citations),
            confidence=float(answer_data.get("llm_confidence", 50)),
            snippets=snippets_data,
            query_time=query_time,
            retrieval_stats=retrieval_stats,
            errors=errors if errors else None
        )
    
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Unexpected error during query: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        return QueryResponse(
            success=False,
            answer="Internal server error during query processing",
            citations=[],
            confidence=0.0,
            snippets=[],
            query_time=time.time() - start_time,
            errors=[error_msg]
        )


@app.delete("/documents/{source}")
async def delete_document(source: str):
    """Delete a document from the knowledge base."""
    try:
        validate_services()
        
        # URL decode the source name
        import urllib.parse
        source = urllib.parse.unquote(source)
        
        removed_chunks = store.remove_document(source)
        
        if removed_chunks > 0:
            return {
                "success": True,
                "message": f"Removed {removed_chunks} chunks from document '{source}'",
                "removed_chunks": removed_chunks
            }
        else:
            return {
                "success": False,
                "message": f"Document '{source}' not found in knowledge base",
                "removed_chunks": 0
            }
    
    except ServiceUnavailableError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )


@app.get("/documents")
async def list_documents():
    """List all documents in the knowledge base."""
    try:
        validate_services()
        
        # Load index if needed
        if store.index is None:
            try:
                store.load()
            except FileNotFoundError:
                return {"documents": [], "total": 0}
        
        # Get document statistics
        document_stats = {}
        for meta in store.metadata:
            source = meta.get('source', 'unknown')
            if source not in document_stats:
                document_stats[source] = {
                    "source": source,
                    "chunk_count": 0,
                    "pages": set()
                }
            
            document_stats[source]["chunk_count"] += 1
            if 'page' in meta:
                document_stats[source]["pages"].add(meta['page'])
        
        # Convert to list and format
        documents = []
        for source, stats in document_stats.items():
            documents.append({
                "source": source,
                "chunk_count": stats["chunk_count"],
                "page_count": len(stats["pages"])
            })
        
        return {
            "documents": documents,
            "total": len(documents)
        }
    
    except ServiceUnavailableError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {str(e)}"
        )


# Helper functions
def cleanup_temp_file(file_path: str) -> None:
    """Clean up temporary file with error handling."""
    try:
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)
            logger.debug(f"Cleaned up temp file: {file_path}")
    except PermissionError:
        logger.warning(f"Permission denied when cleaning up temp file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to clean up temp file {file_path}: {e}")


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    try:
        for unit in ['bytes', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    except Exception:
        return "unknown size"


async def generate_answer_http(query: str, snippets: str) -> Dict[str, Any]:
    """Generate answer using HTTP LLM API with improved error handling."""
    if not llm_client:
        raise RuntimeError("HTTP client not initialized")
    
    try:
        prompt = f"{QUERY_SYSTEM}\n\nQuery: {query}\n\nSnippets:\n{snippets}"
        
        # Ollama API format
        payload = {
            "model": LLM_MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 512
            }
        }
        
        response = await llm_client.post(
            f"{LLM_API_URL}/api/generate",
            json=payload,
            timeout=120.0
        )
        
        if response.status_code == 404:
            raise RuntimeError(f"LLM model '{LLM_MODEL_NAME}' not found on server")
        elif response.status_code == 500:
            raise RuntimeError("LLM server internal error")
        elif response.status_code != 200:
            raise RuntimeError(f"LLM service returned {response.status_code}: {response.text}")
        
        try:
            result = response.json()
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON response from LLM: {e}")
        
        generated_text = result.get("response", "")
        
        if not generated_text.strip():
            raise RuntimeError("LLM returned empty response")
        
        # Try to parse JSON response from LLM
        try:
            parsed_response = json.loads(generated_text)
            # Validate required fields
            if "answer" not in parsed_response:
                raise ValueError("Missing 'answer' field in LLM response")
            return parsed_response
        except json.JSONDecodeError:
            # Fallback to simple text response
            logger.warning("LLM response was not valid JSON, using as plain text")
            return {
                "answer": generated_text,
                "citations": [],
                "llm_confidence": 50
            }
            
    except httpx.TimeoutException:
        logger.error("LLM request timeout")
        raise RuntimeError("Language model request timed out")
    except httpx.ConnectError:
        logger.error("Cannot connect to LLM service")
        raise RuntimeError("Cannot connect to language model service")
    except httpx.RequestError as e:
        logger.error(f"LLM request error: {e}")
        raise RuntimeError(f"Language model request failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected LLM error: {e}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Unexpected language model error: {str(e)}")


def generate_answer_local(query: str, snippets: str) -> Dict[str, Any]:
    """Generate answer using local transformers model."""
    try:
        # Placeholder for local model implementation
        logger.warning("Local LLM generation not implemented, providing fallback")
        
        # Simple fallback that extracts relevant snippets
        if snippets.strip():
            return {
                "answer": "Based on the provided manuals, here are the relevant excerpts. (Note: Local LLM not implemented - showing raw snippets)",
                "citations": ["S1", "S2", "S3"],
                "llm_confidence": 30
            }
        else:
            return {
                "answer": "I don't know based on the provided manuals. No relevant information found.",
                "citations": [],
                "llm_confidence": 0
            }
    except Exception as e:
        logger.error(f"Local LLM fallback failed: {e}")
        return {
            "answer": "Local language model is not available.",
            "citations": [],
            "llm_confidence": 0
        }


# Global exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    error_id = id(exc)
    logger.error(f"Unhandled exception [ID: {error_id}]: {exc}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": f"Internal server error occurred [Error ID: {error_id}]",
            "error_type": type(exc).__name__,
            "success": False
        }
    )


@app.exception_handler(httpx.RequestError)
async def http_request_error_handler(request, exc):
    """Handle HTTP request errors to external services."""
    logger.error(f"HTTP request error: {exc}")
    return JSONResponse(
        status_code=503,
        content={
            "detail": "External service unavailable",
            "error_type": "ServiceError",
            "success": False
        }
    )


@app.exception_handler(httpx.TimeoutException)
async def http_timeout_error_handler(request, exc):
    """Handle HTTP timeout errors."""
    logger.error(f"HTTP timeout error: {exc}")
    return JSONResponse(
        status_code=504,
        content={
            "detail": "Request timeout - external service took too long to respond",
            "error_type": "TimeoutError",
            "success": False
        }
    )


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request, exc):
    """Handle file not found errors."""
    logger.error(f"File not found: {exc}")
    return JSONResponse(
        status_code=404,
        content={
            "detail": "Required file not found",
            "error_type": "FileNotFoundError",
            "success": False
        }
    )


@app.exception_handler(PermissionError)
async def permission_error_handler(request, exc):
    """Handle permission errors."""
    logger.error(f"Permission error: {exc}")
    return JSONResponse(
        status_code=403,
        content={
            "detail": "Permission denied accessing required resources",
            "error_type": "PermissionError",
            "success": False
        }
    )


# Additional utility endpoints
@app.post("/reset-index")
async def reset_index():
    """Reset the entire index (useful for development/testing)."""
    try:
        validate_services()
        
        # Clear the in-memory index and metadata
        store.index = None
        store.metadata = []
        
        # Remove index files if they exist
        try:
            if os.path.exists(INDEX_PATH):
                os.remove(INDEX_PATH)
            if os.path.exists(META_PATH):
                os.remove(META_PATH)
        except Exception as e:
            logger.warning(f"Failed to remove index files: {e}")
        
        logger.info("Index reset successfully")
        
        return {
            "success": True,
            "message": "Index reset successfully"
        }
    
    except ServiceUnavailableError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error resetting index: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset index: {str(e)}"
        )


@app.get("/system-info")
async def get_system_info():
    """Get system information and configuration."""
    try:
        import torch
        import platform
        import psutil
        
        return {
            "system": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2)
            },
            "configuration": {
                "llm_backend": LLM_BACKEND,
                "llm_api_url": LLM_API_URL,
                "llm_model_name": LLM_MODEL_NAME,
                "max_file_size_mb": MAX_FILE_SIZE // (1024*1024),
                "index_path": INDEX_PATH,
                "meta_path": META_PATH
            },
            "torch": {
                "version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        }
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        return {
            "error": f"Failed to get system info: {str(e)}"
        }


if __name__ == "__main__":
    import uvicorn
    
    # Ensure log directory exists
    os.makedirs("logs", exist_ok=True)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )