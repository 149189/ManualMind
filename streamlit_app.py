# streamlit_app.py
import streamlit as st
import requests
import json
import time
from typing import Dict, List, Optional
import os
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:8000"
API_SECRET_KEY = os.getenv("API_SECRET_KEY", "your-super-secret-key-change-in-production")

# Initialize session state
if "api_key" not in st.session_state:
    st.session_state.api_key = API_SECRET_KEY
if "health_status" not in st.session_state:
    st.session_state.health_status = {}
if "index_stats" not in st.session_state:
    st.session_state.index_stats = {}
if "backend_available" not in st.session_state:
    st.session_state.backend_available = False
if "connection_checked" not in st.session_state:
    st.session_state.connection_checked = False

def check_backend_connection():
    """Check if backend API is available"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        st.session_state.backend_available = response.status_code == 200
        if st.session_state.backend_available:
            st.session_state.health_status = response.json()
        st.session_state.connection_checked = True
        return st.session_state.backend_available
    except requests.exceptions.RequestException:
        st.session_state.backend_available = False
        st.session_state.connection_checked = True
        return False

def make_authenticated_request(method, endpoint, **kwargs):
    """Make an authenticated API request"""
    if not st.session_state.backend_available:
        st.error("Backend API is not available. Please check if Docker containers are running.")
        return None
        
    headers = kwargs.get("headers", {})
    headers["Authorization"] = f"Bearer {st.session_state.api_key}"
    kwargs["headers"] = headers
    
    try:
        if method == "get":
            response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=30, **kwargs)
        elif method == "post":
            response = requests.post(f"{API_BASE_URL}{endpoint}", timeout=30, **kwargs)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        # Re-check connection status
        check_backend_connection()
        return None

def check_health():
    """Check API health status"""
    return check_backend_connection()

def get_index_stats():
    """Get index statistics"""
    stats = make_authenticated_request("get", "/stats")
    if stats:
        st.session_state.index_stats = stats
    return stats

def ingest_document(file):
    """Ingest a PDF document"""
    if not st.session_state.backend_available:
        st.error("Backend API is not available. Please check if Docker containers are running.")
        return None
        
    files = {"file": (file.name, file.getvalue(), "application/pdf")}
    
    with st.spinner("Uploading and processing document..."):
        try:
            response = requests.post(
                f"{API_BASE_URL}/ingest",
                files=files,
                headers={"Authorization": f"Bearer {st.session_state.api_key}"},
                timeout=120  # Longer timeout for file uploads
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Upload failed: {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            st.error(f"Upload failed: {e}")
            check_backend_connection()
            return None

def query_knowledge_base(query, top_k=5, min_score=0.0):
    """Query the knowledge base"""
    payload = {
        "q": query,
        "top_k": top_k,
        "min_score": min_score
    }
    
    return make_authenticated_request("post", "/query", json=payload)

def main():
    st.set_page_config(
        page_title="ManualMind - RAG for Product Manuals",
        page_icon="📚",
        layout="wide"
    )
    
    # Check backend connection on app load
    if not st.session_state.connection_checked:
        check_backend_connection()
    
    # Sidebar for configuration
    with st.sidebar:
        st.title("ManualMind Configuration")
        
        # Connection status
        if st.session_state.backend_available:
            st.success("✅ Backend API connected")
        else:
            st.error("❌ Backend API not available")
            if st.button("Retry Connection"):
                check_backend_connection()
                st.rerun()
        
        # API Key input
        api_key = st.text_input(
            "API Secret Key", 
            value=st.session_state.api_key,
            type="password",
            help="Enter your API secret key for authentication"
        )
        if api_key != st.session_state.api_key:
            st.session_state.api_key = api_key
            st.rerun()
        
        # Health check
        st.divider()
        st.subheader("System Status")
        if st.button("Check Health Status"):
            check_backend_connection()
        
        if st.session_state.health_status:
            status = st.session_state.health_status.get("status", "unknown")
            color = "green" if status == "healthy" else "red" if status == "unhealthy" else "orange"
            st.markdown(f"**Overall Status:** :{color}[{status.upper()}]")
            
            for service, status in st.session_state.health_status.get("services", {}).items():
                color = "green" if status == "healthy" else "red" if status == "unavailable" else "orange"
                st.markdown(f"- **{service}:** :{color}[{status}]")
        
        # Index stats
        st.divider()
        st.subheader("Index Statistics")
        if st.button("Refresh Stats"):
            get_index_stats()
        
        if st.session_state.index_stats:
            stats = st.session_state.index_stats
            st.metric("Total Chunks", stats.get("total_chunks", 0))
            st.metric("Total Documents", stats.get("total_documents", 0))
            st.metric("Index Size", stats.get("index_size", "0 bytes"))
    
    # Main content area
    st.title("📚 ManualMind - RAG for Product Manuals")
    
    # Show warning if backend is not available
    if not st.session_state.backend_available:
        st.error("""
        ## Backend API is not available
        
        Please ensure:
        1. Docker is running
        2. The backend containers are started: `docker-compose up -d`
        3. The API is accessible at: `http://localhost:8000`
        
        Check the Docker logs for errors: `docker-compose logs manualmind-backend`
        """)
        
        if st.button("Check Connection Again"):
            check_backend_connection()
            st.rerun()
            
        return
    
    st.caption("Ask questions about your product manuals using AI-powered search")
    
    # Tab interface
    tab1, tab2, tab3 = st.tabs(["Query", "Ingest Documents", "API Documentation"])
    
    # Query tab
    with tab1:
        st.header("Query Knowledge Base")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input(
                "Enter your question",
                placeholder="E.g., How do I troubleshoot my device?",
                help="Ask a question about your product manuals"
            )
        
        with col2:
            top_k = st.slider("Top K", min_value=1, max_value=20, value=5, 
                             help="Number of results to retrieve")
        
        min_score = st.slider("Minimum Score", min_value=0.0, max_value=1.0, 
                             value=0.0, step=0.1,
                             help="Minimum similarity score for results")
        
        if st.button("Search", type="primary") and query:
            with st.spinner("Searching for answers..."):
                result = query_knowledge_base(query, top_k, min_score)
            
            if result:
                st.subheader("Answer")
                st.write(result.get("answer", "No answer found."))
                
                # Display confidence
                confidence = result.get("confidence", 0)
                st.progress(confidence / 100, text=f"Confidence: {confidence:.1f}%")
                
                # Display snippets
                if result.get("snippets"):
                    st.subheader("Source Materials")
                    for snippet in result.get("snippets", []):
                        with st.expander(f"Source: {snippet.get('source', 'Unknown')} - Page {snippet.get('page', '?')} (Score: {snippet.get('score', 0):.3f})"):
                            st.write(snippet.get("text", ""))
                            st.caption(f"Chunk ID: {snippet.get('chunk_id', 'N/A')}")
                
                # Display metrics
                if result.get("query_time"):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Query Time", f"{result.get('query_time', 0):.2f}s")
                    
                    if result.get("retrieval_stats"):
                        stats = result.get("retrieval_stats", {})
                        col2.metric("Retrieved Chunks", stats.get("total_retrieved", 0))
                        col3.metric("Max Score", f"{stats.get('max_score', 0):.3f}")
            else:
                st.error("Failed to get a response from the API. Please check your connection and try again.")
    
    # Ingest tab
    with tab2:
        st.header("Ingest New Documents")
        
        uploaded_file = st.file_uploader(
            "Upload a PDF document",
            type="pdf",
            help="Select a PDF file to add to the knowledge base"
        )
        
        if uploaded_file is not None:
            st.write("**File details:**")
            st.write(f"Name: {uploaded_file.name}")
            st.write(f"Size: {uploaded_file.size} bytes")
            
            if st.button("Process Document", type="primary"):
                result = ingest_document(uploaded_file)
                
                if result:
                    st.success(f"Successfully processed {uploaded_file.name}")
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Chunks Ingested", result.get("ingested_chunks", 0))
                    col2.metric("Processing Time", f"{result.get("processing_time", 0):.2f}s")
                    
                    if result.get("file_info"):
                        file_info = result.get("file_info", {})
                        col3.metric("Pages", file_info.get("page_count", "Unknown"))
                    
                    # Show warnings if any
                    if result.get("warnings"):
                        st.warning("Processing completed with warnings:")
                        for warning in result.get("warnings", []):
                            st.write(f"- {warning}")
                    
                    # Refresh stats
                    get_index_stats()
                else:
                    st.error("Failed to process the document. Please check your connection and try again.")
    
    # API Documentation tab
    with tab3:
        st.header("API Documentation")
        
        st.markdown("""
        ### ManualMind API Endpoints
        
        #### Health Check
        **GET** `/health`
        
        Returns the health status of all services.
        
        **Example Response:**
        ```json
        {
          "status": "healthy",
          "services": {
            "api": "healthy",
            "embedder": "healthy",
            "store": "healthy",
            "llm": "healthy"
          },
          "version": "1.0.0"
        }
        ```
        
        #### Index Statistics
        **GET** `/stats`
        
        Returns statistics about the FAISS index.
        
        **Headers:**
        - `Authorization: Bearer <API_SECRET_KEY>`
        
        **Example Response:**
        ```json
        {
          "total_chunks": 150,
          "total_documents": 5,
          "index_size": "2.1 MB",
          "last_updated": "2023-10-15T14:30:00Z"
        }
        ```
        
        #### Document Ingestion
        **POST** `/ingest`
        
        Uploads and processes a PDF document.
        
        **Headers:**
        - `Authorization: Bearer <API_SECRET_KEY>`
        
        **Body:** Form-data with file upload
        
        **Example Response:**
        ```json
        {
          "ingested_chunks": 42,
          "message": "Successfully indexed 42 chunks from manual.pdf",
          "file_info": {
            "filename": "manual.pdf",
            "page_count": 10,
            "title": "Product Manual"
          },
          "processing_time": 15.23,
          "warnings": null
        }
        ```
        
        #### Query Knowledge Base
        **POST** `/query`
        
        Queries the knowledge base with a question.
        
        **Headers:**
        - `Authorization: Bearer <API_SECRET_KEY>`
        - `Content-Type: application/json`
        
        **Body:**
        ```json
        {
          "q": "How do I troubleshoot my device?",
          "top_k": 5,
          "min_score": 0.5
        }
        ```
        
        **Example Response:**
        ```json
        {
          "answer": "To troubleshoot your device, first ensure it's properly powered on...",
          "citations": ["S1", "S2"],
          "confidence": 85.5,
          "snippets": [
            {
              "id": "S1",
              "text": "Troubleshooting: If your device is not working properly...",
              "source": "manual.pdf",
              "page": 15,
              "score": 0.87,
              "chunk_id": "manual.pdf_p15_c1"
            }
          ],
          "query_time": 1.23,
          "retrieval_stats": {
            "total_retrieved": 3,
            "min_score": 0.65,
            "max_score": 0.87,
            "avg_score": 0.76
          }
        }
        ```
        """)

if __name__ == "__main__":
    main()