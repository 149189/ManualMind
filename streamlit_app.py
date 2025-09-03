# streamlit_app.py
import streamlit as st
import requests
import json
import os
from pathlib import Path
from typing import Optional

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Initialize session state
if "health_status" not in st.session_state:
    st.session_state.health_status = {}
if "index_stats" not in st.session_state:
    st.session_state.index_stats = {}
if "backend_available" not in st.session_state:
    st.session_state.backend_available = False
if "connection_checked" not in st.session_state:
    st.session_state.connection_checked = False

def check_backend_connection():
    """Check if backend API is available (health endpoint usually unauthenticated)."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        st.session_state.backend_available = response.status_code == 200
        if st.session_state.backend_available:
            try:
                st.session_state.health_status = response.json()
            except Exception:
                st.session_state.health_status = {"status": "healthy"}
        else:
            st.session_state.health_status = {}
        st.session_state.connection_checked = True
        return st.session_state.backend_available
    except requests.exceptions.RequestException:
        st.session_state.backend_available = False
        st.session_state.connection_checked = True
        st.session_state.health_status = {}
        return False

def call_api(method: str, endpoint: str, **kwargs) -> Optional[dict]:
    """Call backend API without Authorization header. Returns JSON or None on error."""
    # Ensure we've checked connectivity
    if not st.session_state.connection_checked:
        check_backend_connection()
    if not st.session_state.backend_available:
        st.error("Backend API is not available. Please ensure the backend is running.")
        return None

    url = f"{API_BASE_URL}{endpoint}"
    try:
        if method.lower() == "get":
            resp = requests.get(url, timeout=30, **kwargs)
        elif method.lower() == "post":
            resp = requests.post(url, timeout=120, **kwargs)
        else:
            st.error(f"Unsupported HTTP method: {method}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        check_backend_connection()
        return None

    # handle common HTTP errors
    if resp.status_code >= 500:
        st.error(f"Server error ({resp.status_code}): {resp.text}")
        return None
    if resp.status_code >= 400:
        st.error(f"Request failed ({resp.status_code}): {resp.text}")
        return None

    try:
        return resp.json()
    except ValueError:
        # not JSON — return a text wrapper
        return {"text": resp.text, "status_code": resp.status_code}

def get_index_stats():
    s = call_api("get", "/stats")
    if s:
        st.session_state.index_stats = s
    return s

def ingest_document(file):
    """Upload PDF to /ingest (no auth)."""
    if not st.session_state.backend_available:
        st.error("Backend API is not available. Please ensure the backend is running.")
        return None

    files = {"file": (file.name, file.getvalue(), "application/pdf")}
    with st.spinner("Uploading and processing document..."):
        try:
            resp = requests.post(f"{API_BASE_URL}/ingest", files=files, timeout=120)
        except requests.exceptions.RequestException as e:
            st.error(f"Ingest failed: {e}")
            check_backend_connection()
            return None

    if resp.status_code >= 400:
        st.error(f"Ingest failed ({resp.status_code}): {resp.text}")
        return None

    try:
        return resp.json()
    except ValueError:
        st.error("Ingest returned non-JSON response.")
        return None

def query_knowledge_base(query: str, top_k: int = 5, min_score: float = 0.0):
    payload = {"q": query, "top_k": top_k, "min_score": min_score}
    return call_api("post", "/query", json=payload)

def main():
    st.set_page_config(page_title="ManualMind", page_icon="📚", layout="wide")

    if not st.session_state.connection_checked:
        check_backend_connection()

    with st.sidebar:
        st.title("ManualMind")
        if st.session_state.backend_available:
            st.success("Backend connected")
        else:
            st.error("Backend not available")
            if st.button("Retry connection"):
                check_backend_connection()
                st.experimental_rerun()

        st.divider()
        st.subheader("System Status")
        if st.button("Check Health"):
            check_backend_connection()
        if st.session_state.health_status:
            st.json(st.session_state.health_status)

        st.divider()
        st.subheader("Index Stats")
        if st.button("Refresh Stats"):
            get_index_stats()
        if st.session_state.index_stats:
            stats = st.session_state.index_stats
            st.metric("Total Chunks", stats.get("total_chunks", stats.get("total_vectors", 0)))
            st.metric("Total Documents", stats.get("total_documents", stats.get("unique_sources", 0)))
            idx_size = stats.get("index_size") or stats.get("index_file_size_bytes") or 0
            if isinstance(idx_size, (int, float)):
                st.metric("Index Size", f"{idx_size / (1024*1024):.2f} MB")
            else:
                st.metric("Index Size", str(idx_size))

    st.title("📚 ManualMind - RAG for Product Manuals")
    if not st.session_state.backend_available:
        st.error("Backend is not accessible. Ensure API is running at: " + API_BASE_URL)
        return

    tab1, tab2, tab3 = st.tabs(["Query", "Ingest", "Docs"])

    with tab1:
        st.header("Query Knowledge Base")
        query = st.text_input("Enter your question", "")
        top_k = st.slider("Top K", min_value=1, max_value=20, value=5)
        min_score = st.slider("Min Score", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
        if st.button("Search") and query:
            with st.spinner("Searching..."):
                res = query_knowledge_base(query, top_k, min_score)
            if not res:
                st.error("No result or error from server.")
            else:
                # answer
                st.subheader("Answer")
                st.write(res.get("answer", res.get("text", "No answer.")))
                # confidence
                conf = res.get("confidence", res.get("llm_confidence", 0))
                try:
                    cv = float(conf)
                except Exception:
                    cv = 0.0
                cv = max(0.0, min(100.0, cv))
                st.progress(cv / 100.0)
                st.write(f"Confidence: {cv:.1f}%")
                # snippets
                snippets = res.get("snippets", [])
                if snippets:
                    st.subheader("Source Snippets")
                    for s in snippets:
                        score = s.get("score", 0)
                        with st.expander(f"{s.get('source','unknown')} - page {s.get('page','?')} (score {score})"):
                            st.write(s.get("text",""))
                            st.caption(f"Chunk id: {s.get('chunk_id', s.get('id', 'N/A'))}")

    with tab2:
        st.header("Ingest Document")
        uploaded = st.file_uploader("Upload PDF", type="pdf")
        if uploaded is not None and st.button("Upload & Process"):
            with st.spinner("Ingesting..."):
                res = ingest_document(uploaded)
            if res:
                st.success(res.get("message", "Ingest succeeded"))
                st.write(res)
                # refresh stats
                get_index_stats()

    with tab3:
        st.header("API Documentation")
        st.markdown("""
        Endpoints (no auth):
        - GET /health
        - GET /stats
        - POST /ingest (multipart/form-data: file)
        - POST /query (json: {q, top_k, min_score})
        """)

if __name__ == "__main__":
    main()
