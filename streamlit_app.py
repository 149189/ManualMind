# streamlit_app.py
import streamlit as st
import requests
import json
import os
from typing import Optional, List, Dict, Any

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
if "show_all_snippets" not in st.session_state:
    st.session_state.show_all_snippets = False


def check_backend_connection():
    """Check if backend API is available."""
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
    """Call backend API. Returns JSON or None on error."""
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

    if resp.status_code >= 500:
        st.error(f"Server error ({resp.status_code}): {resp.text}")
        return None
    if resp.status_code >= 400:
        st.error(f"Request failed ({resp.status_code}): {resp.text}")
        return None

    try:
        return resp.json()
    except ValueError:
        return {"text": resp.text, "status_code": resp.status_code}


def get_index_stats():
    s = call_api("get", "/stats")
    if s:
        st.session_state.index_stats = s
    return s


def ingest_document(file):
    """Upload PDF to /ingest."""
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


def _derive_retrieval_confidence(retrieval_stats: Dict[str, Any]) -> float:
    """Derive a confidence percentage from retrieval stats (avg_score expected 0..1)."""
    try:
        avg = retrieval_stats.get("avg_score", 0.0)
        avg = max(0.0, min(1.0, float(avg)))
        return avg * 100.0
    except Exception:
        return 0.0


def _normalize_confidence(raw_conf) -> float:
    """
    Normalize incoming confidence to 0..100 scale.
    Handles both (0..1) and (0..100) inputs.
    """
    try:
        conf = float(raw_conf)
    except Exception:
        return 0.0
    if 0.0 <= conf <= 1.0:
        return conf * 100.0
    if conf > 1.0 and conf <= 100.0:
        return conf
    # Clamp
    return max(0.0, min(100.0, conf))


def _map_snippets_by_id(snippets: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Create a dict mapping snippet id -> snippet dict for quick lookup."""
    mapping = {}
    for s in snippets:
        sid = s.get("id") or s.get("chunk_id") or s.get("id", "")
        if sid:
            mapping[sid] = s
    return mapping


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
        st.header("Ask the ManualMind Agent")

        # Mode toggles
        use_llm = st.checkbox(
            "Use LLM to refine answer (RAG + LLM)",
            value=True,
            help="If checked, the backend's LLM will compose a final summarized answer from the retrieved snippets."
        )
        show_all_button = st.checkbox("Always show all retrieved snippets (expand by default)", value=False)
        st.session_state.show_all_snippets = show_all_button

        query = st.text_input("Enter your question", "")
        top_k = st.slider("Top K", min_value=1, max_value=20, value=5)
        min_score = st.slider("Min Score", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

        if st.button("Search") and query:
            with st.spinner("Thinking..."):
                res = query_knowledge_base(query, top_k, min_score)

            if not res:
                st.error("No result or error from server.")
            else:
                # Backend returns QueryResponse model with fields:
                # answer (str), citations (list), confidence (number 0..100),
                # snippets (list of dicts), retrieval_stats (dict)
                answer_text = res.get("answer", "") or ""
                citations = res.get("citations", []) or []
                llm_conf = res.get("confidence")
                # sometimes backend also uses 'llm_confidence'
                if llm_conf is None:
                    llm_conf = res.get("llm_confidence")
                retrieval_stats = res.get("retrieval_stats", {}) or {}
                snippets = res.get("snippets", []) or []

                # Map snippets by their ID for quick lookup
                snippets_map = _map_snippets_by_id(snippets)

                if use_llm:
                    # Prefer the LLM-provided summary answer
                    st.subheader("🤖 Summarized Answer")
                    # In case backend returned a JSON-string in answer, guard-parse it
                    displayed_answer = answer_text
                    try:
                        # sometimes the LLM may return the JSON as a string; if so, parse
                        parsed = None
                        if isinstance(answer_text, str) and answer_text.strip().startswith("{"):
                            parsed = json.loads(answer_text)
                        if isinstance(parsed, dict) and "answer" in parsed:
                            displayed_answer = parsed.get("answer", displayed_answer)
                            # allow parsed citations/confidence to override
                            parsed_citations = parsed.get("citations")
                            if parsed_citations:
                                citations = parsed_citations
                            parsed_conf = parsed.get("llm_confidence")
                            if parsed_conf is not None:
                                llm_conf = parsed_conf
                    except Exception:
                        # ignore parsing errors and show raw answer_text
                        pass

                    st.write(displayed_answer or "No answer generated.")

                    # Show confidence (normalize 0..100)
                    conf_val = _normalize_confidence(llm_conf) if llm_conf is not None else _derive_retrieval_confidence(retrieval_stats)
                    st.progress(conf_val / 100.0)
                    st.caption(f"Confidence: {conf_val:.1f}% (LLM-reported or derived)")

                    # Show citations (if any)
                    if citations:
                        st.markdown("**Citations used by the summary:**")
                        # clickable list of snippet ids
                        st.write(", ".join(citations))

                        # Show only cited snippets (expanders), with option to show all
                        st.subheader("📖 Cited Snippets")
                        for cid in citations:
                            s = snippets_map.get(cid)
                            if not s:
                                # try to match by "S<number>" pattern if ids differ
                                s = snippets_map.get(cid.upper())
                            if not s:
                                st.warning(f"Snippet {cid} not found in returned snippets.")
                                continue
                            score = s.get("score", 0)
                            with st.expander(f"{cid} — {s.get('source','unknown')} page {s.get('page','?')} (score {score:.2f})", expanded=st.session_state.show_all_snippets):
                                st.write(s.get("text", ""))
                                st.caption(f"Chunk ID: {s.get('chunk_id', s.get('id','N/A'))}")

                    else:
                        st.info("LLM did not cite any snippets. You can view retrieved snippets below.")

                    # Button / toggle to show all retrieved snippets
                    if st.button("Show all retrieved snippets"):
                        st.session_state.show_all_snippets = True

                    if st.session_state.show_all_snippets and snippets:
                        st.subheader("🔎 All retrieved snippets")
                        for i, s in enumerate(snippets, start=1):
                            sid = s.get("id", f"S{i}")
                            score = s.get("score", 0)
                            with st.expander(f"{sid} — {s.get('source','unknown')} page {s.get('page','?')} (score {score:.2f})", expanded=True):
                                st.write(s.get("text", ""))
                                st.caption(f"Chunk ID: {s.get('chunk_id', s.get('id','N/A'))}")

                else:
                    # Raw retrieval-only mode: show short aggregated summary derived from snippets
                    st.subheader("📖 Retrieved Snippets (raw)")
                    if snippets:
                        for i, s in enumerate(snippets, start=1):
                            sid = s.get("id", f"S{i}")
                            score = s.get("score", 0)
                            with st.expander(f"{sid} — {s.get('source','unknown')} page {s.get('page','?')} (score {score:.2f})", expanded=st.session_state.show_all_snippets):
                                st.write(s.get("text", ""))
                                st.caption(f"Chunk ID: {s.get('chunk_id', s.get('id','N/A'))}")
                    else:
                        st.write("No snippets retrieved.")

                    # Show a short derived summary that combines top K snippet sentences (lightweight)
                    if snippets:
                        # build a tiny "summary" from top 3 snippets by score (client-side)
                        try:
                            top_sorted = sorted(snippets, key=lambda x: x.get("score", 0), reverse=True)[:3]
                            st.subheader("🔎 Quick extract (top snippets)")
                            for t in top_sorted:
                                st.write(f"- {t.get('text','')[:600].strip()}...")
                        except Exception:
                            pass

                    # Show retrieval-only confidence derived from retrieval stats
                    derived_conf = _derive_retrieval_confidence(retrieval_stats)
                    st.progress(derived_conf / 100.0)
                    st.caption(f"Retrieval confidence (avg similarity → %): {derived_conf:.1f}%")

    with tab2:
        st.header("Ingest a New Document")
        uploaded = st.file_uploader("Upload PDF", type="pdf")
        if uploaded is not None and st.button("Upload & Process"):
            with st.spinner("Ingesting..."):
                res = ingest_document(uploaded)
            if res:
                st.success(res.get("message", "Ingest succeeded"))
                st.write(res)
                get_index_stats()

    with tab3:
        st.header("API Documentation")
        st.markdown("""
        **Endpoints**:
        - `GET /health`
        - `GET /stats`
        - `POST /ingest` (multipart/form-data: file)
        - `POST /query` (json: {q, top_k, min_score})
        
        Notes:
        - When using "Use LLM to refine answer", the backend asks the LLM to return a concise summarized answer
          and a list of snippet citations (e.g. ["S1","S2"]). The UI shows the summary and the cited snippets.
        - If the LLM does not provide citations, you can inspect all retrieved snippets using "Show all retrieved snippets".
        """)


if __name__ == "__main__":
    main()
