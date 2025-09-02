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

def check_backend_connection():
    """Check if backend API is available"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        st.session_state.backend_available = response.status_code == 200
        if st.session_state.backend_available:
            st.session_state.health_status = response.json()
        return st.session_state.backend_available
    except requests.exceptions.RequestException:
        st.session_state.backend_available = False
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

# Rest of your functions remain the same...

def main():
    st.set_page_config(
        page_title="ManualMind - RAG for Product Manuals",
        page_icon="📚",
        layout="wide"
    )
    
    # Check backend connection on app load
    if not st.session_state.backend_available:
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
        
        # Rest of your sidebar code...
    
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
        return
    
    # Rest of your main content...