# rag-manuals/tests/test_ingestion.py
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from ingestion.pdf_to_text import validate_pdf_file, extract_pages, get_pdf_info
from ingestion.cleaner import clean_pages_batch, clean_text
from ingestion.chunker import chunk_page, validate_chunks

def test_validate_pdf_file():
    """Test PDF validation."""
    # Create a temporary PDF file
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
        # Write minimal PDF content
        tmp.write(b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n1 0 obj\n<<>>\nendobj\nxref\n0 2\n0000000000 65535 f \n0000000010 00000 n \ntrailer\n<<>>\nstartxref\n20\n%%EOF")
        tmp_path = tmp.name
    
    try:
        # Test valid PDF
        assert validate_pdf_file(tmp_path) == True
        
        # Test invalid file
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as txt_file:
            txt_file.write(b"Not a PDF")
            txt_path = txt_file.name
            
        assert validate_pdf_file(txt_path) == False
        os.unlink(txt_path)
        
    finally:
        os.unlink(tmp_path)

def test_extract_pages():
    """Test page extraction from PDF."""
    # Mock a simple PDF
    with patch('ingestion.pdf_to_text.fitz.open') as mock_open:
        mock_doc = MagicMock()
        mock_doc.page_count = 2
        mock_doc.is_encrypted = False
        
        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = "Page 1 text"
        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = "Page 2 text"
        
        mock_doc.load_page.side_effect = [mock_page1, mock_page2]
        mock_open.return_value.__enter__.return_value = mock_doc
        
        # Test extraction
        pages = extract_pages("dummy.pdf")
        assert len(pages) == 2
        assert pages[0]['text'] == "Page 1 text"
        assert pages[1]['text'] == "Page 2 text"

def test_clean_text():
    """Test text cleaning functionality."""
    # Test header removal
    text = "HEADER\nThis is the actual content.\nFooter"
    cleaned = clean_text(text, remove_headers=True, remove_footers=True)
    assert "HEADER" not in cleaned
    assert "Footer" not in cleaned
    assert "actual content" in cleaned
    
    # Test page number removal
    text = "Content here\n123\nMore content"
    cleaned = clean_text(text, remove_page_numbers=True)
    assert "123" not in cleaned

def test_chunk_page():
    """Test text chunking."""
    page = {
        "text": "This is a long text that should be split into multiple chunks. " * 50,
        "page": 1,
        "source": "test.pdf"
    }
    
    chunks = chunk_page(page, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1  # Should be multiple chunks
    assert all("chunk_index" in chunk for chunk in chunks)
    assert all("source" in chunk for chunk in chunks)

def test_validate_chunks():
    """Test chunk validation."""
    chunks = [
        {"text": "Valid chunk with sufficient content", "source": "test.pdf", "page": 1},
        {"text": "Short", "source": "test.pdf", "page": 1},  # Too short
        {"text": "", "source": "test.pdf", "page": 1},  # Empty
    ]
    
    valid = validate_chunks(chunks, min_length=10)
    assert len(valid) == 1  # Only the first chunk should be valid
    assert valid[0]["text"] == "Valid chunk with sufficient content"