# rag-manuals/ingestion/chunker.py
import logging
import re
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

def chunk_page(page: Dict[str, Any], 
               chunk_size: int = 1000, 
               chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Split a page into smaller chunks with overlap.
    
    Args:
        page: Dictionary containing page text and metadata
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of chunks with metadata
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        text = page.get("text", "")
        if not text or not text.strip():
            logger.warning(f"Empty text for page {page.get('page', 'unknown')}")
            return []
        
        # Split the text
        chunks = text_splitter.split_text(text)
        
        # Prepare chunk metadata
        chunk_objects = []
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{page.get('source', 'unknown')}_p{page.get('page', '?')}_c{i+1}"
            
            chunk_obj = {
                "id": chunk_id,
                "text": chunk_text,
                "source": page.get("source", "unknown"),
                "page": page.get("page", "?"),
                "chunk_index": i + 1,
                "total_chunks": len(chunks),
                "original_metadata": {k: v for k, v in page.items() if k != "text"}
            }
            
            chunk_objects.append(chunk_obj)
        
        return chunk_objects
        
    except Exception as e:
        logger.error(f"Failed to chunk page {page.get('page', 'unknown')}: {e}")
        return []

def validate_chunks(chunks: List[Dict[str, Any]], 
                   min_length: int = 50,
                   max_length: int = 2000) -> List[Dict[str, Any]]:
    """
    Validate chunks based on length and content.
    
    Args:
        chunks: List of chunk dictionaries
        min_length: Minimum acceptable chunk length
        max_length: Maximum acceptable chunk length
        
    Returns:
        Filtered list of valid chunks
    """
    valid_chunks = []
    
    for chunk in chunks:
        try:
            text = chunk.get("text", "").strip()
            
            # Skip empty chunks
            if not text:
                continue
                
            # Skip chunks that are too short or too long
            if len(text) < min_length or len(text) > max_length:
                continue
                
            # Skip chunks that are mostly non-text (e.g., code, tables)
            if is_low_quality_text(text):
                continue
                
            valid_chunks.append(chunk)
            
        except Exception as e:
            logger.error(f"Failed to validate chunk {chunk.get('id', 'unknown')}: {e}")
            continue
    
    return valid_chunks

def is_low_quality_text(text: str) -> bool:
    """
    Detect if text is likely low quality (e.g., code, table of contents).
    
    Args:
        text: Text to evaluate
        
    Returns:
        True if text is likely low quality
    """
    # Check for excessive special characters (might be code)
    special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s]', text)) / len(text) if text else 0
    if special_char_ratio > 0.4:
        return True
        
    # Check for very short lines (might be table of contents)
    lines = text.split('\n')
    short_lines = sum(1 for line in lines if len(line.strip()) < 10)
    if short_lines / len(lines) > 0.7:
        return True
        
    # Check for excessive numbers (might be a table)
    digit_ratio = sum(c.isdigit() for c in text) / len(text) if text else 0
    if digit_ratio > 0.3:
        return True
        
    return False

def semantic_chunking(text: str, model=None, threshold: float = 0.85) -> List[str]:
    """
    Experimental: Chunk text based on semantic similarity.
    This requires a sentence embedding model.
    
    Args:
        text: Text to chunk semantically
        model: Sentence embedding model (optional)
        threshold: Similarity threshold for splitting
        
    Returns:
        List of semantic chunks
    """
    # Fallback to recursive splitting if no model provided
    if model is None:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        return text_splitter.split_text(text)
    
    # Implementation for semantic chunking would go here
    # This is a placeholder for future enhancement
    logger.warning("Semantic chunking not fully implemented, using recursive splitting")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return text_splitter.split_text(text)