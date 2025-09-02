# rag-manuals/ingestion/chunker.py
import logging
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def chunk_text_by_chars(text: str, chunk_size: int = 3000, overlap: int = 500) -> List[Dict]:
    """
    Simple char-based chunker. Each chunk keeps offsets for provenance.
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of chunk dictionaries with metadata
        
    Raises:
        ValueError: If chunk_size or overlap are invalid
        TypeError: If text is not a string
    """
    # Input validation
    if not isinstance(text, str):
        raise TypeError(f"Text must be a string, got {type(text)}")
    
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        
    if overlap < 0:
        raise ValueError(f"overlap cannot be negative, got {overlap}")
        
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")
    
    chunks = []
    if not text.strip():  # Handle empty or whitespace-only text
        logger.warning("Empty or whitespace-only text provided to chunker")
        return chunks
    
    try:
        start = 0
        idx = 0
        n = len(text)
        
        while start < n:
            end = min(n, start + chunk_size)
            chunk = text[start:end].strip()
            
            if chunk:  # Only add non-empty chunks
                chunks.append({
                    "chunk_idx": idx,
                    "char_start": start,
                    "char_end": end,
                    "text": chunk,
                    "chunk_length": len(chunk)
                })
                idx += 1
            
            # Calculate next start position
            next_start = start + chunk_size - overlap
            if next_start <= start:  # Prevent infinite loop
                next_start = start + 1
            start = next_start
            
        logger.info(f"Successfully created {len(chunks)} chunks from {n} characters")
        return chunks
        
    except Exception as e:
        logger.error(f"Error during text chunking: {e}")
        raise


def chunk_page(page_meta: Dict, chunk_size: int = 3000, overlap: int = 500) -> List[Dict]:
    """
    Chunk a single page with metadata preservation.
    
    Args:
        page_meta: Dictionary containing page metadata including 'text', 'filename', 'page'
        chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of chunk dictionaries with enhanced metadata
        
    Raises:
        TypeError: If page_meta is not a dictionary
        KeyError: If required keys are missing from page_meta
        ValueError: If chunk parameters are invalid
    """
    # Input validation
    if not isinstance(page_meta, dict):
        raise TypeError(f"page_meta must be a dictionary, got {type(page_meta)}")
    
    # Check for required keys
    required_keys = ['text', 'filename', 'page']
    missing_keys = [key for key in required_keys if key not in page_meta]
    if missing_keys:
        raise KeyError(f"Missing required keys in page_meta: {missing_keys}")
    
    try:
        text = page_meta.get("text", "")
        filename = page_meta.get("filename", "unknown")
        page_num = page_meta.get("page", 0)
        
        # Validate page number
        if not isinstance(page_num, (int, str)):
            logger.warning(f"Invalid page number type: {type(page_num)}, converting to string")
            page_num = str(page_num)
        
        raw_chunks = chunk_text_by_chars(text, chunk_size=chunk_size, overlap=overlap)
        out = []
        
        for c in raw_chunks:
            try:
                meta = {**c}  # Copy chunk metadata
                meta.update({
                    "source": filename,
                    "page": page_num,
                    "original_page_length": len(text)
                })
                
                # Create unique ID with safe formatting
                chunk_id = f"{filename}::p{page_num}::c{c['chunk_idx']}"
                meta["id"] = chunk_id
                
                out.append(meta)
                
            except Exception as e:
                logger.error(f"Error processing chunk {c.get('chunk_idx', 'unknown')}: {e}")
                continue  # Skip problematic chunks but continue processing
        
        logger.info(f"Successfully processed page {page_num} of {filename}: {len(out)} chunks")
        return out
        
    except Exception as e:
        logger.error(f"Error chunking page {page_meta.get('page', 'unknown')} "
                    f"from {page_meta.get('filename', 'unknown')}: {e}")
        raise


def validate_chunks(chunks: List[Dict]) -> List[Dict]:
    """
    Validate and clean a list of chunks, removing invalid ones.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        List of validated chunks
    """
    if not isinstance(chunks, list):
        logger.error(f"Expected list of chunks, got {type(chunks)}")
        return []
    
    valid_chunks = []
    required_fields = ['chunk_idx', 'text', 'id']
    
    for i, chunk in enumerate(chunks):
        try:
            if not isinstance(chunk, dict):
                logger.warning(f"Chunk {i} is not a dictionary, skipping")
                continue
            
            # Check required fields
            missing_fields = [field for field in required_fields if field not in chunk]
            if missing_fields:
                logger.warning(f"Chunk {i} missing required fields: {missing_fields}")
                continue
            
            # Validate text content
            if not chunk.get('text', '').strip():
                logger.warning(f"Chunk {i} has empty text content, skipping")
                continue
            
            valid_chunks.append(chunk)
            
        except Exception as e:
            logger.error(f"Error validating chunk {i}: {e}")
            continue
    
    logger.info(f"Validated {len(valid_chunks)} out of {len(chunks)} chunks")
    return valid_chunks