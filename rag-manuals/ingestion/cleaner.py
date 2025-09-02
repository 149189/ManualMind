# rag-manuals/ingestion/cleaner.py
import logging
import re
from typing import List, Dict, Any
import hashlib

logger = logging.getLogger(__name__)

def clean_pages_batch(pages: List[Dict[str, Any]], 
                     remove_headers: bool = True,
                     remove_footers: bool = True,
                     remove_page_numbers: bool = True) -> List[Dict[str, Any]]:
    """
    Clean a batch of pages by removing headers, footers, etc.
    
    Args:
        pages: List of page dictionaries
        remove_headers: Whether to remove headers
        remove_footers: Whether to remove footers
        remove_page_numbers: Whether to remove page numbers
        
    Returns:
        List of cleaned pages
    """
    cleaned_pages = []
    
    for page in pages:
        try:
            cleaned_page = page.copy()
            text = page.get("text", "")
            
            if remove_headers or remove_footers or remove_page_numbers:
                text = clean_text(text, remove_headers, remove_footers, remove_page_numbers)
                
            cleaned_page["text"] = text
            cleaned_page["text_hash"] = hashlib.md5(text.encode()).hexdigest()
            
            cleaned_pages.append(cleaned_page)
            
        except Exception as e:
            logger.error(f"Failed to clean page {page.get('page', 'unknown')}: {e}")
            # Keep original page if cleaning fails
            cleaned_pages.append(page)
    
    return cleaned_pages

def clean_text(text: str, 
              remove_headers: bool = True,
              remove_footers: bool = True,
              remove_page_numbers: bool = True) -> str:
    """
    Clean text by removing common noise patterns.
    
    Args:
        text: Text to clean
        remove_headers: Whether to remove headers
        remove_footers: Whether to remove footers
        remove_page_numbers: Whether to remove page numbers
        
    Returns:
        Cleaned text
    """
    lines = text.split('\n')
    cleaned_lines = []
    
    # Skip empty lines at beginning and end
    start_idx = 0
    end_idx = len(lines)
    
    for i, line in enumerate(lines):
        if line.strip():
            start_idx = i
            break
            
    for i in range(len(lines)-1, -1, -1):
        if lines[i].strip():
            end_idx = i + 1
            break
    
    if start_idx >= end_idx:
        return text
        
    # Process each line
    for i, line in enumerate(lines[start_idx:end_idx], start=start_idx):
        line = line.strip()
        
        # Skip page numbers
        if remove_page_numbers and is_page_number(line, i, len(lines)):
            continue
            
        # Skip headers (first few lines)
        if remove_headers and i < start_idx + 3 and is_header_footer(line):
            continue
            
        # Skip footers (last few lines)
        if remove_footers and i > end_idx - 3 and is_header_footer(line):
            continue
            
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def is_page_number(text: str, line_num: int, total_lines: int) -> bool:
    """
    Check if text is likely a page number.
    
    Args:
        text: Text to check
        line_num: Line number in document
        total_lines: Total number of lines
        
    Returns:
        True if text is likely a page number
    """
    # Simple numeric check
    if text.isdigit() and 1 <= int(text) <= 9999:
        return True
        
    # Roman numerals (basic check)
    roman_numerals = re.compile(r'^[IVXLCDM]+$', re.IGNORECASE)
    if roman_numerals.match(text):
        return True
        
    # Page X of Y pattern
    page_of_pattern = re.compile(r'page\s+\d+\s+of\s+\d+', re.IGNORECASE)
    if page_of_pattern.search(text):
        return True
        
    return False

def is_header_footer(text: str) -> bool:
    """
    Check if text is likely a header or footer.
    
    Args:
        text: Text to check
        
    Returns:
        True if text is likely a header/footer
    """
    # Common header/footer patterns
    patterns = [
        r'^chapter\s+\d+',
        r'^section\s+\d+',
        r'^\d+\.\d+',  # Section numbers
        r'^confidential',
        r'^draft',
        r'^©|copyright',
        r'^proprietary',
    ]
    
    for pattern in patterns:
        if re.match(pattern, text, re.IGNORECASE):
            return True
            
    # Very short lines in all caps (common in headers)
    if len(text) < 30 and text.isupper():
        return True
        
    return False

def remove_redundant_text(text: str) -> str:
    """
    Remove redundant text that appears across multiple pages.
    
    Args:
        text: Text to process
        
    Returns:
        Text with redundant content removed
    """
    # This would typically use a more sophisticated approach
    # like comparing with previous pages, but for now we'll
    # just remove common repeating patterns
    
    # Common redundant patterns in manuals
    patterns = [
        r"CONFIDENTIAL.*?\.\s*",
        r"Copyright.*?\.\s*",
        r"Proprietary.*?\.\s*",
        r"All rights reserved.*?\.\s*",
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
        
    return text