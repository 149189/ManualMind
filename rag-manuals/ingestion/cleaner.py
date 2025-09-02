# rag-manuals/ingestion/cleaner.py
import re
import logging
from typing import List, Set, Optional, Dict
from collections import Counter

logger = logging.getLogger(__name__)


def find_repeated_headers(pages: List[str], threshold: int = 3, min_length: int = 5) -> List[str]:
    """
    Find lines appearing on multiple pages (simple heuristic for headers/footers).
    
    Args:
        pages: List of page text strings
        threshold: Minimum number of pages a line must appear on to be considered repeated
        min_length: Minimum length of line to be considered (filters out single characters)
        
    Returns:
        List of repeated lines (potential headers/footers)
        
    Raises:
        TypeError: If pages is not a list or threshold is not an integer
        ValueError: If threshold or min_length are invalid
    """
    # Input validation
    if not isinstance(pages, list):
        raise TypeError(f"pages must be a list, got {type(pages)}")
    
    if not isinstance(threshold, int) or threshold <= 0:
        raise ValueError(f"threshold must be a positive integer, got {threshold}")
        
    if not isinstance(min_length, int) or min_length < 0:
        raise ValueError(f"min_length must be a non-negative integer, got {min_length}")
    
    if not pages:
        logger.warning("Empty pages list provided")
        return []
    
    try:
        counters = Counter()
        valid_pages = 0
        
        for i, page in enumerate(pages):
            if not isinstance(page, str):
                logger.warning(f"Page {i} is not a string, skipping")
                continue
            
            if not page.strip():
                logger.debug(f"Page {i} is empty, skipping")
                continue
                
            valid_pages += 1
            lines = [line.strip() for line in page.splitlines() if line.strip()]
            
            # Filter lines by minimum length and exclude very common patterns
            filtered_lines = []
            for line in lines:
                if (len(line) >= min_length and 
                    not _is_likely_content_line(line) and
                    len(line) <= 200):  # Exclude very long lines
                    filtered_lines.append(line)
            
            counters.update(filtered_lines)
        
        # Adjust threshold if we have fewer valid pages
        effective_threshold = min(threshold, max(2, valid_pages // 3))
        
        common = [line for line, count in counters.items() 
                 if count >= effective_threshold]
        
        logger.info(f"Found {len(common)} repeated lines from {valid_pages} pages "
                   f"(threshold: {effective_threshold})")
        
        return common
        
    except Exception as e:
        logger.error(f"Error finding repeated headers: {e}")
        raise


def _is_likely_content_line(line: str) -> bool:
    """
    Check if a line is likely to be content rather than header/footer.
    Returns True if the line should NOT be considered for removal.
    """
    # Skip lines that are likely to be actual content
    content_indicators = [
        len(line.split()) > 8,  # Long lines are likely content
        any(char.isdigit() and char.isalpha() for char in line[:20]),  # Mixed content
        line.count('.') > 2,  # Multiple sentences
        any(word in line.lower() for word in ['the', 'and', 'that', 'with', 'from'])  # Common words
    ]
    
    return any(content_indicators)


def strip_headers_and_footers(page_text: str, repeated_lines: List[str], 
                             aggressive: bool = False) -> str:
    """
    Remove repeated headers and footers from page text.
    
    Args:
        page_text: Text content of a single page
        repeated_lines: List of lines to remove (from find_repeated_headers)
        aggressive: If True, also remove lines that partially match repeated lines
        
    Returns:
        Cleaned page text
        
    Raises:
        TypeError: If inputs are not of expected types
    """
    # Input validation
    if not isinstance(page_text, str):
        raise TypeError(f"page_text must be a string, got {type(page_text)}")
    
    if not isinstance(repeated_lines, list):
        raise TypeError(f"repeated_lines must be a list, got {type(repeated_lines)}")
    
    if not page_text.strip():
        return ""
    
    try:
        lines = page_text.splitlines()
        if not lines:
            return ""
        
        # Create set for faster lookup
        repeated_set = set(line.strip() for line in repeated_lines if line.strip())
        
        cleaned_lines = []
        
        for line in lines:
            stripped_line = line.strip()
            
            # Skip empty lines initially, add back strategically
            if not stripped_line:
                # Keep empty lines that aren't at the very beginning or end
                if cleaned_lines and line != lines[-1]:
                    cleaned_lines.append(line)
                continue
            
            # Check for exact matches
            if stripped_line in repeated_set:
                logger.debug(f"Removing exact match: {stripped_line[:50]}...")
                continue
            
            # Aggressive mode: check for partial matches
            if aggressive:
                should_remove = False
                for repeated in repeated_set:
                    if (len(repeated) > 10 and 
                        (repeated in stripped_line or stripped_line in repeated)):
                        logger.debug(f"Removing partial match: {stripped_line[:50]}...")
                        should_remove = True
                        break
                
                if should_remove:
                    continue
            
            cleaned_lines.append(line)
        
        # Remove excessive blank lines from the result
        result = "\n".join(cleaned_lines)
        result = re.sub(r'\n\s*\n\s*\n', '\n\n', result)  # Max 2 consecutive newlines
        
        return result.strip()
        
    except Exception as e:
        logger.error(f"Error stripping headers and footers: {e}")
        return page_text  # Return original text on error


def clean_text(text: str, preserve_structure: bool = True) -> str:
    """
    Clean and normalize text content.
    
    Args:
        text: Input text to clean
        preserve_structure: If True, maintain paragraph structure
        
    Returns:
        Cleaned text
        
    Raises:
        TypeError: If text is not a string
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text)}")
    
    if not text:
        return ""
    
    try:
        # Basic normalizations
        text = text.replace('\r\n', '\n')
        text = text.replace('\r', '\n')  # Handle old Mac line endings
        
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between camelCase
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)  # Space after sentence endings
        
        # Clean up whitespace
        if preserve_structure:
            # Keep paragraph breaks but limit excessive newlines
            text = re.sub(r'\n{4,}', '\n\n\n', text)  # Max 3 newlines
            text = re.sub(r'[ \t]+', ' ', text)  # Normalize spaces
        else:
            # More aggressive whitespace cleanup
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = re.sub(r'\s+', ' ', text)
        
        # Remove trailing whitespace from lines
        lines = text.splitlines()
        cleaned_lines = [line.rstrip() for line in lines]
        text = '\n'.join(cleaned_lines)
        
        # Final cleanup
        text = text.strip()
        
        logger.debug(f"Text cleaned: {len(text)} characters")
        return text
        
    except Exception as e:
        logger.error(f"Error cleaning text: {e}")
        return text.strip() if text else ""


def clean_pages_batch(pages_data: List[Dict], remove_headers: bool = True) -> List[Dict]:
    """
    Clean multiple pages in batch, optionally removing headers/footers.
    
    Args:
        pages_data: List of page dictionaries with 'text' key
        remove_headers: Whether to detect and remove repeated headers/footers
        
    Returns:
        List of cleaned page dictionaries
        
    Raises:
        TypeError: If pages_data is not a list
    """
    if not isinstance(pages_data, list):
        raise TypeError(f"pages_data must be a list, got {type(pages_data)}")
    
    if not pages_data:
        logger.warning("Empty pages_data provided")
        return []
    
    try:
        # Extract page texts for header detection
        page_texts = []
        for page in pages_data:
            if isinstance(page, dict) and 'text' in page:
                page_texts.append(page['text'])
            else:
                logger.warning(f"Invalid page data structure: {type(page)}")
                page_texts.append("")
        
        # Find repeated headers/footers
        repeated_lines = []
        if remove_headers and len(page_texts) > 2:
            try:
                repeated_lines = find_repeated_headers(page_texts)
                logger.info(f"Found {len(repeated_lines)} repeated lines to remove")
            except Exception as e:
                logger.error(f"Error finding repeated headers: {e}")
        
        # Clean each page
        cleaned_pages = []
        for i, page in enumerate(pages_data):
            try:
                if not isinstance(page, dict):
                    logger.warning(f"Page {i} is not a dictionary, skipping")
                    continue
                
                # Create a copy to avoid modifying original
                cleaned_page = page.copy()
                
                text = page.get('text', '')
                if text:
                    # Remove headers/footers if applicable
                    if repeated_lines:
                        text = strip_headers_and_footers(text, repeated_lines)
                    
                    # Clean the text
                    text = clean_text(text)
                    
                    cleaned_page['text'] = text
                    cleaned_page['cleaned'] = True
                
                cleaned_pages.append(cleaned_page)
                
            except Exception as e:
                logger.error(f"Error cleaning page {i}: {e}")
                # Add original page on error
                cleaned_pages.append(page)
        
        logger.info(f"Successfully cleaned {len(cleaned_pages)} pages")
        return cleaned_pages
        
    except Exception as e:
        logger.error(f"Error in batch cleaning: {e}")
        return pages_data  # Return original data on error