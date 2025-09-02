# rag-manuals/ingestion/pdf_to_text.py
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from .cleaner import clean_text

logger = logging.getLogger(__name__)

# Try to import pdfplumber with fallback options
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False
    logger.warning("pdfplumber not available, PDF extraction will be limited")

# Fallback options
try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

try:
    import pymupdf  # fitz
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False


def extract_pages(pdf_path: Union[str, Path], 
                 extraction_method: str = "auto",
                 max_pages: Optional[int] = None,
                 include_metadata: bool = True) -> List[Dict[str, Any]]:
    """
    Extract page-granular text from a PDF and preserve filename/page metadata.
    
    Args:
        pdf_path: Path to the PDF file
        extraction_method: Method to use ('pdfplumber', 'pypdf2', 'pymupdf', 'auto')
        max_pages: Maximum number of pages to process (None for all)
        include_metadata: Whether to include additional metadata
        
    Returns:
        List of page dictionaries with text and metadata
        
    Raises:
        FileNotFoundError: If PDF file doesn't exist
        ValueError: If extraction method is invalid
        RuntimeError: If no extraction libraries are available
    """
    # Input validation
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    if not pdf_path.suffix.lower() == '.pdf':
        raise ValueError(f"File is not a PDF: {pdf_path}")
    
    valid_methods = ['auto', 'pdfplumber', 'pypdf2', 'pymupdf']
    if extraction_method not in valid_methods:
        raise ValueError(f"Invalid extraction method: {extraction_method}. "
                        f"Must be one of {valid_methods}")
    
    # Check available libraries
    if extraction_method == 'auto':
        if HAS_PDFPLUMBER:
            extraction_method = 'pdfplumber'
        elif HAS_PYMUPDF:
            extraction_method = 'pymupdf'
        elif HAS_PYPDF2:
            extraction_method = 'pypdf2'
        else:
            raise RuntimeError("No PDF extraction libraries available. "
                             "Install pdfplumber, PyPDF2, or pymupdf")
    
    # Validate selected method is available
    method_available = {
        'pdfplumber': HAS_PDFPLUMBER,
        'pypdf2': HAS_PYPDF2,
        'pymupdf': HAS_PYMUPDF
    }
    
    if not method_available.get(extraction_method, False):
        raise RuntimeError(f"Extraction method '{extraction_method}' not available. "
                          f"Please install the required library.")
    
    logger.info(f"Extracting text from {pdf_path} using {extraction_method}")
    
    try:
        # Route to appropriate extraction method
        if extraction_method == 'pdfplumber':
            pages = _extract_with_pdfplumber(pdf_path, max_pages, include_metadata)
        elif extraction_method == 'pymupdf':
            pages = _extract_with_pymupdf(pdf_path, max_pages, include_metadata)
        elif extraction_method == 'pypdf2':
            pages = _extract_with_pypdf2(pdf_path, max_pages, include_metadata)
        else:
            raise ValueError(f"Unsupported extraction method: {extraction_method}")
        
        logger.info(f"Successfully extracted {len(pages)} pages from {pdf_path.name}")
        return pages
        
    except Exception as e:
        logger.error(f"Error extracting from {pdf_path}: {e}")
        # Try fallback method if auto mode failed
        if extraction_method == 'pdfplumber' and HAS_PYMUPDF:
            logger.info("Trying fallback extraction with pymupdf")
            return _extract_with_pymupdf(pdf_path, max_pages, include_metadata)
        raise


def _extract_with_pdfplumber(pdf_path: Path, max_pages: Optional[int], 
                           include_metadata: bool) -> List[Dict[str, Any]]:
    """Extract text using pdfplumber library."""
    pages = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            pages_to_process = min(total_pages, max_pages) if max_pages else total_pages
            
            logger.debug(f"Processing {pages_to_process} of {total_pages} pages")
            
            for i, page in enumerate(pdf.pages[:pages_to_process], start=1):
                try:
                    # Extract text
                    text = page.extract_text() or ""
                    
                    # Clean the text
                    text = clean_text(text)
                    
                    page_data = {
                        "filename": str(pdf_path),
                        "page": i,
                        "text": text,
                        "extraction_method": "pdfplumber"
                    }
                    
                    if include_metadata:
                        # Add additional metadata
                        page_data.update({
                            "char_count": len(text),
                            "word_count": len(text.split()) if text else 0,
                            "page_width": page.width,
                            "page_height": page.height,
                        })
                        
                        # Try to extract tables
                        try:
                            tables = page.extract_tables()
                            page_data["table_count"] = len(tables) if tables else 0
                        except Exception as e:
                            logger.debug(f"Could not extract tables from page {i}: {e}")
                            page_data["table_count"] = 0
                    
                    pages.append(page_data)
                    
                except Exception as e:
                    logger.error(f"Error extracting page {i}: {e}")
                    # Add empty page to maintain page numbering
                    pages.append({
                        "filename": str(pdf_path),
                        "page": i,
                        "text": "",
                        "extraction_method": "pdfplumber",
                        "extraction_error": str(e)
                    })
    
    except Exception as e:
        logger.error(f"Error opening PDF with pdfplumber: {e}")
        raise
    
    return pages


def _extract_with_pymupdf(pdf_path: Path, max_pages: Optional[int],
                         include_metadata: bool) -> List[Dict[str, Any]]:
    """Extract text using pymupdf (fitz) library."""
    pages = []
    
    try:
        import fitz  # pymupdf
        
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        pages_to_process = min(total_pages, max_pages) if max_pages else total_pages
        
        logger.debug(f"Processing {pages_to_process} of {total_pages} pages")
        
        for i in range(pages_to_process):
            try:
                page = doc[i]
                text = page.get_text()
                
                # Clean the text
                text = clean_text(text)
                
                page_data = {
                    "filename": str(pdf_path),
                    "page": i + 1,  # 1-indexed
                    "text": text,
                    "extraction_method": "pymupdf"
                }
                
                if include_metadata:
                    rect = page.rect
                    page_data.update({
                        "char_count": len(text),
                        "word_count": len(text.split()) if text else 0,
                        "page_width": rect.width,
                        "page_height": rect.height,
                    })
                
                pages.append(page_data)
                
            except Exception as e:
                logger.error(f"Error extracting page {i + 1}: {e}")
                pages.append({
                    "filename": str(pdf_path),
                    "page": i + 1,
                    "text": "",
                    "extraction_method": "pymupdf",
                    "extraction_error": str(e)
                })
        
        doc.close()
        
    except Exception as e:
        logger.error(f"Error opening PDF with pymupdf: {e}")
        raise
    
    return pages


def _extract_with_pypdf2(pdf_path: Path, max_pages: Optional[int],
                        include_metadata: bool) -> List[Dict[str, Any]]:
    """Extract text using PyPDF2 library."""
    pages = []
    
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            pages_to_process = min(total_pages, max_pages) if max_pages else total_pages
            
            logger.debug(f"Processing {pages_to_process} of {total_pages} pages")
            
            for i in range(pages_to_process):
                try:
                    page = pdf_reader.pages[i]
                    text = page.extract_text()
                    
                    # Clean the text
                    text = clean_text(text)
                    
                    page_data = {
                        "filename": str(pdf_path),
                        "page": i + 1,  # 1-indexed
                        "text": text,
                        "extraction_method": "pypdf2"
                    }
                    
                    if include_metadata:
                        page_data.update({
                            "char_count": len(text),
                            "word_count": len(text.split()) if text else 0,
                        })
                        
                        # Try to get page dimensions
                        try:
                            mediabox = page.mediabox
                            page_data.update({
                                "page_width": float(mediabox.width),
                                "page_height": float(mediabox.height),
                            })
                        except Exception as e:
                            logger.debug(f"Could not extract page dimensions: {e}")
                    
                    pages.append(page_data)
                    
                except Exception as e:
                    logger.error(f"Error extracting page {i + 1}: {e}")
                    pages.append({
                        "filename": str(pdf_path),
                        "page": i + 1,
                        "text": "",
                        "extraction_method": "pypdf2",
                        "extraction_error": str(e)
                    })
    
    except Exception as e:
        logger.error(f"Error opening PDF with PyPDF2: {e}")
        raise
    
    return pages


def extract_pages_batch(pdf_paths: List[Union[str, Path]], 
                       extraction_method: str = "auto",
                       max_pages_per_pdf: Optional[int] = None,
                       continue_on_error: bool = True) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract text from multiple PDF files.
    
    Args:
        pdf_paths: List of paths to PDF files
        extraction_method: Method to use for extraction
        max_pages_per_pdf: Maximum pages to process per PDF
        continue_on_error: Whether to continue processing other files if one fails
        
    Returns:
        Dictionary mapping file paths to lists of page data
        
    Raises:
        TypeError: If pdf_paths is not a list
    """
    if not isinstance(pdf_paths, list):
        raise TypeError(f"pdf_paths must be a list, got {type(pdf_paths)}")
    
    results = {}
    errors = []
    
    logger.info(f"Starting batch extraction of {len(pdf_paths)} PDF files")
    
    for pdf_path in pdf_paths:
        try:
            pages = extract_pages(pdf_path, 
                                extraction_method=extraction_method,
                                max_pages=max_pages_per_pdf)
            results[str(pdf_path)] = pages
            
        except Exception as e:
            error_msg = f"Failed to extract from {pdf_path}: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            
            if continue_on_error:
                results[str(pdf_path)] = []  # Empty result for failed file
            else:
                raise RuntimeError(f"Batch extraction failed at {pdf_path}: {e}")
    
    logger.info(f"Batch extraction completed. "
                f"Successful: {len([r for r in results.values() if r])}, "
                f"Failed: {len(errors)}")
    
    if errors:
        logger.warning(f"Errors encountered: {errors}")
    
    return results


def validate_pdf_file(pdf_path: Union[str, Path]) -> bool:
    """
    Validate if a file is a readable PDF.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        True if file is a valid PDF, False otherwise
    """
    try:
        pdf_path = Path(pdf_path)
        
        # Check file existence and extension
        if not pdf_path.exists():
            logger.error(f"File does not exist: {pdf_path}")
            return False
        
        if pdf_path.suffix.lower() != '.pdf':
            logger.error(f"File is not a PDF: {pdf_path}")
            return False
        
        # Check file size (empty files are invalid)
        if pdf_path.stat().st_size == 0:
            logger.error(f"File is empty: {pdf_path}")
            return False
        
        # Try to open with available library
        if HAS_PDFPLUMBER:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    return len(pdf.pages) > 0
            except Exception as e:
                logger.debug(f"pdfplumber validation failed: {e}")
        
        if HAS_PYMUPDF:
            try:
                import fitz
                doc = fitz.open(pdf_path)
                page_count = len(doc)
                doc.close()
                return page_count > 0
            except Exception as e:
                logger.debug(f"pymupdf validation failed: {e}")
        
        if HAS_PYPDF2:
            try:
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    return len(reader.pages) > 0
            except Exception as e:
                logger.debug(f"PyPDF2 validation failed: {e}")
        
        logger.error("No PDF libraries available for validation")
        return False
        
    except Exception as e:
        logger.error(f"Error validating PDF {pdf_path}: {e}")
        return False


def get_pdf_info(pdf_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Get basic information about a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary with PDF information or None if extraction fails
    """
    try:
        pdf_path = Path(pdf_path)
        
        if not validate_pdf_file(pdf_path):
            return None
        
        info = {
            "filename": pdf_path.name,
            "filepath": str(pdf_path),
            "file_size": pdf_path.stat().st_size,
            "page_count": 0,
            "extraction_methods_available": []
        }
        
        # Check available extraction methods
        if HAS_PDFPLUMBER:
            info["extraction_methods_available"].append("pdfplumber")
        if HAS_PYMUPDF:
            info["extraction_methods_available"].append("pymupdf")
        if HAS_PYPDF2:
            info["extraction_methods_available"].append("pypdf2")
        
        # Get page count and metadata using best available method
        if HAS_PDFPLUMBER:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    info["page_count"] = len(pdf.pages)
                    if hasattr(pdf, 'metadata') and pdf.metadata:
                        info["metadata"] = dict(pdf.metadata)
            except Exception as e:
                logger.debug(f"Error getting pdfplumber info: {e}")
        
        elif HAS_PYMUPDF:
            try:
                import fitz
                doc = fitz.open(pdf_path)
                info["page_count"] = len(doc)
                try:
                    info["metadata"] = doc.metadata
                except:
                    pass
                doc.close()
            except Exception as e:
                logger.debug(f"Error getting pymupdf info: {e}")
        
        elif HAS_PYPDF2:
            try:
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    info["page_count"] = len(reader.pages)
                    if hasattr(reader, 'metadata') and reader.metadata:
                        info["metadata"] = dict(reader.metadata)
            except Exception as e:
                logger.debug(f"Error getting PyPDF2 info: {e}")
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting PDF info for {pdf_path}: {e}")
        return None