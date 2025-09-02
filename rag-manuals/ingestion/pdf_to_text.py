# rag-manuals/ingestion/pdf_to_text.py
import logging
import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional
import magic
from pathlib import Path

logger = logging.getLogger(__name__)

def validate_pdf_file(file_path: str) -> bool:
    """
    Validate that the file is a legitimate PDF.
    
    Args:
        file_path: Path to the file to validate
        
    Returns:
        bool: True if valid PDF, False otherwise
    """
    try:
        # Check file type using magic numbers
        file_type = magic.from_file(file_path, mime=True)
        if file_type != 'application/pdf':
            logger.error(f"File is not a PDF: {file_type}")
            return False
            
        # Try to open with PyMuPDF to validate PDF structure
        doc = fitz.open(file_path)
        if doc.is_encrypted:
            logger.error("PDF is encrypted and cannot be processed")
            doc.close()
            return False
            
        # Check if we can extract at least one page
        if doc.page_count == 0:
            logger.error("PDF has no pages")
            doc.close()
            return False
            
        doc.close()
        return True
        
    except Exception as e:
        logger.error(f"PDF validation failed: {e}")
        return False

def get_pdf_info(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Extract metadata from PDF.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Dict with PDF metadata or None if failed
    """
    try:
        doc = fitz.open(file_path)
        info = {
            "filename": Path(file_path).name,
            "page_count": doc.page_count,
            "author": doc.metadata.get('author', ''),
            "title": doc.metadata.get('title', ''),
            "subject": doc.metadata.get('subject', ''),
            "creator": doc.metadata.get('creator', ''),
            "producer": doc.metadata.get('producer', ''),
            "creation_date": doc.metadata.get('creationDate', ''),
            "modification_date": doc.metadata.get('modDate', '')
        }
        doc.close()
        return info
    except Exception as e:
        logger.error(f"Failed to extract PDF metadata: {e}")
        return None

def extract_pages(file_path: str, include_metadata: bool = True) -> List[Dict[str, Any]]:
    """
    Extract text from each page of a PDF.
    
    Args:
        file_path: Path to the PDF file
        include_metadata: Whether to include page metadata
        
    Returns:
        List of pages with text and metadata
    """
    pages = []
    
    try:
        doc = fitz.open(file_path)
        
        for page_num in range(doc.page_count):
            try:
                page = doc.load_page(page_num)
                text = page.get_text("text", sort=True)
                
                page_data = {
                    "text": text,
                    "page": page_num + 1,
                    "source": Path(file_path).name
                }
                
                if include_metadata:
                    page_data["dimensions"] = page.rect
                    page_data["rotation"] = page.rotation
                
                pages.append(page_data)
                
            except Exception as e:
                logger.error(f"Failed to extract page {page_num + 1}: {e}")
                # Continue with other pages even if one fails
                continue
                
        doc.close()
        
    except Exception as e:
        logger.error(f"Failed to open PDF for extraction: {e}")
        return []
    
    return pages

def extract_images_from_pdf(file_path: str, output_dir: str, dpi: int = 150) -> List[str]:
    """
    Extract images from PDF and save to directory.
    
    Args:
        file_path: Path to the PDF file
        output_dir: Directory to save extracted images
        dpi: Resolution for image extraction
        
    Returns:
        List of paths to extracted images
    """
    extracted_images = []
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        doc = fitz.open(file_path)
        
        for page_num in range(doc.page_count):
            try:
                page = doc.load_page(page_num)
                image_list = page.get_images(full=True)
                
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    image_filename = f"{Path(file_path).stem}_p{page_num+1}_i{img_index}.{image_ext}"
                    image_path = Path(output_dir) / image_filename
                    
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)
                    
                    extracted_images.append(str(image_path))
                    
            except Exception as e:
                logger.error(f"Failed to extract images from page {page_num + 1}: {e}")
                continue
                
        doc.close()
        
    except Exception as e:
        logger.error(f"Failed to open PDF for image extraction: {e}")
    
    return extracted_images