# rag-manuals/index/build_index.py
#!/usr/bin/env python3
"""
Standalone script to build FAISS index from documents.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.pdf_to_text import extract_pages, get_pdf_info, validate_pdf_file
from ingestion.cleaner import clean_pages_batch
from ingestion.chunker import chunk_page, validate_chunks
from embeddings.embedder import Embedder
from index.faiss_utils import FaissStore

def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def process_document(file_path: str, embedder: Embedder, store: FaissStore) -> int:
    """
    Process a single document and add it to the index.
    
    Args:
        file_path: Path to the document
        embedder: Embedder instance
        store: FAISS store instance
        
    Returns:
        Number of chunks added
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Validate PDF
        if not validate_pdf_file(file_path):
            logger.error(f"Invalid PDF: {file_path}")
            return 0
            
        # Extract info
        file_info = get_pdf_info(file_path)
        logger.info(f"Processing: {file_info.get('filename', file_path)}")
        
        # Extract pages
        pages = extract_pages(file_path, include_metadata=True)
        if not pages:
            logger.error(f"No pages extracted from: {file_path}")
            return 0
            
        logger.info(f"Extracted {len(pages)} pages")
        
        # Clean pages
        cleaned_pages = clean_pages_batch(pages, remove_headers=True)
        logger.info(f"Cleaned {len(cleaned_pages)} pages")
        
        # Chunk pages
        all_chunks = []
        for page in cleaned_pages:
            page_chunks = chunk_page(page)
            all_chunks.extend(page_chunks)
            
        logger.info(f"Created {len(all_chunks)} chunks")
        
        # Validate chunks
        valid_chunks = validate_chunks(all_chunks)
        if len(valid_chunks) != len(all_chunks):
            logger.warning(f"Filtered out {len(all_chunks) - len(valid_chunks)} invalid chunks")
            
        if not valid_chunks:
            logger.error("No valid chunks created")
            return 0
            
        # Generate embeddings
        texts = [chunk["text"] for chunk in valid_chunks]
        embeddings = embedder.embed(texts)
        
        # Add to index
        success = store.build(embeddings, valid_chunks)
        if not success:
            logger.error("Failed to add document to index")
            return 0
            
        return len(valid_chunks)
        
    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")
        return 0

def main():
    """Main function for the index building script."""
    parser = argparse.ArgumentParser(description="Build FAISS index from PDF documents")
    parser.add_argument("input", help="Input PDF file or directory")
    parser.add_argument("-o", "--output", default="index/faiss.index", help="Output index path")
    parser.add_argument("-m", "--meta", default="index/meta.jsonl", help="Output metadata path")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Embedding model name")
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Initialize embedder and store
    try:
        embedder = Embedder(model_name=args.model)
        store = FaissStore(args.output, args.meta, embedder.get_embedding_dimension())
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        return 1
        
    # Process input
    input_path = Path(args.input)
    total_chunks = 0
    
    if input_path.is_file() and input_path.suffix.lower() == '.pdf':
        # Single file
        chunks = process_document(str(input_path), embedder, store)
        total_chunks += chunks
        logger.info(f"Added {chunks} chunks from {input_path.name}")
        
    elif input_path.is_dir():
        # Directory of PDFs
        pdf_files = list(input_path.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        for pdf_file in pdf_files:
            chunks = process_document(str(pdf_file), embedder, store)
            total_chunks += chunks
            logger.info(f"Added {chunks} chunks from {pdf_file.name}")
            
    else:
        logger.error(f"Invalid input: {args.input}")
        return 1
        
    logger.info(f"Index building complete. Total chunks: {total_chunks}")
    return 0

if __name__ == "__main__":
    sys.exit(main())