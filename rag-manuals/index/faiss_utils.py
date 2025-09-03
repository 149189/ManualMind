# rag-manuals/index/faiss_utils.py
import logging
import json
import faiss
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import sqlite3
from datetime import datetime
import traceback
import os

logger = logging.getLogger(__name__)

class FaissStoreError(Exception):
    """Base exception for FAISS store operations."""
    pass

class IndexCorruptedError(FaissStoreError):
    """Raised when index is corrupted."""
    pass

class MetadataMismatchError(FaissStoreError):
    """Raised when metadata doesn't match index."""
    pass


class FaissStore:
    """
    Manages FAISS index and associated metadata with comprehensive error handling.
    """
    
    def __init__(self, index_path: str, meta_path: str, dimension: int = 384):
        """
        Initialize the FAISS store.
        
        Args:
            index_path: Path to save/load the FAISS index
            meta_path: Path to save/load metadata
            dimension: Dimension of embeddings (if creating new index)
        """
        self.index_path = index_path
        self.meta_path = meta_path
        self.dimension = dimension
        self.index = None
        self.metadata = []
        
        # Create directory if it doesn't exist
        try:
            Path(index_path).parent.mkdir(parents=True, exist_ok=True)
            Path(meta_path).parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create index directories: {e}")
            raise FaissStoreError(f"Cannot create index directories: {e}")
        
    def build(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> bool:
        """
        Build or update the FAISS index with comprehensive validation.
        
        Args:
            embeddings: Array of embeddings
            metadata: List of metadata dictionaries
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Input validation
            if embeddings is None or embeddings.size == 0:
                logger.error("No embeddings provided for index building")
                return False
                
            if not isinstance(embeddings, np.ndarray):
                logger.error("Embeddings must be a numpy array")
                return False
                
            if len(embeddings.shape) != 2:
                logger.error(f"Embeddings must be 2D array, got shape: {embeddings.shape}")
                return False
                
            if not metadata:
                logger.error("No metadata provided")
                return False
                
            if len(metadata) != embeddings.shape[0]:
                logger.error(f"Mismatch between embeddings ({embeddings.shape[0]}) and metadata ({len(metadata)}) counts")
                return False
                
            # Validate embedding dimension
            if embeddings.shape[1] != self.dimension:
                if self.index is None:
                    # Update dimension for new index
                    self.dimension = embeddings.shape[1]
                    logger.info(f"Updated dimension to {self.dimension}")
                else:
                    logger.error(f"Embedding dimension ({embeddings.shape[1]}) doesn't match existing index ({self.dimension})")
                    return False
            
            # Validate metadata structure
            for i, meta in enumerate(metadata):
                if not isinstance(meta, dict):
                    logger.error(f"Metadata item {i} is not a dictionary")
                    return False
                if 'text' not in meta:
                    logger.error(f"Metadata item {i} missing 'text' field")
                    return False
                if not meta['text'] or not meta['text'].strip():
                    logger.error(f"Metadata item {i} has empty text")
                    return False
            
            # Create or update index
            if self.index is None:
                self._create_index(embeddings.shape[1])
                
            # Convert to float32 (required by FAISS)
            try:
                embeddings_f32 = embeddings.astype(np.float32)
            except Exception as e:
                logger.error(f"Failed to convert embeddings to float32: {e}")
                return False
            
            # Validate embeddings for NaN or infinity
            if not np.isfinite(embeddings_f32).all():
                logger.error("Embeddings contain NaN or infinity values")
                return False
            
            # Add to existing index
            try:
                start_index = self.index.ntotal
                self.index.add(embeddings_f32)
                logger.info(f"Added {embeddings.shape[0]} vectors to index (total now: {self.index.ntotal})")
            except Exception as e:
                logger.error(f"Failed to add embeddings to index: {e}")
                return False
            
            # Update metadata with index positions
            try:
                for i, meta in enumerate(metadata):
                    meta['faiss_index'] = start_index + i
                
                self.metadata.extend(metadata)
            except Exception as e:
                logger.error(f"Failed to update metadata: {e}")
                # Try to remove the added vectors (if possible)
                try:
                    # Note: FAISS doesn't support removing recently added vectors easily
                    # This is a limitation we need to document
                    pass
                except:
                    pass
                return False
            
            # Save the updated index and metadata
            save_success = self.save()
            if not save_success:
                logger.error("Failed to save index after building")
                return False
            
            logger.info(f"Successfully built index with {embeddings.shape[0]} new vectors")
            return True
            
        except Exception as e:
            logger.error(f"Unexpected error in build: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def _create_index(self, dimension: int) -> None:
        """
        Create a new FAISS index with error handling.
        
        Args:
            dimension: Dimension of the embeddings
        """
        try:
            if dimension <= 0:
                raise ValueError(f"Invalid dimension: {dimension}")
                
            self.dimension = dimension
            
            # Use IndexFlatIP (Inner Product) for better similarity semantics
            # or IndexFlatL2 for L2 distance
            self.index = faiss.IndexFlatL2(dimension)
            
            logger.info(f"Created new FAISS index with dimension {dimension}")
            
        except Exception as e:
            logger.error(f"Failed to create FAISS index: {e}")
            raise FaissStoreError(f"Cannot create FAISS index: {e}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Search the index for similar vectors with comprehensive error handling.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of (score, metadata) tuples
        """
        try:
            # Validate inputs
            if query_embedding is None or query_embedding.size == 0:
                logger.error("Empty query embedding provided")
                return []
            
            if not isinstance(query_embedding, np.ndarray):
                logger.error("Query embedding must be numpy array")
                return []
                
            if top_k <= 0:
                logger.error(f"Invalid top_k value: {top_k}")
                return []
            
            # Check index availability
            if self.index is None or self.index.ntotal == 0:
                logger.warning("Index is empty or not loaded")
                return []
            
            # Validate embedding dimension
            if len(query_embedding.shape) == 1:
                query_embedding = query_embedding.reshape(1, -1)
            elif len(query_embedding.shape) != 2 or query_embedding.shape[0] != 1:
                logger.error(f"Invalid query embedding shape: {query_embedding.shape}")
                return []
            
            if query_embedding.shape[1] != self.dimension:
                logger.error(f"Query embedding dimension ({query_embedding.shape[1]}) doesn't match index ({self.dimension})")
                return []
            
            # Validate for NaN/infinity
            if not np.isfinite(query_embedding).all():
                logger.error("Query embedding contains NaN or infinity values")
                return []
            
            # Ensure proper data type
            query_embedding = query_embedding.astype(np.float32)
            
            # Limit top_k to available vectors
            actual_top_k = min(top_k, self.index.ntotal)
            
            # Search the index
            try:
                distances, indices = self.index.search(query_embedding, actual_top_k)
            except Exception as e:
                logger.error(f"FAISS search failed: {e}")
                return []
            
            # Validate search results
            if distances is None or indices is None:
                logger.error("FAISS search returned None")
                return []
            
            if len(distances.shape) != 2 or len(indices.shape) != 2:
                logger.error(f"Invalid search result shapes: distances={distances.shape}, indices={indices.shape}")
                return []
            
            # Convert distances to similarity scores
            try:
                # For L2 distance: similarity = 1 / (1 + distance)
                scores = 1 / (1 + distances[0])
                
                # Ensure scores are valid
                if not np.isfinite(scores).all():
                    logger.warning("Some similarity scores are invalid, filtering them out")
                    valid_mask = np.isfinite(scores)
                    scores = scores[valid_mask]
                    indices = indices[0][valid_mask]
                else:
                    indices = indices[0]
                
            except Exception as e:
                logger.error(f"Failed to convert distances to scores: {e}")
                return []
            
            # Retrieve metadata for the results
            results = []
            for i, idx in enumerate(indices):
                try:
                    if idx < 0:  # FAISS returns -1 for invalid indices
                        continue
                        
                    if idx >= len(self.metadata):
                        logger.warning(f"Index {idx} out of range for metadata (size: {len(self.metadata)})")
                        continue
                    
                    metadata_item = self.metadata[idx]
                    if not isinstance(metadata_item, dict):
                        logger.warning(f"Invalid metadata at index {idx}")
                        continue
                    
                    # Ensure score is valid
                    score = scores[i] if i < len(scores) else 0.0
                    if not np.isfinite(score):
                        logger.warning(f"Invalid score for result {i}")
                        continue
                    
                    results.append((float(score), metadata_item))
                    
                except Exception as e:
                    logger.warning(f"Error processing search result {i}: {e}")
                    continue
            
            logger.debug(f"Search completed: {len(results)} results returned")
            return results
            
        except Exception as e:
            logger.error(f"Unexpected error in search: {e}")
            logger.error(traceback.format_exc())
            return []
    
    def save(self) -> bool:
        """
        Save the index and metadata to disk with error handling.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate state before saving
            if self.index is None:
                logger.warning("No index to save")
                return True  # Not an error if there's nothing to save
            
            if len(self.metadata) != self.index.ntotal:
                logger.error(f"Metadata count ({len(self.metadata)}) doesn't match index size ({self.index.ntotal})")
                return False
            
            # Create backup of existing files
            backup_created = False
            if Path(self.index_path).exists():
                try:
                    backup_path = f"{self.index_path}.backup"
                    os.rename(self.index_path, backup_path)
                    backup_created = True
                except Exception as e:
                    logger.warning(f"Could not create index backup: {e}")
            
            meta_backup_created = False
            if Path(self.meta_path).exists():
                try:
                    meta_backup_path = f"{self.meta_path}.backup"
                    os.rename(self.meta_path, meta_backup_path)
                    meta_backup_created = True
                except Exception as e:
                    logger.warning(f"Could not create metadata backup: {e}")
            
            try:
                # Save FAISS index
                faiss.write_index(self.index, self.index_path)
                logger.info(f"Saved FAISS index to {self.index_path}")
                
                # Save metadata as JSONL
                with open(self.meta_path, 'w', encoding='utf-8') as f:
                    for meta in self.metadata:
                        try:
                            f.write(json.dumps(meta, ensure_ascii=False) + '\n')
                        except Exception as e:
                            logger.error(f"Failed to serialize metadata item: {e}")
                            raise
                
                logger.info(f"Saved {len(self.metadata)} metadata records to {self.meta_path}")
                
                # Remove backup files on success
                if backup_created:
                    try:
                        os.remove(f"{self.index_path}.backup")
                    except:
                        pass
                if meta_backup_created:
                    try:
                        os.remove(f"{self.meta_path}.backup")
                    except:
                        pass
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to save index: {e}")
                
                # Restore backups on failure
                if backup_created:
                    try:
                        os.rename(f"{self.index_path}.backup", self.index_path)
                        logger.info("Restored index backup")
                    except Exception as restore_error:
                        logger.error(f"Failed to restore index backup: {restore_error}")
                
                if meta_backup_created:
                    try:
                        os.rename(f"{self.meta_path}.backup", self.meta_path)
                        logger.info("Restored metadata backup")
                    except Exception as restore_error:
                        logger.error(f"Failed to restore metadata backup: {restore_error}")
                
                return False
                
        except Exception as e:
            logger.error(f"Unexpected error during save: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def load(self) -> bool:
        """
        Load the index and metadata from disk with comprehensive validation.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if files exist
            index_exists = Path(self.index_path).exists()
            meta_exists = Path(self.meta_path).exists()
            
            if not index_exists and not meta_exists:
                logger.info("No existing index or metadata files found")
                return False
            
            if not index_exists:
                logger.error(f"Index file {self.index_path} does not exist")
                return False
                
            if not meta_exists:
                logger.error(f"Metadata file {self.meta_path} does not exist")
                return False
            
            # Load FAISS index
            try:
                self.index = faiss.read_index(self.index_path)
                self.dimension = self.index.d
                logger.info(f"Loaded FAISS index from {self.index_path} (dimension: {self.dimension}, vectors: {self.index.ntotal})")
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}")
                raise IndexCorruptedError(f"Cannot load index file: {e}")
                
            # Validate index
            if not self.index.is_trained:
                logger.warning("Loaded index is not trained")
            
            if self.index.ntotal == 0:
                logger.warning("Loaded index is empty")
                
            # Load metadata
            self.metadata = []
            try:
                with open(self.meta_path, 'r', encoding='utf-8') as f:
                    line_number = 0
                    for line in f:
                        line_number += 1
                        try:
                            line = line.strip()
                            if not line:  # Skip empty lines
                                continue
                            meta_item = json.loads(line)
                            self.metadata.append(meta_item)
                        except json.JSONDecodeError as e:
                            logger.error(f"Invalid JSON on line {line_number} in {self.meta_path}: {e}")
                            continue
                        except Exception as e:
                            logger.error(f"Error processing line {line_number} in {self.meta_path}: {e}")
                            continue
                
                logger.info(f"Loaded {len(self.metadata)} metadata records from {self.meta_path}")
                
            except Exception as e:
                logger.error(f"Failed to load metadata: {e}")
                return False
                
            # Validate consistency between index and metadata
            if self.index.ntotal != len(self.metadata):
                error_msg = f"Index size ({self.index.ntotal}) doesn't match metadata count ({len(self.metadata)})"
                logger.error(error_msg)
                raise MetadataMismatchError(error_msg)
            
            # Validate metadata structure
            invalid_count = 0
            for i, meta in enumerate(self.metadata):
                if not isinstance(meta, dict):
                    logger.warning(f"Metadata item {i} is not a dictionary")
                    invalid_count += 1
                elif 'text' not in meta or not meta['text']:
                    logger.warning(f"Metadata item {i} has invalid text field")
                    invalid_count += 1
            
            if invalid_count > 0:
                logger.warning(f"Found {invalid_count} invalid metadata items out of {len(self.metadata)}")
                
            return True
            
        except (IndexCorruptedError, MetadataMismatchError):
            # These are expected errors that should be handled by caller
            raise
        except Exception as e:
            logger.error(f"Unexpected error during load: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the index.
        
        Returns:
            Dictionary of index statistics
        """
        try:
            stats = {
                "total_vectors": self.index.ntotal if self.index else 0,
                "dimension": self.dimension,
                "is_trained": self.index.is_trained if self.index else False,
                "metadata_count": len(self.metadata),
                "index_loaded": self.index is not None
            }
            
            # Count documents by source
            sources = {}
            pages_per_source = {}
            
            try:
                for meta in self.metadata:
                    source = meta.get('source', 'unknown')
                    page = meta.get('page', 'unknown')
                    
                    # Count chunks per source
                    sources[source] = sources.get(source, 0) + 1
                    
                    # Track pages per source
                    if source not in pages_per_source:
                        pages_per_source[source] = set()
                    pages_per_source[source].add(page)
                
                # Convert page sets to counts
                for source in pages_per_source:
                    pages_per_source[source] = len(pages_per_source[source])
                
                stats['sources'] = sources
                stats['pages_per_source'] = pages_per_source
                stats['unique_sources'] = len(sources)
                
            except Exception as e:
                logger.error(f"Error calculating source statistics: {e}")
                stats['sources'] = {}
                stats['error'] = "Failed to calculate source statistics"
            
            # Add file size information
            try:
                if Path(self.index_path).exists():
                    index_size = Path(self.index_path).stat().st_size
                    stats['index_file_size_bytes'] = index_size
                    
                if Path(self.meta_path).exists():
                    meta_size = Path(self.meta_path).stat().st_size
                    stats['metadata_file_size_bytes'] = meta_size
            except Exception as e:
                logger.warning(f"Could not get file sizes: {e}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {"error": str(e)}
    
    def remove_document(self, source: str) -> int:
        """
        Remove all chunks from a specific document with error handling.
        
        Args:
            source: Source document to remove
            
        Returns:
            Number of chunks removed
        """
        try:
            if not self.index or not self.metadata:
                logger.warning("No index or metadata to remove from")
                return 0
            
            if not source or not source.strip():
                logger.error("Invalid source provided for removal")
                return 0
                
            # Find indices to remove
            indices_to_remove = []
            try:
                for i, meta in enumerate(self.metadata):
                    if meta.get('source') == source:
                        indices_to_remove.append(i)
            except Exception as e:
                logger.error(f"Error finding indices to remove: {e}")
                return 0
                
            if not indices_to_remove:
                logger.info(f"No chunks found for document: {source}")
                return 0
            
            logger.info(f"Found {len(indices_to_remove)} chunks to remove for document: {source}")
            
            # FAISS doesn't support efficient removal, so we need to rebuild
            try:
                # Get all embeddings except those to remove
                all_embeddings = []
                remaining_metadata = []
                
                for i in range(self.index.ntotal):
                    if i not in indices_to_remove:
                        # Get embedding by reconstructing it
                        # Note: This only works for certain index types
                        try:
                            embedding = self.index.reconstruct(i)
                            all_embeddings.append(embedding)
                            remaining_metadata.append(self.metadata[i])
                        except Exception as e:
                            logger.error(f"Cannot reconstruct embedding {i}: {e}")
                            # Alternative: rebuild entire index from scratch
                            logger.error("Index type doesn't support reconstruction. Manual rebuild required.")
                            return 0
                
                if all_embeddings:
                    # Rebuild index with remaining embeddings
                    embeddings_array = np.array(all_embeddings)
                    self.index = None  # Clear existing index
                    self.metadata = []
                    
                    success = self.build(embeddings_array, remaining_metadata)
                    if not success:
                        logger.error("Failed to rebuild index after removal")
                        return 0
                else:
                    # All embeddings were removed
                    self.index = None
                    self.metadata = []
                    
                    # Remove index files
                    try:
                        if os.path.exists(self.index_path):
                            os.remove(self.index_path)
                        if os.path.exists(self.meta_path):
                            os.remove(self.meta_path)
                    except Exception as e:
                        logger.warning(f"Failed to remove empty index files: {e}")
                
                logger.info(f"Successfully removed {len(indices_to_remove)} chunks from document {source}")
                return len(indices_to_remove)
                
            except Exception as e:
                logger.error(f"Failed to remove chunks: {e}")
                logger.error(traceback.format_exc())
                return 0
            
        except Exception as e:
            logger.error(f"Unexpected error during document removal: {e}")
            logger.error(traceback.format_exc())
            return 0
    
    def verify_integrity(self) -> Dict[str, Any]:
        """
        Verify the integrity of the index and metadata.
        
        Returns:
            Dictionary with integrity check results
        """
        results = {
            "valid": True,
            "issues": [],
            "warnings": [],
            "stats": {}
        }
        
        try:
            # Check if index is loaded
            if self.index is None:
                results["issues"].append("Index not loaded")
                results["valid"] = False
                return results
            
            # Check index-metadata consistency
            if self.index.ntotal != len(self.metadata):
                results["issues"].append(f"Index size ({self.index.ntotal}) != metadata count ({len(self.metadata)})")
                results["valid"] = False
            
            # Check metadata structure
            invalid_metadata = 0
            empty_text = 0
            
            for i, meta in enumerate(self.metadata):
                if not isinstance(meta, dict):
                    invalid_metadata += 1
                elif 'text' not in meta or not meta['text'] or not meta['text'].strip():
                    empty_text += 1
            
            if invalid_metadata > 0:
                results["issues"].append(f"{invalid_metadata} metadata items are invalid")
                results["valid"] = False
            
            if empty_text > 0:
                results["warnings"].append(f"{empty_text} metadata items have empty text")
            
            # Check for duplicate chunks
            try:
                chunk_ids = [meta.get('id') for meta in self.metadata if meta.get('id')]
                if len(chunk_ids) != len(set(chunk_ids)):
                    duplicates = len(chunk_ids) - len(set(chunk_ids))
                    results["warnings"].append(f"{duplicates} duplicate chunk IDs found")
            except Exception as e:
                results["warnings"].append(f"Could not check for duplicates: {e}")
            
            # Add statistics
            results["stats"] = self.get_stats()
            
        except Exception as e:
            logger.error(f"Error during integrity check: {e}")
            results["issues"].append(f"Integrity check failed: {e}")
            results["valid"] = False
        
        return results


class FaissStoreWithDB(FaissStore):
    """
    Enhanced FAISS store with SQLite metadata management and better error handling.
    """
    
    def __init__(self, index_path: str, db_path: str, dimension: int = 384):
        """
        Initialize with SQLite database for metadata.
        
        Args:
            index_path: Path to save/load the FAISS index
            db_path: Path to SQLite database
            dimension: Dimension of embeddings
        """
        super().__init__(index_path, "", dimension)
        self.db_path = db_path
        self.conn = None
        
        try:
            self._init_db()
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise FaissStoreError(f"Database initialization failed: {e}")
    
    def _init_db(self) -> None:
        """Initialize the SQLite database with comprehensive error handling."""
        try:
            # Create database directory
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Connect to database
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.execute("PRAGMA foreign_keys = ON")
            cursor = self.conn.cursor()
            
            # Create tables if they don't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chunk_id TEXT UNIQUE NOT NULL,
                    text TEXT NOT NULL,
                    source TEXT NOT NULL,
                    page INTEGER,
                    chunk_index INTEGER,
                    total_chunks INTEGER,
                    embedding_index INTEGER UNIQUE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source) REFERENCES documents(source)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT UNIQUE NOT NULL,
                    page_count INTEGER DEFAULT 0,
                    chunk_count INTEGER DEFAULT 0,
                    ingested_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunks_embedding_index ON chunks(embedding_index)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source)')
            
            self.conn.commit()
            logger.info("Database initialized successfully")
            
        except sqlite3.Error as e:
            logger.error(f"SQLite error during initialization: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during database initialization: {e}")
            raise
    
    def build(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> bool:
        """
        Build or update the FAISS index and store metadata in SQLite with transactions.
        """
        if not self.conn:
            logger.error("Database connection not available")
            return False
            
        try:
            # Validate inputs first
            if embeddings.shape[0] == 0 or not metadata:
                logger.error("No embeddings or metadata provided")
                return False
                
            if self.index is None:
                self._create_index(embeddings.shape[1])
                
            # Start database transaction
            cursor = self.conn.cursor()
            cursor.execute("BEGIN TRANSACTION")
            
            try:
                start_index = self.index.ntotal
                
                # Add to FAISS index
                self.index.add(embeddings.astype(np.float32))
                
                # Prepare document tracking
                sources_to_update = set()
                
                # Add chunks to database
                for i, meta in enumerate(metadata):
                    embedding_index = start_index + i
                    source = meta.get('source', 'unknown')
                    sources_to_update.add(source)
                    
                    try:
                        cursor.execute('''
                            INSERT OR REPLACE INTO chunks 
                            (chunk_id, text, source, page, chunk_index, total_chunks, embedding_index)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            meta.get('id', f"chunk_{embedding_index}"),
                            meta.get('text', ''),
                            source,
                            meta.get('page'),
                            meta.get('chunk_index'),
                            meta.get('total_chunks'),
                            embedding_index
                        ))
                    except sqlite3.Error as e:
                        logger.error(f"Failed to insert chunk {i}: {e}")
                        raise
                
                # Update document statistics
                for source in sources_to_update:
                    try:
                        # Count chunks and pages for this source
                        cursor.execute('''
                            SELECT COUNT(*), COUNT(DISTINCT page) 
                            FROM chunks WHERE source = ?
                        ''', (source,))
                        
                        chunk_count, page_count = cursor.fetchone()
                        
                        cursor.execute('''
                            INSERT OR REPLACE INTO documents 
                            (source, chunk_count, page_count, last_updated)
                            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                        ''', (source, chunk_count, page_count))
                        
                    except sqlite3.Error as e:
                        logger.error(f"Failed to update document stats for {source}: {e}")
                        raise
                
                # Commit database transaction
                self.conn.commit()
                
                # Save FAISS index
                if not self.save():
                    raise RuntimeError("Failed to save FAISS index")
                
                logger.info(f"Successfully added {embeddings.shape[0]} vectors to index and database")
                return True
                
            except Exception as e:
                # Rollback on any error
                cursor.execute("ROLLBACK")
                logger.error(f"Transaction rolled back due to error: {e}")
                raise
                
        except Exception as e:
            logger.error(f"Failed to build index with database: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Search the index and retrieve metadata from database with error handling.
        """
        if not self.conn:
            logger.error("Database connection not available")
            return []
            
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index not available or empty")
            return []
            
        try:
            # Perform FAISS search using parent method's validation
            faiss_results = super().search(query_embedding, top_k)
            
            if not faiss_results:
                return []
            
            # Get metadata from database
            results = []
            cursor = self.conn.cursor()
            
            for score, _ in faiss_results:
                # Find the embedding index for this result
                # Note: This is a simplified approach - in practice you'd want
                # to track the mapping between FAISS indices and database records
                pass
            
            # For now, fall back to the parent implementation
            # A full implementation would require tracking embedding indices
            logger.warning("Database search not fully implemented, using memory-based search")
            return faiss_results
            
        except Exception as e:
            logger.error(f"Database search failed: {e}")
            logger.error(traceback.format_exc())
            return []
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics from the database."""
        if not self.conn:
            return {"error": "Database not available"}
        
        try:
            cursor = self.conn.cursor()
            
            # Get document counts
            cursor.execute("SELECT COUNT(*) FROM documents")
            doc_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM chunks")
            chunk_count = cursor.fetchone()[0]
            
            # Get recent activity
            cursor.execute('''
                SELECT source, chunk_count, page_count, last_updated 
                FROM documents 
                ORDER BY last_updated DESC 
                LIMIT 10
            ''')
            recent_docs = cursor.fetchall()
            
            return {
                "document_count": doc_count,
                "chunk_count": chunk_count,
                "recent_documents": [
                    {
                        "source": row[0],
                        "chunks": row[1],
                        "pages": row[2],
                        "last_updated": row[3]
                    } for row in recent_docs
                ]
            }
            
        except sqlite3.Error as e:
            logger.error(f"Database stats query failed: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error getting database stats: {e}")
            return {"error": str(e)}
    
    def __del__(self):
        """Clean up database connection."""
        try:
            if self.conn:
                self.conn.close()
                logger.debug("Database connection closed")
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")