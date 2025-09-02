# rag-manuals/index/faiss_utils.py
import logging
import json
import faiss
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import sqlite3
from datetime import datetime

logger = logging.getLogger(__name__)

class FaissStore:
    """
    Manages FAISS index and associated metadata.
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
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        
    def build(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> bool:
        """
        Build or update the FAISS index.
        
        Args:
            embeddings: Array of embeddings
            metadata: List of metadata dictionaries
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if embeddings.shape[0] == 0:
                logger.error("No embeddings provided for index building")
                return False
                
            if len(metadata) != embeddings.shape[0]:
                logger.error("Mismatch between embeddings and metadata counts")
                return False
                
            # Create or update index
            if self.index is None:
                self._create_index(embeddings.shape[1])
                
            # Add to existing index
            self.index.add(embeddings.astype(np.float32))
            self.metadata.extend(metadata)
            
            # Save the updated index and metadata
            self.save()
            
            logger.info(f"Added {embeddings.shape[0]} vectors to index")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build index: {e}")
            return False
    
    def _create_index(self, dimension: int) -> None:
        """
        Create a new FAISS index.
        
        Args:
            dimension: Dimension of the embeddings
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        logger.info(f"Created new FAISS index with dimension {dimension}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Search the index for similar vectors.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of (score, metadata) tuples
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty or not loaded")
            return []
            
        try:
            # Ensure query embedding is the right shape and type
            query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
            
            # Search the index
            distances, indices = self.index.search(query_embedding, top_k)
            
            # Convert distances to similarity scores (1 / (1 + distance))
            scores = 1 / (1 + distances[0])
            
            # Retrieve metadata for the results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx >= 0 and idx < len(self.metadata):  # Valid index
                    results.append((scores[i], self.metadata[idx]))
                elif idx != -1:  # FAISS returns -1 for invalid indices
                    logger.warning(f"Invalid index {idx} returned from FAISS search")
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def save(self) -> bool:
        """
        Save the index and metadata to disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Save FAISS index
            if self.index is not None:
                faiss.write_index(self.index, self.index_path)
                logger.info(f"Saved FAISS index to {self.index_path}")
            
            # Save metadata as JSONL
            with open(self.meta_path, 'w', encoding='utf-8') as f:
                for meta in self.metadata:
                    f.write(json.dumps(meta) + '\n')
                    
            logger.info(f"Saved metadata to {self.meta_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            return False
    
    def load(self) -> bool:
        """
        Load the index and metadata from disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load FAISS index
            if Path(self.index_path).exists():
                self.index = faiss.read_index(self.index_path)
                self.dimension = self.index.d
                logger.info(f"Loaded FAISS index from {self.index_path}")
            else:
                logger.warning(f"Index file {self.index_path} does not exist")
                return False
                
            # Load metadata
            self.metadata = []
            if Path(self.meta_path).exists():
                with open(self.meta_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        self.metadata.append(json.loads(line.strip()))
                logger.info(f"Loaded {len(self.metadata)} metadata records from {self.meta_path}")
            else:
                logger.warning(f"Metadata file {self.meta_path} does not exist")
                return False
                
            # Validate that index and metadata are consistent
            if self.index.ntotal != len(self.metadata):
                logger.error(f"Index size ({self.index.ntotal}) doesn't match metadata count ({len(self.metadata)})")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the index.
        
        Returns:
            Dictionary of index statistics
        """
        stats = {
            "total_vectors": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
            "is_trained": self.index.is_trained if self.index else False,
            "metadata_count": len(self.metadata)
        }
        
        # Count documents by source
        sources = {}
        for meta in self.metadata:
            source = meta.get('source', 'unknown')
            sources[source] = sources.get(source, 0) + 1
            
        stats['sources'] = sources
        
        return stats
    
    def remove_document(self, source: str) -> int:
        """
        Remove all chunks from a specific document.
        
        Args:
            source: Source document to remove
            
        Returns:
            Number of chunks removed
        """
        if not self.index or not self.metadata:
            return 0
            
        try:
            # Find indices to remove
            indices_to_remove = [
                i for i, meta in enumerate(self.metadata) 
                if meta.get('source') == source
            ]
            
            if not indices_to_remove:
                return 0
                
            # Remove from index
            remove_ids = np.array(indices_to_remove, dtype=np.int64)
            self.index.remove_ids(remove_ids)
            
            # Remove from metadata
            for i in sorted(indices_to_remove, reverse=True):
                del self.metadata[i]
                
            # Save the updated index
            self.save()
            
            logger.info(f"Removed {len(indices_to_remove)} chunks from document {source}")
            return len(indices_to_remove)
            
        except Exception as e:
            logger.error(f"Failed to remove document {source}: {e}")
            return 0

class FaissStoreWithDB(FaissStore):
    """
    Enhanced FAISS store with SQLite metadata management.
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
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize the SQLite database."""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT UNIQUE,
                text TEXT,
                source TEXT,
                page INTEGER,
                chunk_index INTEGER,
                total_chunks INTEGER,
                embedding_index INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT UNIQUE,
                page_count INTEGER,
                ingested_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.commit()
    
    def build(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> bool:
        """
        Build or update the FAISS index and store metadata in SQLite.
        """
        try:
            if embeddings.shape[0] == 0:
                return False
                
            if self.index is None:
                self._create_index(embeddings.shape[1])
                
            # Start transaction
            cursor = self.conn.cursor()
            start_index = self.index.ntotal
            
            # Add to index
            self.index.add(embeddings.astype(np.float32))
            
            # Add to database
            for i, meta in enumerate(metadata):
                cursor.execute('''
                    INSERT OR REPLACE INTO chunks 
                    (chunk_id, text, source, page, chunk_index, total_chunks, embedding_index)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    meta.get('id'),
                    meta.get('text'),
                    meta.get('source'),
                    meta.get('page'),
                    meta.get('chunk_index'),
                    meta.get('total_chunks'),
                    start_index + i
                ))
            
            # Update document info
            sources = set(m.get('source') for m in metadata)
            for source in sources:
                cursor.execute('''
                    INSERT OR REPLACE INTO documents (source, page_count)
                    VALUES (?, COALESCE((SELECT page_count FROM documents WHERE source = ?), 0) + 1)
                ''', (source, source))
            
            self.conn.commit()
            self.save()
            
            logger.info(f"Added {embeddings.shape[0]} vectors to index and database")
            return True
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to build index with database: {e}")
            return False
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Search the index and retrieve metadata from database.
        """
        if self.index is None or self.index.ntotal == 0:
            return []
            
        try:
            query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
            distances, indices = self.index.search(query_embedding, top_k)
            scores = 1 / (1 + distances[0])
            
            # Get metadata from database
            results = []
            cursor = self.conn.cursor()
            
            for i, idx in enumerate(indices[0]):
                if idx < 0:
                    continue
                    
                cursor.execute('''
                    SELECT chunk_id, text, source, page, chunk_index, total_chunks
                    FROM chunks WHERE embedding_index = ?
                ''', (idx,))
                
                row = cursor.fetchone()
                if row:
                    meta = {
                        'id': row[0],
                        'text': row[1],
                        'source': row[2],
                        'page': row[3],
                        'chunk_index': row[4],
                        'total_chunks': row[5]
                    }
                    results.append((scores[i], meta))
            
            return results
            
        except Exception as e:
            logger.error(f"Search with database failed: {e}")
            return []