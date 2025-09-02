# rag-manuals/embeddings/embedder.py
import logging
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

logger = logging.getLogger(__name__)

class Embedder:
    """
    Handles text embedding using sentence-transformers models.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None):
        """
        Initialize the embedder with a specific model.
        
        Args:
            model_name: Name of the sentence-transformers model
            device: Device to run the model on (cuda/cpu)
        """
        self.model_name = model_name
        self.device = device if device else "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            logger.info(f"Loading embedding model: {model_name} on {self.device}")
            self.model = SentenceTransformer(model_name, device=self.device)
            logger.info(f"Successfully loaded model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model {model_name}: {e}")
            raise
    
    def embed(self, texts: List[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            **kwargs: Additional arguments to pass to the model
            
        Returns:
            Numpy array of embeddings
        """
        if not texts:
            return np.array([])
            
        try:
            # Ensure all texts are strings
            texts = [str(text) for text in texts]
            
            # Generate embeddings
            embeddings = self.model.encode(
                texts, 
                batch_size=batch_size, 
                convert_to_numpy=True,
                show_progress_bar=False,
                **kwargs
            )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings.
        
        Returns:
            Dimension of the embeddings
        """
        # Create a dummy embedding to get the dimension
        dummy_text = "test"
        embedding = self.embed([dummy_text])
        return embedding.shape[1]
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            return cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {e}")
            return 0.0

class CachedEmbedder(Embedder):
    """
    Embedder with caching to avoid recomputing the same embeddings.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None):
        super().__init__(model_name, device)
        self.cache = {}
        
    def embed(self, texts: List[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        """
        Generate embeddings with caching.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            **kwargs: Additional arguments to pass to the model
            
        Returns:
            Numpy array of embeddings
        """
        if not texts:
            return np.array([])
            
        # Separate cached and non-cached texts
        uncached_texts = []
        uncached_indices = []
        embeddings = np.zeros((len(texts), self.get_embedding_dimension()))
        
        for i, text in enumerate(texts):
            text = str(text)
            if text in self.cache:
                embeddings[i] = self.cache[text]
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Embed uncached texts
        if uncached_texts:
            new_embeddings = super().embed(uncached_texts, batch_size, **kwargs)
            
            # Update cache and result array
            for idx, text in enumerate(uncached_texts):
                self.cache[text] = new_embeddings[idx]
                embeddings[uncached_indices[idx]] = new_embeddings[idx]
        
        return embeddings	