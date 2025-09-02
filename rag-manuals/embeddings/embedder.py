# rag-manuals/embeddings/embedder.py
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List




class Embedder:
	def __init__(self, model_name: str = "all-mpnet-base-v2"):
		self.model_name = model_name
		self.model = SentenceTransformer(model_name)

	def embed(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
		embs = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
		return embs.astype("float32")