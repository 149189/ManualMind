# rag-manuals/index/build_index.py
from ..embeddings.embedder import Embedder
from .faiss_utils import FaissStore
from typing import List, Dict
import numpy as np




def build_index_from_chunks(chunks: List[Dict], model_name: str = "all-mpnet-base-v2") -> FaissStore:
	texts = [c["text"] for c in chunks]
	embedder = Embedder(model_name)
	embeddings = embedder.embed(texts)
	store = FaissStore()
	store.build(embeddings, chunks)
	return store