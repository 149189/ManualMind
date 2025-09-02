# rag-manuals/index/faiss_utils.py
import faiss
import numpy as np
import json
from typing import List, Tuple, Dict, Any
from pathlib import Path




class FaissStore:
	def __init__(self, index_path: str = "index/faiss.index", meta_path: str = "index/meta.jsonl"):
		self.index_path = index_path
		self.meta_path = meta_path
		self.index = None
		self.metadata: List[Dict[str, Any]] = []

	def build(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
		n, d = embeddings.shape
		faiss.normalize_L2(embeddings)
		self.index = faiss.IndexFlatIP(d)
		self.index.add(embeddings)
		Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
		faiss.write_index(self.index, self.index_path)
		Path(self.meta_path).parent.mkdir(parents=True, exist_ok=True)
		with open(self.meta_path, "w", encoding="utf-8") as f:
			for m in metadata:
				f.write(json.dumps(m, ensure_ascii=False) + "\n")
		self.metadata = metadata

	def load(self):
		if not Path(self.index_path).exists() or not Path(self.meta_path).exists():
			raise FileNotFoundError("Index or metadata not found.")
		self.index = faiss.read_index(self.index_path)
		self.metadata = []
		with open(self.meta_path, "r", encoding="utf-8") as f:
			for line in f:
				self.metadata.append(json.loads(line))

	def search(self, query_vec: np.ndarray, top_k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
		if self.index is None:
			self.load()
		q = query_vec.reshape(1, -1).astype("float32")
		faiss.normalize_L2(q)
		scores, idxs = self.index.search(q, top_k)
		out = []
		for s, idx in zip(scores[0], idxs[0]):
			if idx < 0:
				continue
			out.append((float(s), self.metadata[idx]))
		return out