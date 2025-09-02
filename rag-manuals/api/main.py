# rag-manuals/api/main.py
import os
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
import tempfile
import json


from ..ingestion.pdf_to_text import extract_pages
from ..ingestion.chunker import chunk_page
from ..embeddings.embedder import Embedder
from ..index.faiss_utils import FaissStore
from .prompt_templates import QUERY_SYSTEM, format_retrieved_snippets


# LLM backends: local (transformers) or http (docker inference server)
LLM_BACKEND = os.environ.get("LLM_BACKEND", "local") # or "http"
LLM_API_URL = os.environ.get("LLM_API_URL")
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "gpt2")


app = FastAPI(title="ManualMind API")


# Simple globals for demo (replace with proper persistence in prod)
STORE = FaissStore(index_path="index/faiss.index", meta_path="index/meta.jsonl")
EMBEDDER = Embedder()




class IngestResponse(BaseModel):
	ingested_chunks: int
	message: str




class QueryRequest(BaseModel):
	q: str
	top_k: int = 5




class QueryResponse(BaseModel):
	answer: str
	citations: List[str]
	confidence: float
	snippets: List[dict]




@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
	# save file temporarily
	suffix = ".pdf"
	with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
		content = await file.read()
		tmp.write(content)
		tmp_path = tmp.name
	pages = extract_pages(tmp_path)
	chunks = []
	for p in pages:
		chunks.extend(chunk_page(p))
	if not chunks:
		return IngestResponse(ingested_chunks=0, message="No text extracted")
	texts = [c["text"] for c in chunks]
	embs = EMBEDDER.embed(texts)
	STORE.build(embeddings=embs, metadata=chunks)
	return IngestResponse(ingested_chunks=len(chunks), message="Indexed chunks")