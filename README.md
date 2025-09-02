# ManualMind
**ManualMind** is an AI-powered FAQ assistant that leverages Retrieval-Augmented Generation (RAG) to answer questions directly from product manuals. It ingests PDFs, indexes their content, and provides precise, cited answers with confidence scores—making technical support faster, smarter, and more reliable.
                                                            
```
rag-manuals/
├── ingestion/ # PDF processing and text extraction
├── embeddings/ # Text embedding generation
├── index/ # FAISS index management
├── api/ # FastAPI backend
├── tests/ # Unit tests
└── README.md
```                                                            


## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### API Server
Start the FastAPI server:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Building Index
Use the standalone script to build an index from PDFs:

```bash
python index/build_index.py path/to/manuals.pdf
```

Or for a directory of PDFs:

```bash
python index/build_index.py path/to/manuals/
```

### Docker Deployment
Deploy with Docker Compose:

```bash
docker-compose up -d
```

## API Endpoints

- **POST /ingest**: Upload and process a PDF manual  
- **POST /query**: Query the knowledge base  
- **GET /health**: Health check  
- **GET /stats**: Index statistics  

## Configuration

Environment variables:

- `LLM_BACKEND`: LLM integration type (http/local)  
- `LLM_API_URL`: URL for LLM HTTP service  
- `INDEX_PATH`: Path to FAISS index  
- `API_SECRET_KEY`: API authentication key  

## Development

Run tests:

```bash
pytest tests/
```

---

These implementations provide:

1. **Robust error handling** with comprehensive logging  
2. **Additional features** like image extraction, semantic chunking, and SQLite metadata storage  
3. **Validation** at each step of the processing pipeline  
4. **Caching** for embeddings to improve performance  
5. **Standalone scripts** for index building  
6. **Unit tests** to ensure reliability  

The code follows best practices for production-ready applications and should integrate well with your existing `main.py`.
