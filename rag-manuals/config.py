# rag-manuals/config.py
import os
from typing import Optional
from pydantic import BaseSettings, Field, validator
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Configuration
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    api_reload: bool = Field(False, env="API_RELOAD")
    api_secret_key: str = Field(..., env="API_SECRET_KEY")
    max_file_size: int = Field(50 * 1024 * 1024, env="MAX_FILE_SIZE")  # 50MB
    
    # LLM Configuration
    llm_backend: str = Field("http", env="LLM_BACKEND")
    llm_api_url: str = Field("http://localhost:8080", env="LLM_API_URL")
    llm_model_name: str = Field("mistral-7b-instruct", env="LLM_MODEL_NAME")
    llm_timeout: float = Field(60.0, env="LLM_TIMEOUT")
    llm_max_tokens: int = Field(512, env="LLM_MAX_TOKENS")
    llm_temperature: float = Field(0.1, env="LLM_TEMPERATURE")
    llm_do_sample: bool = Field(True, env="LLM_DO_SAMPLE")
    
    # Embedding Configuration
    embedding_model: str = Field("all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    embedding_device: str = Field("auto", env="EMBEDDING_DEVICE")
    embedding_batch_size: int = Field(32, env="EMBEDDING_BATCH_SIZE")
    embedding_cache_enabled: bool = Field(True, env="EMBEDDING_CACHE_ENABLED")
    
    # Vector Store Configuration
    index_path: str = Field("./data/index/faiss.index", env="INDEX_PATH")
    meta_path: str = Field("./data/index/meta.jsonl", env="META_PATH")
    db_path: str = Field("./data/index/metadata.db", env="DB_PATH")
    index_dimension: int = Field(384, env="INDEX_DIMENSION")
    
    # Ingestion Configuration
    chunk_size: int = Field(1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(200, env="CHUNK_OVERLAP")
    min_chunk_length: int = Field(50, env="MIN_CHUNK_LENGTH")
    max_chunk_length: int = Field(2000, env="MAX_CHUNK_LENGTH")
    remove_headers: bool = Field(True, env="REMOVE_HEADERS")
    remove_footers: bool = Field(True, env="REMOVE_FOOTERS")
    remove_page_numbers: bool = Field(True, env="REMOVE_PAGE_NUMBERS")
    
    # Search Configuration
    default_top_k: int = Field(5, env="DEFAULT_TOP_K")
    max_top_k: int = Field(20, env="MAX_TOP_K")
    min_similarity_score: float = Field(0.0, env="MIN_SIMILARITY_SCORE")
    
    # CORS Configuration
    cors_origins: list = Field(["http://localhost:3000", "http://localhost:3001"], env="CORS_ORIGINS")
    
    @validator('cors_origins', pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v
    
    # Logging Configuration
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_file: str = Field("./logs/manualmind.log", env="LOG_FILE")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()

def get_settings() -> Settings:
    """Get the application settings."""
    return settings