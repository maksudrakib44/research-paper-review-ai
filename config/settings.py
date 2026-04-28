"""
config/settings.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Groq API
    groq_api_key: str = Field(..., description="Groq API key")
    llm_model: str = Field("llama-3.3-70b-versatile")
    llm_max_tokens: int = Field(1024)
    llm_temperature: float = Field(0.1)
    
    # Embedding
    embedding_model: str = Field("all-MiniLM-L6-v2")
    
    # Vector DB
    chroma_persist_dir: Path = Field(Path("./data/chroma_db"))
    chroma_collection_name: str = Field("research_papers")
    
    # Chunking
    chunk_size: int = Field(1000)
    chunk_overlap: int = Field(200)
    max_chunks_per_doc: int = Field(500)
    
    # Retrieval
    top_k_dense: int = Field(5)
    top_k_bm25: int = Field(5)
    top_k_rerank: int = Field(3)
    hybrid_alpha: float = Field(0.6)
    
    # Ground Truth
    gt_store_path: Path = Field(Path("./data/ground_truth/research_gt.json"))
    gt_eval_threshold: float = Field(0.75)
    
    # Logging
    log_level: str = Field("INFO")
    log_file: Path = Field(Path("./logs/research_rag.log"))
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()