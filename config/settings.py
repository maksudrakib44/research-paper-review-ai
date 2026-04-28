"""
config/settings.py - Research Paper RAG Configuration
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Research Paper RAG Settings"""
    
    # ============================================================
    # Groq API (Free, Fast LLM)
    # ============================================================
    groq_api_key: str = Field(..., description="Groq API key")
    llm_model: str = Field("llama-3.3-70b-versatile", description="Groq model ID")
    llm_max_tokens: int = Field(1024, ge=256, le=8192)
    llm_temperature: float = Field(0.1, ge=0.0, le=1.0)
    
    # ============================================================
    # Embedding (Simple TF-IDF - No heavy dependencies)
    # ============================================================
    embedding_model: str = Field("simple", description="Embedding model (simple TF-IDF)")
    
    # ============================================================
    # Vector Database
    # ============================================================
    chroma_persist_dir: Path = Field(Path("./data/chroma_db"))
    chroma_collection_name: str = Field("research_papers")
    
    # ============================================================
    # Chunking Strategy
    # ============================================================
    chunk_size: int = Field(1500, ge=256, le=4096, description="Chunk size in characters")
    chunk_overlap: int = Field(200, ge=0, le=1000, description="Overlap between chunks")
    max_chunks_per_doc: int = Field(500, ge=10, le=2000)
    
    # ============================================================
    # Retrieval Settings
    # ============================================================
    top_k_dense: int = Field(5, ge=1, le=50)
    top_k_bm25: int = Field(5, ge=1, le=50)
    top_k_rerank: int = Field(3, ge=1, le=20)
    hybrid_alpha: float = Field(0.6, ge=0.0, le=1.0)
    
    # ============================================================
    # Ground Truth
    # ============================================================
    gt_store_path: Path = Field(Path("./data/ground_truth/research_gt.json"))
    gt_eval_threshold: float = Field(0.75, ge=0.0, le=1.0)
    
    # ============================================================
    # Logging
    # ============================================================
    log_level: str = Field("INFO")
    log_file: Path = Field(Path("./logs/research_rag.log"))
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()