"""
config/settings.py - Research Paper RAG Configuration
Works with both local .env and Streamlit Cloud secrets
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
    """Research Paper RAG Settings"""
    
    # Groq API (Required)
    groq_api_key: str = Field(..., description="Groq API key")
    llm_model: str = Field("llama-3.3-70b-versatile", description="Groq model ID")
    llm_max_tokens: int = Field(1024, ge=256, le=8192)
    llm_temperature: float = Field(0.1, ge=0.0, le=1.0)
    
    # Embedding
    embedding_model: str = Field("simple", description="Embedding model")
    
    # Vector Database
    chroma_persist_dir: Path = Field(Path("./data/chroma_db"))
    chroma_collection_name: str = Field("research_papers")
    
    # Chunking
    chunk_size: int = Field(1500, ge=256, le=4096)
    chunk_overlap: int = Field(200, ge=0, le=1000)
    max_chunks_per_doc: int = Field(500, ge=10, le=2000)
    
    # Retrieval
    top_k_dense: int = Field(5, ge=1, le=50)
    top_k_bm25: int = Field(5, ge=1, le=50)
    top_k_rerank: int = Field(3, ge=1, le=20)
    hybrid_alpha: float = Field(0.6, ge=0.0, le=1.0)
    
    # Ground Truth
    gt_store_path: Path = Field(Path("./data/ground_truth/research_gt.json"))
    gt_eval_threshold: float = Field(0.75, ge=0.0, le=1.0)
    
    # Logging
    log_level: str = Field("INFO")
    log_file: Path = Field(Path("./logs/research_rag.log"))
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Load settings - try .env first (local), then Streamlit secrets (cloud)
def load_settings():
    """Load settings from .env or Streamlit secrets"""
    
    # First, try loading from .env file (local development)
    try:
        settings = Settings()
        if settings.groq_api_key and settings.groq_api_key != "":
            print("✅ Settings loaded from .env file")
            return settings
    except Exception as e:
        print(f"⚠️ .env loading failed: {e}")
    
    # Second, try loading from Streamlit secrets (cloud deployment)
    try:
        import streamlit as st
        
        # Check if running on Streamlit Cloud (secrets exist)
        if hasattr(st, "secrets") and st.secrets:
            groq_key = st.secrets.get("GROQ_API_KEY")
            if groq_key:
                settings = Settings(
                    groq_api_key=groq_key,
                    llm_model=st.secrets.get("LLM_MODEL", "llama-3.3-70b-versatile"),
                    llm_max_tokens=int(st.secrets.get("LLM_MAX_TOKENS", 1024)),
                    llm_temperature=float(st.secrets.get("LLM_TEMPERATURE", 0.1)),
                    embedding_model=st.secrets.get("EMBEDDING_MODEL", "simple"),
                    chunk_size=int(st.secrets.get("CHUNK_SIZE", 1500)),
                    top_k_dense=int(st.secrets.get("TOP_K_DENSE", 5)),
                    top_k_rerank=int(st.secrets.get("TOP_K_RERANK", 3)),
                    hybrid_alpha=float(st.secrets.get("HYBRID_ALPHA", 0.6)),
                    gt_eval_threshold=float(st.secrets.get("GT_EVAL_THRESHOLD", 0.75)),
                )
                print("✅ Settings loaded from Streamlit secrets")
                return settings
    except (ImportError, KeyError, AttributeError):
        pass
    
    # If both fail, show error
    raise ValueError(
        "❌ Could not load Groq API key.\n"
        "For local development: Create a .env file with GROQ_API_KEY=your_key\n"
        "For Streamlit Cloud: Add GROQ_API_KEY to Secrets"
    )


# Load settings at module level
settings = load_settings()