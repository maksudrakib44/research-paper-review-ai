"""
config/settings.py - Research Paper RAG Configuration
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Research Paper RAG Settings"""
    
    # ── LLM (Groq) ───────────────────────────────────────────
    groq_api_key: Optional[str] = Field(None, description="Groq API key")
    llm_model: str = Field("llama-3.3-70b-versatile", description="Groq model ID")
    llm_max_tokens: int = Field(2048, ge=512, le=8192)
    llm_temperature: float = Field(0.2, ge=0.0, le=1.0)
    
    # ── Embedding ────────────────────────────────────────────
    embedding_model: str = Field("all-MiniLM-L6-v2")
    
    # ── Vector Database ──────────────────────────────────────
    chroma_persist_dir: Path = Field(Path("./data/chroma_db"))
    chroma_collection_name: str = Field("research_papers")
    
    # ── Chunking for Research Papers ─────────────────────────
    chunk_size: int = Field(1024, ge=256, le=2048)
    chunk_overlap: int = Field(128, ge=0, le=512)
    max_chunks_per_doc: int = Field(1000, ge=10)
    
    section_patterns: list[str] = Field([
        r"Abstract", r"Introduction", r"Related Work", r"Methodology",
        r"Methods", r"Experiments", r"Results", r"Discussion",
        r"Conclusion", r"References", r"Bibliography"
    ])
    
    # ── Retrieval ────────────────────────────────────────────
    top_k_dense: int = Field(15, ge=1, le=50)
    top_k_bm25: int = Field(15, ge=1, le=50)
    top_k_rerank: int = Field(8, ge=1, le=20)
    hybrid_alpha: float = Field(0.6, ge=0.0, le=1.0)
    
    # ── Ground Truth ─────────────────────────────────────────
    gt_store_path: Path = Field(Path("./data/ground_truth/research_gt.json"))
    gt_eval_threshold: float = Field(0.75, ge=0.0, le=1.0)
    
    # ── Logging ──────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field("INFO")
    log_file: Path = Field(Path("./logs/research_rag.log"))
    
    @field_validator("chroma_persist_dir", "gt_store_path", "log_file", mode="before")
    @classmethod
    def coerce_path(cls, v: str | Path) -> Path:
        return Path(v)
    
    @property
    def section_pattern(self) -> str:
        return r"(?m)^(" + "|".join(self.section_patterns) + r"):"
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


settings = Settings()