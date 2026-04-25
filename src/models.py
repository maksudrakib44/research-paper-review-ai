"""
src/models.py - Research Paper Data Models
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4
from pydantic import BaseModel, Field


class PaperMetadata(BaseModel):
    """Metadata for academic papers"""
    paper_id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    journal: str | None = None
    doi: str | None = None
    filename: str
    section_count: int = 0
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    tags: list[str] = Field(default_factory=list)


class SectionType(str, Enum):
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    RELATED_WORK = "related_work"
    METHODOLOGY = "methodology"
    EXPERIMENTS = "experiments"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    REFERENCES = "references"
    OTHER = "other"


class PaperChunk(BaseModel):
    """A chunk of a research paper"""
    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    paper_id: str
    text: str
    section_type: SectionType
    section_title: str
    chunk_index: int
    page_number: int
    token_count: int
    has_equations: bool = False
    has_citations: bool = False
    metadata: PaperMetadata


class Citation(BaseModel):
    """Academic citation extracted from paper"""
    citation_id: str = Field(default_factory=lambda: str(uuid4()))
    paper_id: str
    cited_authors: list[str]
    cited_year: int
    cited_title: str | None = None
    context: str
    position: int


class RetrievedChunk(BaseModel):
    """Retrieved chunk with relevance score"""
    chunk: PaperChunk
    score: float
    source: str = "dense"
    citation_context: list[Citation] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    """Bundle of top-k chunks"""
    query: str
    chunks: list[RetrievedChunk]
    retrieval_latency_ms: float


class ResearchGroundTruth(BaseModel):
    """Ground truth for research paper evaluation"""
    gt_id: str = Field(default_factory=lambda: str(uuid4()))
    question: str
    ground_truth_answer: str
    required_citations: list[str] = Field(default_factory=list)
    paper_ids: list[str] = Field(default_factory=list)
    domain_tags: list[str] = Field(default_factory=list)
    question_type: Literal["factual", "comparative", "summarization", "methodology"] = "factual"
    created_by: str = "human"
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AcademicEvalMetrics(BaseModel):
    """Specialized metrics for research paper evaluation"""
    faithfulness: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    answer_correctness: float = 0.0
    citation_accuracy: float = 0.0
    factual_consistency: float = 0.0
    semantic_similarity: float = 0.0
    answer_completeness: float = 0.0
    passed: bool = False
    
    @property
    def overall_score(self) -> float:
        return (
            self.faithfulness * 0.25 +
            self.context_precision * 0.15 +
            self.context_recall * 0.15 +
            self.answer_correctness * 0.15 +
            self.citation_accuracy * 0.15 +
            self.factual_consistency * 0.15
        )


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ResearchResponse(BaseModel):
    """Final response for research paper queries"""
    response_id: str = Field(default_factory=lambda: str(uuid4()))
    question: str
    answer: str
    citations: list[Citation]
    paper_references: list[str]
    confidence: ConfidenceLevel
    eval_metrics: AcademicEvalMetrics | None = None
    latency_ms: float
    model_used: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @property
    def has_citations(self) -> bool:
        return len(self.citations) > 0