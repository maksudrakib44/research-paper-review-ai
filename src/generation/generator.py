"""
src/generation/generator.py - Academic Answer Generation
"""
from __future__ import annotations

import time
from textwrap import dedent
from groq import Groq
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.models import (
    AcademicEvalMetrics, Citation, ConfidenceLevel, 
    ResearchResponse, RetrievedChunk, PaperMetadata
)


_SYSTEM_PROMPT = dedent("""\
You are an expert academic research assistant helping scholars understand research papers.

RULES:
1. Answer ONLY from the provided context passages
2. Always cite the source with (Author, Year) format
3. If not found: "I could not find information about this in the provided papers"
4. For methodology questions, explain the approach step-by-step
5. For results, report quantitative findings exactly as stated
6. Note any limitations or assumptions mentioned

CITATION FORMAT:
"The model achieved 92% accuracy on the test set (Smith et al., 2023)"
""")


class ResearchAnswerGenerator:
    def __init__(self, groq_client: Groq, settings):
        self._client = groq_client
        self._settings = settings
    
    def generate(
        self,
        question: str,
        retrieval_chunks: list[RetrievedChunk],
        eval_metrics: AcademicEvalMetrics | None = None,
    ) -> ResearchResponse:
        t0 = time.perf_counter()
        
        if not retrieval_chunks:
            return self._empty_response(question, eval_metrics)
        
        user_message = self._build_prompt(question, retrieval_chunks)
        answer = self._call_llm(user_message)
        citations = self._extract_citations_from_answer(answer)
        paper_refs = list(set([c.chunk.metadata.title for c in retrieval_chunks[:3]]))
        confidence = self._compute_confidence(retrieval_chunks, eval_metrics)
        
        elapsed = (time.perf_counter() - t0) * 1000
        
        return ResearchResponse(
            question=question,
            answer=answer,
            citations=citations,
            paper_references=paper_refs,
            confidence=confidence,
            eval_metrics=eval_metrics,
            latency_ms=elapsed,
            model_used=self._settings.llm_model,
        )
    
    def _build_prompt(self, question: str, chunks: list[RetrievedChunk]) -> str:
        context_parts = []
        for i, rc in enumerate(chunks, 1):
            meta = rc.chunk.metadata
            context_parts.append(
                f"[{i}] From '{meta.title}' by {', '.join(meta.authors[:2])}\n"
                f"{rc.chunk.text}\n"
            )
        
        context = "\n".join(context_parts)
        return f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _call_llm(self, user_message: str) -> str:
        response = self._client.chat.completions.create(
            model=self._settings.llm_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            max_tokens=self._settings.llm_max_tokens,
            temperature=self._settings.llm_temperature,
        )
        return response.choices[0].message.content.strip()
    
    def _extract_citations_from_answer(self, answer: str) -> list[Citation]:
        import re
        pattern = r'\(([A-Z][a-z]+(?:\s+et\s+al\.?)?),\s*(\d{4})\)'
        matches = re.findall(pattern, answer)
        
        citations = []
        for author, year in matches[:5]:
            citations.append(Citation(
                paper_id="",
                cited_authors=[author],
                cited_year=int(year),
                context="",
                position=0
            ))
        return citations
    
    def _compute_confidence(self, chunks, metrics):
        if not chunks:
            return ConfidenceLevel.LOW
        
        top_score = chunks[0].score
        if metrics and top_score > 0.8 and metrics.passed:
            return ConfidenceLevel.HIGH
        elif top_score > 0.55:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW
    
    def _empty_response(self, question, metrics):
        return ResearchResponse(
            question=question,
            answer="I could not find any relevant passages in the uploaded papers to answer this question.",
            citations=[],
            paper_references=[],
            confidence=ConfidenceLevel.LOW,
            eval_metrics=metrics,
            latency_ms=0,
            model_used=self._settings.llm_model,
        )