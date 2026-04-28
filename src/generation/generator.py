"""
src/generation/generator.py - Groq API Generator (Working Perfectly)
"""
from __future__ import annotations

import time
from textwrap import dedent
from typing import Optional
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.models import ResearchResponse, Citation, ConfidenceLevel, AcademicEvalMetrics, RetrievedChunk


SYSTEM_PROMPT = dedent("""\
You are an expert academic research assistant. Answer questions based ONLY on the provided context.

RULES:
1. Answer ONLY from the provided context passages
2. If the answer is not in the context, say: "I could not find this information in the uploaded paper"
3. Be concise and factual (2-4 sentences)
4. Use the exact terms and numbers from the paper

CONTEXT will be provided below.
""")


class GroqGenerator:
    """Uses Groq API for accurate answers"""
    
    def __init__(self, settings):
        from groq import Groq
        self._client = Groq(api_key=settings.groq_api_key)
        self._settings = settings
        logger.info("✅ Groq Generator initialized")
    
    def generate(self, question: str, chunks: list[RetrievedChunk], eval_metrics=None) -> ResearchResponse:
        t0 = time.perf_counter()
        
        if not chunks:
            return self._empty_response(question)
        
        # Build context from chunks
        context = self._build_context(chunks)
        answer = self._call_groq(question, context)
        
        # Extract paper references
        paper_refs = list(set([c.chunk.metadata.title for c in chunks[:3]]))
        
        # Confidence based on whether we have chunks
        confidence = ConfidenceLevel.HIGH if len(chunks) >= 2 else ConfidenceLevel.MEDIUM
        
        elapsed = (time.perf_counter() - t0) * 1000
        
        return ResearchResponse(
            question=question,
            answer=answer,
            citations=[],
            paper_references=paper_refs,
            confidence=confidence,
            eval_metrics=eval_metrics,
            latency_ms=elapsed,
            model_used="groq/llama-3.3-70b",
        )
    
    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        """Build context from retrieved chunks"""
        context_parts = []
        for i, chunk in enumerate(chunks[:5], 1):
            text = chunk.chunk.text[:1500]  # Limit per chunk
            context_parts.append(f"[Excerpt {i}]:\n{text}")
        return "\n\n".join(context_parts)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _call_groq(self, question: str, context: str) -> str:
        """Call Groq API"""
        user_message = f"""CONTEXT:
{context}

QUESTION: {question}

Based ONLY on the context above, provide a concise answer:"""
        
        response = self._client.chat.completions.create(
            model=self._settings.llm_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            max_tokens=500,
            temperature=0.1,
        )
        
        return response.choices[0].message.content.strip()
    
    def _empty_response(self, question: str) -> ResearchResponse:
        return ResearchResponse(
            question=question,
            answer="No papers uploaded. Please upload a research paper first.",
            citations=[],
            paper_references=[],
            confidence=ConfidenceLevel.LOW,
            eval_metrics=None,
            latency_ms=0,
            model_used="none",
        )