"""
src/retrieval/retriever.py - Hybrid Retriever for Research Papers
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Optional
from groq import Groq
from loguru import logger

from src.models import PaperChunk, PaperMetadata, RetrievedChunk, RetrievalResult


class QueryRewriter:
    """HyDE query rewriting using Groq"""
    
    def __init__(self, groq_client: Groq, model: str):
        self._client = groq_client
        self._model = model
    
    def rewrite(self, query: str) -> str:
        prompt = (
            "You are a research expert. Write a 2-3 sentence excerpt from an academic "
            "paper that would directly answer the following question. "
            "Write only the excerpt — no preamble, no explanation.\n\n"
            f"Question: {query}"
        )
        
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3,
            )
            hypothesis = response.choices[0].message.content.strip()
            return hypothesis
        except Exception as e:
            logger.warning(f"HyDE rewrite failed: {e}")
            return query


class BM25Index:
    """BM25 keyword search index"""
    
    def __init__(self):
        self._index = None
        self._chunks = []
    
    def build(self, collection):
        from rank_bm25 import BM25Okapi
        
        result = collection.get(include=["documents", "metadatas"])
        
        if not result.get("documents"):
            return
        
        self._chunks = [
            {"id": rid, "text": doc, "metadata": meta}
            for rid, doc, meta in zip(result["ids"], result["documents"], result["metadatas"])
        ]
        
        tokenized = [c["text"].lower().split() for c in self._chunks]
        self._index = BM25Okapi(tokenized)
        logger.info(f"Built BM25 index over {len(self._chunks)} chunks")
    
    def query(self, text: str, top_k: int):
        if self._index is None:
            return []
        
        scores = self._index.get_scores(text.lower().split())
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        if not top_indices:
            return []
        
        max_score = scores[top_indices[0]]
        if max_score == 0:
            return []
        
        return [(self._chunks[i]["id"], float(scores[i]) / max_score) for i in top_indices]
    
    def get_chunk_by_id(self, chunk_id):
        for c in self._chunks:
            if c["id"] == chunk_id:
                return c
        return None


def reciprocal_rank_fusion(ranked_lists, k=60):
    scores = defaultdict(float)
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, 1):
            scores[doc_id] += 1.0 / (k + rank)
    return scores


class HybridRetriever:
    def __init__(self, collection, settings, groq_client: Optional[Groq] = None):
        self._col = collection
        self._settings = settings
        self._bm25 = BM25Index()
        self._query_rewriter = QueryRewriter(groq_client, settings.llm_model) if groq_client else None
    
    def build_index(self):
        if self._col.count() > 0:
            self._bm25.build(self._col)
    
    def retrieve(self, query: str) -> RetrievalResult:
        t0 = time.perf_counter()
        
        if self._col.count() == 0:
            return RetrievalResult(query=query, chunks=[], retrieval_latency_ms=0)
        
        # HyDE rewriting
        search_text = self._query_rewriter.rewrite(query) if self._query_rewriter else query
        
        # Dense retrieval
        dense_chunks = self._dense_search(search_text, self._settings.top_k_dense)
        
        # BM25 retrieval
        bm25_chunks = self._bm25_search(query, self._settings.top_k_bm25)
        
        # Fusion
        fused = self._fuse(dense_chunks, bm25_chunks)
        final = fused[:self._settings.top_k_rerank]
        
        elapsed = (time.perf_counter() - t0) * 1000
        
        return RetrievalResult(query=query, chunks=final, retrieval_latency_ms=elapsed)
    
    def _dense_search(self, text: str, top_k: int):
        try:
            result = self._col.query(
                query_texts=[text],
                n_results=min(top_k, self._col.count()),
                include=["documents", "metadatas", "distances"],
            )
            
            if not result.get("documents"):
                return []
            
            chunks = []
            for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
                score = 1.0 - min(dist, 1.0)
                chunks.append(RetrievedChunk(chunk=self._build_chunk(doc, meta), score=score, source="dense"))
            
            return chunks
        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            return []
    
    def _bm25_search(self, text: str, top_k: int):
        results = self._bm25.query(text, top_k)
        chunks = []
        for chunk_id, score in results:
            raw = self._bm25.get_chunk_by_id(chunk_id)
            if raw:
                chunks.append(RetrievedChunk(chunk=self._build_chunk(raw["text"], raw["metadata"]), score=score, source="bm25"))
        return chunks
    
    def _fuse(self, dense, bm25):
        if not dense:
            return bm25
        if not bm25:
            return dense
        
        alpha = self._settings.hybrid_alpha
        dense_ids = [c.chunk.chunk_id for c in dense]
        bm25_ids = [c.chunk.chunk_id for c in bm25]
        
        rrf_scores = reciprocal_rank_fusion([dense_ids, bm25_ids])
        
        chunk_map = {c.chunk.chunk_id: c for c in dense + bm25}
        
        fused = []
        for chunk_id, rrf_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
            if chunk_id in chunk_map:
                orig = chunk_map[chunk_id]
                blended = alpha * rrf_score + (1 - alpha) * orig.score
                fused.append(RetrievedChunk(chunk=orig.chunk, score=blended, source=orig.source))
        
        return fused
    
    def _build_chunk(self, text: str, meta: dict):
        from src.models import PaperMetadata, PaperChunk, SectionType
        
        paper_meta = PaperMetadata(
            paper_id=meta.get("paper_id", "unknown"),
            title=meta.get("title", "Unknown"),
            authors=meta.get("authors", "").split(", "),
            year=int(meta.get("year", 0)) if meta.get("year") else None,
            filename=meta.get("filename", "unknown.pdf"),
            tags=meta.get("tags", "").split(", ") if meta.get("tags") else [],
        )
        
        return PaperChunk(
            chunk_id=meta.get("chunk_id", ""),
            paper_id=meta.get("paper_id", "unknown"),
            text=text,
            section_type=SectionType(meta.get("section_type", "other")),
            section_title=meta.get("section_title", ""),
            chunk_index=int(meta.get("chunk_index", 0)),
            page_number=1,
            token_count=len(text) // 4,
            has_equations=meta.get("has_equations", False),
            has_citations=meta.get("has_citations", False),
            metadata=paper_meta,
        )