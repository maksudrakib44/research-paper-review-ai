"""
src/retrieval/retriever.py - Hybrid Retriever for Research Papers
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Optional
from loguru import logger

from src.models import PaperChunk, PaperMetadata, RetrievedChunk, RetrievalResult, SectionType


class BM25Index:
    """BM25 keyword search index"""
    
    def __init__(self):
        self._index = None
        self._chunks = []
    
    def build(self, collection):
        from rank_bm25 import BM25Okapi
        
        try:
            result = collection.get(include=["documents", "metadatas"])
            
            if not result.get("documents") or len(result["documents"]) == 0:
                logger.warning("No documents found for BM25")
                return
            
            self._chunks = [
                {"id": rid, "text": doc, "metadata": meta}
                for rid, doc, meta in zip(result["ids"], result["documents"], result["metadatas"])
            ]
            
            tokenized = [c["text"].lower().split() for c in self._chunks]
            self._index = BM25Okapi(tokenized)
            logger.info(f"BM25 built over {len(self._chunks)} chunks")
        except Exception as e:
            logger.error(f"BM25 build failed: {e}")
    
    def query(self, text: str, top_k: int):
        if self._index is None:
            return []
        
        try:
            scores = self._index.get_scores(text.lower().split())
            
            if not scores:
                return []
            
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            
            if not top_indices:
                return []
            
            max_score = scores[top_indices[0]]
            if max_score == 0:
                return []
            
            return [(self._chunks[i]["id"], float(scores[i]) / max_score) for i in top_indices]
        except Exception as e:
            logger.error(f"BM25 query failed: {e}")
            return []
    
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
    def __init__(self, collection, settings, groq_client=None):
        self._col = collection
        self._settings = settings
        self._bm25 = BM25Index()
    
    def build_index(self):
        if self._col.count() > 0:
            self._bm25.build(self._col)
    
    def retrieve(self, query: str) -> RetrievalResult:
        t0 = time.perf_counter()
        
        if self._col.count() == 0:
            return RetrievalResult(query=query, chunks=[], retrieval_latency_ms=0)
        
        # Dense retrieval
        dense_chunks = self._dense_search(query, self._settings.top_k_dense)
        
        # BM25 retrieval
        bm25_chunks = self._bm25_search(query, self._settings.top_k_bm25)
        
        # Fusion
        fused = self._fuse(dense_chunks, bm25_chunks)
        final = fused[:self._settings.top_k_rerank]
        
        elapsed = (time.perf_counter() - t0) * 1000
        
        return RetrievalResult(query=query, chunks=final, retrieval_latency_ms=elapsed)
    
    def _dense_search(self, text: str, top_k: int):
        if self._col.count() == 0:
            return []
        
        try:
            result = self._col.query(
                query_texts=[text],
                n_results=min(top_k, self._col.count()),
                include=["documents", "metadatas", "distances"],
            )
            
            if not result.get("documents") or not result["documents"][0]:
                return []
            
            chunks = []
            for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
                score = 1.0 - min(dist, 1.0)
                chunk = self._build_chunk_safe(doc, meta)
                if chunk:
                    chunks.append(RetrievedChunk(chunk=chunk, score=score, source="dense"))
            
            return chunks
        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            return []
    
    def _bm25_search(self, text: str, top_k: int):
        if self._col.count() == 0:
            return []
        
        try:
            results = self._bm25.query(text, top_k)
            chunks = []
            for chunk_id, score in results:
                raw = self._bm25.get_chunk_by_id(chunk_id)
                if raw:
                    chunk = self._build_chunk_safe(raw["text"], raw["metadata"])
                    if chunk:
                        chunks.append(RetrievedChunk(chunk=chunk, score=score, source="bm25"))
            return chunks
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []
    
    def _fuse(self, dense, bm25):
        if not dense and not bm25:
            return []
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
                fused.append(RetrievedChunk(
                    chunk=orig.chunk,
                    score=blended,
                    source=orig.source
                ))
        return fused
    
    def _build_chunk_safe(self, text: str, meta: dict) -> Optional[PaperChunk]:
        """Safely build PaperChunk with fallbacks for missing fields"""
        try:
            # Handle authors - could be string or list
            authors = meta.get("authors", ["Unknown"])
            if isinstance(authors, str):
                authors = [a.strip() for a in authors.split(",")]
            elif not isinstance(authors, list):
                authors = ["Unknown"]
            
            paper_meta = PaperMetadata(
                paper_id=meta.get("paper_id", "unknown"),
                title=meta.get("title", "Unknown Paper"),
                authors=authors,
                year=meta.get("year", None),
                filename=meta.get("filename", "unknown.txt"),
            )
            
            return PaperChunk(
                chunk_id=meta.get("chunk_id", f"chunk_{hash(text)}"),
                paper_id=meta.get("paper_id", "unknown"),
                text=text,
                section_type=SectionType.OTHER,
                section_title=meta.get("section_title", "Content"),
                chunk_index=int(meta.get("chunk_index", 0)),
                page_number=int(meta.get("page_number", 1)),
                token_count=max(1, len(text) // 4),
                has_equations=False,
                has_citations=False,
                metadata=paper_meta,
            )
        except Exception as e:
            logger.error(f"Failed to build chunk: {e}")
            return None