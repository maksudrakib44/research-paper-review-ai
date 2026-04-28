"""
src/retrieval/retriever.py - Hybrid Retriever for Research Papers
No Groq dependency - uses only local BM25 and FAISS
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Optional
from loguru import logger

from src.models import PaperChunk, PaperMetadata, RetrievedChunk, RetrievalResult


# ── Query Rewriter (Simple version - no LLM fallback) ─────────────────────
class QueryRewriter:
    """
    Simple query rewriter (no LLM - keeps original query)
    Can be extended with local models if needed
    """
    
    def __init__(self):
        logger.info("QueryRewriter initialized (simple mode - no LLM)")
    
    def rewrite(self, query: str) -> str:
        """Returns original query (no rewriting)"""
        return query


# ── BM25 Index ────────────────────────────────────────────────────────────
class BM25Index:
    """BM25 keyword search index"""
    
    def __init__(self):
        self._index = None
        self._chunks = []
    
    def build(self, collection):
        from rank_bm25 import BM25Okapi
        
        result = collection.get(include=["documents", "metadatas"])
        
        if not result.get("documents"):
            logger.warning("No documents found for BM25 index")
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


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────
def reciprocal_rank_fusion(ranked_lists, k=60):
    """RRF combines multiple ranked lists without normalization"""
    scores = defaultdict(float)
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, 1):
            scores[doc_id] += 1.0 / (k + rank)
    return scores


# ── Main Retriever ─────────────────────────────────────────────────────────
class HybridRetriever:
    """
    Hybrid retriever combining dense (FAISS) and sparse (BM25) search.
    No LLM dependency - works completely offline.
    """
    
    def __init__(self, collection, settings, groq_client=None):
        """
        Initialize hybrid retriever.
        
        Args:
            collection: FAISS vector store
            settings: Application settings
            groq_client: Ignored (kept for compatibility)
        """
        self._col = collection
        self._settings = settings
        self._bm25 = BM25Index()
        # Use simple query rewriter (no LLM)
        self._query_rewriter = QueryRewriter()
        logger.info("HybridRetriever initialized (no LLM dependency)")
    
    def build_index(self):
        """Build BM25 index from collection"""
        if self._col.count() > 0:
            self._bm25.build(self._col)
            logger.info(f"BM25 index built with {self._col.count()} chunks")
        else:
            logger.warning("No documents to build index")
    
    def retrieve(self, query: str) -> RetrievalResult:
        """Retrieve relevant chunks for query"""
        t0 = time.perf_counter()
        
        if self._col.count() == 0:
            logger.warning("No documents in collection")
            return RetrievalResult(query=query, chunks=[], retrieval_latency_ms=0)
        
        # Simple query (no HyDE rewrite to avoid LLM dependency)
        search_text = self._query_rewriter.rewrite(query)
        
        # Dense retrieval (FAISS)
        dense_chunks = self._dense_search(search_text, self._settings.top_k_dense)
        
        # BM25 retrieval (keyword)
        bm25_chunks = self._bm25_search(query, self._settings.top_k_bm25)
        
        # Fusion
        fused = self._fuse(dense_chunks, bm25_chunks)
        final = fused[:self._settings.top_k_rerank]
        
        elapsed = (time.perf_counter() - t0) * 1000
        
        logger.info(f"Retrieved {len(final)} chunks in {elapsed:.0f}ms")
        
        return RetrievalResult(query=query, chunks=final, retrieval_latency_ms=elapsed)
    
    def _dense_search(self, text: str, top_k: int):
        """Dense retrieval using FAISS"""
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
                # Convert distance to similarity score (0-1 range)
                score = 1.0 - min(dist, 1.0)
                chunks.append(RetrievedChunk(
                    chunk=self._build_chunk(doc, meta),
                    score=score,
                    source="dense"
                ))
            return chunks
        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            return []
    
    def _bm25_search(self, text: str, top_k: int):
        """BM25 keyword search"""
        if self._col.count() == 0:
            return []
        
        results = self._bm25.query(text, top_k)
        chunks = []
        for chunk_id, score in results:
            raw = self._bm25.get_chunk_by_id(chunk_id)
            if raw:
                chunks.append(RetrievedChunk(
                    chunk=self._build_chunk(raw["text"], raw["metadata"]),
                    score=score,
                    source="bm25"
                ))
        return chunks
    
    def _fuse(self, dense, bm25):
        """Reciprocal Rank Fusion to combine results"""
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
        
        # Build lookup map
        chunk_map = {}
        for c in dense + bm25:
            if c.chunk.chunk_id not in chunk_map:
                chunk_map[c.chunk.chunk_id] = c
        
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
    
    def _build_chunk(self, text: str, meta: dict) -> PaperChunk:
        """Build PaperChunk from metadata"""
        from src.models import PaperMetadata, PaperChunk, SectionType
        
        paper_meta = PaperMetadata(
            paper_id=meta.get("paper_id", "unknown"),
            title=meta.get("title", "Unknown"),
            authors=meta.get("authors", "").split(", ") if meta.get("authors") else [],
            year=int(meta.get("year", 0)) if meta.get("year") else None,
            filename=meta.get("filename", "unknown.pdf"),
            tags=meta.get("tags", "").split(", ") if meta.get("tags") else [],
        )
        
        # Get section type from metadata or default to OTHER
        section_type_str = meta.get("section_type", "other")
        try:
            section_type = SectionType(section_type_str)
        except ValueError:
            section_type = SectionType.OTHER
        
        return PaperChunk(
            chunk_id=meta.get("chunk_id", ""),
            paper_id=meta.get("paper_id", "unknown"),
            text=text,
            section_type=section_type,
            section_title=meta.get("section_title", ""),
            chunk_index=int(meta.get("chunk_index", 0)),
            page_number=1,
            token_count=len(text) // 4,
            has_equations=meta.get("has_equations", False),
            has_citations=meta.get("has_citations", False),
            metadata=paper_meta,
        )