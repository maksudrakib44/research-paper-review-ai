"""
tests/test_pipeline.py - Unit tests for Research Paper RAG
"""
from __future__ import annotations

import pytest
import json
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestImports:
    """Test all modules import correctly"""
    
    def test_models_import(self):
        from src.models import (
            PaperMetadata, PaperChunk, SectionType, 
            Citation, ResearchResponse, AcademicEvalMetrics
        )
        assert PaperMetadata is not None
        assert SectionType is not None
    
    def test_ingestion_import(self):
        from src.ingestion.faiss_store import FAISSStore
        from src.ingestion.section_chunker import AcademicPaperChunker
        assert FAISSStore is not None
        assert AcademicPaperChunker is not None
    
    def test_retrieval_import(self):
        from src.retrieval.retriever import HybridRetriever, BM25Index
        assert HybridRetriever is not None
    
    def test_generation_import(self):
        from src.generation.generator import ResearchAnswerGenerator
        assert ResearchAnswerGenerator is not None
    
    def test_validation_import(self):
        from src.validation.ground_truth import ResearchGroundTruthStore
        from src.validation.metrics import ResearchEvaluator
        assert ResearchGroundTruthStore is not None


class TestModels:
    """Test data models"""
    
    def test_paper_metadata(self):
        from src.models import PaperMetadata
        
        meta = PaperMetadata(
            title="Test Paper",
            authors=["Author 1"],
            filename="test.pdf"
        )
        assert meta.title == "Test Paper"
        assert meta.paper_id is not None
    
    def test_paper_chunk(self):
        from src.models import PaperChunk, PaperMetadata, SectionType
        
        meta = PaperMetadata(title="Test", authors=["A"], filename="test.pdf")
        chunk = PaperChunk(
            paper_id="test_id",
            text="Test content",
            section_type=SectionType.ABSTRACT,
            section_title="Abstract",
            chunk_index=0,
            page_number=1,
            token_count=50,
            metadata=meta
        )
        assert chunk.text == "Test content"
        assert chunk.section_type == SectionType.ABSTRACT
    
    def test_research_response(self):
        from src.models import ResearchResponse, ConfidenceLevel
        
        response = ResearchResponse(
            question="Test?",
            answer="Test answer",
            citations=[],
            paper_references=[],
            confidence=ConfidenceLevel.HIGH,
            latency_ms=100,
            model_used="test_model"
        )
        assert response.confidence == ConfidenceLevel.HIGH
        assert response.has_citations is False


class TestChunker:
    """Test academic paper chunking"""
    
    def test_section_detection(self):
        from src.ingestion.section_chunker import AcademicPaperChunker
        
        chunker = AcademicPaperChunker(["Abstract", "Introduction", "Conclusion"])
        
        # Test abstract detection
        section_type = chunker.detect_section_type("Abstract")
        from src.models import SectionType
        assert section_type == SectionType.ABSTRACT
    
    def test_equation_extraction(self):
        from src.ingestion.section_chunker import AcademicPaperChunker
        
        chunker = AcademicPaperChunker([])
        text = "The equation $$E = mc^2$$ is important"
        equations = chunker.extract_equations(text)
        assert len(equations) > 0
    
    def test_citation_extraction(self):
        from src.ingestion.section_chunker import AcademicPaperChunker
        
        chunker = AcademicPaperChunker([])
        text = "As shown in (Smith et al., 2020), this is important"
        citations = chunker.extract_citations(text)
        assert len(citations) > 0


class TestGroundTruth:
    """Test ground truth store"""
    
    def test_store_creation(self, tmp_path):
        from src.validation.ground_truth import ResearchGroundTruthStore
        
        store_path = tmp_path / "test_gt.json"
        store = ResearchGroundTruthStore(store_path)
        assert store.count == 0
    
    def test_add_and_find(self, tmp_path):
        from src.validation.ground_truth import ResearchGroundTruthStore
        from src.models import ResearchGroundTruth
        
        store_path = tmp_path / "test_gt.json"
        store = ResearchGroundTruthStore(store_path)
        
        pair = ResearchGroundTruth(
            question="What is RAG?",
            ground_truth_answer="Retrieval-Augmented Generation",
            domain_tags=["RAG"]
        )
        store.add(pair)
        
        found = store.find_by_question("What is RAG?")
        assert found is not None
        assert found.ground_truth_answer == "Retrieval-Augmented Generation"


class TestMetrics:
    """Test evaluation metrics"""
    
    def test_token_f1(self):
        from src.validation.metrics import ResearchEvaluator
        
        evaluator = ResearchEvaluator()
        f1 = evaluator._token_f1("hello world", "hello world")
        assert f1 == 1.0
    
    def test_faithfulness(self):
        from src.validation.metrics import ResearchEvaluator
        
        evaluator = ResearchEvaluator()
        answer = "The model achieves high accuracy."
        contexts = ["The model achieves high accuracy on test data."]
        
        score = evaluator._faithfulness(answer, contexts)
        assert 0 <= score <= 1
    
    def test_citation_accuracy(self):
        from src.validation.metrics import ResearchEvaluator
        from src.models import RetrievedChunk, PaperChunk, PaperMetadata, SectionType
        
        evaluator = ResearchEvaluator()
        answer = "As shown by (Smith, 2020), this works."
        
        meta = PaperMetadata(title="Test", authors=["Smith"], filename="test.pdf")
        chunk = PaperChunk(
            paper_id="id",
            text="Smith (2020) showed this works",
            section_type=SectionType.ABSTRACT,
            section_title="Abstract",
            chunk_index=0,
            page_number=1,
            token_count=50,
            metadata=meta
        )
        retrieval = [RetrievedChunk(chunk=chunk, score=0.9)]
        
        score = evaluator._check_citation_accuracy(answer, retrieval)
        assert score >= 0


class TestRetriever:
    """Test retriever functionality"""
    
    def test_rrf_fusion(self):
        from src.retrieval.retriever import reciprocal_rank_fusion
        
        list1 = ["doc1", "doc2", "doc3"]
        list2 = ["doc2", "doc1", "doc4"]
        
        scores = reciprocal_rank_fusion([list1, list2])
        assert "doc1" in scores
        assert "doc2" in scores
        assert scores["doc2"] > scores["doc3"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])