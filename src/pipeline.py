"""
src/pipeline.py - Main Research RAG Pipeline
"""
from __future__ import annotations

from pathlib import Path
from groq import Groq
from loguru import logger

from src.generation.generator import ResearchAnswerGenerator
from src.ingestion.ingester import AcademicPaperIngester
from src.ingestion.faiss_store import FAISSStore
from src.models import ResearchGroundTruth, ResearchResponse
from src.retrieval.retriever import HybridRetriever
from src.validation.ground_truth import ResearchGroundTruthStore
from src.validation.metrics import ResearchEvaluator


class ResearchPipeline:
    def __init__(self, settings):
        self._settings = settings
        self._setup_logging()
        
        logger.info("Initializing Research RAG Pipeline...")
        
        # LLM client
        self._groq_client = Groq(api_key=settings.groq_api_key)
        
        # Vector store
        settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self._collection = FAISSStore(
            persist_dir=settings.chroma_persist_dir,
            embedding_model=settings.embedding_model
        )
        
        # Pipeline components
        self._ingester = AcademicPaperIngester(self._collection, settings)
        self._retriever = HybridRetriever(self._collection, settings, self._groq_client)
        self._gt_store = ResearchGroundTruthStore(settings.gt_store_path)
        self._evaluator = ResearchEvaluator(threshold=settings.gt_eval_threshold)
        self._generator = ResearchAnswerGenerator(self._groq_client, settings)
        
        # Build index if documents exist
        if self._collection.count() > 0:
            self._retriever.build_index()
        
        logger.info("Pipeline ready!")
    
    def ingest(self, path: Path, **kwargs) -> int:
        chunks = self._ingester.ingest(path, **kwargs)
        return len(chunks)
    
    def build_index(self):
        self._retriever.build_index()
    
    def ask(self, question: str) -> ResearchResponse:
        retrieval = self._retriever.retrieve(question)
        gt_pair = self._gt_store.find_by_question(question)
        eval_metrics = self._evaluator.evaluate(
            question=question,
            answer="",
            retrieval=retrieval.chunks,
            gt_pair=gt_pair
        )
        response = self._generator.generate(question, retrieval.chunks, eval_metrics)
        
        if gt_pair:
            final_metrics = self._evaluator.evaluate(
                question=question,
                answer=response.answer,
                retrieval=retrieval.chunks,
                gt_pair=gt_pair
            )
            response.eval_metrics = final_metrics
        
        return response
    
    def add_ground_truth(self, pair: ResearchGroundTruth):
        self._gt_store.add(pair)
    
    def list_ground_truth(self):
        return self._gt_store.get_all()
    
    def delete_ground_truth(self, gt_id: str) -> bool:
        return self._gt_store.delete(gt_id)
    
    def run_eval_suite(self) -> list[dict]:
        results = []
        for pair in self._gt_store.get_all():
            response = self.ask(pair.question)
            results.append({
                "question": pair.question,
                "ground_truth": pair.ground_truth_answer,
                "generated_answer": response.answer,
                "faithfulness": response.eval_metrics.faithfulness if response.eval_metrics else None,
                "overall_score": response.eval_metrics.overall_score if response.eval_metrics else None,
                "passed": response.eval_metrics.passed if response.eval_metrics else None,
                "latency_ms": response.latency_ms,
            })
        return results
    
    @property
    def paper_count(self):
        return self._collection.count()
    
    @property
    def gt_count(self):
        return self._gt_store.count
    
    def _setup_logging(self):
        from loguru import logger as log
        self._settings.log_file.parent.mkdir(parents=True, exist_ok=True)
        log.add(str(self._settings.log_file), level=self._settings.log_level)