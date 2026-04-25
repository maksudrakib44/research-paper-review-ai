"""
src/validation/metrics.py - Academic Evaluation Metrics
"""
from __future__ import annotations

import re
import numpy as np
from sentence_transformers import SentenceTransformer
from src.models import AcademicEvalMetrics, RetrievedChunk, ResearchGroundTruth


class ResearchEvaluator:
    """Specialized evaluator for research papers"""
    
    def __init__(self, threshold: float = 0.75):
        self._threshold = threshold
        self._embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def evaluate(
        self,
        question: str,
        answer: str,
        retrieval: list[RetrievedChunk],
        gt_pair: ResearchGroundTruth | None,
    ) -> AcademicEvalMetrics:
        context_texts = [c.chunk.text for c in retrieval]
        
        faithfulness = self._faithfulness(answer, context_texts)
        context_precision = self._context_precision(question, context_texts)
        context_recall = self._context_recall(gt_pair, context_texts)
        answer_correctness = self._answer_correctness(answer, gt_pair)
        citation_accuracy = self._check_citation_accuracy(answer, retrieval)
        factual_consistency = self._check_factual_consistency(answer, context_texts)
        
        passed = (
            faithfulness >= self._threshold and
            context_precision >= self._threshold - 0.1 and
            answer_correctness >= self._threshold - 0.1
        )
        
        return AcademicEvalMetrics(
            faithfulness=faithfulness,
            context_precision=context_precision,
            context_recall=context_recall,
            answer_correctness=answer_correctness,
            citation_accuracy=citation_accuracy,
            factual_consistency=factual_consistency,
            semantic_similarity=0.8,
            answer_completeness=0.8,
            passed=passed,
        )
    
    def _faithfulness(self, answer: str, contexts: list[str]) -> float:
        sentences = self._split_sentences(answer)
        if not sentences:
            return 0.0
        
        combined_context = " ".join(contexts)
        supported = sum(1 for sent in sentences if self._token_f1(sent, combined_context) >= 0.15)
        return supported / len(sentences)
    
    def _context_precision(self, question: str, contexts: list[str]) -> float:
        if not contexts:
            return 0.0
        relevant = sum(1 for ctx in contexts if self._token_f1(question, ctx) >= 0.1)
        return relevant / len(contexts)
    
    def _context_recall(self, gt_pair, contexts) -> float:
        if gt_pair is None:
            return 1.0
        combined_context = " ".join(contexts)
        return min(1.0, self._token_f1(gt_pair.ground_truth_answer, combined_context) * 2)
    
    def _answer_correctness(self, answer, gt_pair) -> float:
        if gt_pair is None:
            return 1.0
        return self._token_f1(answer, gt_pair.ground_truth_answer)
    
    def _check_citation_accuracy(self, answer: str, retrieval: list[RetrievedChunk]) -> float:
        pattern = r'\(([A-Z][a-z]+(?:\s+et\s+al\.?)?),\s*(\d{4})\)'
        cited = re.findall(pattern, answer)
        
        if not cited:
            return 1.0
        
        context_text = " ".join([c.chunk.text for c in retrieval])
        accurate = sum(1 for author, year in cited if author.lower() in context_text.lower() and year in context_text)
        return accurate / len(cited)
    
    def _check_factual_consistency(self, answer: str, contexts: list[str]) -> float:
        if not contexts or not answer:
            return 0.0
        
        combined_context = " ".join(contexts)
        sentences = self._split_sentences(answer)
        
        if not sentences:
            return 0.0
        
        consistent = sum(1 for sent in sentences if self._token_f1(sent, combined_context) >= 0.2)
        return consistent / len(sentences)
    
    @staticmethod
    def _token_f1(pred: str, ref: str) -> float:
        pred_tokens = set(pred.lower().split())
        ref_tokens = set(ref.lower().split())
        
        if not pred_tokens or not ref_tokens:
            return 0.0
        
        intersection = pred_tokens & ref_tokens
        precision = len(intersection) / len(pred_tokens)
        recall = len(intersection) / len(ref_tokens)
        
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
    
    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        return [p.strip() for p in parts if len(p.strip()) > 20]