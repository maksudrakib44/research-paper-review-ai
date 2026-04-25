"""
src/validation/ground_truth.py - Research Ground Truth Management
"""
from __future__ import annotations

import json
from pathlib import Path
from loguru import logger
from src.models import ResearchGroundTruth


class ResearchGroundTruthStore:
    """JSON-backed store for research Q&A ground truth"""
    
    def __init__(self, store_path: Path):
        self._path = Path(store_path)
        self._pairs: list[ResearchGroundTruth] = []
        self._load()
    
    def add(self, pair: ResearchGroundTruth) -> None:
        self._pairs.append(pair)
        self._save()
        logger.info(f"Added GT pair: {pair.gt_id}")
    
    def delete(self, gt_id: str) -> bool:
        before = len(self._pairs)
        self._pairs = [p for p in self._pairs if p.gt_id != gt_id]
        if len(self._pairs) < before:
            self._save()
            return True
        return False
    
    def get_all(self) -> list[ResearchGroundTruth]:
        return list(self._pairs)
    
    def find_by_question(self, question: str) -> ResearchGroundTruth | None:
        q_lower = question.strip().lower()
        for p in self._pairs:
            if p.question.strip().lower() == q_lower:
                return p
        return None
    
    @property
    def count(self) -> int:
        return len(self._pairs)
    
    def _load(self) -> None:
        if not self._path.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text("[]", encoding="utf-8")
            return
        
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._pairs = [ResearchGroundTruth(**item) for item in raw]
            logger.info(f"Loaded {len(self._pairs)} GT pairs")
        except Exception as e:
            logger.error(f"Failed to load GT: {e}")
            self._pairs = []
    
    def _save(self) -> None:
        data = [p.model_dump(mode="json") for p in self._pairs]
        self._path.write_text(json.dumps(data, indent=2, default=str))