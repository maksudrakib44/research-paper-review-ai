"""
src/ingestion/faiss_store.py - FAISS vector store for research papers
"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer


class FAISSStore:
    """FAISS-based vector store for research papers"""
    
    def __init__(self, persist_dir: Path, embedding_model: str = "all-MiniLM-L6-v2"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.embedding_model = SentenceTransformer(embedding_model)
        self.index = None
        self.documents: List[str] = []
        self.metadatas: List[Dict] = []
        self.ids: List[str] = []
        
        self._load()
        print(f"✅ FAISSStore initialized at {persist_dir}")
    
    def add(self, ids: List[str], documents: List[str], metadatas: List[Dict]) -> None:
        """Add documents to the index"""
        if not documents:
            return
        
        print(f"📊 Generating embeddings for {len(documents)} chunks...")
        embeddings = self.embedding_model.encode(documents, show_progress_bar=False)
        
        if self.index is None:
            import faiss
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
        
        import faiss
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype(np.float32))
        
        self.ids.extend(ids)
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        
        self._save()
        print(f"✅ Added {len(documents)} chunks. Total: {self.index.ntotal}")
    
    def query(self, query_texts: List[str], n_results: int = 10, include: List[str] = None) -> Dict:
        """Query the index"""
        if self.index is None or self.index.ntotal == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}
        
        query_embedding = self.embedding_model.encode([query_texts[0]])
        import faiss
        faiss.normalize_L2(query_embedding)
        
        k = min(n_results, self.index.ntotal)
        scores, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        results_ids = []
        results_docs = []
        results_metadatas = []
        results_distances = []
        
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.ids):
                results_ids.append(self.ids[idx])
                results_docs.append(self.documents[idx])
                results_metadatas.append(self.metadatas[idx])
                results_distances.append(float(1 - scores[0][i]))
        
        result = {}
        if not include or "documents" in include:
            result["documents"] = [results_docs]
        if not include or "metadatas" in include:
            result["metadatas"] = [results_metadatas]
        if not include or "distances" in include:
            result["distances"] = [results_distances]
        if not include or "ids" in include:
            result["ids"] = [results_ids]
        
        return result
    
    def get(self, where: Optional[Dict] = None, limit: int = 100, include: List[str] = None) -> Dict:
        """Get documents with optional filtering"""
        if where is None:
            filtered_indices = list(range(len(self.ids)))
        else:
            filtered_indices = []
            for i, meta in enumerate(self.metadatas):
                match = True
                for key, value in where.items():
                    if str(meta.get(key, "")) != str(value):
                        match = False
                        break
                if match:
                    filtered_indices.append(i)
        
        filtered_indices = filtered_indices[:limit]
        
        result = {"ids": [self.ids[i] for i in filtered_indices]}
        
        if not include or "documents" in include:
            result["documents"] = [self.documents[i] for i in filtered_indices]
        if not include or "metadatas" in include:
            result["metadatas"] = [self.metadatas[i] for i in filtered_indices]
        
        return result
    
    def count(self) -> int:
        return len(self.ids) if self.index else 0
    
    def _save(self) -> None:
        """Save index and metadata"""
        import faiss
        if self.index is not None:
            faiss.write_index(self.index, str(self.persist_dir / "index.faiss"))
        
        with open(self.persist_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump({
                "ids": self.ids,
                "documents": self.documents,
                "metadatas": self.metadatas
            }, f, indent=2)
    
    def _load(self) -> None:
        """Load index and metadata"""
        import faiss
        index_path = self.persist_dir / "index.faiss"
        metadata_path = self.persist_dir / "metadata.json"
        
        if index_path.exists():
            self.index = faiss.read_index(str(index_path))
        
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.ids = data.get("ids", [])
                self.documents = data.get("documents", [])
                self.metadatas = data.get("metadatas", [])