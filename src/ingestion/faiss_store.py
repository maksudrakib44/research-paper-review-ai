"""
src/ingestion/faiss_store.py - Pure Python FAISS Store
No sklearn, no pyarrow, no DLL issues
"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter
import math


class SimpleEmbedder:
    """Pure Python TF-IDF embedder - no dependencies"""
    
    def __init__(self, max_features: int = 384):
        self.max_features = max_features
        self.vocabulary = {}
        self.idf = {}
        self.is_fitted = False
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        text = text.lower()
        # Remove punctuation
        for char in '.,!?;:()[]{}"\'-':
            text = text.replace(char, ' ')
        return [w for w in text.split() if len(w) > 2]
    
    def _compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        """Compute term frequency"""
        counter = Counter(tokens)
        total = len(tokens)
        return {term: count / total for term, count in counter.items()}
    
    def fit(self, texts: List[str]):
        """Build vocabulary and compute IDF"""
        # Count document frequency
        doc_freq = Counter()
        for text in texts:
            unique_terms = set(self._tokenize(text))
            for term in unique_terms:
                doc_freq[term] += 1
        
        # Select top features
        most_common = doc_freq.most_common(self.max_features)
        self.vocabulary = {term: idx for idx, (term, _) in enumerate(most_common)}
        
        # Compute IDF
        total_docs = len(texts)
        for term, idx in self.vocabulary.items():
            df = doc_freq[term]
            self.idf[idx] = math.log((total_docs + 1) / (df + 1)) + 1
        
        self.is_fitted = True
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Convert texts to TF-IDF vectors"""
        vectors = np.zeros((len(texts), len(self.vocabulary)), dtype=np.float32)
        
        for i, text in enumerate(texts):
            tokens = self._tokenize(text)
            tf = self._compute_tf(tokens)
            
            for term, tf_val in tf.items():
                if term in self.vocabulary:
                    idx = self.vocabulary[term]
                    vectors[i, idx] = tf_val * self.idf[idx]
        
        # Normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vectors = vectors / norms
        
        return vectors
    
    def encode(self, texts: List[str], show_progress_bar: bool = False) -> np.ndarray:
        """Encode texts to vectors"""
        if not texts:
            return np.array([]).reshape(0, self.max_features)
        
        if not self.is_fitted:
            self.fit(texts)
        
        return self.transform(texts)


class FAISSStore:
    """FAISS-based vector store - minimal dependencies"""
    
    def __init__(self, persist_dir: Path, embedding_model: str = "simple"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.embedding_model = SimpleEmbedder()
        self.index = None
        self.documents: List[str] = []
        self.metadatas: List[Dict] = []
        self.ids: List[str] = []
        
        self._load()
        print(f"✅ FAISSStore initialized (pure Python)")
    
    def add(self, ids: List[str], documents: List[str], metadatas: List[Dict]) -> None:
        if not documents:
            return
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(documents)
        
        if len(embeddings) == 0:
            return
        
        # Initialize FAISS index
        if self.index is None:
            try:
                import faiss
                dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dimension)
                print(f"✅ FAISS index created (dimension: {dimension})")
            except ImportError:
                print("⚠️ FAISS not available, using numpy fallback")
                self.index = None
        
        if self.index is not None:
            import faiss
            self.index.add(embeddings.astype(np.float32))
        
        self.ids.extend(ids)
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        
        self._save()
        print(f"✅ Added {len(documents)} chunks. Total: {len(self.ids)}")
    
    def query(self, query_texts: List[str], n_results: int = 10, include: List[str] = None) -> Dict:
        if len(self.ids) == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query_texts)
        
        if len(query_embedding) == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}
        
        # Linear search (fallback if FAISS not available)
        k = min(n_results, len(self.ids))
        
        # Compute cosine similarity with all documents
        similarities = []
        for doc_vec in self.embedding_model.transform(self.documents):
            sim = np.dot(query_embedding[0], doc_vec)
            similarities.append(sim)
        
        # Get top k indices
        indices = np.argsort(similarities)[::-1][:k]
        scores = [similarities[i] for i in indices]
        
        results_ids = []
        results_docs = []
        results_metadatas = []
        results_distances = []
        
        for idx in indices:
            if idx < len(self.ids):
                results_ids.append(self.ids[idx])
                results_docs.append(self.documents[idx])
                results_metadatas.append(self.metadatas[idx])
                results_distances.append(float(1 - scores[len(results_distances)]))
        
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
        return len(self.ids)
    
    def _save(self) -> None:
        try:
            with open(self.persist_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump({
                    "ids": self.ids,
                    "documents": self.documents,
                    "metadatas": self.metadatas
                }, f, indent=2)
            print(f"💾 Saved {len(self.ids)} documents")
        except Exception as e:
            print(f"⚠️ Save failed: {e}")
    
    def _load(self) -> None:
        try:
            metadata_path = self.persist_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.ids = data.get("ids", [])
                    self.documents = data.get("documents", [])
                    self.metadatas = data.get("metadatas", [])
                print(f"📂 Loaded {len(self.ids)} documents")
        except Exception as e:
            print(f"⚠️ Load failed: {e}")