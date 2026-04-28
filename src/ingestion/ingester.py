"""
src/ingestion/ingester.py - PDF and TXT Ingester
Supports both PDF and text files with proper text extraction
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Optional
from loguru import logger

# Try multiple PDF libraries
try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

from src.models import PaperChunk, PaperMetadata, SectionType


class AcademicPaperIngester:
    """Ingests both PDF and TXT files"""
    
    def __init__(self, collection, settings):
        self._col = collection
        self._settings = settings
    
    def ingest(self, path: Path, title=None, authors=None, year=None, tags=None) -> list[PaperChunk]:
        logger.info(f"📄 Processing: {path.name}")
        
        paper_id = f"paper_{hashlib.md5(path.name.encode()).hexdigest()[:8]}"
        
        # Check if already ingested
        if self._already_ingested(paper_id):
            logger.info(f"⏭️ Already ingested: {path.name}")
            return []
        
        # Extract text based on file type
        suffix = path.suffix.lower()
        text = ""
        
        if suffix == '.pdf':
            text = self._extract_pdf_text(path)
        elif suffix == '.txt':
            text = self._extract_txt_text(path)
        else:
            logger.error(f"❌ Unsupported file type: {suffix}. Use .pdf or .txt")
            return []
        
        if not text or len(text.strip()) < 100:
            logger.error(f"❌ No text extracted from {path.name}")
            return []
        
        logger.info(f"📝 Extracted {len(text)} characters from {path.name}")
        
        # Create metadata
        metadata = PaperMetadata(
            paper_id=paper_id,
            title=title or path.stem.replace("_", " ").replace("-", " "),
            authors=authors or ["Unknown"],
            year=year,
            filename=path.name,
            tags=tags or [],
        )
        
        # Create chunks
        chunks = self._create_chunks(text, metadata)
        
        if not chunks:
            logger.warning(f"⚠️ No chunks created for {path.name}")
            return []
        
        # Store in FAISS
        self._store_chunks(chunks)
        logger.success(f"✅ {path.name}: {len(chunks)} chunks")
        return chunks
    
    def _extract_pdf_text(self, path: Path) -> str:
        """Extract text from PDF using multiple methods"""
        
        # Method 1: Try pypdf first
        if HAS_PYPDF:
            try:
                reader = PdfReader(path)
                text_parts = []
                for i, page in enumerate(reader.pages, 1):
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text_parts.append(f"--- Page {i} ---\n{page_text}")
                
                full_text = "\n\n".join(text_parts)
                if len(full_text) > 200:
                    logger.info(f"✅ pypdf extracted {len(full_text)} chars from PDF")
                    return full_text
            except Exception as e:
                logger.warning(f"pypdf failed: {e}")
        
        # Method 2: Try pdfplumber (better for some PDFs)
        if HAS_PDFPLUMBER:
            try:
                import pdfplumber
                text_parts = []
                with pdfplumber.open(path) as pdf:
                    for i, page in enumerate(pdf.pages, 1):
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text_parts.append(f"--- Page {i} ---\n{page_text}")
                
                full_text = "\n\n".join(text_parts)
                if len(full_text) > 200:
                    logger.info(f"✅ pdfplumber extracted {len(full_text)} chars from PDF")
                    return full_text
            except Exception as e:
                logger.warning(f"pdfplumber failed: {e}")
        
        logger.error(f"❌ Could not extract text from PDF: {path.name}")
        logger.info(f"   Tip: Make sure the PDF has selectable text (not scanned images)")
        return ""
    
    def _extract_txt_text(self, path: Path) -> str:
        """Extract text from TXT file"""
        try:
            text = path.read_text(encoding='utf-8', errors='ignore')
            if text and text.strip():
                logger.info(f"✅ TXT extracted {len(text)} chars")
                return text
        except Exception as e:
            logger.error(f"Failed to read TXT: {e}")
        return ""
    
    def _create_chunks(self, text: str, metadata: PaperMetadata) -> list[PaperChunk]:
        """Create overlapping chunks from text"""
        chunks = []
        chunk_size = 1500  # characters
        overlap = 200
        
        for i in range(0, len(text), chunk_size - overlap):
            chunk_text = text[i:i + chunk_size]
            if len(chunk_text) < 100:
                continue
            
            chunks.append(PaperChunk(
                chunk_id=f"{metadata.paper_id}_{i}",
                paper_id=metadata.paper_id,
                text=chunk_text,
                section_type=SectionType.OTHER,
                section_title="Content",
                chunk_index=len(chunks),
                page_number=1,
                token_count=len(chunk_text) // 4,
                has_equations=False,
                has_citations=False,
                metadata=metadata,
            ))
            
            # Limit chunks per document
            if len(chunks) >= self._settings.max_chunks_per_doc:
                break
        
        logger.info(f"📦 Created {len(chunks)} chunks")
        return chunks
    
    def _already_ingested(self, paper_id: str) -> bool:
        """Check if paper already in store"""
        try:
            result = self._col.get(where={"paper_id": paper_id}, limit=1)
            return len(result.get("ids", [])) > 0
        except:
            return False
    
    def _store_chunks(self, chunks: list[PaperChunk]):
        """Store chunks in FAISS"""
        if not chunks:
            return
        
        self._col.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "paper_id": c.paper_id,
                    "title": c.metadata.title,
                    "authors": ", ".join(c.metadata.authors),
                    "filename": c.metadata.filename,
                    "chunk_index": c.chunk_index,
                }
                for c in chunks
            ],
        )
        logger.info(f"💾 Stored {len(chunks)} chunks in FAISS")