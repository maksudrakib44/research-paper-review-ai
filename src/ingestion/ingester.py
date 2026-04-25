"""
src/ingestion/ingester.py - Fixed PDF/Text Ingester
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Optional
from loguru import logger

# Try multiple PDF libraries
try:
    import pypdf
    HAS_PYPDF = True
except:
    HAS_PYPDF = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except:
    HAS_PDFPLUMBER = False

from src.models import PaperChunk, PaperMetadata, SectionType
from src.ingestion.section_chunker import AcademicPaperChunker


class AcademicPaperIngester:
    """Ingests academic papers from PDF or TXT files"""
    
    def __init__(self, collection, settings):
        self._col = collection
        self._settings = settings
        self._chunker = AcademicPaperChunker(settings.section_patterns)
        logger.info(f"PDF Support: pypdf={HAS_PYPDF}, pdfplumber={HAS_PDFPLUMBER}")
    
    def ingest(
        self,
        path: Path,
        title: Optional[str] = None,
        authors: Optional[list[str]] = None,
        year: Optional[int] = None,
        tags: Optional[list[str]] = None,
    ) -> list[PaperChunk]:
        """Ingest a research paper from PDF or TXT"""
        
        logger.info(f"📄 Processing: {path.name}")
        
        paper_id = self._stable_paper_id(path)
        
        # Check if already ingested
        if self._already_ingested(paper_id):
            logger.info(f"⏭️ Skipping already-ingested: {path.name}")
            return []
        
        # Extract text based on file type
        suffix = path.suffix.lower()
        
        if suffix == '.pdf':
            pages = self._extract_pdf_text_pypdf(path)
            if not pages:
                pages = self._extract_pdf_text_pdfplumber(path)
        elif suffix == '.txt':
            pages = self._extract_text_file(path)
        else:
            logger.error(f"Unsupported file type: {suffix}")
            return []
        
        if not pages:
            logger.error(f"❌ No text extracted from {path.name}")
            return []
        
        logger.info(f"📊 Extracted {len(pages)} pages, {sum(len(t) for _, t in pages)} chars")
        
        # Extract metadata
        metadata = self._extract_metadata(path, paper_id, title, authors, year, tags, pages)
        
        # Build chunks
        chunks = self._build_chunks(pages, metadata)
        
        if not chunks:
            logger.warning(f"⚠️ No chunks created for {path.name}")
            return []
        
        # Store in FAISS
        self._store_chunks(chunks)
        
        logger.success(f"✅ {path.name}: {len(chunks)} chunks indexed")
        return chunks
    
    def _extract_pdf_text_pypdf(self, path: Path) -> list[tuple[int, str]]:
        """Extract text using pypdf"""
        if not HAS_PYPDF:
            return []
        
        try:
            reader = pypdf.PdfReader(path)
            pages = []
            for i, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text and text.strip():
                    pages.append((i, text.strip()))
                else:
                    pages.append((i, ""))
            
            if any(t for _, t in pages):
                logger.info(f"✅ pypdf extracted {len([p for p in pages if p[1]])} pages")
                return pages
        except Exception as e:
            logger.warning(f"pypdf failed: {e}")
        
        return []
    
    def _extract_pdf_text_pdfplumber(self, path: Path) -> list[tuple[int, str]]:
        """Extract text using pdfplumber (better for academic papers)"""
        if not HAS_PDFPLUMBER:
            return []
        
        try:
            import pdfplumber
            pages = []
            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text and text.strip():
                        pages.append((i, text.strip()))
                    else:
                        pages.append((i, ""))
            
            if any(t for _, t in pages):
                logger.info(f"✅ pdfplumber extracted {len([p for p in pages if p[1]])} pages")
                return pages
        except Exception as e:
            logger.warning(f"pdfplumber failed: {e}")
        
        return []
    
    def _extract_text_file(self, path: Path) -> list[tuple[int, str]]:
        """Extract text from plain text file"""
        try:
            text = path.read_text(encoding='utf-8', errors='ignore')
            if text.strip():
                # Split into pages (approx 3000 chars per page for display)
                page_size = 3000
                pages = []
                for i in range(0, len(text), page_size):
                    page_text = text[i:i+page_size]
                    pages.append((i//page_size + 1, page_text.strip()))
                logger.info(f"✅ Extracted {len(pages)} pages from text file")
                return pages
        except Exception as e:
            logger.error(f"Failed to read text file: {e}")
        
        return []
    
    def _extract_metadata(self, path, paper_id, title, authors, year, tags, pages):
        """Extract metadata from paper"""
        full_text = " ".join(text for _, text in pages if text)
        
        # Try to extract title from first few lines
        if not title and pages:
            first_page = pages[0][1]
            lines = first_page.split('\n')[:10]
            for line in lines:
                line = line.strip()
                if len(line) > 20 and len(line) < 200 and not line.startswith(('Abstract', 'ABSTRACT', '1.', '2.')):
                    title = line
                    break
        
        # Try to extract authors
        if not authors and pages:
            first_page = pages[0][1]
            # Look for author patterns
            author_match = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?(?:\s+et\s+al\.?)?(?:\s*,\s*[A-Z][a-z]+)*)', first_page[:500])
            if author_match:
                authors = [author_match.group(1)]
        
        # Try to extract year
        if not year:
            year_match = re.search(r'(19|20)\d{2}', full_text[:2000])
            if year_match:
                year = int(year_match.group(0))
        
        return PaperMetadata(
            paper_id=paper_id,
            title=title or path.stem.replace("_", " ").replace("-", " "),
            authors=authors or ["Unknown"],
            year=year,
            filename=path.name,
            tags=tags or [],
        )
    
    def _build_chunks(self, pages, metadata):
        """Build chunks from extracted text"""
        chunks = []
        full_text = "\n\n".join(text for _, text in pages if text)
        
        if not full_text.strip():
            return []
        
        # Simple chunking if section chunker fails
        words = full_text.split()
        chunk_size = self._settings.chunk_size
        chunk_overlap = self._settings.chunk_overlap
        step = chunk_size - chunk_overlap
        
        for i in range(0, len(words), step):
            chunk_words = words[i:i + chunk_size]
            if not chunk_words:
                continue
            chunk_text = ' '.join(chunk_words)
            
            chunks.append(
                PaperChunk(
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
                )
            )
            
            if len(chunks) >= self._settings.max_chunks_per_doc:
                break
        
        logger.info(f"📦 Created {len(chunks)} chunks")
        return chunks
    
    def _stable_paper_id(self, path):
        """Generate stable ID from content"""
        try:
            content_hash = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        except:
            content_hash = hashlib.sha256(path.name.encode()).hexdigest()[:16]
        return f"paper_{content_hash}"
    
    def _already_ingested(self, paper_id):
        """Check if paper already in store"""
        try:
            result = self._col.get(where={"paper_id": paper_id}, limit=1)
            return len(result.get("ids", [])) > 0
        except:
            return False
    
    def _store_chunks(self, chunks):
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
                    "section_type": c.section_type.value,
                    "section_title": c.section_title,
                    "chunk_index": c.chunk_index,
                    "has_equations": c.has_equations,
                    "has_citations": c.has_citations,
                    "year": c.metadata.year or "",
                    "tags": ", ".join(c.metadata.tags),
                    "filename": c.metadata.filename,
                }
                for c in chunks
            ],
        )
        logger.info(f"💾 Stored {len(chunks)} chunks in FAISS")