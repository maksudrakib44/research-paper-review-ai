"""
src/ingestion/section_chunker.py - Academic Paper Chunking
"""
from __future__ import annotations

import re
from typing import Generator, Tuple, List
from src.models import SectionType


class AcademicPaperChunker:
    """Specialized chunker for academic papers"""
    
    def __init__(self, section_patterns: List[str]):
        self.section_patterns = section_patterns
        # Match section headers (e.g., "Abstract:", "1. Introduction", "METHODS")
        self.section_pattern = re.compile(
            r'(?m)^(?:' + '|'.join(re.escape(p) for p in section_patterns) + r'):?\s*$',
            re.IGNORECASE
        )
    
    def detect_section_type(self, section_title: str) -> SectionType:
        """Map section title to SectionType enum"""
        title_lower = section_title.lower()
        
        if 'abstract' in title_lower:
            return SectionType.ABSTRACT
        elif 'introduction' in title_lower:
            return SectionType.INTRODUCTION
        elif 'related' in title_lower or 'literature' in title_lower:
            return SectionType.RELATED_WORK
        elif 'method' in title_lower or 'approach' in title_lower:
            return SectionType.METHODOLOGY
        elif 'experiment' in title_lower:
            return SectionType.EXPERIMENTS
        elif 'result' in title_lower:
            return SectionType.RESULTS
        elif 'discussion' in title_lower:
            return SectionType.DISCUSSION
        elif 'conclusion' in title_lower or 'future' in title_lower:
            return SectionType.CONCLUSION
        elif 'reference' in title_lower or 'bibliography' in title_lower:
            return SectionType.REFERENCES
        else:
            return SectionType.OTHER
    
    def extract_equations(self, text: str) -> List[str]:
        """Extract LaTeX equations from text"""
        patterns = [
            r'\$\$(.+?)\$\$',
            r'\\\((.+?)\\\)',
            r'\\\[(.+?)\\\]'
        ]
        equations = []
        for pattern in patterns:
            equations.extend(re.findall(pattern, text, re.DOTALL))
        return equations
    
    def extract_citations(self, text: str) -> List[Tuple[str, int]]:
        """Extract academic citations"""
        citation_patterns = [
            r'\(([A-Z][a-z]+(?:\s+et\s+al\.?)?(?:\s+and\s+[A-Z][a-z]+)?),\s*(\d{4})\)',
            r'\[(\d+(?:[–-]\d+)?)\]',
            r'\\cite\{([^}]+)\}'
        ]
        
        citations = []
        for pattern in citation_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                citations.append((match.group(0), match.start()))
        
        return citations
    
    def chunk_by_sections(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> Generator[Tuple[str, SectionType, str, bool, bool], None, None]:
        """Chunk paper by sections first, then by token limit"""
        lines = text.split('\n')
        current_section = "Abstract"
        current_content = []
        
        for line in lines:
            # Check if line is a section header
            if self.section_pattern.match(line.strip()):
                # Process previous section
                if current_content:
                    section_text = '\n'.join(current_content)
                    section_type = self.detect_section_type(current_section)
                    equations = self.extract_equations(section_text)
                    citations = self.extract_citations(section_text)
                    
                    yield from self._chunk_text(
                        section_text,
                        current_section,
                        section_type,
                        bool(equations),
                        bool(citations),
                        chunk_size,
                        chunk_overlap
                    )
                
                # Start new section
                current_section = line.strip().rstrip(':')
                current_content = []
            else:
                current_content.append(line)
        
        # Process last section
        if current_content:
            section_text = '\n'.join(current_content)
            section_type = self.detect_section_type(current_section)
            equations = self.extract_equations(section_text)
            citations = self.extract_citations(section_text)
            
            yield from self._chunk_text(
                section_text,
                current_section,
                section_type,
                bool(equations),
                bool(citations),
                chunk_size,
                chunk_overlap
            )
    
    def _chunk_text(
        self,
        text: str,
        section_title: str,
        section_type: SectionType,
        has_equations: bool,
        has_citations: bool,
        chunk_size: int,
        chunk_overlap: int,
    ) -> Generator[Tuple[str, SectionType, str, bool, bool], None, None]:
        """Split long sections into smaller chunks"""
        words = text.split()
        
        if len(words) <= chunk_size:
            yield (text, section_type, section_title, has_equations, has_citations)
            return
        
        # Sliding window for long sections
        step = chunk_size - chunk_overlap
        for i in range(0, len(words), step):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            yield (chunk_text, section_type, section_title, has_equations, has_citations)