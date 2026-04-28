"""
src/ingestion/section_chunker.py - Academic Paper Chunking
"""
from __future__ import annotations

import re
from typing import Generator, Tuple, List
from src.models import SectionType


class AcademicPaperChunker:
    """Specialized chunker for academic papers with section detection"""
    
    def __init__(self, section_patterns: List[str]):
        self.section_patterns = section_patterns
        # Compile regex for section detection
        pattern = '|'.join(re.escape(p) for p in section_patterns)
        self.section_regex = re.compile(f'^(?:{pattern}):?\\s*$', re.IGNORECASE | re.MULTILINE)
    
    def detect_section_type(self, section_title: str) -> SectionType:
        """Map section title to SectionType enum"""
        title_lower = section_title.lower().strip()
        
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
            matches = re.findall(pattern, text, re.DOTALL)
            equations.extend(matches)
        return equations
    
    def extract_citations(self, text: str) -> List[Tuple[str, int]]:
        """Extract academic citations from text"""
        patterns = [
            r'\(([A-Z][a-z]+(?:\s+et\s+al\.?)?),\s*(\d{4})\)',
            r'\[(\d+(?:[–-]\d+)?)\]',
            r'\\cite\{([^}]+)\}'
        ]
        citations = []
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                citations.append((match.group(0), match.start()))
        return citations
    
    def chunk_by_sections(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> Generator[Tuple[str, SectionType, str, bool, bool], None, None]:
        """
        Chunk paper by sections first, then by token limit.
        
        Yields: (chunk_text, section_type, section_title, has_equations, has_citations)
        """
        if not text or not text.strip():
            return
        
        lines = text.split('\n')
        current_section = "Content"
        current_content = []
        
        for line in lines:
            line_stripped = line.strip()
            
            # Check if line is a section header
            if self.section_regex.match(line_stripped):
                # Yield previous section if it has content
                if current_content:
                    section_text = '\n'.join(current_content).strip()
                    if section_text:
                        section_type = self.detect_section_type(current_section)
                        equations = self.extract_equations(section_text)
                        citations = self.extract_citations(section_text)
                        
                        # Chunk this section
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
                current_section = line_stripped.rstrip(':')
                current_content = []
            else:
                current_content.append(line)
        
        # Yield final section
        if current_content:
            section_text = '\n'.join(current_content).strip()
            if section_text:
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
        """
        Split long text into smaller chunks using sliding window.
        """
        if not text:
            return
        
        words = text.split()
        
        # If text is short enough, yield as one chunk
        if len(words) <= chunk_size:
            yield (text, section_type, section_title, has_equations, has_citations)
            return
        
        # Sliding window for long sections
        step = max(1, chunk_size - chunk_overlap)
        
        for i in range(0, len(words), step):
            chunk_words = words[i:i + chunk_size]
            if not chunk_words:
                continue
            
            chunk_text = ' '.join(chunk_words)
            
            # Check if this chunk has equations or citations
            chunk_has_eq = has_equations and any(eq in chunk_text for eq in self.extract_equations(chunk_text))
            chunk_has_cite = has_citations and any(cite[0] in chunk_text for cite in self.extract_citations(chunk_text))
            
            yield (chunk_text, section_type, section_title, chunk_has_eq, chunk_has_cite)