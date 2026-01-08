"""
Advanced chunking strategies for finance documents
Handles text, tables, images, and maintains document structure
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter, SemanticChunker
from langchain_openai import OpenAIEmbeddings
import tiktoken


class ChunkType(Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    HEADER = "header"
    FOOTER = "footer"


@dataclass
class Chunk:
    content: str
    chunk_type: ChunkType
    metadata: Dict[str, Any]
    page_number: int
    section: str
    chapter: str
    token_count: int
    

class FinanceChunkingStrategy:
    """
    Advanced chunking strategies specifically designed for finance documents
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.encoding = tiktoken.encoding_for_model(model_name)
        self.max_chunk_size = 1024  # tokens
        self.chunk_overlap = 200    # tokens
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def semantic_chunking(self, text: str, metadata: Dict) -> List[Chunk]:
        """
        Semantic chunking using embeddings to find natural breakpoints
        Best for preserving financial concepts and context
        """
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        
        text_splitter = SemanticChunker(
            embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95,
            number_of_chunks=None
        )
        
        chunks = text_splitter.create_documents([text])
        
        result_chunks = []
        for i, chunk in enumerate(chunks):
            result_chunks.append(Chunk(
                content=chunk.page_content,
                chunk_type=ChunkType.TEXT,
                metadata={
                    **metadata,
                    'chunk_id': i,
                    'chunking_strategy': 'semantic'
                },
                page_number=metadata.get('page_number', 0),
                section=metadata.get('section', ''),
                chapter=metadata.get('chapter', ''),
                token_count=self.count_tokens(chunk.page_content)
            ))
            
        return result_chunks
    
    def hierarchical_chunking(self, text: str, metadata: Dict) -> List[Chunk]:
        """
        Hierarchical chunking that preserves document structure
        Creates parent-child relationships between chunks
        """
        # Extract document hierarchy
        sections = self._extract_sections(text)
        
        all_chunks = []
        
        for section_title, section_content in sections.items():
            # Create parent chunk for section
            parent_chunk = Chunk(
                content=f"{section_title}\n{section_content[:500]}...",
                chunk_type=ChunkType.HEADER,
                metadata={
                    **metadata,
                    'section_title': section_title,
                    'is_parent': True,
                    'chunking_strategy': 'hierarchical'
                },
                page_number=metadata.get('page_number', 0),
                section=section_title,
                chapter=metadata.get('chapter', ''),
                token_count=self.count_tokens(section_content)
            )
            all_chunks.append(parent_chunk)
            
            # Create child chunks within section
            child_chunks = self._split_section_content(
                section_content, section_title, metadata
            )
            all_chunks.extend(child_chunks)
            
        return all_chunks
    
    def table_aware_chunking(self, text: str, metadata: Dict) -> List[Chunk]:
        """
        Special handling for tables - keep tables intact as single chunks
        """
        chunks = []
        
        # Find and extract tables
        table_pattern = r'(\|.*\|(?:\n\|.*\|)*)'
        tables = re.finditer(table_pattern, text, re.MULTILINE)
        
        last_end = 0
        
        for match in tables:
            start, end = match.span()
            
            # Add text before table as regular chunk
            if start > last_end:
                text_before = text[last_end:start].strip()
                if text_before:
                    text_chunks = self._split_text_content(text_before, metadata)
                    chunks.extend(text_chunks)
            
            # Add table as single chunk
            table_content = match.group(1)
            table_chunk = Chunk(
                content=table_content,
                chunk_type=ChunkType.TABLE,
                metadata={
                    **metadata,
                    'chunking_strategy': 'table_aware',
                    'contains_table': True
                },
                page_number=metadata.get('page_number', 0),
                section=metadata.get('section', ''),
                chapter=metadata.get('chapter', ''),
                token_count=self.count_tokens(table_content)
            )
            chunks.append(table_chunk)
            
            last_end = end
        
        # Add remaining text
        if last_end < len(text):
            remaining_text = text[last_end:].strip()
            if remaining_text:
                text_chunks = self._split_text_content(remaining_text, metadata)
                chunks.extend(text_chunks)
        
        return chunks
    
    def multi_modal_chunking(self, content: Dict, metadata: Dict) -> List[Chunk]:
        """
        Handle multi-modal content (text + images + tables)
        """
        chunks = []
        
        # Process text content
        if 'text' in content:
            text_chunks = self.semantic_chunking(content['text'], metadata)
            chunks.extend(text_chunks)
        
        # Process images with context
        if 'images' in content:
            for i, image_info in enumerate(content['images']):
                # Include surrounding text context with image
                context_text = image_info.get('surrounding_text', '')
                
                image_chunk = Chunk(
                    content=f"Image: {image_info['caption']}\nContext: {context_text}",
                    chunk_type=ChunkType.IMAGE,
                    metadata={
                        **metadata,
                        'image_path': image_info['path'],
                        'image_caption': image_info['caption'],
                        'chunking_strategy': 'multimodal'
                    },
                    page_number=metadata.get('page_number', 0),
                    section=metadata.get('section', ''),
                    chapter=metadata.get('chapter', ''),
                    token_count=self.count_tokens(context_text)
                )
                chunks.append(image_chunk)
        
        # Process tables
        if 'tables' in content:
            for table_info in content['tables']:
                table_chunk = Chunk(
                    content=table_info['content'],
                    chunk_type=ChunkType.TABLE,
                    metadata={
                        **metadata,
                        'table_caption': table_info.get('caption', ''),
                        'chunking_strategy': 'multimodal'
                    },
                    page_number=metadata.get('page_number', 0),
                    section=metadata.get('section', ''),
                    chapter=metadata.get('chapter', ''),
                    token_count=self.count_tokens(table_info['content'])
                )
                chunks.append(table_chunk)
        
        return chunks
    
    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract sections based on headers"""
        sections = {}
        
        # Pattern for financial document sections
        section_pattern = r'^((?:\d+\.)*\d*\s*[A-Z][^.\n]*(?:\n|$))'
        
        sections_found = re.split(section_pattern, text, flags=re.MULTILINE)
        
        current_section = "Introduction"
        for i in range(1, len(sections_found), 2):
            if i < len(sections_found) - 1:
                section_title = sections_found[i].strip()
                section_content = sections_found[i + 1].strip()
                sections[section_title] = section_content
        
        return sections
    
    def _split_section_content(self, content: str, section_title: str, 
                              metadata: Dict) -> List[Chunk]:
        """Split section content into smaller chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.count_tokens,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.create_documents([content])
        
        result_chunks = []
        for i, chunk in enumerate(chunks):
            result_chunks.append(Chunk(
                content=chunk.page_content,
                chunk_type=ChunkType.TEXT,
                metadata={
                    **metadata,
                    'parent_section': section_title,
                    'chunk_id': i,
                    'is_child': True
                },
                page_number=metadata.get('page_number', 0),
                section=section_title,
                chapter=metadata.get('chapter', ''),
                token_count=self.count_tokens(chunk.page_content)
            ))
            
        return result_chunks
    
    def _split_text_content(self, text: str, metadata: Dict) -> List[Chunk]:
        """Split regular text content"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.count_tokens
        )
        
        chunks = text_splitter.create_documents([text])
        
        result_chunks = []
        for i, chunk in enumerate(chunks):
            result_chunks.append(Chunk(
                content=chunk.page_content,
                chunk_type=ChunkType.TEXT,
                metadata={**metadata, 'chunk_id': i},
                page_number=metadata.get('page_number', 0),
                section=metadata.get('section', ''),
                chapter=metadata.get('chapter', ''),
                token_count=self.count_tokens(chunk.page_content)
            ))
            
        return result_chunks


class AdaptiveChunker:
    """
    Adaptive chunker that selects the best strategy based on content type
    """
    
    def __init__(self):
        self.strategy = FinanceChunkingStrategy()
    
    def chunk_document(self, content: str, metadata: Dict) -> List[Chunk]:
        """
        Automatically select and apply the best chunking strategy
        """
        # Analyze content to determine best strategy
        content_analysis = self._analyze_content(content)
        
        if content_analysis['has_tables'] and content_analysis['has_images']:
            # Multi-modal content
            return self._prepare_multimodal_content(content, metadata)
        elif content_analysis['has_tables']:
            # Table-heavy content
            return self.strategy.table_aware_chunking(content, metadata)
        elif content_analysis['has_clear_structure']:
            # Well-structured document
            return self.strategy.hierarchical_chunking(content, metadata)
        else:
            # Default to semantic chunking
            return self.strategy.semantic_chunking(content, metadata)
    
    def _analyze_content(self, content: str) -> Dict[str, bool]:
        """Analyze content to determine characteristics"""
        return {
            'has_tables': bool(re.search(r'\|.*\|', content)),
            'has_images': bool(re.search(r'!\[.*\]', content)),
            'has_clear_structure': bool(re.search(r'^\d+\.', content, re.MULTILINE)),
            'is_financial': bool(re.search(r'\$[\d,]+|\d+%|[A-Z]{3,4}\s+\d+', content))
        }
    
    def _prepare_multimodal_content(self, content: str, metadata: Dict) -> List[Chunk]:
        """Prepare content for multimodal chunking"""
        # This would integrate with image extraction and table parsing
        multimodal_content = {
            'text': content,
            'images': [],  # Would be populated by image extractor
            'tables': []   # Would be populated by table extractor
        }
        
        return self.strategy.multi_modal_chunking(multimodal_content, metadata)


# Usage example
if __name__ == "__main__":
    chunker = AdaptiveChunker()
    
    sample_text = """
    1. Executive Summary
    
    This quarterly financial report provides an overview of our performance.
    
    | Metric | Q1 2024 | Q2 2024 | Change |
    |--------|---------|---------|--------|
    | Revenue | $1.2M | $1.5M | +25% |
    | Profit | $200K | $300K | +50% |
    
    2. Financial Analysis
    
    Our revenue growth of 25% demonstrates strong market performance...
    """
    
    metadata = {
        'page_number': 1,
        'section': 'Financial Report',
        'chapter': 'Q2 2024',
        'document_type': 'financial_report'
    }
    
    chunks = chunker.chunk_document(sample_text, metadata)
    
    for chunk in chunks:
        print(f"Type: {chunk.chunk_type.value}")
        print(f"Content: {chunk.content[:100]}...")
        print(f"Tokens: {chunk.token_count}")
        print("---")
