"""
Advanced chunking strategies for different document types.
Preserves document structure (headers, lists, code blocks, tables).
"""
import logging
import re
from typing import List, Dict, Any, Optional
from pathlib import Path

from llama_index.core import Document
from llama_index.core.node_parser import (
    SentenceSplitter,
    MarkdownNodeParser,
    CodeSplitter,
)
from llama_index.core.schema import BaseNode, TextNode

logger = logging.getLogger(__name__)


def detect_document_type(document: Document) -> str:
    """Detect document type based on file extension and content."""
    metadata = document.metadata or {}
    file_path = (
        metadata.get("file_path") or 
        metadata.get("file_name") or 
        metadata.get("filename") or 
        metadata.get("source") or 
        ""
    )
    
    if not file_path:
        # Try to detect from content
        content = document.get_content()
        if content.startswith("#") or "```" in content:
            return "markdown"
        if "<html" in content.lower() or "<body" in content.lower():
            return "html"
        return "text"
    
    path = Path(file_path)
    ext = path.suffix.lower()
    
    type_mapping = {
        ".md": "markdown",
        ".markdown": "markdown",
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".java": "java",
        ".cpp": "cpp",
        ".c": "c",
        ".html": "html",
        ".htm": "html",
        ".pdf": "pdf",
        ".docx": "docx",
        ".txt": "text",
    }
    
    return type_mapping.get(ext, "text")


def detect_code_blocks(text: str) -> List[Dict[str, Any]]:
    """Detect code blocks in text and return their positions."""
    code_blocks = []
    # Match fenced code blocks (```...```)
    pattern = r'```[\s\S]*?```'
    for match in re.finditer(pattern, text):
        code_blocks.append({
            "start": match.start(),
            "end": match.end(),
            "content": match.group(),
            "type": "fenced"
        })
    # Match indented code blocks (4+ spaces or tabs)
    lines = text.split('\n')
    in_code_block = False
    code_start = 0
    for i, line in enumerate(lines):
        if re.match(r'^(\s{4,}|\t+)', line) and line.strip():
            if not in_code_block:
                in_code_block = True
                code_start = sum(len(l) + 1 for l in lines[:i])
        else:
            if in_code_block:
                code_end = sum(len(l) + 1 for l in lines[:i])
                code_blocks.append({
                    "start": code_start,
                    "end": code_end,
                    "content": text[code_start:code_end],
                    "type": "indented"
                })
                in_code_block = False
    
    return code_blocks


def detect_tables(text: str) -> List[Dict[str, Any]]:
    """Detect markdown tables and return their positions."""
    tables = []
    lines = text.split('\n')
    current_table = []
    table_start_line = None
    
    for i, line in enumerate(lines):
        # Check if line looks like a table row (contains |)
        if '|' in line and line.strip().startswith('|') and line.strip().endswith('|'):
            if not current_table:
                table_start_line = i
            current_table.append(line)
        elif re.match(r'^\|[\s\-:]+\|$', line.strip()):  # Table separator
            if current_table:
                current_table.append(line)
        else:
            if current_table and len(current_table) >= 2:  # At least header + separator
                # Calculate positions
                start_pos = sum(len(l) + 1 for l in lines[:table_start_line])
                end_pos = sum(len(l) + 1 for l in lines[:i])
                tables.append({
                    "start": start_pos,
                    "end": end_pos,
                    "content": '\n'.join(current_table),
                    "type": "markdown_table"
                })
            current_table = []
            table_start_line = None
    
    # Handle table at end of document
    if current_table and len(current_table) >= 2:
        start_pos = sum(len(l) + 1 for l in lines[:table_start_line])
        end_pos = len(text)
        tables.append({
            "start": start_pos,
            "end": end_pos,
            "content": '\n'.join(current_table),
            "type": "markdown_table"
        })
    
    return tables


def detect_headers(text: str) -> List[Dict[str, Any]]:
    """Detect markdown headers and return their positions and levels."""
    headers = []
    lines = text.split('\n')
    
    for i, line in enumerate(lines):
        # ATX-style headers (# ## ###)
        match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            pos = sum(len(l) + 1 for l in lines[:i])
            headers.append({
                "start": pos,
                "end": pos + len(line),
                "level": level,
                "title": title,
                "type": "atx"
            })
        # Setext-style headers (=== or ---)
        elif i > 0 and re.match(r'^[=\-]+$', line.strip()):
            prev_line = lines[i-1].strip()
            if prev_line:
                level = 1 if '=' in line else 2
                pos = sum(len(l) + 1 for l in lines[:i-1])
                headers.append({
                    "start": pos,
                    "end": pos + len(lines[i-1]) + 1 + len(line),
                    "level": level,
                    "title": prev_line,
                    "type": "setext"
                })
    
    return headers


def preserve_structure_in_metadata(node: BaseNode, document: Document) -> BaseNode:
    """Add structure information to node metadata."""
    content = node.get_content()
    
    # Detect structure elements
    code_blocks = detect_code_blocks(content)
    tables = detect_tables(content)
    headers = detect_headers(content)
    
    # Update metadata
    metadata = node.metadata or {}
    
    if code_blocks:
        metadata["has_code"] = True
        metadata["code_block_count"] = len(code_blocks)
    
    if tables:
        metadata["has_table"] = True
        metadata["table_count"] = len(tables)
    
    if headers:
        metadata["has_headers"] = True
        metadata["header_count"] = len(headers)
        # Include header hierarchy
        header_titles = [h["title"] for h in headers[:3]]  # First 3 headers
        metadata["section_headers"] = " > ".join(header_titles)
    
    # Detect lists
    list_pattern = r'^[\s]*[-*+]\s+|^[\s]*\d+\.\s+'
    if re.search(list_pattern, content, re.MULTILINE):
        metadata["has_list"] = True
    
    node.metadata = metadata
    return node


def chunk_markdown_document(document: Document) -> List[BaseNode]:
    """Chunk markdown documents preserving structure."""
    try:
        parser = MarkdownNodeParser()
        nodes = parser.get_nodes_from_documents([document])
        
        # Post-process to preserve structure
        processed_nodes = []
        for node in nodes:
            node = preserve_structure_in_metadata(node, document)
            processed_nodes.append(node)
        
        logger.info(f"Chunked markdown document into {len(processed_nodes)} nodes")
        return processed_nodes
    except Exception as e:
        logger.warning(f"Markdown parsing failed, falling back to sentence splitter: {e}")
        return chunk_with_sentence_splitter(document, chunk_size=800, chunk_overlap=150)


def chunk_code_document(document: Document, language: Optional[str] = None) -> List[BaseNode]:
    """Chunk code documents preserving code blocks."""
    try:
        # Use CodeSplitter for code files
        parser = CodeSplitter(
            language=language,
            chunk_lines=50,  # Smaller chunks for code
            chunk_lines_overlap=10,
            max_chars=1500,  # Max characters per chunk
        )
        nodes = parser.get_nodes_from_documents([document])
        
        # Post-process to preserve structure
        processed_nodes = []
        for node in nodes:
            node = preserve_structure_in_metadata(node, document)
            # Add language info
            if language:
                node.metadata = node.metadata or {}
                node.metadata["code_language"] = language
            processed_nodes.append(node)
        
        logger.info(f"Chunked code document into {len(processed_nodes)} nodes")
        return processed_nodes
    except Exception as e:
        logger.warning(f"Code parsing failed, falling back to sentence splitter: {e}")
        return chunk_with_sentence_splitter(document, chunk_size=600, chunk_overlap=100)


def chunk_with_sentence_splitter(
    document: Document, 
    chunk_size: int = 1000, 
    chunk_overlap: int = 200
) -> List[BaseNode]:
    """Chunk using sentence splitter with custom sizes."""
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="\n\n",  # Prefer splitting on paragraph breaks
    )
    nodes = splitter.get_nodes_from_documents([document])
    
    # Post-process to preserve structure
    processed_nodes = []
    for node in nodes:
        node = preserve_structure_in_metadata(node, document)
        processed_nodes.append(node)
    
    return processed_nodes


def chunk_pdf_document(document: Document) -> List[BaseNode]:
    """Chunk PDF documents with special handling for tables and images."""
    content = document.get_content()
    
    # Detect if content has tables (common in PDFs)
    has_tables = (
        '|' in content or 
        '\t' in content or
        re.search(r'\s{3,}', content)  # Multiple spaces (potential table)
    )
    
    # Use larger chunks for PDFs to preserve context
    # But smaller overlap to avoid too much duplication
    chunk_size = 1200 if has_tables else 1000
    chunk_overlap = 150
    
    # Use sentence splitter but with paragraph awareness
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="\n\n",  # Prefer paragraph breaks
        paragraph_separator="\n\n\n",  # Strong paragraph breaks
    )
    nodes = splitter.get_nodes_from_documents([document])
    
    # Post-process: detect and preserve tables
    processed_nodes = []
    for node in nodes:
        node_content = node.get_content()
        
        # Try to detect and format tables
        formatted_content = format_tables_in_text(node_content)
        if formatted_content != node_content:
            node.set_content(formatted_content)
            node.metadata = node.metadata or {}
            node.metadata["has_table"] = True
        
        # Preserve other structure
        node = preserve_structure_in_metadata(node, document)
        
        # Add PDF-specific metadata
        node.metadata = node.metadata or {}
        node.metadata["document_type"] = "pdf"
        if document.metadata.get("page_label"):
            node.metadata["page"] = document.metadata.get("page_label")
        
        processed_nodes.append(node)
    
    logger.info(f"Chunked PDF document into {len(processed_nodes)} nodes")
    return processed_nodes


def format_tables_in_text(text: str) -> str:
    """Detect and format tables in text to markdown format."""
    lines = text.split('\n')
    result_lines = []
    current_table = []
    in_table = False
    
    for i, line in enumerate(lines):
        # Detect table-like patterns
        # Pattern 1: Multiple columns separated by tabs or multiple spaces
        parts_tab = line.split('\t')
        parts_space = re.split(r'\s{2,}', line.strip())
        
        is_table_row = False
        if len(parts_tab) >= 2 and all(p.strip() for p in parts_tab[:3]):
            is_table_row = True
            columns = [p.strip() for p in parts_tab if p.strip()]
        elif len(parts_space) >= 2 and all(p.strip() for p in parts_space[:3]):
            is_table_row = True
            columns = [p.strip() for p in parts_space if p.strip()]
        
        if is_table_row:
            if not in_table:
                in_table = True
                current_table = []
            current_table.append(columns)
        else:
            if in_table and len(current_table) >= 2:
                # Format as markdown table
                if current_table:
                    # Header row
                    header = '| ' + ' | '.join(current_table[0]) + ' |'
                    result_lines.append(header)
                    # Separator
                    separator = '| ' + ' | '.join(['---'] * len(current_table[0])) + ' |'
                    result_lines.append(separator)
                    # Data rows
                    for row in current_table[1:]:
                        # Pad row to match header length
                        padded_row = row + [''] * (len(current_table[0]) - len(row))
                        row_str = '| ' + ' | '.join(padded_row[:len(current_table[0])]) + ' |'
                        result_lines.append(row_str)
                current_table = []
                in_table = False
            
            result_lines.append(line)
    
    # Handle table at end
    if in_table and len(current_table) >= 2:
        header = '| ' + ' | '.join(current_table[0]) + ' |'
        result_lines.append(header)
        separator = '| ' + ' | '.join(['---'] * len(current_table[0])) + ' |'
        result_lines.append(separator)
        for row in current_table[1:]:
            padded_row = row + [''] * (len(current_table[0]) - len(row))
            row_str = '| ' + ' | '.join(padded_row[:len(current_table[0])]) + ' |'
            result_lines.append(row_str)
    
    return '\n'.join(result_lines)


def smart_chunk_documents(documents: List[Document]) -> List[BaseNode]:
    """
    Intelligently chunk documents based on their type.
    Preserves structure (headers, lists, code blocks, tables).
    """
    all_nodes = []
    
    for doc in documents:
        doc_type = detect_document_type(doc)
        logger.info(f"Chunking document type: {doc_type}")
        
        try:
            if doc_type == "markdown":
                nodes = chunk_markdown_document(doc)
            elif doc_type in ["python", "javascript", "typescript", "java", "cpp", "c"]:
                nodes = chunk_code_document(doc, language=doc_type)
            elif doc_type == "pdf":
                nodes = chunk_pdf_document(doc)
            elif doc_type == "html":
                # HTML can be complex, use sentence splitter with larger chunks
                nodes = chunk_with_sentence_splitter(doc, chunk_size=1000, chunk_overlap=200)
            else:
                # Default: use sentence splitter with adaptive sizing
                content = doc.get_content()
                # Adjust chunk size based on content characteristics
                if len(content) < 2000:
                    # Small document: single chunk or small chunks
                    chunk_size = min(1500, len(content) + 100)
                    chunk_overlap = 100
                elif 'table' in content.lower() or '|' in content:
                    # Has tables: larger chunks
                    chunk_size = 1200
                    chunk_overlap = 200
                else:
                    # Standard chunking
                    chunk_size = 1000
                    chunk_overlap = 150
                
                nodes = chunk_with_sentence_splitter(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            
            # Ensure all nodes have proper metadata
            for node in nodes:
                node.metadata = node.metadata or {}
                node.metadata["document_type"] = doc_type
                # Preserve original document metadata
                if doc.metadata:
                    for key, value in doc.metadata.items():
                        if key not in node.metadata:
                            node.metadata[key] = value
            
            all_nodes.extend(nodes)
            
        except Exception as e:
            logger.error(f"Error chunking document: {e}", exc_info=True)
            # Fallback to basic chunking
            logger.warning("Falling back to basic sentence splitter")
            try:
                nodes = chunk_with_sentence_splitter(doc)
                all_nodes.extend(nodes)
            except Exception as fallback_error:
                logger.error(f"Fallback chunking also failed: {fallback_error}", exc_info=True)
                # Last resort: create single node
                node = TextNode(
                    text=doc.get_content(),
                    metadata=doc.metadata or {}
                )
                all_nodes.append(node)
    
    logger.info(f"Total chunks created: {len(all_nodes)}")
    return all_nodes

