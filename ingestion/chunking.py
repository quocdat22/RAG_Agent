"""
Advanced chunking strategies for different document types.
Preserves document structure (headers, lists, code blocks, tables).
"""
import logging
import os
import re
import warnings
from typing import List, Dict, Any, Optional
import io
from pathlib import Path

from llama_index.core import Document
from llama_index.core.node_parser import (
    SentenceSplitter,
    MarkdownNodeParser,
    CodeSplitter,
)
from llama_index.core.schema import BaseNode, TextNode

from ingestion.metadata_schema import normalize_metadata, get_file_path, MetadataFields
from rag.llm import generate_image_description

logger = logging.getLogger(__name__)

# Suppress noisy PDF library warnings using Python's warnings module
warnings.filterwarnings('ignore', message='.*Cannot set gray non-stroke color.*')
warnings.filterwarnings('ignore', message='.*invalid float value.*')
warnings.filterwarnings('ignore', category=UserWarning, module='pdfplumber')
warnings.filterwarnings('ignore', category=UserWarning, module='pdfminer')

# Filter to suppress noisy PDF library warnings in logging
class PDFWarningFilter(logging.Filter):
    """Filter to suppress common PDF processing warnings that don't affect functionality."""
    def filter(self, record):
        # Suppress warnings about invalid float values in PDF color settings
        # These are common in PDFs and don't affect table extraction
        message = record.getMessage()
        if "Cannot set gray non-stroke color" in message:
            return False
        if "invalid float value" in message.lower():
            return False
        if "pdfplumber" in record.name.lower() and record.levelno == logging.WARNING:
            # Only suppress specific pdfplumber warnings, not all
            if any(phrase in message.lower() for phrase in ["color", "float", "invalid"]):
                return False
        return True

# Apply filter to root logger to catch all PDF library warnings
pdf_filter = PDFWarningFilter()
logging.getLogger().addFilter(pdf_filter)

# Also suppress warnings from specific PDF libraries
for lib_name in ['pdfplumber', 'pdfminer', 'PIL', 'Pillow']:
    lib_logger = logging.getLogger(lib_name)
    lib_logger.addFilter(pdf_filter)
    lib_logger.setLevel(logging.ERROR)  # Only show errors, not warnings

# Try to import PDF table extraction libraries
try:
    # Suppress warnings before importing pdfplumber
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import pdfplumber
    PDFPLUMBER_AVAILABLE = True
    # Suppress pdfplumber's internal warnings
    pdfplumber_logger = logging.getLogger('pdfplumber')
    pdfplumber_logger.setLevel(logging.ERROR)
    # Also suppress pdfminer warnings (used by pdfplumber)
    pdfminer_logger = logging.getLogger('pdfminer')
    pdfminer_logger.setLevel(logging.ERROR)
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber not available. PDF table extraction will be limited.")

try:
    from pdf2image import convert_from_path
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logger.warning("OCR libraries (pdf2image, pytesseract) not available. OCR table detection will be disabled.")


def detect_document_type(document: Document) -> str:
    """Detect document type based on file extension and content."""
    metadata = document.metadata or {}
    # Use normalized metadata helper
    file_path = get_file_path(metadata)
    
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
    """
    Detect markdown tables and return their positions.
    Improved to avoid false positives with text containing | characters.
    """
    tables = []
    lines = text.split('\n')
    current_table = []
    table_start_line = None
    expected_columns = None
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Check if line is a table separator (---|:---|:---:|---)
        is_separator = bool(re.match(r'^\|[\s\-:]+\|$', stripped))
        
        # Check if line looks like a table row
        is_table_row = False
        if '|' in line and stripped.startswith('|') and stripped.endswith('|'):
            # Count columns (split by |, remove empty first/last)
            columns = [c.strip() for c in stripped.split('|') if c.strip() or len([x for x in stripped.split('|')]) > 2]
            # Must have at least 2 columns to be a table
            if len(columns) >= 2:
                is_table_row = True
                # Set expected columns on first row
                if expected_columns is None:
                    expected_columns = len(columns)
                # Allow some flexibility in column count (for merged cells representation)
                elif abs(len(columns) - expected_columns) > expected_columns // 2:
                    # Column count differs too much, probably not a table
                    is_table_row = False
        
        if is_table_row:
            if not current_table:
                table_start_line = i
                expected_columns = len([c.strip() for c in stripped.split('|') if c.strip()])
            current_table.append(line)
        elif is_separator:
            if current_table:
                current_table.append(line)
        else:
            # End of potential table
            if current_table and len(current_table) >= 2:
                # Verify it's actually a table:
                # 1. Must have at least one separator line
                # 2. Must have consistent structure
                has_separator = any(re.match(r'^\|[\s\-:]+\|$', l.strip()) for l in current_table)
                # 3. Must have at least 2 data rows (excluding separator)
                data_rows = [l for l in current_table if not re.match(r'^\|[\s\-:]+\|$', l.strip())]
                
                if has_separator and len(data_rows) >= 2:
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
            expected_columns = None
    
    # Handle table at end of document
    if current_table and len(current_table) >= 2:
        has_separator = any(re.match(r'^\|[\s\-:]+\|$', l.strip()) for l in current_table)
        data_rows = [l for l in current_table if not re.match(r'^\|[\s\-:]+\|$', l.strip())]
        
        if has_separator and len(data_rows) >= 2:
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
    
    # Normalize existing metadata first
    raw_metadata = node.metadata or {}
    metadata = normalize_metadata(raw_metadata)
    
    # Add structure information using standard field names
    if code_blocks:
        metadata[MetadataFields.HAS_CODE] = True
        metadata[MetadataFields.CODE_BLOCK_COUNT] = len(code_blocks)
    
    if tables:
        metadata[MetadataFields.HAS_TABLE] = True
        metadata[MetadataFields.TABLE_COUNT] = len(tables)
    
    if headers:
        metadata[MetadataFields.HAS_HEADERS] = True
        metadata[MetadataFields.HEADER_COUNT] = len(headers)
        # Include header hierarchy
        header_titles = [h["title"] for h in headers[:3]]  # First 3 headers
        metadata[MetadataFields.SECTION_HEADERS] = " > ".join(header_titles)
    
    # Detect lists
    list_pattern = r'^[\s]*[-*+]\s+|^[\s]*\d+\.\s+'
    if re.search(list_pattern, content, re.MULTILINE):
        metadata[MetadataFields.HAS_LIST] = True
    
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
                node.metadata[MetadataFields.CODE_LANGUAGE] = language
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


def extract_tables_from_pdf(file_path: str) -> Dict[int, List[Dict[str, Any]]]:
    """
    Extract tables from PDF using pdfplumber.
    Returns a dictionary mapping page numbers to list of tables.
    Each table is a dict with 'content' (markdown format) and 'bbox' (bounding box).
    """
    if not PDFPLUMBER_AVAILABLE:
        return {}
    
    tables_by_page = {}
    
    try:
        # Suppress warnings during PDF processing
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    # Extract tables with warnings suppressed
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        tables = page.extract_tables()
                    if tables:
                        table_list = []
                        for table in tables:
                            # Convert table to markdown format
                            markdown_table = table_to_markdown(table)
                            if markdown_table:
                                table_list.append({
                                    "content": markdown_table,
                                    "bbox": page.bbox,
                                    "type": "pdf_table"
                                })
                        if table_list:
                            tables_by_page[page_num] = table_list
    except Exception as e:
        logger.warning(f"Failed to extract tables from PDF {file_path}: {e}")
    
    return tables_by_page


def table_to_markdown(table: List[List[Any]]) -> str:
    """
    Convert a table (list of lists) to markdown format.
    Handles merged cells by detecting empty cells and preserving structure.
    """
    if not table or len(table) < 1:
        return ""
    
    # Normalize table: handle None values and empty cells
    normalized_table = []
    for row in table:
        normalized_row = [str(cell).strip() if cell is not None else "" for cell in row]
        # Pad rows to same length (handle merged cells)
        if normalized_table:
            max_len = max(len(normalized_table[0]), len(normalized_row))
            # Pad previous rows
            for prev_row in normalized_table:
                while len(prev_row) < max_len:
                    prev_row.append("")
            # Pad current row
            while len(normalized_row) < max_len:
                normalized_row.append("")
        normalized_table.append(normalized_row)
    
    if not normalized_table:
        return ""
    
    # Ensure all rows have same length
    max_cols = max(len(row) for row in normalized_table) if normalized_table else 0
    for row in normalized_table:
        while len(row) < max_cols:
            row.append("")
    
    # Build markdown table
    lines = []
    
    # Header row
    if normalized_table:
        header = '| ' + ' | '.join(normalized_table[0]) + ' |'
        lines.append(header)
        # Separator
        separator = '| ' + ' | '.join(['---'] * len(normalized_table[0])) + ' |'
        lines.append(separator)
        # Data rows
        for row in normalized_table[1:]:
            row_str = '| ' + ' | '.join(row[:len(normalized_table[0])]) + ' |'
            lines.append(row_str)
    
    return '\n'.join(lines)


def extract_tables_with_ocr(file_path: str) -> Dict[int, List[Dict[str, Any]]]:
    """
    Extract tables from PDF using OCR (for scanned PDFs).
    Falls back to regular extraction if OCR is not available.
    """
    if not OCR_AVAILABLE:
        logger.warning("OCR not available, falling back to regular PDF extraction")
        return extract_tables_from_pdf(file_path)
    
    # For now, return empty - OCR table extraction is complex and requires
    # additional libraries like table-transformer or custom ML models
    # This is a placeholder for future enhancement
    logger.info("OCR table extraction not yet implemented, using regular extraction")
    return extract_tables_from_pdf(file_path)


def extract_chart_descriptions_from_pdf(file_path: str) -> Dict[int, str]:
    """
    Extract descriptions of charts/images from PDF using Vision LLM.
    Returns a dictionary mapping page numbers to descriptions.
    """
    if not OCR_AVAILABLE:
        return {}
        
    descriptions = {}
    
    try:
        # Check if we should process this file (heuristics using pdfplumber if available)
        pages_to_process = []
        if PDFPLUMBER_AVAILABLE:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with pdfplumber.open(file_path) as pdf:
                    for i, page in enumerate(pdf.pages):
                        # Simple heuristic: if page has images/figures
                        # Filter out small images (logos, icons)
                        has_significant_images = False
                        if hasattr(page, 'images') and page.images:
                            for img in page.images:
                                # Check dimensions (arbitrary threshold like 100x100)
                                w = float(img.get('width', 0))
                                h = float(img.get('height', 0))
                                if w > 100 and h > 100:
                                    has_significant_images = True
                                    break
                        
                        if has_significant_images:
                            pages_to_process.append(i + 1) # 1-based index
        else:
            # If pdfplumber not available, maybe process all pages? or first few?
            # Safe default: skip to avoid processing everything
            logger.warning("pdfplumber not available, skipping intelligent chart detection")
            return {}
            
        if not pages_to_process:
            return {}
            
        logger.info(f"Detected potential charts on pages {pages_to_process} of {file_path}")
        
        # Convert specific pages to images
        # pdf2image allows converting specific pages
        for page_num in pages_to_process:
            try:
                # Convert single page
                images = convert_from_path(
                    file_path, 
                    first_page=page_num, 
                    last_page=page_num,
                    fmt='jpeg'
                )
                
                if images:
                    img = images[0]
                    # Convert to bytes
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='JPEG')
                    img_bytes = img_byte_arr.getvalue()
                    
                    # Generate description
                    description = generate_image_description(
                        img_bytes,
                        prompt="Describe any charts, graphs, or data visualizations on this page. If there are no charts or significant data visualizations, return 'NO_CHART' only. If there are, provide a detailed description of the data, trends, and labels."
                    )
                    
                    if description and "NO_CHART" not in description:
                        # Format the description
                        formatted_desc = f"\n\n[MÔ TẢ BIỂU ĐỒ/HÌNH ẢNH TRANG {page_num}]:\n{description}\n"
                        descriptions[page_num] = formatted_desc
                        logger.info(f"Generated chart description for page {page_num}")
            except Exception as e:
                logger.warning(f"Error processing page {page_num} for chart description: {e}")
                
    except Exception as e:
        logger.warning(f"Failed to extract chart descriptions: {e}")
        
    return descriptions


def chunk_pdf_document(document: Document) -> List[BaseNode]:
    """
    Chunk PDF documents with special handling for tables and images.
    Now supports:
    - Better table detection using pdfplumber
    - OCR support for scanned PDFs (if available)
    - Merged cells and nested tables handling
    """
    content = document.get_content()
    file_path = get_file_path(document.metadata or {})
    
    # Try to extract tables from PDF if file path is available
    extracted_tables = {}
    extracted_charts = {}
    if file_path and os.path.exists(file_path):
        if PDFPLUMBER_AVAILABLE:
            try:
                extracted_tables = extract_tables_from_pdf(file_path)
                logger.info(f"Extracted {sum(len(t) for t in extracted_tables.values())} tables from PDF")
            except Exception as e:
                logger.warning(f"Failed to extract tables from PDF: {e}")
            
            # Extract chart descriptions
            try:
                extracted_charts = extract_chart_descriptions_from_pdf(file_path)
            except Exception as e:
                logger.warning(f"Failed to extract chart descriptions: {e}")
    
    # Detect if content has tables (common in PDFs)
    has_tables = (
        '|' in content or 
        '\t' in content or
        re.search(r'\s{3,}', content) or  # Multiple spaces (potential table)
        bool(extracted_tables)  # Tables extracted by pdfplumber
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
    features_injected_pages = set() # Track pages where we injected features
    
    for node in nodes:
        node_content = node.get_content()
        
        # Get page number for this node
        page_value = node.metadata.get("page_label") or node.metadata.get("page")
        page_num = None
        if page_value:
            try:
                page_num = int(str(page_value))
            except (ValueError, TypeError):
                pass
        
        # Inject chart descriptions (only once per page)
        if page_num and page_num in extracted_charts:
            if page_num not in features_injected_pages:
                desc = extracted_charts[page_num]
                if not node_content.endswith('\n'):
                    node_content += '\n'
                node_content += desc
                node.set_content(node_content)
                # Mark as injected (we use a separate set for charts to ensure we don't interfere with tables logic, 
                # although reusing features_injected_pages for both might be cleaner if we wanted to group them)
                # But here I'll just use a set for "chart injected" logic implicitly? 
                # Actually, let's just use a specific set for charts if I want to be precise, 
                # but "features_injected_pages" is fine if I define what "features" means.
                # However, tables are injected below based on page_num match.
                
        # If we have extracted tables for this page, enhance content with them
        if page_num and page_num in extracted_tables:
            # Check if content already has tables
            existing_tables = detect_tables(node_content)
            if not existing_tables:
                # No tables detected in text, try to add extracted tables
                # Append extracted tables to content
                extracted_table_texts = [t["content"] for t in extracted_tables[page_num]]
                if extracted_table_texts:
                    enhanced_content = node_content
                    if not enhanced_content.endswith('\n'):
                        enhanced_content += '\n'
                    enhanced_content += '\n\n'.join(extracted_table_texts)
                    node_content = enhanced_content
                    node.set_content(enhanced_content)
        
        # Mark page as having features injected (charts handled above)
        if page_num and page_num in extracted_charts:
             features_injected_pages.add(page_num)

        # Try to detect and format tables in text
        formatted_content = format_tables_in_text(node_content)
        if formatted_content != node_content:
            node.set_content(formatted_content)
            node.metadata = node.metadata or {}
            node.metadata["has_table"] = True
        
        # Preserve other structure
        node = preserve_structure_in_metadata(node, document)
        
        # Add PDF-specific metadata
        node.metadata = node.metadata or {}
        node.metadata[MetadataFields.DOCUMENT_TYPE] = "pdf"
        # Normalize page field
        if page_value:
            node.metadata[MetadataFields.PAGE] = str(page_value)
        
        processed_nodes.append(node)
    
    logger.info(f"Chunked PDF document into {len(processed_nodes)} nodes")
    return processed_nodes


def format_tables_in_text(text: str) -> str:
    """
    Detect and format tables in text to markdown format.
    Improved to handle merged cells and nested tables better.
    """
    lines = text.split('\n')
    result_lines = []
    current_table = []
    in_table = False
    max_columns = 0
    
    for i, line in enumerate(lines):
        # Skip if line is already a markdown table row or separator
        if re.match(r'^\|[\s\S]*\|$', line.strip()) or re.match(r'^\|[\s\-:]+\|$', line.strip()):
            result_lines.append(line)
            continue
        
        # Detect table-like patterns
        # Pattern 1: Multiple columns separated by tabs
        parts_tab = line.split('\t')
        # Pattern 2: Multiple columns separated by multiple spaces (2+)
        parts_space = re.split(r'\s{2,}', line.strip())
        
        is_table_row = False
        columns = []
        
        # Check tab-separated (more reliable for tables)
        if len(parts_tab) >= 2:
            # Filter out empty parts but keep structure
            columns_tab = [p.strip() for p in parts_tab]
            # Must have at least 2 non-empty columns
            non_empty = [c for c in columns_tab if c]
            if len(non_empty) >= 2:
                is_table_row = True
                columns = columns_tab
                max_columns = max(max_columns, len(columns))
        
        # Check space-separated (less reliable, need more validation)
        if not is_table_row and len(parts_space) >= 2:
            columns_space = [p.strip() for p in parts_space if p.strip()]
            # More strict: need at least 3 columns and reasonable length
            if len(columns_space) >= 3 and all(len(c) > 0 and len(c) < 100 for c in columns_space):
                # Check if it's not just regular text with multiple spaces
                # Look for consistent column-like structure
                avg_len = sum(len(c) for c in columns_space) / len(columns_space)
                if avg_len < 50:  # Reasonable column width
                    is_table_row = True
                    columns = columns_space
                    max_columns = max(max_columns, len(columns))
        
        if is_table_row:
            if not in_table:
                in_table = True
                current_table = []
                max_columns = len(columns)
            
            # Normalize column count (handle merged cells by padding)
            while len(columns) < max_columns:
                columns.append("")
            # Update max if this row has more columns
            if len(columns) > max_columns:
                # Pad all previous rows
                for prev_row in current_table:
                    while len(prev_row) < len(columns):
                        prev_row.append("")
                max_columns = len(columns)
            
            current_table.append(columns)
        else:
            if in_table and len(current_table) >= 2:
                # Format as markdown table
                # Normalize all rows to same column count
                for row in current_table:
                    while len(row) < max_columns:
                        row.append("")
                
                # Header row (first row)
                header = '| ' + ' | '.join(current_table[0][:max_columns]) + ' |'
                result_lines.append(header)
                # Separator
                separator = '| ' + ' | '.join(['---'] * max_columns) + ' |'
                result_lines.append(separator)
                # Data rows
                for row in current_table[1:]:
                    # Handle merged cells: empty cells are represented as empty strings
                    row_str = '| ' + ' | '.join(row[:max_columns]) + ' |'
                    result_lines.append(row_str)
                
                current_table = []
                in_table = False
                max_columns = 0
            
            result_lines.append(line)
    
    # Handle table at end
    if in_table and len(current_table) >= 2:
        # Normalize all rows
        for row in current_table:
            while len(row) < max_columns:
                row.append("")
        
        header = '| ' + ' | '.join(current_table[0][:max_columns]) + ' |'
        result_lines.append(header)
        separator = '| ' + ' | '.join(['---'] * max_columns) + ' |'
        result_lines.append(separator)
        for row in current_table[1:]:
            row_str = '| ' + ' | '.join(row[:max_columns]) + ' |'
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
            for idx, node in enumerate(nodes):
                raw_metadata = node.metadata or {}
                # Normalize metadata
                normalized_metadata = normalize_metadata(raw_metadata)
                normalized_metadata[MetadataFields.DOCUMENT_TYPE] = doc_type
                # Preserve original document metadata (will be normalized)
                if doc.metadata:
                    doc_metadata_normalized = normalize_metadata(doc.metadata)
                    for key, value in doc_metadata_normalized.items():
                        if key not in normalized_metadata:
                            normalized_metadata[key] = value
                node.metadata = normalized_metadata
            
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

