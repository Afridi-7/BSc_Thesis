"""
PDF processing utilities for extracting and chunking medical documents.

Provides text extraction from hematology PDFs and chunking for retrieval.
"""

import logging
import re
from pathlib import Path
from typing import List, Dict, Any
from pypdf import PdfReader

logger = logging.getLogger(__name__)


def extract_pdf_text(pdf_path: str) -> str:
    """
    Extract text content from PDF file.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Extracted text as string (empty string on failure)
        
    Examples:
        >>> text = extract_pdf_text('essentials_haematology.pdf')
        >>> print(f"Extracted {len(text)} characters")
    """
    try:
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            logger.error(f"PDF not found: {pdf_path}")
            return ""
        
        logger.info(f"Extracting text from: {pdf_path.name}")
        
        reader = PdfReader(str(pdf_path))
        text_pages = []
        
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text_pages.append(page_text)
            except Exception as e:
                logger.warning(f"Failed to extract page {page_num} from {pdf_path.name}: {e}")
        
        full_text = "\n".join(text_pages)
        logger.info(f"Extracted {len(full_text)} characters from {len(text_pages)} pages")
        
        return full_text
        
    except Exception as e:
        logger.error(f"Failed to extract text from {pdf_path}: {e}")
        return ""


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    min_chunk_length: int = 100
) -> List[str]:
    """
    Split text into overlapping word-based chunks.
    
    Args:
        text: Raw text to chunk
        chunk_size: Chunk size in words
        overlap: Overlap in words between chunks
        min_chunk_length: Minimum character length for valid chunk
        
    Returns:
        List of text chunks
        
    Examples:
        >>> text = "Long medical document..."
        >>> chunks = chunk_text(text, chunk_size=500, overlap=50)
        >>> print(f"Created {len(chunks)} chunks")
    """
    if not text:
        return []
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Split into words
    words = text.split()
    
    if len(words) <= chunk_size:
        # Text is shorter than chunk size
        return [text] if len(text) >= min_chunk_length else []
    
    chunks = []
    start = 0
    
    while start < len(words):
        # Extract chunk
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text = ' '.join(chunk_words)
        
        # Only add if meets minimum length
        if len(chunk_text) >= min_chunk_length:
            chunks.append(chunk_text)
        
        # Move window with overlap
        start += chunk_size - overlap
        
        # Prevent infinite loop
        if start >= len(words):
            break
    
    logger.info(f"Created {len(chunks)} chunks from {len(words)} words")
    return chunks


def process_pdf_library(
    pdf_directory: str,
    pdf_filenames: List[str],
    chunk_size: int = 500,
    overlap: int = 50
) -> List[Dict[str, Any]]:
    """
    Process multiple PDFs into chunks with metadata.
    
    Args:
        pdf_directory: Directory containing PDF files
        pdf_filenames: List of PDF filenames to process
        chunk_size: Chunk size in words
        overlap: Overlap in words between chunks
        
    Returns:
        List of chunk dictionaries:
            [
                {
                    'text': str,
                    'source': str,
                    'chunk_id': int,
                    'total_chunks': int
                },
                ...
            ]
            
    Examples:
        >>> chunks = process_pdf_library(
        ...     'pdfs/',
        ...     ['essentials_haematology.pdf', 'concise_haematology.pdf']
        ... )
        >>> print(f"Total chunks: {len(chunks)}")
    """
    pdf_dir = Path(pdf_directory)
    all_chunks = []
    
    for pdf_filename in pdf_filenames:
        pdf_path = pdf_dir / pdf_filename
        
        if not pdf_path.exists():
            logger.warning(f"PDF not found, skipping: {pdf_path}")
            continue
        
        # Extract text
        text = extract_pdf_text(str(pdf_path))
        
        if not text:
            logger.warning(f"No text extracted from {pdf_filename}")
            continue
        
        # Create chunks
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        
        # Add metadata
        for chunk_idx, chunk_text in enumerate(chunks):
            all_chunks.append({
                'text': chunk_text,
                'source': pdf_filename,
                'chunk_id': chunk_idx,
                'total_chunks': len(chunks)
            })
    
    logger.info(f"Processed {len(pdf_filenames)} PDFs into {len(all_chunks)} total chunks")
    return all_chunks


def validate_pdf_library(pdf_directory: str, pdf_filenames: List[str]) -> Dict[str, Any]:
    """
    Validate PDF library and return status.
    
    Args:
        pdf_directory: Directory containing PDFs
        pdf_filenames: Expected PDF filenames
        
    Returns:
        Dictionary with validation results:
            {
                'total_files': int,
                'found_files': int,
                'missing_files': List[str],
                'valid': bool
            }
    """
    pdf_dir = Path(pdf_directory)
    
    if not pdf_dir.exists():
        return {
            'total_files': len(pdf_filenames),
            'found_files': 0,
            'missing_files': pdf_filenames,
            'valid': False,
            'error': f"Directory not found: {pdf_directory}"
        }
    
    missing = []
    found = 0
    
    for filename in pdf_filenames:
        pdf_path = pdf_dir / filename
        if pdf_path.exists():
            found += 1
        else:
            missing.append(filename)
    
    return {
        'total_files': len(pdf_filenames),
        'found_files': found,
        'missing_files': missing,
        'valid': len(missing) == 0
    }
