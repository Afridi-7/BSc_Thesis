"""
Clinical knowledge retriever using ChromaDB or numpy fallback.

Provides semantic search over hematology knowledge base with embeddings.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

from src.config.config_loader import Config
from src.rag.pdf_processor import process_pdf_library, validate_pdf_library

logger = logging.getLogger(__name__)


class ClinicalRetriever:
    """Retrieval system for hematology clinical knowledge."""
    
    def __init__(self, config: Config):
        """
        Initialize clinical retriever.
        
        Args:
            config: Configuration object with RAG parameters
        """
        self.config = config
        
        # Load configuration
        self.pdf_directory = config.get('rag.pdf_directory', 'LLM_RAG_Pipline/pdfs')
        self.pdf_sources = config.get('rag.pdf_sources', [])
        self.pdf_missing_strategy = config.get('rag.pdf_missing_strategy', 'fail')
        self.pdf_download_base_url = config.get('rag.pdf_download_base_url', '')
        self.pdf_download_timeout_seconds = config.get('rag.pdf_download_timeout_seconds', 30)
        self.chunk_size = config.get('rag.chunking.chunk_size', 500)
        self.overlap = config.get('rag.chunking.overlap', 50)
        self.top_k = config.get('rag.retrieval.top_k', 5)
        self.min_chunk_length = config.get('rag.retrieval.min_chunk_length', 80)
        
        # Embedding configuration
        embedding_model_name = config.get('rag.embedding.model_name', 'sentence-transformers/all-MiniLM-L6-v2')
        self.embedding_dim = config.get('rag.embedding.embedding_dim', 384)
        self.batch_size = config.get('rag.embedding.batch_size', 64)
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model_name}")
        self.embedder = SentenceTransformer(embedding_model_name)
        
        # Vector store configuration
        self.vector_store_type = config.get('rag.vector_store.type', 'chromadb')
        self.chromadb_path = config.get('rag.vector_store.chromadb_path', 'LLM_RAG_Pipline/chroma_db')
        self.collection_name = config.get('rag.vector_store.collection_name', 'hematology_knowledge')
        
        # Storage
        self.chunks = []
        self.embeddings = None
        self.chroma_client = None
        self.chroma_collection = None
        
        logger.info("ClinicalRetriever initialized")
    
    def build_index(self) -> Dict[str, Any]:
        """
        Build retrieval index from PDF library.
        
        Returns:
            Dictionary with build statistics:
                {
                    'total_chunks': int,
                    'pdfs_processed': int,
                    'index_type': str,
                    'embedding_dim': int
                }
                
        Examples:
            >>> retriever = ClinicalRetriever(config)
            >>> stats = retriever.build_index()
            >>> print(f"Indexed {stats['total_chunks']} chunks")
        """
        logger.info("Building retrieval index...")
        
        # Validate PDF library
        validation = validate_pdf_library(self.pdf_directory, self.pdf_sources)
        if not validation['valid']:
            logger.warning(f"Missing PDFs: {validation['missing_files']}")
            self._handle_missing_pdfs(validation)
        
        # Process PDFs into chunks
        self.chunks = process_pdf_library(
            self.pdf_directory,
            self.pdf_sources,
            chunk_size=self.chunk_size,
            overlap=self.overlap
        )
        
        if not self.chunks:
            raise ValueError("No chunks extracted from PDF library")
        
        logger.info(f"Extracted {len(self.chunks)} chunks from PDFs")
        
        # Create embeddings
        chunk_texts = [chunk['text'] for chunk in self.chunks]
        logger.info(f"Encoding {len(chunk_texts)} chunks...")
        
        self.embeddings = self.embedder.encode(
            chunk_texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        logger.info(f"Created embeddings: {self.embeddings.shape}")
        
        # Initialize vector store
        if self.vector_store_type == 'chromadb':
            try:
                self._init_chromadb()
                index_type = 'chromadb'
            except Exception as e:
                logger.warning(f"ChromaDB initialization failed: {e}")
                logger.info("Falling back to numpy-based retrieval")
                index_type = 'numpy_fallback'
        else:
            index_type = 'numpy_fallback'
        
        return {
            'total_chunks': len(self.chunks),
            'pdfs_processed': len(set(c['source'] for c in self.chunks)),
            'index_type': index_type,
            'embedding_dim': self.embedding_dim
        }

    def _handle_missing_pdfs(self, validation: Dict[str, Any]) -> None:
        """Apply configured strategy when required PDFs are missing."""
        strategy = str(self.pdf_missing_strategy).lower()
        missing = validation.get('missing_files', [])
        if not missing:
            return

        if strategy == 'download':
            self._download_missing_pdfs(missing)
            post_check = validate_pdf_library(self.pdf_directory, self.pdf_sources)
            if not post_check['valid']:
                raise FileNotFoundError(
                    f"PDF auto-download incomplete. Missing files: {post_check['missing_files']}. "
                    f"Place files under '{self.pdf_directory}' or set rag.pdf_missing_strategy to 'warn'."
                )
            return

        if strategy == 'warn':
            logger.warning(
                "Continuing with partial PDF library. Missing files: %s",
                missing
            )
            return

        raise FileNotFoundError(
            f"Missing required PDF files: {missing}. "
            f"Expected directory: '{self.pdf_directory}'. "
            "Either add the files, set rag.pdf_missing_strategy to 'warn', "
            "or configure rag.pdf_missing_strategy='download' with rag.pdf_download_base_url."
        )

    def _download_missing_pdfs(self, missing_files: List[str]) -> None:
        """Download missing PDF files from configured base URL."""
        if not self.pdf_download_base_url:
            raise ValueError(
                "rag.pdf_missing_strategy is 'download' but rag.pdf_download_base_url is empty"
            )

        target_dir = Path(self.pdf_directory)
        target_dir.mkdir(parents=True, exist_ok=True)

        base = self.pdf_download_base_url.rstrip('/')
        for filename in missing_files:
            url = f"{base}/{filename}"
            target_path = target_dir / filename
            logger.info(f"Downloading missing PDF: {url}")
            try:
                response = requests.get(url, timeout=self.pdf_download_timeout_seconds)
                response.raise_for_status()
                target_path.write_bytes(response.content)
                logger.info(f"Downloaded PDF to: {target_path}")
            except Exception as exc:
                logger.error(f"Failed to download {filename}: {exc}")
    
    def _init_chromadb(self) -> None:
        """Initialize ChromaDB vector store."""
        import chromadb
        from chromadb.config import Settings
        
        # Create persistent client
        self.chroma_client = chromadb.PersistentClient(
            path=str(Path(self.chromadb_path)),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Delete existing collection if it exists
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted existing collection: {self.collection_name}")
        except Exception as e:
            logger.debug(f"Collection did not exist or could not be deleted: {e}")
        
        # Create collection
        self.chroma_collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Add documents
        ids = [f"chunk_{i}" for i in range(len(self.chunks))]
        documents = [chunk['text'] for chunk in self.chunks]
        metadatas = [
            {
                'source': chunk['source'],
                'chunk_id': chunk['chunk_id'],
                'total_chunks': chunk['total_chunks']
            }
            for chunk in self.chunks
        ]
        
        # Add in batches to avoid memory issues
        batch_size = 1000
        for i in range(0, len(ids), batch_size):
            batch_end = min(i + batch_size, len(ids))
            self.chroma_collection.add(
                ids=ids[i:batch_end],
                embeddings=self.embeddings[i:batch_end].tolist(),
                documents=documents[i:batch_end],
                metadatas=metadatas[i:batch_end]
            )
        
        logger.info(f"Added {len(ids)} chunks to ChromaDB collection")
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for query.
        
        Args:
            query: Search query
            top_k: Number of chunks to retrieve (None to use config default)
            
        Returns:
            List of retrieved chunks with metadata and scores:
                [
                    {
                        'text': str,
                        'source': str,
                        'score': float,
                        'chunk_id': int
                    },
                    ...
                ]
                
        Examples:
            >>> results = retriever.retrieve("What causes leukocytosis?")
            >>> for result in results:
            ...     print(f"[{result['source']}]: {result['text'][:100]}...")
        """
        k = top_k if top_k is not None else self.top_k
        
        logger.info(f"Retrieving top-{k} chunks for query: {query[:50]}...")
        
        # Encode query
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)[0]
        
        # Retrieve using available backend
        if self.chroma_collection is not None:
            return self._retrieve_chromadb(query_embedding, k)
        else:
            return self._retrieve_numpy(query_embedding, k)
    
    def _retrieve_chromadb(self, query_embedding: np.ndarray, k: int) -> List[Dict[str, Any]]:
        """Retrieve using ChromaDB."""
        results = self.chroma_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k
        )
        
        # Format results
        retrieved = []
        for i in range(len(results['documents'][0])):
            retrieved.append({
                'text': results['documents'][0][i],
                'source': results['metadatas'][0][i]['source'],
                'score': float(1.0 - results['distances'][0][i]),  # Convert distance to similarity
                'chunk_id': results['metadatas'][0][i]['chunk_id']
            })
        
        logger.info(f"Retrieved {len(retrieved)} chunks using ChromaDB")
        return retrieved
    
    def _retrieve_numpy(self, query_embedding: np.ndarray, k: int) -> List[Dict[str, Any]]:
        """Retrieve using numpy cosine similarity."""
        # Compute cosine similarities with numerical stability
        epsilon = 1e-8
        similarities = np.dot(self.embeddings, query_embedding)
        
        # Normalize with epsilon to prevent division by zero
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding) + epsilon
        similarities = similarities / norms
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        # Format results
        retrieved = []
        for idx in top_indices:
            retrieved.append({
                'text': self.chunks[idx]['text'],
                'source': self.chunks[idx]['source'],
                'score': float(similarities[idx]),
                'chunk_id': self.chunks[idx]['chunk_id']
            })
        
        logger.info(f"Retrieved {len(retrieved)} chunks using numpy fallback")
        return retrieved
    
    def format_retrieved_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks into context string with citations.
        
        Args:
            retrieved_chunks: List of retrieved chunk dictionaries
            
        Returns:
            Formatted context string with numbered citations
            
        Examples:
            >>> context = retriever.format_retrieved_context(results)
            >>> print(context)
            [Reference 1 — essentials_haematology.pdf]
            Leukocytosis is an increase in white blood cell count...
            
            [Reference 2 — concise_haematology.pdf]
            Common causes include infection, inflammation...
        """
        if not retrieved_chunks:
            return ""
        
        formatted_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            citation = f"[Reference {i} — {chunk['source']}]"
            formatted_parts.append(f"{citation}\n{chunk['text']}")
        
        return "\n\n".join(formatted_parts)
    
    def check_retrieval_quality(self, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Check quality of retrieved results.
        
        Args:
            retrieved_chunks: List of retrieved chunks
            
        Returns:
            Dictionary with quality metrics:
                {
                    'sufficient_evidence': bool,
                    'total_length': int,
                    'num_chunks': int,
                    'avg_score': float
                }
        """
        if not retrieved_chunks:
            return {
                'sufficient_evidence': False,
                'total_length': 0,
                'num_chunks': 0,
                'avg_score': 0.0
            }
        
        total_text = ''.join(c['text'] for c in retrieved_chunks)
        avg_score = np.mean([c['score'] for c in retrieved_chunks])
        
        return {
            'sufficient_evidence': len(total_text) >= self.min_chunk_length,
            'total_length': len(total_text),
            'num_chunks': len(retrieved_chunks),
            'avg_score': float(avg_score)
        }
