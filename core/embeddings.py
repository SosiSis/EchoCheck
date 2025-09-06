"""Embedding utilities for RAG-Guardian."""

import logging
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from utils.config import config

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Manages embeddings for the RAG system."""
    
    def __init__(self, use_openai: bool = True):
        """Initialize embedding manager.
        
        Args:
            use_openai: Whether to use OpenAI embeddings (requires API key)
        """
        self.use_openai = use_openai
        self._embeddings = None
        
    @property
    def embeddings(self):
        """Get embeddings instance."""
        if self._embeddings is None:
            self._embeddings = self._create_embeddings()
        return self._embeddings
    
    def _create_embeddings(self):
        """Create embeddings instance."""
        if self.use_openai and config.OPENAI_API_KEY:
            logger.info("Using OpenAI embeddings")
            return OpenAIEmbeddings(
                model=config.EMBEDDING_MODEL,
                openai_api_key=config.OPENAI_API_KEY
            )
        else:
            logger.info("Using local HuggingFace embeddings")
            return HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise
