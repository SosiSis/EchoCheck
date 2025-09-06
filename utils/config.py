"""Configuration management for RAG-Guardian."""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration."""
    
    # API Keys
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    
    # Vector Database
    CHROMA_PERSIST_DIRECTORY: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db")
    
    # Application Settings
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Model Settings
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "llama-3.1-8b-instant")  # Updated Groq model
    EMBEDDING_MODEL: str = "text-embedding-3-small"  # Keep OpenAI for embeddings
    USE_GROQ: bool = os.getenv("USE_GROQ", "True").lower() == "true"
    TEMPERATURE: float = 0.1
    MAX_TOKENS: int = 2000
    
    # Retrieval Settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RETRIEVAL: int = 5
    
    # Reflection Settings
    MAX_REFLECTION_CYCLES: int = 2
    CONFIDENCE_THRESHOLD: float = 0.7
    
    # Document Loading Settings
    ENABLE_REACT_DOCS: bool = os.getenv("ENABLE_REACT_DOCS", "True").lower() == "true"
    ENABLE_NEXTJS_DOCS: bool = os.getenv("ENABLE_NEXTJS_DOCS", "True").lower() == "true"
    ENABLE_LOCAL_DOCS: bool = os.getenv("ENABLE_LOCAL_DOCS", "True").lower() == "true"
    ENABLE_SAMPLE_DOCS: bool = os.getenv("ENABLE_SAMPLE_DOCS", "True").lower() == "true"
    USE_DOCUMENT_CACHE: bool = os.getenv("USE_DOCUMENT_CACHE", "True").lower() == "true"
    CACHE_EXPIRY_HOURS: int = int(os.getenv("CACHE_EXPIRY_HOURS", "24"))
    DOCUMENT_SOURCE_MODE: str = os.getenv("DOCUMENT_SOURCE_MODE", "auto")  # auto, local, remote, sample
    CACHE_DIR: str = os.getenv("CACHE_DIR", "./data/cache")
    
    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration."""
        if cls.USE_GROQ and not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is required when USE_GROQ is True")
        elif not cls.USE_GROQ and not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required when USE_GROQ is False")
        return True

# Create global config instance
config = Config()
