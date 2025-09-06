"""Configuration management for RAG-Guardian."""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Handle Streamlit secrets
try:
    import streamlit as st
    def get_secret(key: str, default: str = "") -> str:
        """Get secret from Streamlit secrets or environment."""
        try:
            return st.secrets.get(key, os.getenv(key, default))
        except:
            return os.getenv(key, default)
except ImportError:
    def get_secret(key: str, default: str = "") -> str:
        """Get secret from environment."""
        return os.getenv(key, default)

class Config:
    """Application configuration."""
    
    # API Keys
    GROQ_API_KEY: str = get_secret("GROQ_API_KEY", "")
    OPENAI_API_KEY: str = get_secret("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = get_secret("ANTHROPIC_API_KEY", "")
    
    # Vector Database
    CHROMA_PERSIST_DIRECTORY: str = get_secret("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db")
    
    # Application Settings
    DEBUG: bool = get_secret("DEBUG", "False").lower() == "true"
    LOG_LEVEL: str = get_secret("LOG_LEVEL", "INFO")
    
    # Model Settings
    DEFAULT_MODEL: str = get_secret("DEFAULT_MODEL", "llama-3.1-8b-instant")  # Updated Groq model
    EMBEDDING_MODEL: str = "text-embedding-3-small"  # Keep OpenAI for embeddings
    USE_GROQ: bool = get_secret("USE_GROQ", "True").lower() == "true"
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
    ENABLE_REACT_DOCS: bool = get_secret("ENABLE_REACT_DOCS", "True").lower() == "true"
    ENABLE_NEXTJS_DOCS: bool = get_secret("ENABLE_NEXTJS_DOCS", "True").lower() == "true"
    ENABLE_LOCAL_DOCS: bool = get_secret("ENABLE_LOCAL_DOCS", "True").lower() == "true"
    ENABLE_SAMPLE_DOCS: bool = get_secret("ENABLE_SAMPLE_DOCS", "True").lower() == "true"
    USE_DOCUMENT_CACHE: bool = get_secret("USE_DOCUMENT_CACHE", "True").lower() == "true"
    CACHE_EXPIRY_HOURS: int = int(get_secret("CACHE_EXPIRY_HOURS", "24"))
    DOCUMENT_SOURCE_MODE: str = get_secret("DOCUMENT_SOURCE_MODE", "auto")  # auto, local, remote, sample
    CACHE_DIR: str = get_secret("CACHE_DIR", "./data/cache")
    
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
