"""Configuration management for RAG-Guardian."""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def safe_bool_conversion(value, default="False"):
    """Safely convert value to boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes", "on")
    return str(default).lower() in ("true", "1", "yes", "on")

def safe_int_conversion(value, default="0"):
    """Safely convert value to integer."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return int(default)

def get_config_value(key: str, default: str = "", convert_type=None):
    """Get configuration value from Streamlit secrets or environment."""
    try:
        import streamlit as st
        try:
            value = st.secrets.get(key)
            if value is not None:
                if convert_type == bool:
                    return safe_bool_conversion(value, default)
                elif convert_type == int:
                    return safe_int_conversion(value, default)
                return str(value)
        except:
            pass
    except ImportError:
        pass
    
    # Fallback to environment variables
    env_value = os.getenv(key, default)
    if convert_type == bool:
        return safe_bool_conversion(env_value, default)
    elif convert_type == int:
        return safe_int_conversion(env_value, default)
    return env_value

class Config:
    """Application configuration."""
    
    # API Keys
    GROQ_API_KEY: str = get_config_value("GROQ_API_KEY", "")
    OPENAI_API_KEY: str = get_config_value("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = get_config_value("ANTHROPIC_API_KEY", "")
    
    # Vector Database
    CHROMA_PERSIST_DIRECTORY: str = get_config_value("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db")
    
    # Application Settings
    DEBUG: bool = get_config_value("DEBUG", "False", bool)
    LOG_LEVEL: str = get_config_value("LOG_LEVEL", "INFO")
    
    # Model Settings
    DEFAULT_MODEL: str = get_config_value("DEFAULT_MODEL", "llama-3.1-8b-instant")
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    USE_GROQ: bool = get_config_value("USE_GROQ", "True", bool)
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
    ENABLE_REACT_DOCS: bool = get_config_value("ENABLE_REACT_DOCS", "True", bool)
    ENABLE_NEXTJS_DOCS: bool = get_config_value("ENABLE_NEXTJS_DOCS", "True", bool)
    ENABLE_LOCAL_DOCS: bool = get_config_value("ENABLE_LOCAL_DOCS", "True", bool)
    ENABLE_SAMPLE_DOCS: bool = get_config_value("ENABLE_SAMPLE_DOCS", "True", bool)
    USE_DOCUMENT_CACHE: bool = get_config_value("USE_DOCUMENT_CACHE", "True", bool)
    CACHE_EXPIRY_HOURS: int = get_config_value("CACHE_EXPIRY_HOURS", "24", int)
    DOCUMENT_SOURCE_MODE: str = get_config_value("DOCUMENT_SOURCE_MODE", "auto")
    CACHE_DIR: str = get_config_value("CACHE_DIR", "./data/cache")
    
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
