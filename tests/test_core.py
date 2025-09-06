"""Tests for RAG-Guardian core functionality."""

import pytest
import os
import sys
from unittest.mock import Mock, patch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import EmbeddingManager, DocumentRetriever, ResponseGenerator, ResponseCritic
from data import DocumentLoader
from utils import config

class TestEmbeddingManager:
    """Test the embedding manager."""
    
    def test_init_without_openai(self):
        """Test initialization without OpenAI key."""
        with patch.object(config, 'OPENAI_API_KEY', ''):
            manager = EmbeddingManager(use_openai=False)
            assert manager.use_openai is False
    
    def test_init_with_openai(self):
        """Test initialization with OpenAI key."""
        with patch.object(config, 'OPENAI_API_KEY', 'test-key'):
            manager = EmbeddingManager(use_openai=True)
            assert manager.use_openai is True

class TestDocumentLoader:
    """Test the document loader."""
    
    def test_load_sample_docs(self):
        """Test loading sample documents."""
        loader = DocumentLoader()
        docs = loader.load_sample_docs()
        
        assert len(docs) > 0
        assert all(hasattr(doc, 'page_content') for doc in docs)
        assert all(hasattr(doc, 'metadata') for doc in docs)
    
    def test_get_document_stats(self):
        """Test document statistics."""
        loader = DocumentLoader()
        docs = loader.load_sample_docs()
        stats = loader.get_document_stats(docs)
        
        assert 'count' in stats
        assert 'total_chars' in stats
        assert 'avg_chars' in stats
        assert 'sources' in stats
        assert stats['count'] == len(docs)

class TestResponseCritic:
    """Test the response critic."""
    
    @patch('core.critic.ChatOpenAI')
    def test_critique_response(self, mock_llm):
        """Test response critique."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = "APPROVED"
        mock_llm.return_value.invoke.return_value = mock_response
        
        critic = ResponseCritic()
        result = critic.critique_response(
            "Test query",
            "Test response", 
            ["Test source"]
        )
        
        assert 'is_approved' in result
        assert 'confidence_score' in result
        assert 'flaws' in result
        assert 'suggestions' in result

def test_config_validation():
    """Test configuration validation."""
    # Should pass with valid API key
    with patch.object(config, 'OPENAI_API_KEY', 'test-key'):
        assert config.validate() is True
    
    # Should fail without API key
    with patch.object(config, 'OPENAI_API_KEY', ''):
        with pytest.raises(ValueError):
            config.validate()

if __name__ == "__main__":
    pytest.main([__file__])
