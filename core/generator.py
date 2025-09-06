"""Response generation for RAG-Guardian."""

import logging
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.schema import Document, HumanMessage, SystemMessage
from utils.config import config
from utils.helpers import clean_code_snippet

logger = logging.getLogger(__name__)

class ResponseGenerator:
    """Generates responses using retrieved context."""
    
    def __init__(self):
        """Initialize the response generator."""
        if config.USE_GROQ:
            self.llm = ChatGroq(
                model=config.DEFAULT_MODEL,
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_TOKENS,
                groq_api_key=config.GROQ_API_KEY
            )
        else:
            self.llm = ChatOpenAI(
                model=config.DEFAULT_MODEL,
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_TOKENS,
                openai_api_key=config.OPENAI_API_KEY
            )
    
    def generate_initial_response(
        self, 
        query: str, 
        context_docs: List[Document]
    ) -> Dict[str, Any]:
        """Generate initial response using retrieved context.
        
        Args:
            query: User query
            context_docs: Retrieved documents
            
        Returns:
            Dictionary containing response and metadata
        """
        try:
            # Format context
            context = self._format_context(context_docs)
            
            # Create prompt
            system_prompt = self._create_system_prompt()
            user_prompt = self._create_user_prompt(query, context)
            
            # Generate response
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            # Clean and format response
            content = clean_code_snippet(response.content)
            
            result = {
                "content": content,
                "context_used": len(context_docs),
                "sources": [doc.metadata.get("source", "Unknown") for doc in context_docs],
                "raw_response": response.content
            }
            
            logger.info(f"Generated initial response for query: {query[:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def generate_improved_response(
        self,
        original_query: str,
        original_response: str,
        critique_feedback: str,
        new_context_docs: List[Document]
    ) -> Dict[str, Any]:
        """Generate improved response based on critique.
        
        Args:
            original_query: Original user query
            original_response: Previous response that was critiqued
            critique_feedback: Feedback from the critic
            new_context_docs: New context documents
            
        Returns:
            Dictionary containing improved response and metadata
        """
        try:
            # Format new context
            context = self._format_context(new_context_docs)
            
            # Create improvement prompt
            system_prompt = self._create_improvement_system_prompt()
            user_prompt = self._create_improvement_user_prompt(
                original_query, original_response, critique_feedback, context
            )
            
            # Generate improved response
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            # Clean and format response
            content = clean_code_snippet(response.content)
            
            result = {
                "content": content,
                "context_used": len(new_context_docs),
                "sources": [doc.metadata.get("source", "Unknown") for doc in new_context_docs],
                "raw_response": response.content,
                "improved_from_critique": True
            }
            
            logger.info("Generated improved response based on critique")
            return result
            
        except Exception as e:
            logger.error(f"Error generating improved response: {e}")
            raise
    
    def _format_context(self, documents: List[Document]) -> str:
        """Format documents into context string."""
        if not documents:
            return "No relevant context found."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            content = doc.page_content.strip()
            context_parts.append(f"[Source {i}: {source}]\n{content}")
        
        return "\n\n".join(context_parts)
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for initial response generation."""
        return """You are an expert coding assistant specializing in modern web development frameworks and best practices. 

Your role is to provide accurate, up-to-date, and practical coding advice based on the most recent documentation and examples.

Guidelines:
- Always use the most recent stable APIs and patterns
- Provide complete, runnable code examples when possible
- Explain the reasoning behind your recommendations
- Highlight any important gotchas or common mistakes
- Focus on modern, production-ready solutions
- If you're uncertain about something, say so explicitly

Format your response clearly with proper code blocks and explanations."""
    
    def _create_user_prompt(self, query: str, context: str) -> str:
        """Create user prompt with query and context."""
        return f"""Based on the following documentation and context, please answer the user's question:

CONTEXT:
{context}

USER QUESTION:
{query}

Please provide a comprehensive answer that uses the context provided above. Include relevant code examples and explanations."""
    
    def _create_improvement_system_prompt(self) -> str:
        """Create system prompt for improved response generation."""
        return """You are an expert coding assistant tasked with improving a previous response based on critique feedback.

Your role is to:
- Address all the specific flaws mentioned in the critique
- Use the new context to provide more accurate information
- Maintain the helpful tone while being more precise
- Provide better, more current examples if needed

Focus on fixing the specific issues raised rather than completely rewriting the response."""
    
    def _create_improvement_user_prompt(
        self, 
        original_query: str, 
        original_response: str, 
        critique: str, 
        new_context: str
    ) -> str:
        """Create prompt for response improvement."""
        return f"""Please improve the following response based on the critique provided:

ORIGINAL QUERY:
{original_query}

PREVIOUS RESPONSE:
{original_response}

CRITIQUE FEEDBACK:
{critique}

NEW CONTEXT:
{new_context}

Please provide an improved response that addresses the critique while maintaining helpfulness and clarity."""
