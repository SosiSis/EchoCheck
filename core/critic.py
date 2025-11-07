"""Self-critique system for RAG-Guardian."""

import logging
from typing import Dict, Any, List

# Guard LangChain imports
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None

try:
    from langchain.schema import HumanMessage, SystemMessage
except ImportError:
    # Minimal fallbacks
    from dataclasses import dataclass
    @dataclass
    class HumanMessage:
        content: str
    
    @dataclass
    class SystemMessage:
        content: str

from utils.config import config
from utils.helpers import parse_critique_response

logger = logging.getLogger(__name__)

class ResponseCritic:
    """Provides self-critique functionality for generated responses."""
    
    def __init__(self):
        """Initialize the critic."""
        if config.USE_GROQ:
            self.llm = ChatGroq(
                model=config.DEFAULT_MODEL,
                temperature=0.2,  # Lower temperature for more consistent critique
                max_tokens=1000,
                groq_api_key=config.GROQ_API_KEY
            )
        else:
            self.llm = ChatOpenAI(
                model=config.DEFAULT_MODEL,
                temperature=0.2,  # Lower temperature for more consistent critique
                max_tokens=1000,
                openai_api_key=config.OPENAI_API_KEY
            )
    
    def critique_response(
        self, 
        query: str, 
        response: str, 
        context_sources: List[str]
    ) -> Dict[str, Any]:
        """Critique a generated response for accuracy and completeness.
        
        Args:
            query: Original user query
            response: Generated response to critique
            context_sources: Sources used for the response
            
        Returns:
            Dictionary containing critique results
        """
        try:
            # Create critique prompt
            system_prompt = self._create_critique_system_prompt()
            user_prompt = self._create_critique_user_prompt(query, response, context_sources)
            
            # Get critique
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            critique_response = self.llm.invoke(messages)
            critique_text = critique_response.content
            
            # Parse critique
            parsed_critique = parse_critique_response(critique_text)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(parsed_critique)
            
            result = {
                "is_approved": parsed_critique["is_approved"],
                "confidence_score": confidence_score,
                "flaws": parsed_critique["flaws"],
                "suggestions": parsed_critique["suggestions"],
                "raw_critique": critique_text,
                "needs_improvement": not parsed_critique["is_approved"]
            }
            
            logger.info(f"Critique completed - Approved: {result['is_approved']}, Confidence: {confidence_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error during critique: {e}")
            raise
    
    def generate_improved_query(
        self, 
        original_query: str, 
        critique_feedback: Dict[str, Any]
    ) -> str:
        """Generate an improved search query based on critique feedback.
        
        Args:
            original_query: Original search query
            critique_feedback: Feedback from critique
            
        Returns:
            Improved search query
        """
        try:
            if critique_feedback["is_approved"]:
                return original_query
            
            # Create query improvement prompt
            system_prompt = self._create_query_improvement_system_prompt()
            user_prompt = self._create_query_improvement_user_prompt(
                original_query, critique_feedback
            )
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            improved_query = response.content.strip()
            
            logger.info(f"Generated improved query: {improved_query}")
            return improved_query
            
        except Exception as e:
            logger.error(f"Error generating improved query: {e}")
            return original_query  # Fallback to original
    
    def _create_critique_system_prompt(self) -> str:
        """Create system prompt for critique."""
        return """You are a harsh but fair senior developer and technical reviewer. Your job is to critically evaluate responses to coding questions.

Evaluate responses based on:
1. **CORRECTNESS**: Is the code accurate and will it actually work?
2. **MODERNITY**: Does it use current, stable APIs and best practices?
3. **COMPLETENESS**: Does it fully address the question asked?
4. **CLARITY**: Is the explanation clear and helpful?

Be specific about any flaws you find. If you find significant issues, list them clearly starting with "FLAW:".

If the response is good and addresses the question accurately with current information, simply respond with "APPROVED".

If there are issues, provide specific feedback about what's wrong and what should be improved."""
    
    def _create_critique_user_prompt(
        self, 
        query: str, 
        response: str, 
        sources: List[str]
    ) -> str:
        """Create user prompt for critique."""
        sources_text = "\n".join([f"- {source}" for source in sources]) if sources else "None provided"
        
        return f"""Please critically evaluate this response to a coding question:

ORIGINAL QUESTION:
{query}

RESPONSE TO EVALUATE:
{response}

SOURCES USED:
{sources_text}

Provide your evaluation focusing on correctness, modernity, completeness, and clarity. Be specific about any issues."""
    
    def _create_query_improvement_system_prompt(self) -> str:
        """Create system prompt for query improvement."""
        return """You are an expert at crafting search queries to find better technical documentation.

Your task is to improve a search query based on critique feedback to help find more accurate and current information.

Focus on:
- Adding specific technology versions or frameworks mentioned in the critique
- Including keywords that would find more current documentation
- Removing ambiguous terms that led to outdated results
- Adding context that would help find examples that actually work

Return only the improved search query, nothing else."""
    
    def _create_query_improvement_user_prompt(
        self, 
        original_query: str, 
        critique_feedback: Dict[str, Any]
    ) -> str:
        """Create user prompt for query improvement."""
        flaws_text = "\n".join(critique_feedback["flaws"]) if critique_feedback["flaws"] else "None"
        suggestions_text = "\n".join(critique_feedback["suggestions"]) if critique_feedback["suggestions"] else "None"
        
        return f"""Improve this search query based on the critique feedback:

ORIGINAL QUERY:
{original_query}

IDENTIFIED FLAWS:
{flaws_text}

SUGGESTIONS:
{suggestions_text}

FULL CRITIQUE:
{critique_feedback["raw_critique"]}

Create a better search query that would find more accurate, current documentation."""
    
    def _calculate_confidence_score(self, parsed_critique: Dict[str, Any]) -> float:
        """Calculate confidence score based on critique."""
        if parsed_critique["is_approved"]:
            return 0.9  # High confidence for approved responses
        
        # Lower confidence based on number of flaws
        num_flaws = len(parsed_critique["flaws"])
        if num_flaws == 0:
            return 0.7  # Medium confidence even if not explicitly approved
        elif num_flaws <= 2:
            return 0.5  # Medium-low confidence
        else:
            return 0.3  # Low confidence for many flaws
