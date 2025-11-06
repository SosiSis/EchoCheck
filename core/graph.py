"""LangGraph workflow for Reflective RAG system."""

import logging
from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict

from core.retriever import DocumentRetriever
from core.generator import ResponseGenerator
from core.critic import ResponseCritic
from utils.config import config

logger = logging.getLogger(__name__)

class RAGState(TypedDict):
    """State for the RAG workflow."""
    # Input
    query: str
    
    # Retrieval
    retrieved_docs: List[Any]
    context_sources: List[str]
    
    # Generation
    response: str
    response_metadata: Dict[str, Any]
    
    # Critique
    critique_result: Dict[str, Any]
    is_approved: bool
    confidence_score: float
    
    # Iteration control
    iteration_count: int
    max_iterations: int
    
    # Final output
    final_response: str
    thinking_process: List[Dict[str, Any]]

class ReflectiveRAGWorkflow:
    """Implements the Reflective RAG workflow using LangGraph."""
    
    def __init__(self, retriever: Optional[DocumentRetriever] = None):
        """Initialize the workflow components.
        
        Args:
            retriever: Optional DocumentRetriever instance to share. If None, creates new one.
        """
        self.retriever = retriever or DocumentRetriever()
        self.generator = ResponseGenerator()
        self.critic = ResponseCritic()
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        self.compiled_workflow = self.workflow.compile()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("retrieve", self._retrieve_documents)
        workflow.add_node("generate", self._generate_response)
        workflow.add_node("critique", self._critique_response)
        workflow.add_node("improve_query", self._improve_query)
        workflow.add_node("retrieve_improved", self._retrieve_improved)
        workflow.add_node("generate_improved", self._generate_improved)
        workflow.add_node("finalize", self._finalize_response)
        
        # Set entry point
        workflow.set_entry_point("retrieve")
        
        # Add edges
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "critique")
        
        # Conditional edge from critique
        workflow.add_conditional_edges(
            "critique",
            self._should_improve,
            {
                "improve": "improve_query",
                "finalize": "finalize"
            }
        )
        
        # Improvement path
        workflow.add_edge("improve_query", "retrieve_improved")
        workflow.add_edge("retrieve_improved", "generate_improved")
        workflow.add_edge("generate_improved", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow
    
    def _retrieve_documents(self, state: RAGState) -> RAGState:
        """Retrieve relevant documents."""
        try:
            docs = self.retriever.retrieve_documents(state["query"])
            sources = [doc.metadata.get("source", "Unknown") for doc in docs]
            
            # Add to thinking process
            thinking_step = {
                "step": "retrieve",
                "description": f"Retrieved {len(docs)} documents",
                "details": {"sources": sources}
            }
            
            return {
                **state,
                "retrieved_docs": docs,
                "context_sources": sources,
                "thinking_process": [thinking_step]
            }
            
        except Exception as e:
            logger.error(f"Error in retrieve_documents: {e}")
            return {
                **state,
                "retrieved_docs": [],
                "context_sources": [],
                "thinking_process": [{"step": "retrieve", "error": str(e)}]
            }
    
    def _generate_response(self, state: RAGState) -> RAGState:
        """Generate initial response."""
        try:
            response_data = self.generator.generate_initial_response(
                state["query"], 
                state["retrieved_docs"]
            )
            
            # Add to thinking process
            thinking_step = {
                "step": "generate",
                "description": "Generated initial response",
                "details": {
                    "context_used": response_data["context_used"],
                    "sources": response_data["sources"]
                }
            }
            
            thinking_process = state.get("thinking_process", [])
            thinking_process.append(thinking_step)
            
            return {
                **state,
                "response": response_data["content"],
                "response_metadata": response_data,
                "thinking_process": thinking_process
            }
            
        except Exception as e:
            logger.error(f"Error in generate_response: {e}")
            thinking_process = state.get("thinking_process", [])
            thinking_process.append({"step": "generate", "error": str(e)})
            
            return {
                **state,
                "response": "Error generating response",
                "response_metadata": {},
                "thinking_process": thinking_process
            }
    
    def _critique_response(self, state: RAGState) -> RAGState:
        """Critique the generated response."""
        try:
            critique_result = self.critic.critique_response(
                state["query"],
                state["response"],
                state["context_sources"]
            )
            
            # Add to thinking process
            thinking_step = {
                "step": "critique",
                "description": "🧠 Self-critique analysis",
                "details": {
                    "approved": critique_result["is_approved"],
                    "confidence": critique_result["confidence_score"],
                    "flaws": critique_result["flaws"]
                }
            }
            
            thinking_process = state.get("thinking_process", [])
            thinking_process.append(thinking_step)
            
            return {
                **state,
                "critique_result": critique_result,
                "is_approved": critique_result["is_approved"],
                "confidence_score": critique_result["confidence_score"],
                "iteration_count": 1,
                "max_iterations": config.MAX_REFLECTION_CYCLES,
                "thinking_process": thinking_process
            }
            
        except Exception as e:
            logger.error(f"Error in critique_response: {e}")
            thinking_process = state.get("thinking_process", [])
            thinking_process.append({"step": "critique", "error": str(e)})
            
            return {
                **state,
                "critique_result": {"is_approved": True, "confidence_score": 0.5},
                "is_approved": True,
                "confidence_score": 0.5,
                "thinking_process": thinking_process
            }
    
    def _should_improve(self, state: RAGState) -> str:
        """Decide whether to improve the response or finalize."""
        if state["is_approved"]:
            return "finalize"
        
        if state["iteration_count"] >= state["max_iterations"]:
            return "finalize"
        
        return "improve"
    
    def _improve_query(self, state: RAGState) -> RAGState:
        """Generate improved query based on critique."""
        try:
            improved_query = self.critic.generate_improved_query(
                state["query"],
                state["critique_result"]
            )
            
            # Add to thinking process
            thinking_step = {
                "step": "improve_query",
                "description": "⚠️ Generating improved search query",
                "details": {
                    "original_query": state["query"],
                    "improved_query": improved_query
                }
            }
            
            thinking_process = state.get("thinking_process", [])
            thinking_process.append(thinking_step)
            
            return {
                **state,
                "query": improved_query,
                "thinking_process": thinking_process
            }
            
        except Exception as e:
            logger.error(f"Error in improve_query: {e}")
            thinking_process = state.get("thinking_process", [])
            thinking_process.append({"step": "improve_query", "error": str(e)})
            
            return {
                **state,
                "thinking_process": thinking_process
            }
    
    def _retrieve_improved(self, state: RAGState) -> RAGState:
        """Retrieve documents with improved query."""
        try:
            docs = self.retriever.retrieve_documents(state["query"])
            sources = [doc.metadata.get("source", "Unknown") for doc in docs]
            
            # Add to thinking process
            thinking_step = {
                "step": "retrieve_improved",
                "description": "🔄 Re-retrieving with improved query",
                "details": {"sources": sources}
            }
            
            thinking_process = state.get("thinking_process", [])
            thinking_process.append(thinking_step)
            
            return {
                **state,
                "retrieved_docs": docs,
                "context_sources": sources,
                "thinking_process": thinking_process
            }
            
        except Exception as e:
            logger.error(f"Error in retrieve_improved: {e}")
            thinking_process = state.get("thinking_process", [])
            thinking_process.append({"step": "retrieve_improved", "error": str(e)})
            
            return {
                **state,
                "thinking_process": thinking_process
            }
    
    def _generate_improved(self, state: RAGState) -> RAGState:
        """Generate improved response."""
        try:
            response_data = self.generator.generate_improved_response(
                state["query"],
                state["response"],
                state["critique_result"]["raw_critique"],
                state["retrieved_docs"]
            )
            
            # Add to thinking process
            thinking_step = {
                "step": "generate_improved",
                "description": "✨ Generating improved response",
                "details": {
                    "context_used": response_data["context_used"],
                    "sources": response_data["sources"]
                }
            }
            
            thinking_process = state.get("thinking_process", [])
            thinking_process.append(thinking_step)
            
            return {
                **state,
                "response": response_data["content"],
                "response_metadata": response_data,
                "thinking_process": thinking_process
            }
            
        except Exception as e:
            logger.error(f"Error in generate_improved: {e}")
            thinking_process = state.get("thinking_process", [])
            thinking_process.append({"step": "generate_improved", "error": str(e)})
            
            return {
                **state,
                "thinking_process": thinking_process
            }
    
    def _finalize_response(self, state: RAGState) -> RAGState:
        """Finalize the response."""
        thinking_step = {
            "step": "finalize",
            "description": "✅ Finalizing response",
            "details": {
                "final_confidence": state.get("confidence_score", 0.5),
                "iterations": state.get("iteration_count", 1)
            }
        }
        
        thinking_process = state.get("thinking_process", [])
        thinking_process.append(thinking_step)
        
        return {
            **state,
            "final_response": state["response"],
            "thinking_process": thinking_process
        }
    
    def run(self, query: str) -> Dict[str, Any]:
        """Run the complete RAG workflow.
        
        Args:
            query: User query
            
        Returns:
            Complete workflow result
        """
        try:
            initial_state = {
                "query": query,
                "retrieved_docs": [],
                "context_sources": [],
                "response": "",
                "response_metadata": {},
                "critique_result": {},
                "is_approved": False,
                "confidence_score": 0.0,
                "iteration_count": 0,
                "max_iterations": config.MAX_REFLECTION_CYCLES,
                "final_response": "",
                "thinking_process": []
            }
            
            # Run the workflow
            final_state = self.compiled_workflow.invoke(initial_state)
            
            logger.info(f"Workflow completed for query: {query[:50]}...")
            return final_state
            
        except Exception as e:
            logger.error(f"Error running workflow: {e}")
            raise
