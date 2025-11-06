"""EchoCheck: Streamlit Dashboard for Reflective RAG System."""

# Resource limit guards (Unix-only) - prevent "Too many open files" errors
try:
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = min(65536, hard) if hard != resource.RLIM_INFINITY else 65536
    if soft < target:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        print(f'EchoCheck: Increased RLIMIT_NOFILE from {soft} to {target}')
except Exception as e:
    # Expected on Windows or environments where process can't modify limits
    pass

import streamlit as st
import logging
import time
from typing import Dict, Any, List
import json

# Import our modules
from core import ReflectiveRAGWorkflow, DocumentRetriever
from data import DocumentLoader
from utils import config, setup_logging, format_confidence_score

# Setup logging
setup_logging(config.LOG_LEVEL)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="EchoCheck 🛡️",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS for modern look
st.markdown("""
<style>
    body {
        background: linear-gradient(135deg, #f8fafc 0%, #e3e9f7 100%);
    }
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        color: #1a237e;
        letter-spacing: -1px;
        margin-bottom: 0.2em;
        text-shadow: 0 2px 8px #e3e9f7;
    }
    .main-subheader {
        font-size: 1.2rem;
        color: #3949ab;
        margin-bottom: 1.5em;
    }
    .thinking-step {
        padding: 16px 18px;
        margin: 10px 0;
        border-radius: 10px;
        border-left: 6px solid #1976d2;
        background: #f5faff;
        box-shadow: 0 2px 8px #e3e9f7;
    }
    .critique-step {
        border-left-color: #ff9800;
        background: #fff8e1;
    }
    .improvement-step {
        border-left-color: #43a047;
        background: #e8f5e9;
    }
    .error-step {
        border-left-color: #e53935;
        background: #ffebee;
    }
    .confidence-high {
        color: #43a047;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ff9800;
        font-weight: bold;
    }
    .confidence-low {
        color: #e53935;
        font-weight: bold;
    }
    .source-badge {
        display: inline-block;
        padding: 3px 10px;
        margin: 2px 2px 2px 0;
        background: linear-gradient(90deg, #e3f2fd 60%, #bbdefb 100%);
        border-radius: 14px;
        font-size: 0.9em;
        color: #1565c0;
        font-weight: 600;
        box-shadow: 0 1px 4px #e3e9f7;
    }
    .stButton>button {
        background: linear-gradient(90deg, #1976d2 60%, #43a047 100%);
        color: white;
        font-weight: 700;
        border-radius: 8px;
        border: none;
        padding: 0.6em 1.5em;
        margin-top: 0.5em;
        box-shadow: 0 2px 8px #e3e9f7;
        transition: 0.2s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #43a047 60%, #1976d2 100%);
        color: #fffde7;
        transform: translateY(-2px) scale(1.03);
    }
    .stTextArea textarea {
        border-radius: 8px;
        border: 1.5px solid #1976d2;
        background: #f5faff;
        font-size: 1.1em;
    }
    .stMetric {
        background: #e3f2fd;
        border-radius: 8px;
        padding: 0.5em 1em;
        margin-bottom: 0.5em;
    }
    .sidebar-logo {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 1.5em;
    }
    .sidebar-logo img {
        width: 38px;
        height: 38px;
        border-radius: 8px;
        box-shadow: 0 2px 8px #e3e9f7;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_system():
    """Initialize the RAG system components."""
    try:
        # Validate configuration
        config.validate()
        
        # Initialize components (share retriever to avoid duplicates)
        retriever = DocumentRetriever()
        workflow = ReflectiveRAGWorkflow(retriever=retriever)
        loader = DocumentLoader()
        
        # Load sample documents if collection is empty
        stats = retriever.get_collection_stats()
        if stats["document_count"] == 0:
            with st.spinner("Loading documents..."):
                docs = loader.load_all_documents()
                if docs:
                    retriever.add_documents(docs)
                    st.success(f"Loaded {len(docs)} documents")
                else:
                    # Fallback to sample docs
                    sample_docs = loader.load_sample_docs()
                    retriever.add_documents(sample_docs)
                    st.success(f"Loaded {len(sample_docs)} sample documents")
        
        return workflow, retriever, loader
        
    except Exception as e:
        st.error(f"Failed to initialize system: {e}")
        st.stop()

def display_thinking_process(thinking_process: List[Dict[str, Any]]):
    """Display the AI's thinking process."""
    st.subheader("🧠 AI Thinking Process")
    
    for i, step in enumerate(thinking_process, 1):
        step_type = step["step"]
        description = step.get("description", f"Step {i}: {step_type}")
        
        # Determine CSS class based on step type
        css_class = "thinking-step"
        if "critique" in step_type:
            css_class += " critique-step"
        elif "improve" in step_type or "generate_improved" in step_type:
            css_class += " improvement-step"
        elif "error" in step:
            css_class += " error-step"
        
        # Display step
        st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
        st.write(f"**{i}. {description}**")
        
        # Show details if available
        if "details" in step:
            details = step["details"]
            
            if "sources" in details:
                sources = details["sources"]
                if sources:
                    st.write("📚 Sources:")
                    for source in sources[:3]:  # Show max 3 sources
                        st.markdown(f'<span class="source-badge">{source}</span>', unsafe_allow_html=True)
            
            if "approved" in details:
                status = "✅ Approved" if details["approved"] else "❌ Needs Improvement"
                st.write(f"Status: {status}")
            
            if "confidence" in details:
                confidence = details["confidence"]
                formatted_confidence = format_confidence_score(confidence)
                st.write(f"Confidence: {formatted_confidence}")
            
            if "flaws" in details and details["flaws"]:
                st.write("⚠️ Issues found:")
                for flaw in details["flaws"]:
                    st.write(f"• {flaw}")
        
        if "error" in step:
            st.error(f"Error: {step['error']}")
        
        st.markdown('</div>', unsafe_allow_html=True)

def display_final_response(result: Dict[str, Any]):
    """Display the final response with metadata."""
    st.subheader("✅ Final Response")
    
    # Show confidence score
    confidence = result.get("confidence_score", 0.5)
    formatted_confidence = format_confidence_score(confidence)
    st.markdown(f"**Confidence:** {formatted_confidence}")
    
    # Show the response
    final_response = result.get("final_response", "No response generated")
    st.markdown(final_response)
    
    # Show sources
    if "response_metadata" in result and "sources" in result["response_metadata"]:
        sources = result["response_metadata"]["sources"]
        if sources:
            st.subheader("📚 Sources")
            for source in sources:
                st.markdown(f'<span class="source-badge">{source}</span>', unsafe_allow_html=True)

def main():
    """Main Streamlit application."""

    # Modern header with icon and subtitle
    st.markdown('<div class="main-header">EchoCheck 🛡️</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subheader">Your LLM\'s Harsh (but Fair) Code Reviewer &mdash; Preventing AI Hallucinations in Real-Time</div>', unsafe_allow_html=True)

    # Initialize system
    workflow, retriever, loader = initialize_system()

    # Sidebar with logo and configuration
    with st.sidebar:
        st.markdown('<div class="sidebar-logo"><img src="https://em-content.zobj.net/source/microsoft-teams/363/shield_1f6e1-fe0f.png" alt="logo"/> <span style="font-size:1.3em;font-weight:700;color:#1a237e;">EchoCheck</span></div>', unsafe_allow_html=True)
        st.header("⚙️ Configuration")

        # Show system stats
        stats = retriever.get_collection_stats()
        st.metric("Documents Loaded", stats["document_count"])

        # Document loading mode selector
        st.subheader("📋 Document Mode")
        mode_options = {
            "hybrid": "🔄 Hybrid (cache if recent)",
            "cache_only": "📁 Cache Only (fastest)",
            "live_only": "🌐 Live Only (always fresh)"
        }
        
        selected_mode = st.selectbox(
            "Source Mode:",
            options=list(mode_options.keys()),
            index=0,
            format_func=lambda x: mode_options[x],
            key="doc_mode"
        )

        # Cache information
        try:
            cache_info = loader.get_cache_info()
            if cache_info.get("exists"):
                cache_age = cache_info.get("cache_age_hours", 0)
                is_expired = cache_info.get("is_expired", False)
                
                st.subheader("📊 Cache Status")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Documents", cache_info.get("document_count", 0))
                
                with col2:
                    status = "🟡 Expired" if is_expired else "🟢 Fresh"
                    st.metric("Status", status)
                
                st.write(f"**Age:** {cache_age:.1f} hours")
                
                # Source breakdown
                sources_used = cache_info.get("sources_used", {})
                if sources_used:
                    st.write("**Sources:**")
                    for source, enabled in sources_used.items():
                        icon = "✅" if enabled else "❌"
                        st.write(f"{icon} {source}")
            else:
                st.warning("📭 No cache found")
        except Exception as e:
            st.warning(f"Cache info unavailable: {e}")

        # Model settings
        st.subheader("Model Settings")
        if config.USE_GROQ:
            st.info(f"🚀 Model: {config.DEFAULT_MODEL} (Groq)")
        else:
            st.info(f"🤖 Model: {config.DEFAULT_MODEL} (OpenAI)")
        st.info(f"Max Iterations: {config.MAX_REFLECTION_CYCLES}")

        # Sample queries
        st.subheader("💡 Try These Sample Queries")
        sample_queries = [
            "How do I use React's new 'use' hook in a Client Component?",
            "What's the best way to handle state in Next.js 15?",
            "How do I implement streaming with the new OpenAI SDK?",
            "How to fetch data in a Next.js Client Component?",
        ]
        for query in sample_queries:
            if st.button(query, key=f"sample_{hash(query)}"):
                st.session_state.user_query = query

        # Document management
        st.subheader("📚 Document Management")
        colA, colB = st.columns(2)
        with colA:
            if st.button("� Refresh Documents"):
                with st.spinner("Refreshing documents..."):
                    # Force fresh load
                    docs = loader.load_all_documents(mode="live_only")
                    retriever.clear_collection()
                    if docs:
                        retriever.add_documents(docs)
                st.success(f"Refreshed with {len(docs) if 'docs' in locals() else 0} documents")
                st.rerun()
        
        with colB:
            if st.button("�️ Clear Cache"):
                if loader.clear_cache():
                    st.success("Cache cleared!")
                else:
                    st.error("Failed to clear cache")
                st.rerun()
        
        # Load with selected mode
        if st.button("📥 Load with Selected Mode"):
            with st.spinner(f"Loading documents in {selected_mode} mode..."):
                docs = loader.load_all_documents(mode=selected_mode)
                retriever.clear_collection()
                if docs:
                    retriever.add_documents(docs)
            st.success(f"Loaded {len(docs) if 'docs' in locals() else 0} documents in {selected_mode} mode!")
            st.rerun()

    # Main interface
    st.markdown("""
        <div style="margin-top:1.5em;"></div>
    """, unsafe_allow_html=True)
    st.markdown('<h3 style="color:#1976d2;">💬 Ask a Question</h3>', unsafe_allow_html=True)

    # Query input in expander for focus
    with st.expander("Type your coding question here", expanded=True):
        user_query = st.text_area(
            "Enter your coding question:",
            value=st.session_state.get("user_query", ""),
            height=100,
            placeholder="e.g., How do I use the new React hooks in Next.js 15?"
        )

    # Process query
    if st.button("🚀 Get Answer", type="primary") and user_query.strip():
        # Clear previous results
        st.session_state.pop("last_result", None)

        # Create columns for real-time display
        thinking_col, response_col = st.columns([1, 1])

        with thinking_col:
            thinking_placeholder = st.empty()

        with response_col:
            response_placeholder = st.empty()

        # Show initial thinking
        with thinking_placeholder.container():
            st.subheader("🧠 AI Thinking Process")
            st.info("🔄 Starting analysis...")

        try:
            # Run the workflow
            start_time = time.time()
            result = workflow.run(user_query)
            end_time = time.time()

            # Store result
            st.session_state.last_result = result

            # Display thinking process in expander for clarity
            with thinking_placeholder.container():
                with st.expander("See AI Thinking Process", expanded=True):
                    display_thinking_process(result.get("thinking_process", []))
                st.success(f"⏱️ Completed in {end_time - start_time:.2f} seconds")

            # Display final response in a visually distinct panel
            with response_placeholder.container():
                with st.expander("See Final Answer", expanded=True):
                    display_final_response(result)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            logger.error(f"Error processing query: {e}")

    # Show previous result if available
    elif "last_result" in st.session_state and not user_query.strip():
        st.subheader("📋 Previous Result")
        col1, col2 = st.columns([1, 1])
        with col1:
            with st.expander("See AI Thinking Process", expanded=True):
                display_thinking_process(st.session_state.last_result.get("thinking_process", []))
        with col2:
            with st.expander("See Final Answer", expanded=True):
                display_final_response(st.session_state.last_result)

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;font-size:1.1em;color:#3949ab;'>"
    "<b>EchoCheck</b> demonstrates Reflective RAG architecture using LangGraph.<br>"
        "Built for hackathon demo. 🏆"
        "</div>", unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
