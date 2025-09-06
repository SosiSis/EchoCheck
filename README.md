# EchoCheck 🛡️

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://echocheck-w7wuxpftjefzfxhcsrautd.streamlit.app/)

**Your LLM's Harsh (but Fair) Code Reviewer. Preventing AI Hallucinations in Real-Time.**

## 🎯 The Problem

Large Language Models are powerful, but they are notoriously confident, even when they are wrong. In high-stakes environments like coding, legal research, or medical queries, a single hallucination can be catastrophic. Current chatbots don't double-check their work.

## 🚀 The Solution

EchoCheck is a **Reflective RAG system** that acts as its own fact-checker before it ever gives you an answer. It retrieves, generates, **critiques**, and then **adapts** - creating a self-improving AI assistant.

## 🏗️ Architecture

- **LangGraph**: Orchestrates the reflective workflow
- **Streamlit**: Interactive dashboard with real-time thinking visualization
- **ChromaDB**: Vector database for document storage and retrieval
- **Groq LLaMA**: Primary LLM for generation and critique (fast inference)
- **OpenAI**: Fallback option and embeddings

## 🔄 The Reflection Loop

1. **Initial Retrieval & Generation**: Generate first answer
2. **Self-Critique**: AI critically evaluates its own response
3. **Adaptive Refinement**: If flaws found, improve query and regenerate
4. **Verification**: Deliver verified, high-quality answer

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

## 🔧 Setup

1. Create a `.env` file with your API keys:
```
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  # Optional, for embeddings
USE_GROQ=True
DEFAULT_MODEL=llama3-8b-8192
```

2. Run the application:
```bash
streamlit run app.py
```

## 📁 Project Structure

```
echocheck/
├── app.py                 # Streamlit dashboard
├── core/
│   ├── __init__.py
│   ├── graph.py          # LangGraph workflow definition
│   ├── retriever.py      # Document retrieval logic
│   ├── generator.py      # Response generation
│   ├── critic.py         # Self-critique system
│   └── embeddings.py     # Embedding utilities
├── data/
│   ├── __init__.py
│   ├── loader.py         # Document loading and processing
│   └── sources/          # Documentation sources
├── utils/
│   ├── __init__.py
│   ├── config.py         # Configuration management
│   └── helpers.py        # Utility functions
└── tests/
    ├── __init__.py
    └── test_core.py       # Unit tests
```

## 🎯 Features

- **Real-time Reflection Visualization**: See the AI thinking process
- **Confidence Scoring**: Get reliability metrics for each answer
- **Source Citations**: Hover to see document sources
- **Multi-domain Support**: Currently optimized for coding queries
- **Modular Architecture**: Easy to extend and customize

## 🏆 Why This Wins

- **Solves Real Problem**: Addresses LLM hallucination concerns
- **Cutting-edge Tech**: Implements Reflective RAG architecture
- **Amazing Demo**: Visual thinking process impresses judges
- **Clear Value**: Makes AI more reliable and trustworthy

## 🚀 Demo Scenarios

Try these queries to see the reflection in action:
- "How do I use React's new 'use' hook in a Client Component?"
- "What's the best way to handle state in Next.js 15?"
- "How do I implement streaming with the new OpenAI SDK?"

## 📈 Future Enhancements

- Multi-turn reflection cycles
- HyDE (Hypothetical Document Embeddings) implementation
- Support for additional domains (legal, medical, etc.)
- Advanced confidence scoring algorithms

---

**Built for NSK Hackathon  2025** 🏆
