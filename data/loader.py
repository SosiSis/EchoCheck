"""Enhanced document loader with multiple sources and caching."""

import os
import json
import requests
import hashlib
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal
from urllib.parse import urljoin, urlparse
import zipfile
import tempfile

import markdown
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import (
    TextLoader, 
    DirectoryLoader,
    WebBaseLoader
)

# Import configuration
from utils.config import Config
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
import markdown

logger = logging.getLogger(__name__)

# Create config instance
config = Config()

class DocumentLoader:
    """Loads and processes documents for the RAG system."""
    
    def __init__(self):
        """Initialize the document loader."""
        self.sources_dir = Path("data/sources")
        self.sources_dir.mkdir(parents=True, exist_ok=True)
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def fetch_react_docs(self) -> List[Dict[str, Any]]:
        """Fetch latest React documentation."""
        docs = []
        react_urls = [
            "https://react.dev/reference/react/use",
            "https://react.dev/learn/synchronizing-with-effects",
            "https://react.dev/reference/react/useState",
            "https://react.dev/reference/react/useEffect"
        ]
        
        for url in react_urls:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Extract main content
                    content_div = soup.find('article') or soup.find('main') or soup.find('div', class_='content')
                    if content_div:
                        text_content = content_div.get_text(strip=True)
                        if len(text_content) > 100:  # Only add if substantial content
                            docs.append({
                                "content": text_content,
                                "metadata": {
                                    "source": "react-official",
                                    "url": url,
                                    "type": "documentation",
                                    "framework": "react",
                                    "fetched_at": time.time()
                                }
                            })
                time.sleep(1)  # Rate limiting
            except Exception as e:
                logger.warning(f"Failed to fetch {url}: {e}")
                
        return docs
    
    def fetch_nextjs_docs(self) -> List[Dict[str, Any]]:
        """Fetch Next.js documentation."""
        docs = []
        nextjs_urls = [
            "https://nextjs.org/docs/app/building-your-application/data-fetching",
            "https://nextjs.org/docs/app/building-your-application/rendering/client-components",
            "https://nextjs.org/docs/app/building-your-application/rendering/server-components"
        ]
        
        for url in nextjs_urls:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    content_div = soup.find('article') or soup.find('main') or soup.find('div', class_='content')
                    if content_div:
                        text_content = content_div.get_text(strip=True)
                        if len(text_content) > 100:
                            docs.append({
                                "content": text_content,
                                "metadata": {
                                    "source": "nextjs-official",
                                    "url": url,
                                    "type": "documentation",
                                    "framework": "nextjs",
                                    "fetched_at": time.time()
                                }
                            })
                time.sleep(1)
            except Exception as e:
                logger.warning(f"Failed to fetch {url}: {e}")
                
        return docs
    
    def load_local_docs(self) -> List[Dict[str, Any]]:
        """Load documents from local files."""
        docs = []
        
        # Load markdown files from sources directory
        for md_file in self.sources_dir.glob("*.md"):
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Convert markdown to text
                    html = markdown.markdown(content)
                    soup = BeautifulSoup(html, 'html.parser')
                    text_content = soup.get_text()
                    
                    docs.append({
                        "content": text_content,
                        "metadata": {
                            "source": "local-docs",
                            "filename": md_file.name,
                            "type": "markdown",
                            "path": str(md_file)
                        }
                    })
            except Exception as e:
                logger.warning(f"Failed to load {md_file}: {e}")
        
        # Load JSON files
        for json_file in self.sources_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        docs.extend(data)
                    else:
                        docs.append(data)
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")
        
        return docs
    
    def load_all_documents(
        self, 
        mode: Optional[Literal["cache_only", "live_only", "hybrid"]] = None
    ) -> List[Document]:
        """Load documents based on configuration mode.
        
        Args:
            mode: Loading mode - cache_only, live_only, or hybrid
            
        Returns:
            List of Document objects
        """
        if mode is None:
            mode = config.DOCUMENT_SOURCE_MODE
            
        cache_file = self.sources_dir / "cached_docs.json"
        
        # Handle different modes
        if mode == "cache_only":
            if cache_file.exists():
                logger.info("📁 Loading from cache only...")
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    return self._convert_to_documents(cached_data.get("documents", []))
            else:
                logger.warning("❌ No cache found, falling back to sample documents")
                return self.load_sample_docs()
        
        elif mode == "live_only":
            logger.info("🌐 Fetching fresh documents (ignoring cache)...")
            return self._fetch_fresh_documents(cache_file)
        
        else:  # hybrid mode
            if config.USE_DOCUMENT_CACHE and cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    cached_at = cached_data.get("cached_at", 0)
                    cache_age = (time.time() - cached_at) / 3600  # hours
                    
                    if cache_age < config.CACHE_EXPIRY_HOURS:
                        logger.info(f"📁 Using cached documents ({cache_age:.1f}h old)")
                        return self._convert_to_documents(cached_data.get("documents", []))
                    else:
                        logger.info(f"⏰ Cache expired ({cache_age:.1f}h old), fetching fresh...")
            
            return self._fetch_fresh_documents(cache_file)
    
    def _fetch_fresh_documents(self, cache_file: Path) -> List[Document]:
        """Fetch fresh documents from all enabled sources."""
        all_docs = []
        
        # Fetch from enabled sources
        if config.ENABLE_REACT_DOCS:
            try:
                react_docs = self.fetch_react_docs()
                all_docs.extend(react_docs)
                logger.info(f"✅ Loaded {len(react_docs)} React documents")
            except Exception as e:
                logger.error(f"❌ Failed to load React docs: {e}")
        
        if config.ENABLE_NEXTJS_DOCS:
            try:
                nextjs_docs = self.fetch_nextjs_docs()
                all_docs.extend(nextjs_docs)
                logger.info(f"✅ Loaded {len(nextjs_docs)} Next.js documents")
            except Exception as e:
                logger.error(f"❌ Failed to load Next.js docs: {e}")
        
        if config.ENABLE_LOCAL_DOCS:
            local_docs = self.load_local_docs()
            all_docs.extend(local_docs)
            logger.info(f"✅ Loaded {len(local_docs)} local documents")
        
        # Fallback to sample documents if nothing loaded
        if not all_docs and config.ENABLE_SAMPLE_DOCS:
            logger.info("📝 No documents loaded, using samples...")
            return self.load_sample_docs()
        
        # Cache the fresh documents
        if all_docs:
            cache_data = {
                "documents": all_docs,
                "cached_at": time.time(),
                "total_count": len(all_docs),
                "sources_used": {
                    "react": config.ENABLE_REACT_DOCS,
                    "nextjs": config.ENABLE_NEXTJS_DOCS,
                    "local": config.ENABLE_LOCAL_DOCS,
                    "sample": config.ENABLE_SAMPLE_DOCS
                }
            }
            cache_file.parent.mkdir(exist_ok=True, parents=True)
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Cached {len(all_docs)} documents")
        
        return self._convert_to_documents(all_docs)
    
    def _convert_to_documents(self, doc_dicts: List[Dict[str, Any]]) -> List[Document]:
        """Convert dictionary format to LangChain Document objects."""
        documents = []
        for doc_data in doc_dicts:
            doc = Document(
                page_content=doc_data.get("content", ""),
                metadata=doc_data.get("metadata", {})
            )
            documents.append(doc)
        return documents
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the document cache."""
        cache_file = self.sources_dir / "cached_docs.json"
        if not cache_file.exists():
            return {"exists": False}
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                
            cached_at = cache_data.get("cached_at", 0)
            cache_age_hours = (time.time() - cached_at) / 3600
            
            return {
                "exists": True,
                "document_count": cache_data.get("total_count", 0),
                "cached_at": cached_at,
                "cache_age_hours": cache_age_hours,
                "is_expired": cache_age_hours > config.CACHE_EXPIRY_HOURS,
                "sources_used": cache_data.get("sources_used", {}),
                "documents": cache_data.get("documents", [])
            }
        except Exception as e:
            logger.error(f"Error reading cache info: {e}")
            return {"exists": False, "error": str(e)}
    
    def clear_cache(self) -> bool:
        """Clear the document cache."""
        cache_file = self.sources_dir / "cached_docs.json"
        try:
            if cache_file.exists():
                cache_file.unlink()
                logger.info("🗑️ Document cache cleared")
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    def load_sample_docs(self) -> List[Document]:
        """Load sample documentation for demo purposes."""
        sample_docs = [
            {
                "content": """
# React 18+ Modern Patterns

## Using the `use` Hook

The `use` hook is a new React hook for handling asynchronous operations in Server Components.

**Important**: The `use` hook is designed for Server Components, not Client Components.

For Client Components, continue using:
- `useEffect` with `fetch`
- Data fetching libraries like SWR or React Query
- `useSWR` for simple cases

Example for Client Component:
```jsx
import useSWR from 'swr'

function Profile() {
  const { data, error } = useSWR('/api/user', fetch)
  
  if (error) return <div>Failed to load</div>
  if (!data) return <div>Loading...</div>
  
  return <div>Hello {data.name}!</div>
}
```
                """,
                "metadata": {"source": "React 18 Official Docs", "type": "documentation"}
            },
            {
                "content": """
# Next.js 15 App Router

## Client vs Server Components

Server Components run on the server and can directly access databases.
Client Components run in the browser and need to fetch data through APIs.

For data fetching in Client Components:
1. Use `fetch` with `useEffect`
2. Use SWR for caching: `npm install swr`
3. Use React Query for complex state management

Example with SWR:
```jsx
'use client'
import useSWR from 'swr'

const fetcher = (url) => fetch(url).then(res => res.json())

export default function Posts() {
  const { data, error, isLoading } = useSWR('/api/posts', fetcher)
  
  if (error) return <div>Failed to load posts</div>
  if (isLoading) return <div>Loading...</div>
  
  return (
    <ul>
      {data.map(post => (
        <li key={post.id}>{post.title}</li>
      ))}
    </ul>
  )
}
```
                """,
                "metadata": {"source": "Next.js 15 Documentation", "type": "documentation"}
            },
            {
                "content": """
# OpenAI SDK v4+ Streaming

## New Streaming API

The OpenAI SDK v4 introduced a new streaming API.

```javascript
import OpenAI from 'openai'

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
})

const stream = await openai.chat.completions.create({
  model: 'gpt-4',
  messages: [{ role: 'user', content: 'Hello!' }],
  stream: true,
})

for await (const chunk of stream) {
  process.stdout.write(chunk.choices[0]?.delta?.content || '')
}
```

## React Integration

```jsx
import { useState } from 'react'

function ChatComponent() {
  const [response, setResponse] = useState('')
  
  const handleStream = async () => {
    const response = await fetch('/api/chat', {
      method: 'POST',
      body: JSON.stringify({ message: 'Hello' })
    })
    
    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      
      const chunk = decoder.decode(value)
      setResponse(prev => prev + chunk)
    }
  }
  
  return (
    <div>
      <button onClick={handleStream}>Start Chat</button>
      <div>{response}</div>
    </div>
  )
}
```
                """,
                "metadata": {"source": "OpenAI SDK v4 Docs", "type": "documentation"}
            },
            {
                "content": """
# Common React Mistakes in 2024

## Outdated Patterns to Avoid

### 1. Using `use` hook in Client Components
**Wrong:**
```jsx
'use client'
import { use } from 'react'

function UserProfile() {
  const user = use(fetchUser()) // This will NOT work in Client Components
  return <div>{user.name}</div>
}
```

**Correct:**
```jsx
'use client'
import { useState, useEffect } from 'react'

function UserProfile() {
  const [user, setUser] = useState(null)
  
  useEffect(() => {
    fetchUser().then(setUser)
  }, [])
  
  if (!user) return <div>Loading...</div>
  return <div>{user.name}</div>
}
```

### 2. Old React 17 Patterns
Avoid using class components for new code. Use function components with hooks.

### 3. Deprecated Next.js Patterns
Don't use `getServerSideProps` in App Router. Use Server Components instead.
                """,
                "metadata": {"source": "React Best Practices 2024", "type": "best_practices"}
            }
        ]
        
        documents = []
        for doc_data in sample_docs:
            doc = Document(
                page_content=doc_data["content"],
                metadata=doc_data["metadata"]
            )
            documents.append(doc)
        
        logger.info(f"Loaded {len(documents)} sample documents")
        return documents
    
    def load_from_directory(self, directory_path: str) -> List[Document]:
        """Load documents from a directory.
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            List of loaded documents
        """
        try:
            loader = DirectoryLoader(
                directory_path,
                glob="**/*.{txt,md,py,js,jsx,ts,tsx}",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"}
            )
            
            documents = loader.load()
            
            # Add source metadata
            for doc in documents:
                if "source" not in doc.metadata:
                    doc.metadata["source"] = os.path.basename(doc.metadata.get("file_path", "Unknown"))
            
            logger.info(f"Loaded {len(documents)} documents from {directory_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading from directory {directory_path}: {e}")
            return []
    
    def load_from_urls(self, urls: List[str]) -> List[Document]:
        """Load documents from web URLs.
        
        Args:
            urls: List of URLs to load
            
        Returns:
            List of loaded documents
        """
        documents = []
        
        for url in urls:
            try:
                loader = WebBaseLoader(url)
                docs = loader.load()
                
                # Clean and process web content
                for doc in docs:
                    doc.page_content = self._clean_web_content(doc.page_content)
                    doc.metadata["source"] = url
                    doc.metadata["type"] = "web"
                
                documents.extend(docs)
                logger.info(f"Loaded document from {url}")
                
            except Exception as e:
                logger.error(f"Error loading from URL {url}: {e}")
                continue
        
        return documents
    
    def load_markdown_file(self, file_path: str) -> List[Document]:
        """Load and process a markdown file.
        
        Args:
            file_path: Path to markdown file
            
        Returns:
            List of document chunks
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Convert markdown to text
            html = markdown.markdown(content)
            soup = BeautifulSoup(html, 'html.parser')
            text_content = soup.get_text()
            
            # Create document
            doc = Document(
                page_content=text_content,
                metadata={
                    "source": os.path.basename(file_path),
                    "type": "markdown",
                    "file_path": file_path
                }
            )
            
            # Split into chunks
            chunks = self.text_splitter.split_documents([doc])
            
            logger.info(f"Loaded markdown file {file_path} into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error loading markdown file {file_path}: {e}")
            return []
    
    def save_document(self, content: str, filename: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save content as a document file.
        
        Args:
            content: Document content
            filename: Name for the file
            metadata: Optional metadata
            
        Returns:
            Path to saved file
        """
        try:
            file_path = self.sources_dir / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Saved document to {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error saving document {filename}: {e}")
            raise
    
    def _clean_web_content(self, content: str) -> str:
        """Clean web content for better processing."""
        # Remove excessive whitespace
        content = " ".join(content.split())
        
        # Remove common web artifacts
        content = content.replace("\\n", "\n")
        content = content.replace("\\t", "\t")
        
        # Limit length
        if len(content) > 10000:
            content = content[:10000] + "..."
        
        return content
    
    def get_document_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """Get statistics about loaded documents.
        
        Args:
            documents: List of documents
            
        Returns:
            Statistics dictionary
        """
        if not documents:
            return {"count": 0, "total_chars": 0, "avg_chars": 0, "sources": []}
        
        total_chars = sum(len(doc.page_content) for doc in documents)
        sources = list(set(doc.metadata.get("source", "Unknown") for doc in documents))
        
        return {
            "count": len(documents),
            "total_chars": total_chars,
            "avg_chars": total_chars // len(documents),
            "sources": sources
        }
