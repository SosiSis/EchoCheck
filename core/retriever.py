"""Document retrieval system for RAG-Guardian."""

import logging
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from core.embeddings import EmbeddingManager
from utils.config import config

logger = logging.getLogger(__name__)

class DocumentRetriever:
    """Handles document storage and retrieval using ChromaDB."""
    
    def __init__(self, collection_name: str = "rag_guardian_docs"):
        """Initialize the retriever.
        
        Args:
            collection_name: Name of the ChromaDB collection
        """
        self.collection_name = collection_name
        self.embedding_manager = EmbeddingManager()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        self._vectorstore = None
    
    @property
    def vectorstore(self):
        """Get or create the vector store."""
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_manager.embeddings,
                persist_directory=config.CHROMA_PERSIST_DIRECTORY
            )
        return self._vectorstore
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store.
        
        Args:
            documents: List of documents to add
            
        Returns:
            List of document IDs
        """
        try:
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Add to vector store
            ids = self.vectorstore.add_documents(chunks)
            
            # Persist the changes
            self.vectorstore.persist()
            
            logger.info(f"Added {len(chunks)} document chunks to vector store")
            return ids
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def retrieve_documents(
        self, 
        query: str, 
        k: int = None, 
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            filter_dict: Optional metadata filter
            
        Returns:
            List of relevant documents
        """
        try:
            k = k or config.TOP_K_RETRIEVAL
            
            # Perform similarity search
            if filter_dict:
                docs = self.vectorstore.similarity_search(
                    query, k=k, filter=filter_dict
                )
            else:
                docs = self.vectorstore.similarity_search(query, k=k)
            
            logger.info(f"Retrieved {len(docs)} documents for query: {query[:50]}...")
            return docs
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            raise
    
    def retrieve_with_scores(
        self, 
        query: str, 
        k: int = None
    ) -> List[tuple[Document, float]]:
        """Retrieve documents with similarity scores.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of (document, score) tuples
        """
        try:
            k = k or config.TOP_K_RETRIEVAL
            
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            logger.info(f"Retrieved {len(results)} documents with scores")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving documents with scores: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the document collection."""
        try:
            collection = self.vectorstore._collection
            count = collection.count()
            
            return {
                "document_count": count,
                "collection_name": self.collection_name
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"document_count": 0, "collection_name": self.collection_name}
    
    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        try:
            # Delete the collection and recreate it
            self.vectorstore.delete_collection()
            self._vectorstore = None  # Force recreation
            logger.info("Cleared document collection")
            
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            raise
