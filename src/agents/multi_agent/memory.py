"""
Long-term memory management using FAISS and vector search.
"""

import os
import pickle
import torch

from typing import Dict, List, Any, Optional
from datetime import datetime
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

from src.core.logger import Logger


class MemoryManager:
    """
    Manages long-term memory for UML diagram generation using LangChain's FAISS.
    
    Supports semantic search to find similar past solutions.
    """
    
    def __init__(
        self,
        embedder: SentenceTransformer,
        db_path: str,
        model_name: str,
    ):
        """
        Initialize memory manager with LangChain FAISS.
        
        Args:
            embedder: SentenceTransformer model for semantic search
            db_path: Path to the FAISS index file
            model_name: Name of the HuggingFace model to use for embeddings
        """
        self.embedder = embedder
        self.db_path = db_path

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        
        # Create embedding function
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Load or create FAISS vector store
        index_file = os.path.join(db_path, "index.faiss")
        if os.path.exists(index_file):
            self.vector_store = FAISS.load_local(
                db_path,
                self.embedding_function,
                allow_dangerous_deserialization=True
            )
            Logger.log_info(f"MemoryManager loaded existing FAISS index from {db_path}")
        else:
            # Create empty vector store
            self.vector_store = None
            Logger.log_info(f"MemoryManager initialized with empty FAISS index")

    def save_diagram(
        self,
        requirements: str,
        diagram: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a validated diagram to FAISS long-term memory.
        
        Args:
            requirements: Original requirements text
            diagram: PlantUML diagram code
            metadata: Optional metadata
            
        Returns:
            ID of the stored record
        """
        timestamp = datetime.now().isoformat()
        
        full_metadata = metadata or {}
        full_metadata.update({
            "diagram": diagram,
            "timestamp": timestamp
        })
        
        doc = Document(
            page_content=requirements,
            metadata=full_metadata
        )
        
        # Create vector store if it doesn't exist
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents([doc], self.embedding_function)
        else:
            self.vector_store.add_documents([doc])
        
        # Save to disk
        self.vector_store.save_local(self.db_path)
        
        Logger.log_info("Diagram saved to FAISS memory")
        return timestamp
    
    def retrieve_similar_diagrams(
        self,
        requirements: str,
        limit: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Retrieve similar diagrams from FAISS memory using vector search.
        
        Args:
            requirements: Requirements text to search for
            limit: Maximum number of results
            
        Returns:
            List of similar diagram records
        """
        try:
            # Check if vector store exists
            if self.vector_store is None:
                Logger.log_info("No memories stored yet, returning empty list")
                return []
            
            results = self.vector_store.similarity_search(requirements, k=limit)
            
            diagrams = []
            for doc in results:
                diagrams.append({
                    "requirements": doc.page_content,
                    "diagram": doc.metadata.get("diagram", ""),
                    "timestamp": doc.metadata.get("timestamp", ""),
                    "metadata": {
                        k: v for k, v in doc.metadata.items()
                        if k not in ["diagram", "timestamp"]
                    }
                })
            
            Logger.log_info(f"Retrieved {len(diagrams)} similar diagrams from FAISS")
            return diagrams
            
        except Exception as e:
            Logger.log_warning(f"Memory retrieval failed: {e}")
            return []
    
    def clear_memory(self) -> None:
        """Clear all memory from the database and reinitialize."""
        try:
            # Remove FAISS files
            if os.path.exists(self.db_path):
                for file in os.listdir(self.db_path):
                    file_path = os.path.join(self.db_path, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                os.rmdir(self.db_path)
            
            # Reset vector store
            self.vector_store = None
            
            Logger.log_info("Memory cleared and reinitialized")
        except Exception as e:
            Logger.log_error(f"Failed to clear memory: {e}")


