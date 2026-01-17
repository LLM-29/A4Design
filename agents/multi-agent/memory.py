"""
Long-term memory management using SQLite and vector search.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

# sqlite-vec expects the standard library sqlite3 module.
# Older versions of this notebook replaced sqlite3 with sqlean via sys.modules; undo that if present.
if "sqlite3" in sys.modules and "sqlean" in sys.modules and sys.modules.get("sqlite3") is sys.modules.get("sqlean"):
    del sys.modules["sqlite3"]
import sqlite3

from langchain_core.documents import Document
from langchain_community.vectorstores import SQLiteVec
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Manages long-term memory for UML diagram generation using LangChain's SQLiteVec.
    
    Supports semantic search to find similar past solutions.
    """
    
    def __init__(
        self,
        embedder: SentenceTransformer,
        db_path: str = "./../../data/uml_knowledge.db",
        embedding_dims: int = 1024
    ):
        """
        Initialize memory manager with LangChain SQLiteVec.
        
        Args:
            embedder: SentenceTransformer model for semantic search
            db_path: Path to the SQLite database file
            embedding_dims: Dimensions of the embeddings
        """
        self.embedder = embedder
        self.db_path = db_path
        self.embedding_dims = embedding_dims
        
        # Create embedding function
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=(
                embedder.model_name
                if hasattr(embedder, 'model_name')
                else "BAAI/bge-large-en-v1.5"
            ),
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Create directory and connection
        os.makedirs(
            os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else ".",
            exist_ok=True
        )
        
        # Create connection using sqlean instead of default sqlite3
        # (which lacks extension support on macOS)
        try:
            import sqlean
            import sqlite_vec
            
            self.connection = sqlean.connect(self.db_path)
            self.connection.row_factory = sqlean.Row
            self.connection.enable_load_extension(True)
            sqlite_vec.load(self.connection)
            self.connection.enable_load_extension(False)
            logger.info(
                "Used sqlean for SQLite connection (extension support enabled)"
            )
        except ImportError:
            logger.warning(
                "sqlean not found, falling back to SQLiteVec.create_connection "
                "(may fail on macOS)"
            )
            self.connection = SQLiteVec.create_connection(db_file=self.db_path)
        
        # Initialize vector store with connection
        self.vector_store = SQLiteVec(
            table="uml_memories",
            connection=self.connection,
            embedding=self.embedding_function
        )
        
        logger.info(
            f"MemoryManager initialized with LangChain SQLiteVec at {db_path} "
            f"(dims={embedding_dims})"
        )

    def save_diagram(
        self,
        requirements: str,
        diagram: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Save a validated diagram to SQLite long-term memory.
        
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
        
        ids = self.vector_store.add_documents([doc])
        
        logger.info("Diagram saved to SQLite memory using LangChain SQLiteVec")
        return ids[0] if ids else 0
    
    def retrieve_similar_diagrams(
        self,
        requirements: str,
        limit: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Retrieve similar diagrams from SQLite memory using vector search.
        
        Args:
            requirements: Requirements text to search for
            limit: Maximum number of results
            
        Returns:
            List of similar diagram records
        """
        try:
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
            
            logger.info(f"Retrieved {len(diagrams)} similar diagrams from SQLite")
            return diagrams
            
        except Exception as e:
            logger.warning(f"Memory retrieval failed: {e}")
            return []
    
    def clear_memory(self) -> None:
        """Clear all memory from the database and reinitialize."""
        try:
            # Close existing connection
            if hasattr(self, 'connection'):
                self.connection.close()
            
            # Remove database file
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            
            # Recreate connection and vector store
            import sqlean
            import sqlite_vec
            
            self.connection = sqlean.connect(self.db_path)
            self.connection.row_factory = sqlean.Row
            self.connection.enable_load_extension(True)
            sqlite_vec.load(self.connection)
            self.connection.enable_load_extension(False)
            
            self.vector_store = SQLiteVec(
                table="uml_memories",
                connection=self.connection,
                embedding=self.embedding_function
            )

            logger.info("Memory cleared and reinitialized")
        except Exception as e:
            logger.error(f"Failed to clear memory: {e}")


def seed_memory_from_shots(
    memory_manager: MemoryManager,
    shots_json_path: str = "./../data/complete_shots.json",
    force_reseed: bool = False
) -> int:
    """
    Seed the memory database with few-shot examples from JSON file.
    Skips seeding if database already contains data (unless force_reseed=True).
    
    Args:
        memory_manager: MemoryManager instance to seed
        shots_json_path: Path to the complete_shots.json file
        force_reseed: If True, clears existing data and reseeds
        
    Returns:
        Number of shots seeded (0 if skipped)
    """
    logger.info("="*60)
    logger.info("CHECKING MEMORY SEEDING STATUS")
    logger.info("="*60)
    
    # Check if database already has data
    try:
        existing_docs = memory_manager.vector_store.similarity_search("test", k=1)
        if existing_docs and not force_reseed:
            logger.info(
                f"Database already contains data ({len(existing_docs)} docs found)"
            )
            logger.info(
                "Skipping seeding operation. Set force_reseed=True to override."
            )
            return 0
    except Exception as e:
        logger.info(f"Database appears empty or uninitialized: {e}")
    
    if force_reseed:
        logger.warning("Force reseed enabled - clearing existing memory")
        memory_manager.clear_memory()
    
    if not os.path.exists(shots_json_path):
        logger.error(f"Shots file not found at {shots_json_path}")
        return 0
    
    logger.info(f"Loading shots from {shots_json_path}")
    with open(shots_json_path, 'r', encoding='utf-8') as f:
        shots = json.load(f)
    
    logger.info(f"Found {len(shots)} shots to seed")
    
    # Prepare documents
    documents = []
    for shot in shots:
        requirements = shot["requirements"]
        diagram = shot["diagram"]
        
        metadata = {
            "diagram": diagram,
            "timestamp": datetime.now().isoformat(),
            "plan": shot.get("plan"),
            "is_static": True,
            "title": shot.get("title", "Untitled")
        }
        
        logger.info(f"  Processing: {metadata['title']}")
        
        doc = Document(
            page_content=requirements,
            metadata=metadata
        )
        documents.append(doc)
    
    if documents:
        memory_manager.vector_store.add_documents(documents)
        logger.info("="*60)
        logger.info(f"✓ Successfully seeded {len(documents)} shots to memory")
        logger.info("="*60)
        return len(documents)
    
    return 0
