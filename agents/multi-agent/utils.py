"""
Utility functions for the UML generation system.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple

from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer

try:
    from .config import SystemConfig
    from .plantuml_tools import PlantUMLTool
    from .memory import MemoryManager, seed_memory_from_shots
    from .agents import UMLNodes
    from .workflow import create_uml_graph
    from .model_manager import ModelManager, create_default_model_manager
except ImportError:
    from config import SystemConfig
    from plantuml_tools import PlantUMLTool
    from memory import MemoryManager, seed_memory_from_shots
    from agents import UMLNodes
    from workflow import create_uml_graph
    from model_manager import ModelManager, create_default_model_manager

logger = logging.getLogger(__name__)


def load_test_exercises(
    json_path: str = "./../../data/test_exercises.json"
) -> List[Dict[str, Any]]:
    """
    Load test exercises from JSON file.
    
    Args:
        json_path: Path to test exercises JSON
        
    Returns:
        List of exercise dictionaries
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If JSON is invalid
    """
    logger.info(f"Loading test exercises from {json_path}")
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Test exercises file not found: {json_path}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        exercises = json.load(f)
    
    logger.info(f"Loaded {len(exercises)} test exercises")
    return exercises


def initialize_system(
    config: Optional[SystemConfig] = None,
    enable_long_term_memory: bool = True
) -> Tuple[UMLNodes, Any, SystemConfig, Optional[MemoryManager]]:
    """
    Initialize all system components.
    
    Args:
        config: Optional system configuration
        enable_long_term_memory: Whether to enable long-term memory
        
    Returns:
        Tuple of (nodes, compiled_workflow, config, memory_manager)
    """
    cfg = config or SystemConfig()
    logger.info("="*60)
    logger.info("INITIALIZING UML GENERATION SYSTEM")
    logger.info("="*60)
    
    try:
        logger.info("Using DYNAMIC MODEL LOADING")
        logger.info(f"  Decompose model: {cfg.decompose_model}")
        logger.info(f"  Generate model:  {cfg.generate_model}")
        
        model_mgr = create_default_model_manager(
            decompose_model=cfg.decompose_model,
            generate_model=cfg.generate_model,
            base_url=cfg.lmstudio_base_url,
            temperature=cfg.temperature,
            timeout=cfg.llm_timeout
        )
        
        logger.info("Initializing PlantUML tool...")
        puml_tool = PlantUMLTool(cfg.plantuml_host)
        
        memory_mgr = None
        if enable_long_term_memory:
            logger.info(f"Initializing long-term memory with {cfg.embedder_model}...")

            dims = 1024 if "large" in cfg.embedder_model.lower() else 384
            
            memory_mgr = MemoryManager(
                embedder=SentenceTransformer(cfg.embedder_model),
                db_path=cfg.db_path,
                embedding_dims=dims
            )

            seeded_count = seed_memory_from_shots(
                memory_manager=memory_mgr,
                shots_json_path=cfg.shots_json_path,
                force_reseed=False
            )

            logger.info("Long-term memory (SQLite + sqlite-vec) enabled")

            if seeded_count > 0:
                logger.info(f"Seeded {seeded_count} few-shot examples into memory")
        else:
            logger.info("Long-term memory disabled")
        
        logger.info("Building LangGraph workflow...")
        nodes = UMLNodes(puml_tool, memory_mgr, cfg, model_mgr)
        app = create_uml_graph(nodes, cfg)
        
        logger.info("="*60)
        logger.info("SYSTEM INITIALIZED SUCCESSFULLY")
        logger.info("="*60)
        
        return nodes, app, cfg, memory_mgr
        
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        raise
