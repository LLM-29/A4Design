"""
Utility functions for the UML generation system.
"""

import os
import json

from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

from src.agents.multi_agent.config import SystemConfig
from src.agents.multi_agent.memory import MemoryManager
from src.agents.multi_agent.agents import UMLNodes
from src.core.model_manager import ModelManager, ModelSpec
from src.agents.multi_agent.workflow import create_critic_workflow, create_scoring_workflow
from src.agents.single_agent.agent import Node
from src.agents.single_agent.workflow import create_single_agent
from src.core.models import AgentState, SingleAgentState
from src.core.plantuml import PlantUMLTool
from src.core.models import EvaluationMode, TaskType
from src.core.logger import Logger


def load_test_exercises(json_path: str) -> List[Dict[str, Any]]:
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
    Logger.log_info(f"Loading test exercises from {json_path}")
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Test exercises file not found: {json_path}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        exercises = json.load(f)
    
    Logger.log_info(f"Loaded {len(exercises)} test exercises")
    return exercises


def initialize_multi_agent_system(
    cfg: SystemConfig,
) -> Any:
    """
    Initialize all system components.
    
    Args:
        config: Optional system configuration
        enable_long_term_memory: Whether to enable long-term memory
        
    Returns:
        Tuple of (nodes, compiled_workflow, config, memory_manager)
    """
    Logger.log_title("INITIALIZING UML GENERATION SYSTEM")
    
    try:
        Logger.log_models(cfg.decompose_model, cfg.generate_model)
        model_mgr = create_model_manager(cfg=cfg)
        
        Logger.log_info("Initializing PlantUML tool...")
        puml_tool = PlantUMLTool(cfg.plantuml_host)
        
        
        Logger.log_info(f"Initializing long-term memory with {cfg.embedder_model}...")
        memory_mgr = MemoryManager(
            embedder=SentenceTransformer(cfg.embedder_model),
            db_path=cfg.db_path,
            model_name=cfg.embedder_model
        )

        seeded_count = seed_memory_from_shots(
            memory_manager=memory_mgr,
            shots_json_path=cfg.diagrams_json_path,
            force_reseed=False
        )

        Logger.log_info("Long-term memory enabled")
        if seeded_count > 0:
            Logger.log_info(f"Seeded {seeded_count} few-shot examples into memory")
        
        Logger.log_info("Building LangGraph workflow...")
        nodes = UMLNodes(puml_tool, memory_mgr, cfg, model_mgr)

        if cfg.evaluation_mode == EvaluationMode.CRITIC:
            app = create_critic_workflow(nodes, cfg)
        else:
            app = create_scoring_workflow(nodes, cfg)
        
        Logger.log_title("SYSTEM INITIALIZATION COMPLETE")
        
        return app
        
    except Exception as e:
        Logger.log_error(f"System initialization failed: {e}")
        raise


def initialize_single_agent_system(
    cfg: SystemConfig,
) -> Any:
    """
    Initialize single agent system components.
    
    Args:
        cfg: System configuration
        
    Returns:
        Compiled workflow for single agent
    """
    Logger.log_title("INITIALIZING SINGLE AGENT UML GENERATION SYSTEM")
    
    try:
        Logger.log_models(cfg.decompose_model, cfg.generate_model)
        model_mgr = create_model_manager(cfg=cfg)
        
        Logger.log_info("Building single agent workflow...")
        nodes = Node(model_mgr)
        app = create_single_agent(nodes, cfg)
        
        Logger.log_title("SINGLE AGENT SYSTEM INITIALIZATION COMPLETE")
        
        return app
        
    except Exception as e:
        Logger.log_error(f"Single agent system initialization failed: {e}")
        raise


def seed_memory_from_shots(
    memory_manager: MemoryManager,
    shots_json_path: str,
    force_reseed: bool = False
) -> int:
    """
    Seed the memory database with few-shot examples from JSON file.
    Skips seeding if database already contains data (unless force_reseed=True).
    
    Args:
        memory_manager: MemoryManager instance to seed
        shots_json_path: Path to the few_shot_examples.json file
        force_reseed: If True, clears existing data and reseeds
        
    Returns:
        Number of shots seeded (0 if skipped)
    """
    Logger.log_title("SEEDING MEMORY FROM FEW-SHOT EXAMPLES")
    
    # Check if database already has data
    try:
        if memory_manager.vector_store is not None:
            existing_docs = memory_manager.vector_store.similarity_search("test", k=1)
            if existing_docs and not force_reseed:
                Logger.log_info(
                    f"Database already contains data ({len(existing_docs)} docs found)"
                )
                Logger.log_info(
                    "Skipping seeding operation. Set force_reseed=True to override."
                )
                return 0
        else:
            Logger.log_info("Database is empty, proceeding with seeding")
    except Exception as e:
        Logger.log_info(f"Database appears empty or uninitialized: {e}")
    
    if force_reseed:
        Logger.log_warning("Force reseed enabled - clearing existing memory")
        memory_manager.clear_memory()
    
    if not os.path.exists(shots_json_path):
        Logger.log_error(f"Shots file not found at {shots_json_path}")
        return 0
    
    Logger.log_info(f"Loading shots from {shots_json_path}")
    with open(shots_json_path, 'r', encoding='utf-8') as f:
        shots = json.load(f)
    
    shots = shots[:3]  # Limit to first 3 shots for seeding
    Logger.log_info(f"Found {len(shots)} shots to seed")
    
    # Seed each shot individually
    seeded_count = 0
    for shot in shots:
        requirements = shot["requirements"]
        diagram = shot["diagram"]
        
        metadata = {
            "plan": shot.get("plan"),
            "is_static": True,
            "title": shot.get("title", "Untitled")
        }
        
        Logger.log_info(f"  Processing: {metadata['title']}")
        
        # Use save_diagram which handles None vector_store
        memory_manager.save_diagram(
            requirements=requirements,
            diagram=diagram,
            metadata=metadata
        )
        seeded_count += 1
    
    if seeded_count > 0:
        Logger.log_title("MEMORY SEEDING COMPLETE")
    
    return seeded_count


def create_model_manager(
    cfg: SystemConfig
) -> ModelManager:
    """
    Create a model manager with sensible defaults.
    
    Args:
        cfg: System configuration object
        
    Returns:
        Configured ModelManager instance
    """
    # Define model specifications
    analysis_model = ModelSpec(
        name=cfg.decompose_model,
        api_key=cfg.api_key,
        base_url=cfg.openrouter_base_url,
        temperature=cfg.temperature_decompose,
        timeout=cfg.llm_timeout
    )
    

    code_model = ModelSpec(
        name=cfg.generate_model,
        api_key=cfg.api_key,
        base_url=cfg.openrouter_base_url,
        temperature=cfg.temperature_generation,
        timeout=cfg.llm_timeout
    )
    
    # Map tasks to models
    task_mapping = {
        TaskType.DECOMPOSE: analysis_model, 
        TaskType.GENERATE: code_model,       
        TaskType.CRITIQUE: analysis_model,       
    }
    
    return ModelManager(
        task_model_mapping=task_mapping,
        default_model=code_model,  # Default to code model if no mapping
    )


def create_initial_single_agent_state(requirements: str) -> SingleAgentState:
    """
    Create an initial state for the single agent workflow.
    
    Args:
        requirements: Software requirements text
        
    Returns:
        Initial SingleAgentState dictionary
    """
    return {
        "requirements": requirements,
        "current_diagram": None,
    }


def create_initial_state(requirements: str) -> AgentState:

    """
    Create an initial state for the workflow.
    
    Args:
        requirements: Software requirements text
        
    Returns:
        Initial AgentState dictionary
    """
    return {
        "requirements": requirements,
        "plan": None,
        "examples": [],
        "current_diagram": None,
        "best_diagram": None,
        "summary": None,
        "syntax_valid": False,
        "logic_valid": False,
        "iterations": 0,
        "error_message": None,
        "failed_attempts": [],
        "critique_cache": {},
        "plan_valid": False,
        "audit_feedback": None,
        "plan_audit_attempts": 0,
        "best_score": 0.0,
        "best_code": "",
        "current_validation": None,
        "audit_suggestions": None,
    }


def run_exercise(
    app: Any,
    requirements: str,
    exercise_name: str,
    is_single_agent: bool = False
) -> Any:
    """
    Run the workflow on a single exercise.
    
    Args:
        app: Compiled LangGraph workflow
        requirements: Software requirements text
        exercise_name: Name for logging purposes
        is_single_agent: Whether this is a single agent workflow
        
    Returns:
        Final workflow state
    """
    
    Logger.log_run_start(exercise_name, requirements)
    
    if is_single_agent:
        initial_state = create_initial_single_agent_state(requirements)
    else:
        initial_state = create_initial_state(requirements)
    
    try:
        final_output = app.invoke(initial_state, config={"recursion_limit": 130})
        Logger.log_generation(final_output)
        return final_output
    except Exception as e:
        Logger.log_error(f"Workflow execution failed: {e}")
        raise


