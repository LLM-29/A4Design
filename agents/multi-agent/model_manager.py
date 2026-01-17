"""
Dynamic model management for loading and unloading LLMs based on task.

This module enables using different models for different tasks:
- Lighter models for decomposition/analysis (finding requirements/relationships)
- More powerful models for code generation (PlantUML generation)

Supports both local models (via LM Studio) and cloud models (OpenAI, etc.)
"""

import logging
from typing import Optional, Dict, Any
from enum import Enum

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    """Types of tasks that may require different models."""
    DECOMPOSE = "decompose"  # Analyzing requirements, extracting structure
    GENERATE = "generate"    # Generating PlantUML code
    CRITIQUE = "critique"    # Evaluating diagram quality
    REFLECT = "reflect"      # Reflecting on feedback
    REFINE = "refine"        # Refining structural plans
    AUDIT = "audit"          # Auditing plans for completeness


class ModelSpec:
    """Specification for a model to use for specific tasks."""
    
    def __init__(
        self,
        name: str,
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "lm-studio",
        temperature: float = 0.15,
        timeout: int = 600,
        max_tokens: Optional[int] = None
    ):
        """
        Initialize model specification.
        
        Args:
            name: Model name/identifier
            base_url: API endpoint URL
            api_key: API key (use 'lm-studio' for local)
            temperature: Sampling temperature
            timeout: Request timeout in seconds
            max_tokens: Maximum tokens to generate (None = model default)
        """
        self.name = name
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        self.max_tokens = max_tokens
    
    def __repr__(self) -> str:
        return f"ModelSpec({self.name} @ {self.base_url})"


class ModelManager:
    """
    Manages dynamic loading and unloading of LLM instances.
    
    Allows using different models for different tasks to optimize
    performance, cost, and quality. Models are loaded on-demand and
    can be cached for reuse.
    """
    
    def __init__(
        self,
        task_model_mapping: Optional[Dict[TaskType, ModelSpec]] = None,
        default_model: Optional[ModelSpec] = None,
        cache_models: bool = True
    ):
        """
        Initialize the model manager.
        
        Args:
            task_model_mapping: Mapping of task types to model specs
            default_model: Default model to use if no specific mapping exists
            cache_models: Whether to cache loaded models for reuse
        """
        self.task_model_mapping = task_model_mapping or {}
        self.default_model = default_model
        self.cache_models = cache_models
        self._model_cache: Dict[str, ChatOpenAI] = {}
        
        logger.info("ModelManager initialized")
        if self.task_model_mapping:
            logger.info("Task-specific models configured:")
            for task, spec in self.task_model_mapping.items():
                logger.info(f"  {task.value}: {spec.name}")
        else:
            logger.info("Using default model for all tasks")
    
    def get_model(
        self,
        task_type: TaskType,
        **override_kwargs: Any
    ) -> ChatOpenAI:
        """
        Get an LLM instance for a specific task type.
        
        Models are loaded dynamically and cached if caching is enabled.
        Previously loaded models are reused to avoid reloading overhead.
        
        Args:
            task_type: The type of task requiring a model
            **override_kwargs: Optional overrides for model parameters
            
        Returns:
            Configured ChatOpenAI instance for the task
        """
        # Determine which model spec to use
        model_spec = self.task_model_mapping.get(task_type, self.default_model)
        
        if model_spec is None:
            raise ValueError(
                f"No model specified for task {task_type} and no default model set"
            )
        
        # Create cache key
        cache_key = f"{task_type.value}:{model_spec.name}"
        
        # Check cache first
        if self.cache_models and cache_key in self._model_cache:
            logger.debug(f"Using cached model for {task_type.value}: {model_spec.name}")
            return self._model_cache[cache_key]
        
        # Load model
        logger.info(f"Loading model for {task_type.value}: {model_spec.name}")
        
        model_kwargs = {
            "base_url": model_spec.base_url,
            "api_key": model_spec.api_key,
            "model": model_spec.name,
            "temperature": model_spec.temperature,
            "timeout": model_spec.timeout,
        }
        
        # Apply overrides
        model_kwargs.update(override_kwargs)
        
        llm = ChatOpenAI(**model_kwargs)
        
        # Cache if enabled
        if self.cache_models:
            self._model_cache[cache_key] = llm
            logger.debug(f"Cached model: {cache_key}")
        
        return llm
    
    def unload_model(self, task_type: TaskType) -> bool:
        """
        Unload (remove from cache) a specific model.
        
        Args:
            task_type: Task type whose model should be unloaded
            
        Returns:
            True if model was unloaded, False if not in cache
        """
        model_spec = self.task_model_mapping.get(task_type)
        if model_spec is None:
            return False
        
        cache_key = f"{task_type.value}:{model_spec.name}"
        
        if cache_key in self._model_cache:
            del self._model_cache[cache_key]
            logger.info(f"Unloaded model: {cache_key}")
            return True
        
        return False
    
    def unload_all(self) -> None:
        """Unload all cached models."""
        count = len(self._model_cache)
        self._model_cache.clear()
        logger.info(f"Unloaded {count} cached models")
    
    def get_loaded_models(self) -> list[str]:
        """Get list of currently loaded (cached) model keys."""
        return list(self._model_cache.keys())


def create_default_model_manager(
    decompose_model: str = "qwen3-vl-8b-instruct-mlx",
    generate_model: str = "mistralai/devstral-small-2-2512",
    base_url: str = "http://localhost:1234/v1",
    temperature: float = 0.15,
    timeout: int = 600
) -> ModelManager:
    """
    Create a model manager with sensible defaults.
    
    Args:
        decompose_model: Model for analysis/decomposition tasks
        generate_model: Model for code generation tasks
        base_url: LM Studio base URL
        temperature: Sampling temperature
        timeout: Request timeout
        
    Returns:
        Configured ModelManager instance
    """
    # Define model specifications
    analysis_model = ModelSpec(
        name=decompose_model,
        base_url=base_url,
        temperature=temperature,
        timeout=timeout
    )
    
    code_model = ModelSpec(
        name=generate_model,
        base_url=base_url,
        temperature=temperature,
        timeout=timeout
    )
    
    # Map tasks to models
    task_mapping = {
        TaskType.DECOMPOSE: analysis_model,  # Use lighter model for analysis
        TaskType.GENERATE: code_model,       # Use code model for generation
        TaskType.CRITIQUE: analysis_model,       # Code model for code critique
        TaskType.REFLECT: analysis_model,    # Analysis model for reflection
        TaskType.REFINE: analysis_model,     # Analysis model for refinement
        TaskType.AUDIT: analysis_model,      # Analysis model for auditing
    }
    
    return ModelManager(
        task_model_mapping=task_mapping,
        default_model=code_model,  # Default to code model if no mapping
        cache_models=True
    )
