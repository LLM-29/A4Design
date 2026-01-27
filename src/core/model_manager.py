from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI

from src.core.models import TaskType
from src.core.logger import Logger


class ModelSpec:
    """Specification for a model to use for specific tasks."""
    
    def __init__(
        self,
        name: str,
        api_key: str,
        base_url: str,
        temperature: float,
        timeout: int,
        max_tokens: Optional[int] = None
    ):
        """
        Initialize model specification.
        
        Args:
            name: Model name/identifier
            api_key: API key for local models (use 'lm-studio' for local)
            base_url: Base URL for the API (default: OpenRouter)
            temperature: Sampling temperature
            timeout: Request timeout in seconds
            max_tokens: Maximum tokens to generate (None = model default)
        """
        self.name = name
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.timeout = timeout
        self.max_tokens = max_tokens
    
    def __repr__(self) -> str:
        return f"ModelSpec({self.name}) with temp={self.temperature}"


class ModelManager:
    """
    Manages dynamic loading and unloading of LLM instances.
    
    Allows using different models for different tasks to optimize
    performance, cost, and quality.
    """
    
    def __init__(
        self,
        task_model_mapping: Optional[Dict[TaskType, ModelSpec]] = None,
        default_model: Optional[ModelSpec] = None,
    ):
        """
        Initialize the model manager.
        
        Args:
            task_model_mapping: Mapping of task types to model specs
            default_model: Default model to use if no specific mapping exists
        """
        self.task_model_mapping = task_model_mapping or {}
        self.default_model = default_model
        
        Logger.log_info("ModelManager initialized")
        if self.task_model_mapping:
            Logger.log_info("Task-specific models configured:")
            for task, spec in self.task_model_mapping.items():
                Logger.log_info(f"  {task.value}: {spec.name}")
        else:
            Logger.log_info("Using default model for all tasks")
    
    def get_model(
        self,
        task_type: TaskType,
        **override_kwargs: Any
    ) -> ChatOpenAI:
        """
        Get an LLM instance for a specific task type.
        
        Args:
            task_type: The type of task requiring a model
            **override_kwargs: Optional overrides for model parameters
            
        Returns:
            Configured LLM instance for the task
        """
        # Determine which model spec to use
        model_spec = self.task_model_mapping.get(task_type, self.default_model)
        
        if model_spec is None:
            raise ValueError(
                f"No model specified for task {task_type} and no default model set"
            )
        
        # Base model parameters
        model_kwargs = {
            "temperature": model_spec.temperature,
            "timeout": override_kwargs.get("timeout", model_spec.timeout),
            "model_name": model_spec.name,
            "api_key": model_spec.api_key,
            "base_url": model_spec.base_url,
        }
    
        # Apply remaining overrides
        model_kwargs.update({k: v for k, v in override_kwargs.items() if k not in ["timeout"]})
        
        llm = ChatOpenAI(**model_kwargs)

        return llm


