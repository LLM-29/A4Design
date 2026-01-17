"""
Configuration management for the UML generation system.
"""

from pydantic import BaseModel, Field


class SystemConfig(BaseModel):
    """System configuration for UML generation."""
    
    lmstudio_base_url: str = Field(
        default="http://localhost:1234/v1",
        description="LMStudio API endpoint"
    )
    decompose_model: str = Field(
        default="qwen3-vl-8b-instruct-mlx",
        description="Model for decomposition/analysis tasks"
    )
    generate_model: str = Field(
        default="mistralai/devstral-small-2-2512",
        description="Model for code generation tasks"
    )
    embedder_model: str = Field(
        default="BAAI/bge-large-en-v1.5",
        description="Embedder model for database retrieval/search tasks"
    )
    evaluation_embedder_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Embedder model for evaluation semantic similarity (optimized for STS tasks)"
    )
    db_path: str = Field(
        default="./../../data/uml_knowledge.db",
        description="Path to SQLite database"
    )
    shots_json_path: str = Field(
        default="./../../data/complete_shots.json",
        description="Path to few-shot examples"
    )
    plantuml_host: str = Field(
        default="http://localhost:8080",
        description="PlantUML server host"
    )
    max_iterations: int = Field(
        default=6,
        ge=1,
        description="Maximum workflow iterations"
    )
    max_tokens_decompose: int = Field(
        default=2048,
        description="Max tokens for decompose step"
    )
    max_tokens_generate: int = Field(
        default=2048,
        description="Max tokens for generate step"
    )
    max_tokens_critique: int = Field(
        default=2048,
        description="Max tokens for critique step"
    )
    max_tokens_reflect: int = Field(
        default=2048,
        description="Max tokens for reflect step"
    )
    max_tokens_refine: int = Field(
        default=2048,
        description="Max tokens for structure refine step"
    )
    temperature: float = Field(
        default=0.15,
        ge=0.0,
        le=2.0,
        description="Base temperature for LLM"
    )
    num_few_shots: int = Field(
        default=3,
        ge=0,
        description="Number of few-shot examples"
    )
    request_timeout: int = Field(
        default=5,
        ge=1,
        description="Timeout for PlantUML server requests"
    )
    llm_timeout: int = Field(
        default=600,
        ge=1,
        description="Timeout for LLM operations"
    )
    plateau_window: int = Field(
        default=3,
        ge=2,
        description="Number of iterations to consider for plateau detection"
    )
    plateau_threshold: float = Field(
        default=0.1,
        ge=0.0,
        description="Score delta threshold for plateau detection"
    )
