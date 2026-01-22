"""
Configuration management for the UML generation system.
"""

from pydantic import BaseModel, Field
from os import getenv
import src.config as cfg

class SystemConfig(BaseModel):
    """System configuration for UML generation."""
    openrouter_base_url: str = Field(
        default=cfg.OPENROUTER_BASE_URL,
        description="Base URL for OpenAI-compatible API"
    )
    model: str = Field(
        default=cfg.GENERATE_MODEL,
        description="Model for decomposition/analysis tasks"
    )
    api_key: str = Field(
        default=getenv("OPENROUTER_API_KEY"),
        description="API key for decomposition model"
    )
    evaluation_embedder_model: str = Field(
        default=cfg.EVALUATION_EMBEDDER_MODEL,
        description="Embedder model for evaluation semantic similarity (optimized for STS tasks)"
    )
    db_path: str = Field(
        default=cfg.DATABASE,
        description="Path to FAISS index"
    )
    shots_json_path: str = Field(
        default=cfg.FEW_SHOT_EXAMPLES,
        description="Path to few-shot examples"
    )
    diagrams_json_path: str = Field(
        default=cfg.DIAGRAMS,
        description="Path to diagrams data"
    )
    test_exercises_path: str = Field(
        default=cfg.TEST_EXERCISES,
        description="Path to test exercises"
    )
    plantuml_host: str = Field(
        default=cfg.PLANTUML_HOST,
        description="PlantUML server host"
    )
    max_tokens: int = Field(
        default=cfg.MAX_TOKENS_GENERATE,
        description="Max tokens for generate step"
    )
    temperature: float = Field(
        default=cfg.TEMPERATURE_DECOMPOSE,
        ge=0.0,
        le=2.0,
        description="Base temperature for LLM"
    )
    num_few_shots: int = Field(
        default=cfg.NUM_FEW_SHOTS,
        ge=0,
        description="Number of few-shot examples"
    )
    evaluation_similarity_threshold: float = Field(
        default=cfg.EVALUATION_SIMILARITY_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for evaluation"
    )
    request_timeout: int = Field(
        default=5,
        ge=1,
        description="Timeout for PlantUML server requests"
    )
    llm_timeout: int = Field(
        default=15,
        ge=1,
        description="Timeout for LLM operations"
    )
