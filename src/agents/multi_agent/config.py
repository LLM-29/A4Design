"""
Configuration management for the UML generation system.
"""

from pydantic import BaseModel, Field
from os import getenv
import src.config as cfg

class SystemConfig(BaseModel):
    """System configuration for UML generation."""
    evaluation_mode: str = Field(
        default="critic",
        description="Evaluation mode (critic or scorer)"
    )
    openrouter_base_url: str = Field(
        default=cfg.OPENROUTER_BASE_URL,
        description="Base URL for OpenAI-compatible API"
    )
    decompose_model: str = Field(
        default=cfg.DECOMPOSE_MODEL,
        description="Model for decomposition/analysis tasks"
    )
    api_key: str = Field(
        default=getenv("OPENROUTER_API_KEY"),
        description="API key for decomposition model"
    )
    generate_model: str = Field(
        default=cfg.GENERATE_MODEL,
        description="Model for code generation tasks"
    )
    embedder_model: str = Field(
        default=cfg.EMBEDDER_MODEL,
        description="Embedder model for database retrieval/search tasks"
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
    max_tokens_decompose: int = Field(
        default=cfg.MAX_TOKENS_DECOMPOSE,
        description="Max tokens for decompose step"
    )
    max_tokens_generate: int = Field(
        default=cfg.MAX_TOKENS_GENERATE,
        description="Max tokens for generate step"
    )
    max_tokens_critique: int = Field(
        default=cfg.MAX_TOKENS_CRITIQUE,
        description="Max tokens for critique step"
    )
    max_tokens_scoring: int = Field(
        default=cfg.MAX_TOKENS_SCORING,
        description="Max tokens for scoring step"
    )
    temperature_generation: float = Field(
        default=cfg.TEMPERATURE_GENERATION,
        ge=0.0,
        le=2.0,
        description="Base temperature for LLM"
    )
    temperature_decompose: float = Field(
        default=cfg.TEMPERATURE_DECOMPOSE,
        ge=0.0,
        le=2.0,
        description="Temperature for decomposition model"
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
    convergence_similarity_threshold: float = Field(
        default=cfg.CONVERGENCE_SIMILARITY_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for convergence detection in logical fixer"
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
    scoring_threshold: float = Field(
        default=cfg.SCORE_THRESHOLD,
        ge=0.0,
        le=5.0,
        description="Scoring threshold for diagram quality"
    )
