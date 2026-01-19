"""
Data models and type definitions for the UML generation system.
"""

import operator
from enum import Enum
from typing import Annotated, List, TypedDict, Optional, Dict, Any
from pydantic import BaseModel, Field, computed_field


class AgentMode:
    SINGLE = "single"
    MULTI = "multi"


class EvaluationMode:
    CRITIC = "critic"
    SCORER = "scorer"


class TaskType(str, Enum):
    """Types of tasks that may require different models."""
    DECOMPOSE = "decompose"  # Analyzing requirements, extracting classes/relationships
    GENERATE = "generate"    # Generating PlantUML code
    CRITIQUE = "critique"    # Evaluating diagram quality


class NodeNames(str, Enum):
    """Enum for node names to avoid string literals."""
    RETRIEVE = "retrieve"
    EXTRACT_CLASSES = "extract_classes"
    EXTRACT_RELATIONSHIPS = "extract_relationships"
    GENERATE = "generate"
    SYNTAX_CHECK = "syntax_check"
    CRITIC = "critic"
    SYNTAX_FIXER = "syntax_fixer"
    LOGICAL_FIXER = "logical_fixer"
    SCORER = "scorer"


class Attribute(BaseModel):
    """Model for a class attribute."""
    name: str = Field(description="Attribute name")
    type: str = Field(description="Attribute type")


class Class(BaseModel):
    """Model for a UML class."""
    name: str = Field(description="Class name")
    attributes: List[Attribute] = Field(
        default_factory=list,
        description="List of class attributes"
    )


class Relationship(BaseModel):
    """Model for a relationship between classes."""
    source: str = Field(description="Source class name")
    target: str = Field(description="Target class name")
    type: str = Field(
        description="Relationship type (e.g., association, composition, inheritance)"
    )


class ClassExtractionResult(BaseModel):
    """Structured output from the class extraction step."""
    thought: str = Field(
        description="Detailed step-by-step reasoning about domain entities and constraints."
    )
    classes: List[Class] = Field(
        default_factory=list,
        description="List of identified classes with their attributes"
    )


class RelationshipExtractionResult(BaseModel):
    """Structured output from the relationship extraction step."""
    thought: str = Field(
        description="Deliberation on relationship types (e.g. Association vs Composition) based on lifecycle."
    )
    relationships: List[Relationship] = Field(
        default_factory=list,
        description="List of relationships between classes"
    )


class PlantUMLResult(BaseModel):
    """Result from PlantUML validation."""
    is_valid: bool = Field(description="Whether the PlantUML syntax is valid")
    error: Optional[str] = Field(
        default=None,
        description="Error message if validation failed"
    )
    url: Optional[str] = Field(
        default=None,
        description="URL to view the diagram"
    )
    svg_url: Optional[str] = Field(
        default=None,
        description="URL to view the diagram as SVG"
    )


class FindingCategory(str, Enum):
    """Enum for finding categories based on conceptual error taxonomy."""
    classes = "classes"
    attributes = "attributes"
    relationships = "relationships"


class CritiqueFinding(BaseModel):
    """Model for a single critique finding."""
    category: FindingCategory = Field(description="Error category")
    description: str = Field(description="Clear description of the issue")
    expected_correction: str = Field(description="How to fix this issue")


class CritiqueReport(BaseModel):
    """Complete critique report with findings."""
    findings: List[CritiqueFinding] = Field(
        default_factory=list,
        description="List of issues found in the diagram"
    )

    @computed_field
    @property
    def is_valid(self) -> bool:
        """Check if the diagram is valid (no findings)."""
        return len(self.findings) == 0


class AgentState(TypedDict):
    """Shared state for the LangGraph workflow."""
    requirements: str
    plan: Optional[str]
    extracted_classes: Optional[List[Class]]
    examples: List[Dict[str, str]]
    current_diagram: Optional[str]
    best_diagram: Optional[str]
    syntax_valid: bool
    logic_valid: bool
    error_message: Optional[str]
    best_score: float
    best_code: str
    current_validation: Optional[CritiqueReport]
    failed_attempts: Annotated[List[Dict[str, Any]], operator.add]
    iterations: int
    critique_cache: Dict[str, Dict[str, Any]]


class EvaluationMetrics(BaseModel):
    """Container for evaluation metrics."""
    precision: float = Field(ge=0.0, le=1.0, description="Precision score")
    recall: float = Field(ge=0.0, le=1.0, description="Recall score")
    f1: float = Field(ge=0.0, le=1.0, description="F1 score")
    
    def __str__(self) -> str:
        return f"P={self.precision:.2f}, R={self.recall:.2f}, F1={self.f1:.2f}"
