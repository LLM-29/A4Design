"""
Agent nodes for the UML generation workflow.
"""

import json
import time
import hashlib
import logging
from typing import Dict, List, Any, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

try:
    from .config import SystemConfig
    from .models import (
        NodeNames,
        ClassExtractionResult,
        RelationshipExtractionResult,
        CritiqueReport,
        AgentState,
        Class,
        Relationship,
    )
    from .plantuml_tools import PlantUMLTool
    from .memory import MemoryManager
    from .model_manager import ModelManager, TaskType
except ImportError:
    from config import SystemConfig
    from models import (
        NodeNames,
        ClassExtractionResult,
        RelationshipExtractionResult,
        CritiqueReport,
        AgentState,
        Class,
        Relationship,
    )
    from plantuml_tools import PlantUMLTool
    from memory import MemoryManager
    from model_manager import ModelManager, TaskType

# Import prompts from parent package
import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from prompts import (
    CLASS_EXTRACTOR_SYSTEM,
    RELATIONSHIP_EXTRACTOR_SYSTEM,
    GENERATOR_SYSTEM,
    CRITIC_SYSTEM,
)

logger = logging.getLogger(__name__)


class UMLNodes:
    """
    Collection of agent nodes for the UML generation workflow.
    
    Each method represents a node in the LangGraph workflow and
    follows the pattern of taking AgentState and returning a dict
    with state updates.
    
    Supports dynamic model loading via ModelManager for using different
    models for different tasks (e.g., lightweight for analysis, powerful for code).
    """
    
    def __init__(
        self,
        plantuml_tool: PlantUMLTool,
        memory_manager: Optional[MemoryManager],
        config: SystemConfig,
        model_manager: ModelManager
    ):
        """
        Initialize UML nodes with required dependencies.
        
        Args:
            plantuml_tool: Tool for PlantUML validation
            memory_manager: long-term memory manager
            config: System configuration
            model_manager: Model manager for dynamic model loading
        """
        self.plantuml_tool = plantuml_tool
        self.memory_manager = memory_manager
        self.config = config
        self.model_manager = model_manager
        
        logger.info("UMLNodes initialized with DYNAMIC MODEL LOADING")
    
    def _get_model_for_task(self, task_type: TaskType) -> ChatOpenAI:
        """
        Get the appropriate model for a given task.
        
        Args:
            task_type: The type of task requiring a model
            
        Returns:
            ChatOpenAI instance from model manager
        """
        return self.model_manager.get_model(task_type)

    @staticmethod
    def _normalize_diagram(diagram: str) -> str:
        """Normalize diagram for consistent hashing (remove whitespace variations)."""
        lines = [line.strip() for line in diagram.strip().split('\n') if line.strip()]
        return '\n'.join(sorted(lines))  # Sort for order-independent comparison
    
    @staticmethod
    def _hash_diagram(diagram: str) -> str:
        """Create a hash of the diagram content for caching."""
        normalized = UMLNodes._normalize_diagram(diagram)
        return hashlib.md5(normalized.encode()).hexdigest()

    def _safe_invoke(self, runnable: Any, input_data: Any, **kwargs) -> Any:
        """
        Invoke a runnable (LLM or chain) with retry logic.
        
        Args:
            runnable: The runnable to invoke
            input_data: Input data for the runnable
            **kwargs: Additional keyword arguments
            
        Returns:
            Result from the runnable
            
        Raises:
            Exception: If all retries fail
        """
        max_retries = 3
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                return runnable.invoke(input_data, **kwargs)
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"LLM call failed (attempt {attempt+1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
        
        logger.error(f"Max retries reached for LLM call: {last_exception}")
        raise last_exception

    @staticmethod
    def _format_plan(classes: List[Class], relationships: List[Relationship]) -> str:
        """
        Format classes and relationships into a readable plan string.
        
        Args:
            classes: List of classes with attributes
            relationships: List of relationships between classes
            
        Returns:
            Formatted plan as a string
        """
        lines = ["## STRUCTURAL DECOMPOSITION\n"]
        
        # Format classes
        if classes:
            lines.append("### Classes:")
            for cls in classes:
                attrs_str = ", ".join([
                    f"{attr.name}: {attr.type}"
                    for attr in cls.attributes
                ])
                lines.append(
                    f"- {cls.name}" + (f" ({attrs_str})" if attrs_str else "")
                )
            lines.append("")
        
        # Format relationships
        if relationships:
            lines.append("### Relationships:")
            for rel in relationships:
                lines.append(
                    f"- {rel.source} --{rel.type}--> "
                    f"{rel.target}"
                )
        
        return "\n".join(lines)

    def retrieve(self, state: AgentState) -> Dict[str, Any]:
        """
        Retrieve relevant few-shot examples based on requirements.
        
        Args:
            state: Current workflow state
            
        Returns:
            Dict with 'examples' key containing formatted shots
        """
        logger.info(f"--- NODE: {NodeNames.RETRIEVE.upper()} ---")
        
        try:
            memories = self.memory_manager.retrieve_similar_diagrams(
                state["requirements"],
                limit=self.config.num_few_shots
            )
            
            formatted_shots = []
            for mem in memories:
                formatted_shots.append(
                    HumanMessage(content=f"Requirements:\n{mem['requirements']}")
                )
                
                meta = mem.get("metadata", {})
                plan = meta.get("plan", "No plan available.")
                
                assistant_content = (
                    f"1. DESIGN PLAN:\n{plan}\n\n"
                    f"2. PLANTUML DIAGRAM:\n```plantuml\n{mem['diagram']}\n```"
                )
                
                formatted_shots.append(
                    AIMessage(content=assistant_content)
                )
            
            logger.info(
                f"Retrieved {len(memories)} relevant examples from unified memory"
            )
            return {"examples": formatted_shots}
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return {"examples": []}

    def extract_classes(self, state: AgentState) -> Dict[str, Any]:
        """
        Extract classes and their attributes from requirements.
        
        Args:
            state: Current workflow state
            
        Returns:
            Dict with 'extracted_classes' update
        """
        logger.info(f"--- NODE: {NodeNames.EXTRACT_CLASSES.upper()} ---")
        
        messages = [
            SystemMessage(content=CLASS_EXTRACTOR_SYSTEM),
            HumanMessage(content=f"REQUIREMENTS:\n{state['requirements']}")
        ]
        
        try:
            llm = self._get_model_for_task(TaskType.DECOMPOSE)
            
            structured_llm = llm.bind(
                max_tokens=self.config.max_tokens_decompose
            ).with_structured_output(ClassExtractionResult)
            
            result: ClassExtractionResult = self._safe_invoke(structured_llm, messages)
            
            logger.info(f"Extracted {len(result.classes)} classes")
            
            return {"extracted_classes": result.classes}
            
        except Exception as e:
            logger.error(f"Class extraction failed: {e}")
            return {"extracted_classes": []}
    
    def extract_relationships(self, state: AgentState) -> Dict[str, Any]:
        """
        Extract relationships between the previously extracted classes.
        
        Args:
            state: Current workflow state
            
        Returns:
            Dict with 'plan' update containing formatted plan
        """
        logger.info(f"--- NODE: {NodeNames.EXTRACT_RELATIONSHIPS.upper()} ---")
        
        extracted_classes = state.get("extracted_classes", [])
        
        if not extracted_classes:
            logger.warning("No classes extracted; skipping relationship extraction")
            return {"plan": "## STRUCTURAL DECOMPOSITION\n\nNo classes found."}
        
        # Format class list for the LLM
        class_names = [cls.name for cls in extracted_classes]
        class_list_str = "\n".join([f"- {name}" for name in class_names])
        
        messages = [
            SystemMessage(content=RELATIONSHIP_EXTRACTOR_SYSTEM),
            HumanMessage(content=f"""REQUIREMENTS:
{state['requirements']}

EXTRACTED CLASSES:
{class_list_str}

Extract the relationships between these classes based on the requirements.""")
        ]
        
        try:
            llm = self._get_model_for_task(TaskType.DECOMPOSE)
            
            structured_llm = llm.bind(
                max_tokens=self.config.max_tokens_decompose
            ).with_structured_output(RelationshipExtractionResult)
            
            result: RelationshipExtractionResult = self._safe_invoke(
                structured_llm,
                messages
            )
            
            logger.info(f"Extracted {len(result.relationships)} relationships")
            
            # Format the complete plan
            formatted_plan = self._format_plan(extracted_classes, result.relationships)
            
            return {"plan": formatted_plan}
            
        except Exception as e:
            logger.error(f"Relationship extraction failed: {e}")
            # Return plan with just classes
            formatted_plan = self._format_plan(extracted_classes, [])
            return {"plan": formatted_plan}
        
    def generate(self, state: AgentState) -> Dict[str, Any]:
        """
        Generate PlantUML diagram using chain-of-thought reasoning.
        
        Args:
            state: Current workflow state
            
        Returns:
            Dict with 'current_diagram' and 'iterations' updates
        """
        logger.info(f"--- NODE: {NodeNames.GENERATE.upper()} ---")
        
        messages = [SystemMessage(content=GENERATOR_SYSTEM)]
        
        # Add few-shot examples if available
        if state.get("examples"):
            messages.extend(state["examples"])
            logger.debug(f"Added {len(state['examples'])} example messages")
            
        user_content = f"""
        # VALIDATED REQUIREMENTS
        {state['requirements']}

        # VALIDATED DESIGN PLAN
        {state['plan']}

        # TASK
        Render the PlantUML class diagram exactly from the design plan.
        """
        
        # Add syntax error feedback if we're retrying after syntax check failure
        if not state.get("syntax_valid", True) and state.get("error_message"):
            user_content += f"""
        
        # PREVIOUS ATTEMPT HAD SYNTAX ERROR
        {state['error_message']}
        
        Fix the syntax error and regenerate the diagram.
        """
            logger.info("Added syntax error feedback to generation prompt")
        
        # Add critique feedback if we're fixing semantic issues
        critique_report = state.get("current_validation")
        if critique_report and critique_report.findings:
            findings_summary = []
            for finding in critique_report.findings:
                finding_desc = (
                    f"- [{finding.category.value.upper()}] "
                    f"{finding.description}"
                )
                if finding.expected_correction:
                    finding_desc += f"\n  → FIX: {finding.expected_correction}"
                findings_summary.append(finding_desc)
            
            user_content += f"""

        # CRITIQUE FINDINGS TO ADDRESS
        The previous diagram had the following issues:

        {chr(10).join(findings_summary)}

        Apply ALL corrections above to fix the diagram.
        """
            logger.info(
                f"Added {len(critique_report.findings)} critique findings "
                "to generation prompt"
            )
        
        # Add previous diagram if available for reference
        if state.get("current_diagram") and (
            critique_report or not state.get("syntax_valid", True)
        ):
            user_content += f"""

        # PREVIOUS DIAGRAM (for reference)
        ```plantuml
        {state['current_diagram']}
        ```
        """
        
        messages.append(HumanMessage(content=user_content))

        logger.debug(f"Generation prompt messages: {messages}")
        
        try:
            # Get appropriate model for code generation task
            llm = self._get_model_for_task(TaskType.GENERATE)
            
            response = self._safe_invoke(
                llm,
                messages,
                max_tokens=self.config.max_tokens_generate
            )
            diagram = self.plantuml_tool.extract_plantuml(response.content)
            
            logger.info(f"Generation completed (iteration {state['iterations'] + 1})")
            return {
                "current_diagram": diagram,
                "iterations": state["iterations"] + 1,
            }
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {
                "current_diagram": f"Error: {str(e)}",
                "iterations": state["iterations"] + 1
            }

    def syntax_check(self, state: AgentState) -> Dict[str, Any]:
        """
        Validate PlantUML syntax through server.
        
        Args:
            state: Current workflow state
            
        Returns:
            Dict with 'syntax_valid' and optional 'error_message'
        """
        logger.info(f"--- NODE: {NodeNames.SYNTAX_CHECK.upper()} ---")
        
        try:
            result = self.plantuml_tool.check_syntax(
                state["current_diagram"],
                timeout=self.config.request_timeout
            )
            
            if result.is_valid:
                logger.info(f"Syntax valid. View at: {result.url}")
            else:
                logger.warning(f"Syntax error: {result.error}")
            
            return {
                "syntax_valid": result.is_valid,
                "error_message": result.error if not result.is_valid else None,
            }
            
        except Exception as e:
            logger.error(f"Syntax check failed: {e}")
            return {
                "syntax_valid": False,
                "error_message": f"Syntax check error: {str(e)}"
            }

    def critic(self, state: AgentState) -> Dict[str, Any]:
        """
        Critic node: produces a formal CritiqueReport.
        
        Args:
            state: Current workflow state
            
        Returns:
            Dict with 'current_validation' update
        """
        logger.info(f"--- NODE: {NodeNames.CRITIC.upper()} ---")

        try:
            requirements = state["requirements"]
            diagram = self.plantuml_tool.extract_plantuml(state["current_diagram"])

            messages = [
                SystemMessage(content=CRITIC_SYSTEM),
                HumanMessage(content=json.dumps({
                    "requirements": requirements,
                    "diagram": diagram
                }))
            ]

            # Get appropriate model for critique task
            llm = self._get_model_for_task(TaskType.CRITIQUE)
            
            structured_llm = llm.bind(
                max_tokens=self.config.max_tokens_critique
            ).with_structured_output(CritiqueReport)
            report: CritiqueReport = self._safe_invoke(structured_llm, messages)

            logger.info(f"Critic findings: {len(report.findings)} issues found")
            
            if report.findings:
                logger.info(f"Categories: {[f.category.value for f in report.findings]}")

            return {
                "current_validation": report,
            }

        except Exception as e:
            logger.error(f"Critic node failed: {e}")
            return {
                "logic_valid": False,
                "current_validation": None,
                "error_message": str(e)
            }
