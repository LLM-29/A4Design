"""
Agent nodes for the UML generation workflow.
"""

import json
import Levenshtein

from typing import Dict, List, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from src.agents.multi_agent.config import SystemConfig
from src.core.models import (
    NodeNames,
    ClassExtractionResult,
    RelationshipExtractionResult,
    CritiqueReport,
    AgentState,
    Class,
    Relationship,
    ScoredCritiqueReport
)
from src.core.plantuml import PlantUMLTool
from src.core.logger import Logger
from src.core.invoke import safe_invoke
from src.agents.multi_agent.memory import MemoryManager
from src.core.model_manager import ModelManager, TaskType
from src.core.few_shot_loader import FewShotLoader
from src.core.prompts import (
    CLASS_EXTRACTOR_SYSTEM,
    RELATIONSHIP_EXTRACTOR_SYSTEM,
    GENERATOR_SYSTEM,
    CRITIC_SYSTEM,
    PLANTUML_SYNTAX_CHECKER_SYSTEM,
    PLANTUML_LOGICAL_FIXER_SYSTEM,
    SCORER_SYSTEM
)



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
        self.few_shot_loader = FewShotLoader(
            examples_path=config.shots_json_path,
            diagrams_path=config.diagrams_json_path
        )
        
        Logger.log_info("UMLNodes initialized with DYNAMIC MODEL LOADING")
    
    def _get_model_for_task(self, task_type: TaskType, **override_kwargs) -> ChatOpenAI:
        """
        Get the appropriate model for a given task.
        
        Args:
            task_type: The type of task requiring a model
            
        Returns:
            LLM instance from model manager
        """
        return self.model_manager.get_model(task_type, **override_kwargs)

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
        Logger.log_info(f"--- NODE: {NodeNames.RETRIEVE.upper()} ---")
        
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
            
            Logger.log_info(
                f"Retrieved {len(memories)} relevant examples from unified memory"
            )
            return {"examples": formatted_shots}
        except Exception as e:
            Logger.log_error(f"Retrieval failed: {e}")
            return {"examples": []}

    def extract_classes(self, state: AgentState) -> Dict[str, Any]:
        """
        Extract classes and their attributes from requirements.
        
        Args:
            state: Current workflow state
            
        Returns:
            Dict with 'extracted_classes' update
        """
        Logger.log_info(f"--- NODE: {NodeNames.EXTRACT_CLASSES.upper()} ---")
        
        # Format few-shot examples for class extraction
        few_shots = self.few_shot_loader.format_for_class_extraction(
            num_examples=self.config.num_few_shots
        )
        
        messages = [
            SystemMessage(content=CLASS_EXTRACTOR_SYSTEM),
            HumanMessage(content=f"""
            # EXAMPLES
            {few_shots}

            REQUIREMENTS:
            {state['requirements']}""")
        ]
        
        try:
            llm = self._get_model_for_task(TaskType.DECOMPOSE)
            
            structured_llm = llm.bind(
                max_tokens=self.config.max_tokens_decompose
            ).with_structured_output(ClassExtractionResult)
            
            result: ClassExtractionResult = safe_invoke(structured_llm, messages)
            
            Logger.log_classes(result.classes)
            
            return {"extracted_classes": result.classes}
            
        except Exception as e:
            Logger.log_error(f"Class extraction failed: {e}")
            return {"extracted_classes": []}
    
    def extract_relationships(self, state: AgentState) -> Dict[str, Any]:
        """
        Extract relationships between the previously extracted classes.
        
        Args:
            state: Current workflow state
            
        Returns:
            Dict with 'plan' update containing formatted plan
        """
        Logger.log_info(f"--- NODE: {NodeNames.EXTRACT_RELATIONSHIPS.upper()} ---")
        
        extracted_classes = state.get("extracted_classes", [])
        
        if not extracted_classes:
            Logger.log_warning("No classes extracted; skipping relationship extraction")
            return {"plan": "## STRUCTURAL DECOMPOSITION\n\nNo classes found."}
        
        # Format class list for the LLM
        class_names = [cls.name for cls in extracted_classes]
        class_list_str = "\n".join([f"- {name}" for name in class_names])
        
        # Format few-shot examples for relationship extraction
        few_shots = self.few_shot_loader.format_for_relationship_extraction(
            num_examples=self.config.num_few_shots
        )
        
        messages = [
            SystemMessage(content=RELATIONSHIP_EXTRACTOR_SYSTEM),
            HumanMessage(content=f"""
            # EXAMPLES
            {few_shots}

            REQUIREMENTS:
            {state['requirements']}

            EXTRACTED CLASSES:
            {class_list_str}

            Extract the relationships between these classes based on the requirements.""")
        ]
        
        try:
            Logger.log_info("Getting LLM model for relationship extraction...")
            llm = self._get_model_for_task(
                TaskType.DECOMPOSE,
                max_tokens=self.config.max_tokens_decompose
            )
            
            structured_llm = llm.with_structured_output(RelationshipExtractionResult)
            
            result: RelationshipExtractionResult = safe_invoke(
                structured_llm,
                messages
            )
            
            Logger.log_relationships(result.relationships)
            
            # Format the complete plan
            formatted_plan = self._format_plan(extracted_classes, result.relationships)
            
            return {"plan": formatted_plan}
            
        except Exception as e:
            Logger.log_error(f"Relationship extraction failed: {type(e).__name__}: {e}", exc_info=True)
            Logger.log_warning("Falling back to plan with classes only (no relationships)")
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
        Logger.log_info(f"--- NODE: {NodeNames.GENERATE.upper()} ---")
        
        # Format few-shot examples for generator
        few_shots = self.few_shot_loader.format_for_generator(
            num_examples=self.config.num_few_shots
        )
        
        messages = [SystemMessage(content=GENERATOR_SYSTEM)]
            
        user_content = f"""
        # EXAMPLES
        {few_shots}

        # VALIDATED DESIGN PLAN
        {state['plan']}

        Render the PlantUML class diagram exactly from the design plan.
        """
        
        messages.append(HumanMessage(content=user_content))
        Logger.log_debug(f"Generation prompt messages: {messages}")
        
        try:
            llm = self._get_model_for_task(
                TaskType.GENERATE,
                max_tokens=self.config.max_tokens_generate
            )
            
            response = safe_invoke(
                llm,
                messages
            )
            diagram = self.plantuml_tool.extract_plantuml(response.content)
            
            Logger.log_info(f"Generation completed (iteration {state['iterations'] + 1})")
            return {
                "current_diagram": diagram,
                "iterations": state["iterations"] + 1,
            }
            
        except Exception as e:
            Logger.log_error(f"Generation failed: {e}")
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
        Logger.log_info(f"--- NODE: {NodeNames.SYNTAX_CHECK.upper()} ---")
        
        try:
            result = self.plantuml_tool.check_syntax(
                state["current_diagram"],
                timeout=self.config.request_timeout
            )
            
            if result.is_valid:
                Logger.log_info(f"Syntax valid. View at: {result.url}")
            else:
                Logger.log_warning(f"Syntax error: {result.error}")
            
            return {
                "syntax_valid": result.is_valid,
                "error_message": result.error if not result.is_valid else None,
            }
            
        except Exception as e:
            Logger.log_error(f"Syntax check failed: {e}")
            return {
                "syntax_valid": False,
                "error_message": f"Syntax check error: {str(e)}"
            }

    def syntax_fixer(self, state: AgentState) -> Dict[str, Any]:
        """
        Fix PlantUML syntax errors using the PLANTUML_SYNTAX_CHECKER_SYSTEM prompt.
        
        Args:
            state: Current workflow state
            
        Returns:
            Dict with 'current_diagram' and 'iterations' updates
        """
        Logger.log_info(f"--- NODE: {NodeNames.SYNTAX_FIXER.upper()} ---")
        
        messages = [SystemMessage(content=PLANTUML_SYNTAX_CHECKER_SYSTEM)]
        
        user_content = f"""
        # PLANTUML DIAGRAM WITH SYNTAX ERROR
        ```plantuml
        {state['current_diagram']}
        ```
        
        # ERROR MESSAGE
        {state['error_message']}
        
        # TASK
        Fix the syntax error and provide a corrected version of the PlantUML diagram.
        """
        
        messages.append(HumanMessage(content=user_content))
        
        try:
            llm = self._get_model_for_task(
                TaskType.GENERATE,
                max_tokens=self.config.max_tokens_generate
            )
            
            response = safe_invoke(llm, messages)
            diagram = self.plantuml_tool.extract_plantuml(response.content)
            
            Logger.log_info(f"Syntax fixing completed (iteration {state['iterations'] + 1})")
            return {
                "current_diagram": diagram,
            }
            
        except Exception as e:
            Logger.log_error(f"Syntax fixing failed: {e}")
            return {
                "current_diagram": f"Error: {str(e)}",
            }

    def logical_fixer(self, state: AgentState) -> Dict[str, Any]:
        """
        Fix logical errors in the PlantUML diagram using the PLANTUML_LOGICAL_FIXER_SYSTEM prompt.
        
        Args:
            state: Current workflow state
            
        Returns:
            Dict with 'current_diagram' and 'iterations' updates
        """
        Logger.log_info(f"--- NODE: {NodeNames.LOGICAL_FIXER.upper()} ---")
        
        messages = [SystemMessage(content=PLANTUML_LOGICAL_FIXER_SYSTEM)]
        
        critique_report = state.get("current_validation")
        current_diagram = state.get("current_diagram", "")
        current_iteration = state.get("iterations", 0)

        if critique_report and critique_report.findings:
            findings_summary = []
            for finding in critique_report.findings:
                finding_desc = (
                    f"- [{finding.category.value.upper()}] "
                    f"{finding.description}"
                )
                if finding.expected_correction:
                    finding_desc += f"\n  -> FIX: {finding.expected_correction}"
                findings_summary.append(finding_desc)
            
            user_content = f"""
            # PLANTUML DIAGRAM WITH LOGICAL ERRORS
            ```plantuml
            {current_diagram}
            ```
            
            # CRITIQUE FINDINGS TO ADDRESS
            The diagram has the following issues:
            
            {chr(10).join(findings_summary)}
            
            # TASK
            Apply ALL corrections above to fix the diagram.
            """
            
            messages.append(HumanMessage(content=user_content))
            
            try:
                llm = self._get_model_for_task(
                    TaskType.GENERATE,
                    max_tokens=self.config.max_tokens_generate
                )
                
                response = safe_invoke(llm, messages)
                generated_diagram = self.plantuml_tool.extract_plantuml(response.content)

                similarity = Levenshtein.ratio(
                    current_diagram,
                    generated_diagram
                )

                if similarity > self.config.convergence_similarity_threshold:
                    Logger.log_warning("No meaningful changes made to diagram")
                    return {
                        "current_diagram": current_diagram,
                        "no_improvements_iteration": current_iteration,
                    }
                
                Logger.log_info(f"Logical fixing completed (iteration {state['iterations'] + 1})")
                return {
                    "current_diagram": generated_diagram,
                    "iterations": state["iterations"] + 1,
                }
                
            except Exception as e:
                Logger.log_error(f"Logical fixing failed: {e}")
                return {
                    "current_diagram": f"Error: {str(e)}",
                    "iterations": state["iterations"] + 1
                }
        else:
            Logger.log_warning("No critique findings; skipping logical fixing")
            return {
                "current_diagram": state.get("current_diagram", ""),
                "iterations": state["iterations"]
            }

    def critic(self, state: AgentState) -> Dict[str, Any]:
        """
        Critic node: produces a formal CritiqueReport.
        
        Args:
            state: Current workflow state
            
        Returns:
            Dict with 'current_validation' update
        """
        Logger.log_info(f"--- NODE: {NodeNames.CRITIC.upper()} ---")

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

            Logger.log_info("Getting model for critique task")
            llm = self._get_model_for_task(TaskType.CRITIQUE)
            
            structured_llm = llm.bind(
                max_tokens=self.config.max_tokens_critique
            ).with_structured_output(CritiqueReport)
            report: CritiqueReport = safe_invoke(structured_llm, messages)

            Logger.log_critique_report(report)

            return {
                "current_validation": report,
            }

        except Exception as e:
            Logger.log_error(f"Critic node failed: {e}")
            return {
                "logic_valid": False,
                "current_validation": None,
                "error_message": str(e)
            }
    
    def scorer(self, state: AgentState) -> Dict[str, Any]:
        """
        Score the current diagram based on syntax and logic validity.
        
        Args:
            state: Current workflow state
        Returns:
            Dict with 'syntax_score', 'semantic_score', 'pragmatic_score' updates
        """

        Logger.log_info(f"--- NODE: {NodeNames.SCORER.upper()} ---")

        try:
            requirements = state["requirements"]
            diagram = self.plantuml_tool.extract_plantuml(state["current_diagram"])

            messages = [
                SystemMessage(content=SCORER_SYSTEM),
                HumanMessage(content=json.dumps({
                    "requirements": requirements,
                    "diagram": diagram
                }))
            ]

            Logger.log_info("Getting model for scoring task")
            llm = self._get_model_for_task(TaskType.SCORING)

            structured_llm = llm.bind(
                max_tokens=self.config.max_tokens_scoring
            ).with_structured_output(ScoredCritiqueReport)
            report: ScoredCritiqueReport = safe_invoke(structured_llm, messages)

            Logger.log_info(f"Report type: {type(report)}, value: {report}")
            
            if isinstance(report, dict):
                scores = report['scores']
                syntax_score = scores['syntax_score']
                semantic_score = scores['semantic_score']
                pragmatic_score = scores['pragmatic_score']
                current_validation = report['report']
            else:
                scores = report.scores
                syntax_score = report.scores.syntax_score
                semantic_score = report.scores.semantic_score
                pragmatic_score = report.scores.pragmatic_score
                current_validation = report.report

            Logger.log_scored_report(report)

            return {
                "syntax_score": syntax_score,
                "semantic_score": semantic_score,
                "pragmatic_score": pragmatic_score,
                "scores": scores,
                "current_validation": current_validation,
            }

        except Exception as e:
            Logger.log_error(f"Scoring node failed: {e}")
            return {
                "syntax_score": 0.0,
                "semantic_score": 0.0,
                "pragmatic_score": 0.0,
                "scores": None,
                "current_validation": None,
                "error_message": str(e)
            }
