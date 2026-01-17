"""
Workflow creation and state management for the UML generation system.
"""

import logging
from typing import Any, Optional

from langgraph.graph import StateGraph, START, END

try:
    from .config import SystemConfig
    from .models import NodeNames, AgentState
    from .agents import UMLNodes
except ImportError:
    from config import SystemConfig
    from models import NodeNames, AgentState
    from agents import UMLNodes

logger = logging.getLogger(__name__)


def create_uml_graph(
    nodes: UMLNodes,
    config: Optional[SystemConfig] = None
) -> Any:
    """
    Create the LangGraph workflow for UML diagram generation.
    
    Args:
        nodes: UMLNodes instance with all agent methods
        config: Optional system configuration
        
    Returns:
        Compiled LangGraph workflow
    """
    cfg = config or SystemConfig()
    logger.info("Creating UML generation workflow")
    
    workflow = StateGraph(AgentState)

    # Add all nodes
    workflow.add_node(NodeNames.RETRIEVE, nodes.retrieve)
    workflow.add_node(NodeNames.EXTRACT_CLASSES, nodes.extract_classes)
    workflow.add_node(NodeNames.EXTRACT_RELATIONSHIPS, nodes.extract_relationships)
    workflow.add_node(NodeNames.GENERATE, nodes.generate)
    workflow.add_node(NodeNames.SYNTAX_CHECK, nodes.syntax_check)
    workflow.add_node(NodeNames.CRITIC, nodes.critic)
    
    logger.debug("Added 6 nodes to workflow")

    # Define edges
    workflow.add_edge(START, NodeNames.RETRIEVE)
    workflow.add_edge(NodeNames.RETRIEVE, NodeNames.EXTRACT_CLASSES)
    workflow.add_edge(NodeNames.EXTRACT_CLASSES, NodeNames.EXTRACT_RELATIONSHIPS)
    workflow.add_edge(NodeNames.EXTRACT_RELATIONSHIPS, NodeNames.GENERATE)
    workflow.add_edge(NodeNames.GENERATE, NodeNames.SYNTAX_CHECK)

    def route_after_syntax_check(state: AgentState) -> str:
        """
        Route based on syntax validation results and iteration limits.
        
        Args:
            state: Current workflow state
            
        Returns:
            Next node name
        """
        if state.get("syntax_valid", False):
            logger.debug("Routing: syntax_check -> critic")
            return NodeNames.CRITIC
            
        if state["iterations"] >= cfg.max_iterations:
            logger.warning(
                f"Max iterations ({cfg.max_iterations}) reached during syntax check"
            )
            return END
        
        logger.debug("Routing: syntax_check -> generate (syntax error)")
        return NodeNames.GENERATE

    workflow.add_conditional_edges(
        NodeNames.SYNTAX_CHECK,
        route_after_syntax_check,
        {
            NodeNames.CRITIC: NodeNames.CRITIC,
            NodeNames.GENERATE: NodeNames.GENERATE,
            END: END
        }
    )

    def route_after_critic(state: AgentState) -> str:
        """
        Route based on critic results.
        
        Args:
            state: Current workflow state
            
        Returns:
            Next node name
        """
        validation = state.get("current_validation")
        if not validation:
            logger.error("No critic report; stopping workflow.")
            return END

        if validation.is_valid:
            logger.info("CRITIC passed: diagram is fully valid -> END")
            state["logic_valid"] = True
            return END

        # If there are findings, attempt to regenerate the diagram with critique feedback
        if len(validation.findings) > 0:
            logger.debug(
                "Routing: CRITIC -> GENERATE (issues found, regenerating diagram)"
            )
            return NodeNames.GENERATE

        logger.warning("CRITIC findings not actionable; ending workflow")
        return END

    workflow.add_conditional_edges(
        NodeNames.CRITIC,
        route_after_critic,
        {
            NodeNames.GENERATE: NodeNames.GENERATE,
            END: END
        }
    )
        
    logger.info("Workflow graph created successfully")
    return workflow.compile()


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
