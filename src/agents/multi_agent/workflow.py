"""
Workflow creation and state management for the UML generation system.
"""


from typing import Any
from langgraph.graph import StateGraph, START, END
from src.agents.multi_agent.config import SystemConfig
from src.core.models import NodeNames, AgentState
from src.agents.multi_agent.agents import UMLNodes

from src.core.logger import Logger


def create_critic_workflow(
    nodes: UMLNodes,
    cfg: SystemConfig
) -> Any:
    """
    Create the LangGraph workflow for UML diagram generation.
    
    Args:
        nodes: UMLNodes instance with all agent methods
        cfg: SystemConfig instance
        
    Returns:
        Compiled LangGraph workflow
    """
    Logger.log_info("Creating UML generation workflow")
    
    workflow = StateGraph(AgentState)

    # Add all nodes
    workflow.add_node(NodeNames.RETRIEVE, nodes.retrieve)
    workflow.add_node(NodeNames.EXTRACT_CLASSES, nodes.extract_classes)
    workflow.add_node(NodeNames.EXTRACT_RELATIONSHIPS, nodes.extract_relationships)
    workflow.add_node(NodeNames.GENERATE, nodes.generate)
    workflow.add_node(NodeNames.SYNTAX_CHECK, nodes.syntax_check)
    workflow.add_node(NodeNames.CRITIC, nodes.critic)
    workflow.add_node(NodeNames.SYNTAX_FIXER, nodes.syntax_fixer)
    workflow.add_node(NodeNames.LOGICAL_FIXER, nodes.logical_fixer)
    Logger.log_debug("Added 8 nodes to workflow")

    # Define edges
    workflow.add_edge(START, NodeNames.RETRIEVE)
    workflow.add_edge(NodeNames.RETRIEVE, NodeNames.EXTRACT_CLASSES)
    workflow.add_edge(NodeNames.EXTRACT_CLASSES, NodeNames.EXTRACT_RELATIONSHIPS)
    workflow.add_edge(NodeNames.EXTRACT_RELATIONSHIPS, NodeNames.GENERATE)
    workflow.add_edge(NodeNames.GENERATE, NodeNames.SYNTAX_CHECK)

    def route_after_syntax_check(state: AgentState) -> str:
        """
        Route based on syntax validation results.
        
        Args:
            state: Current workflow state
            
        Returns:
            Next node name
        """
        if state.get("syntax_valid", False):
            Logger.log_debug("Routing: syntax_check -> critic")
            return NodeNames.CRITIC
         
        Logger.log_debug("Routing: syntax_check -> syntax_fixer (syntax error)")
        return NodeNames.SYNTAX_FIXER

    workflow.add_conditional_edges(
        NodeNames.SYNTAX_CHECK,
        route_after_syntax_check,
        {
            NodeNames.CRITIC: NodeNames.CRITIC,
            NodeNames.SYNTAX_FIXER: NodeNames.SYNTAX_FIXER,
            END: END
        }
    )
    
    workflow.add_edge(NodeNames.SYNTAX_FIXER, NodeNames.SYNTAX_CHECK)

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
            Logger.log_error("No critic report; stopping workflow.")
            return END

        if validation.is_valid:
            Logger.log_info("CRITIC passed: diagram is fully valid -> END")
            state["logic_valid"] = True
            return END

        # If there are findings, attempt to fix the diagram with logical fixer
        if len(validation.findings) > 0:
            Logger.log_debug(
                "Routing: CRITIC -> LOGICAL_FIXER (issues found, fixing diagram)"
            )
            return NodeNames.LOGICAL_FIXER

        Logger.log_warning("CRITIC findings not actionable; ending workflow")
        return END

    def route_after_logical_fixer(state: AgentState) -> str:
        """
        Route based on logical fixer results.
        
        Args:
            state: Current workflow state
            
        Returns:
            Next node name
        """
        if "no_improvements_iteration" in state:
            Logger.log_info("Converged: no further improvements detected -> END")
            return END
        
        Logger.log_debug("Routing: logical_fixer -> critic")
        return NodeNames.CRITIC

    workflow.add_conditional_edges(
        NodeNames.CRITIC,
        route_after_critic,
        {
            NodeNames.LOGICAL_FIXER: NodeNames.LOGICAL_FIXER,
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        NodeNames.LOGICAL_FIXER,
        route_after_logical_fixer,
        {
            NodeNames.CRITIC: NodeNames.CRITIC,
            END: END
        }
    )
          
    Logger.log_info("Workflow graph created successfully")
    return workflow.compile()


def create_scoring_workflow(
    nodes: UMLNodes,
    cfg: SystemConfig
) -> StateGraph:
    """
    Create a simplified scoring workflow for UML diagrams.
    
    Returns:
        Compiled LangGraph scoring workflow
    """
    Logger.log_info("Creating UML scoring workflow")
    
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node(NodeNames.RETRIEVE, nodes.retrieve)
    workflow.add_node(NodeNames.EXTRACT_CLASSES, nodes.extract_classes)
    workflow.add_node(NodeNames.EXTRACT_RELATIONSHIPS, nodes.extract_relationships)
    workflow.add_node(NodeNames.GENERATE, nodes.generate)
    workflow.add_node(NodeNames.SCORER, nodes.scorer)
    workflow.add_node(NodeNames.SYNTAX_CHECK, nodes.syntax_check)
    workflow.add_node(NodeNames.LOGICAL_FIXER, nodes.logical_fixer)
    workflow.add_node(NodeNames.SYNTAX_FIXER, nodes.syntax_fixer)

    Logger.log_debug("Added 7 nodes to scoring workflow")

    # Define edges
    workflow.add_edge(START, NodeNames.RETRIEVE)
    workflow.add_edge(NodeNames.RETRIEVE, NodeNames.EXTRACT_CLASSES)
    workflow.add_edge(NodeNames.EXTRACT_CLASSES, NodeNames.EXTRACT_RELATIONSHIPS)
    workflow.add_edge(NodeNames.EXTRACT_RELATIONSHIPS, NodeNames.GENERATE)
    workflow.add_edge(NodeNames.GENERATE, NodeNames.SYNTAX_CHECK)
    workflow.add_edge(NodeNames.SYNTAX_FIXER, NodeNames.SYNTAX_CHECK)

    def route_after_syntax_check(state: AgentState) -> str:
        """
        Route based on syntax validation results.
        
        Args:
            state: Current workflow state
            
        Returns:
            Next node name
        """
        if state.get("syntax_valid", False):
            Logger.log_debug("Routing: syntax_check -> scorer (valid syntax)")
            return NodeNames.SCORER
         
        Logger.log_debug("Routing: syntax_check -> syntax_fixer (syntax error)")
        return NodeNames.SYNTAX_FIXER

    workflow.add_conditional_edges(
        NodeNames.SYNTAX_CHECK,
        route_after_syntax_check,
        {
            NodeNames.SCORER: NodeNames.SCORER,
            NodeNames.SYNTAX_FIXER: NodeNames.SYNTAX_FIXER
        }
    )

    workflow.add_edge(NodeNames.SYNTAX_FIXER, NodeNames.SYNTAX_CHECK)

    def route_after_scorer(state: AgentState) -> str:
        """
        Route based on scoring results.
        
        Args:
            state: Current workflow state
            
        Returns:
            Next node name
        """
        validation = state.get("current_validation")
        if not validation:
            Logger.log_error("No scoring report; stopping workflow.")
            return END
        
        avg_score = (state.syntax_score + state.semantic_score + state.pragmatic_score) / 3.0
        Logger.log_info(f"Average diagram score: {avg_score:.2f}")
        if avg_score < cfg.scoring_threshold:
            if abs(avg_score - cfg.plateau_threshold) <= cfg.plateau_threshold:
                if state.get("no_improvements_iteration") > cfg.plateau_window:
                    Logger.log_info("Converged: no further improvements detected -> END")
                    return END
                else:
                    state["no_improvements_iteration"] = state.get("no_improvements_iteration", 0) + 1
            Logger.log_debug("Routing: scorer -> logical_fixer (score below threshold)")
            return NodeNames.LOGICAL_FIXER
        else:
            Logger.log_debug("Routing: scorer -> END (score above threshold)")
            return END
    
    workflow.add_conditional_edges(
        NodeNames.SCORER,
        route_after_scorer,
        {
            NodeNames.LOGICAL_FIXER: NodeNames.LOGICAL_FIXER,
            END: END
        }
    )

    def route_after_logical_fixer(state: AgentState) -> str:
        """
        Route based on logical fixer results.
        
        Args:
            state: Current workflow state
            
        Returns:
            Next node name
        """
        Logger.log_debug("Routing: logical_fixer -> scorer")
        return NodeNames.SCORER
    
    workflow.add_conditional_edges(
        NodeNames.LOGICAL_FIXER,
        route_after_logical_fixer,
        {
            NodeNames.SCORER: NodeNames.SCORER
        }
    )
          
    Logger.log_info("Scoring workflow graph created successfully")
    return workflow.compile()



