"""
Workflow creation and state management for the UML generation system.
"""

from typing import Any
from langgraph.graph import StateGraph, START, END
from src.agents.multi_agent.config import SystemConfig
from src.core.models import NodeNames, AgentState
from src.agents.single_agent.agent import Node

from src.core.logger import Logger


def create_single_agent(nodes: Node,) -> Any:
    """
    Create the LangGraph workflow for UML diagram generation.
    
    Args:
        nodes: Node instance with all agent methods
        cfg: SystemConfig instance
        
    Returns:
        Compiled LangGraph workflow
    """
    Logger.log_info("Creating UML generation workflow")
    
    workflow = StateGraph(AgentState)

    workflow.add_node(NodeNames.SINGLE_AGENT, nodes.generate)
    
    Logger.log_debug("Added 1 node to workflow")

    # Define edges
    workflow.add_edge(START, NodeNames.SINGLE_AGENT)
    workflow.add_edge(NodeNames.SINGLE_AGENT, END)

    return workflow.compile()