from typing import Any, Dict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from src.core.models import SingleAgentState, TaskType, SingleAgentOutput
from src.core.logger import Logger
from src.core.utils import safe_invoke
from src.core.model_manager import ModelManager
from src.core.prompts import SINGLE_AGENT_SYSTEM
from src.agents.single_agent.config import SystemConfig


class Node:
    def __init__(self, model_manager: ModelManager, config: SystemConfig):
        """
        Initialize the Node with a model manager.
        
        Args:
            model_manager: Model manager for dynamic model loading
        """
        self.model_manager = model_manager
        self.config = config

    def _get_model_for_task(self, task_type: TaskType, **override_kwargs) -> ChatOpenAI:
        """
        Get a model instance for a specific task type.
        
        Args:
            task_type: The type of task requiring a model
            
        Returns:
            LLM instance from model manager
        """
        return self.model_manager.get_model(task_type, **override_kwargs)

    def generate(self, state: SingleAgentState) -> Dict[str, Any]:
        """
        Generate a diagram based on the requirements.

        Args:
            state (SingleAgentState): The current state of the agent.

        Returns:
            Dict[str, Any]: The updated state with the generated diagram.
        """
        Logger.log_info("Generating diagram based on requirements.")

        requirements = state.get("requirements", "")

        try:
            
            llm = self._get_model_for_task(TaskType.GENERATE)
            messages = [
                SystemMessage(content=SINGLE_AGENT_SYSTEM),
                HumanMessage(content=f"""
                    # REQUIREMENTS:
                    {requirements}

                    Generate a PlantUML diagram based on the requirements above.
                """)
            ]
            
            structured_llm = llm.bind(
                max_tokens=self.config.max_tokens
            ).with_structured_output(SingleAgentOutput)
            
            result: SingleAgentOutput = safe_invoke(structured_llm, messages)
            

            return {
                "current_diagram": result.diagram,
            }
        except Exception as e:
            Logger.log_error(f"Generation failed: {e}")
            return {}
