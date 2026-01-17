"""
Main entry point for the UML generation system.
"""

import logging
import sys
from pathlib import Path
from typing import Any

# Handle both direct execution and module import
try:
    from .config import SystemConfig
    from .models import AgentState
    from .plantuml_tools import PlantUMLTool
    from .workflow import create_initial_state
    from .evaluation import evaluate_diagram
    from .utils import initialize_system, load_test_exercises
except ImportError:
    # Running as script, add parent to path
    sys.path.insert(0, str(Path(__file__).parent))
    from config import SystemConfig
    from models import AgentState
    from plantuml_tools import PlantUMLTool
    from workflow import create_initial_state
    from evaluation import evaluate_diagram
    from utils import initialize_system, load_test_exercises

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_single_test(
    app: Any,
    requirements: str,
    config: SystemConfig,
    exercise_name: str = "Test Exercise"
) -> AgentState:
    """
    Run the workflow on a single exercise.
    
    Args:
        app: Compiled LangGraph workflow
        requirements: Software requirements text
        config: System configuration
        exercise_name: Name for logging purposes
        
    Returns:
        Final workflow state
    """
    logger.info("="*60)
    logger.info(f"RUNNING: {exercise_name}")
    logger.info("="*60)
    logger.info(f"Requirements preview: {requirements[:150]}...")
    
    initial_state = create_initial_state(requirements)
    
    try:
        final_output = app.invoke(initial_state, config={"recursion_limit": 50})
        
        logger.info("="*60)
        logger.info("WORKFLOW COMPLETED")
        logger.info("="*60)
        logger.info(f"Iterations: {final_output['iterations']}")
        logger.info(f"Syntax Valid: {final_output['syntax_valid']}")
        logger.info(f"Logic Valid: {final_output['logic_valid']}")
        
        if final_output.get('best_diagram') and not final_output['logic_valid']:
            if final_output['best_diagram'] != final_output['current_diagram']:
                logger.info(
                    "Using BEST diagram instead of final (prevented regression)"
                )
                final_output['current_diagram'] = final_output['best_diagram']
        
        return final_output
        
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        raise


def main():
    """Main function to run the system."""
    # Initialize system
    nodes, app, config, memory_manager = initialize_system(
        enable_long_term_memory=True
    )
    
    print("\nSystem ready for diagram generation")
    print(f"Long-term memory: {'ENABLED' if memory_manager else 'DISABLED'}")
    
    # Load test exercises
    try:
        test_exercises = load_test_exercises()
        print(f"\nLoaded {len(test_exercises)} test exercises")
    except Exception as e:
        logger.error(f"Failed to load test exercises: {e}")
        return
    
    # Select and run a test exercise
    test_idx = 2  # Change this to test different exercises
    requirements = test_exercises[test_idx]["requirements"]
    
    final_output = run_single_test(
        app,
        requirements,
        config,
        f"Exercise {test_idx + 1}"
    )
    
    # Display results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Iterations: {final_output['iterations']}")
    print(f"Syntax Valid: {final_output['syntax_valid']}")
    print(f"Logic Valid: {final_output['logic_valid']}")
    
    if final_output['current_diagram']:
        puml_tool = PlantUMLTool(config.plantuml_host)
        diagram_url = puml_tool.get_diagram_url(final_output['current_diagram'])
        print(f"\nDiagram URL: {diagram_url}")
        
        print("\nGenerated Diagram:")
        print(final_output['current_diagram'])
        
        # Evaluate if gold standard is available
        if "solution_plantuml" in test_exercises[test_idx]:
            gold_standard = test_exercises[test_idx]["solution_plantuml"]
            generated_diagram = final_output["current_diagram"]
            
            metrics = evaluate_diagram(gold_standard, generated_diagram, config.evaluation_embedder_model)
            
            print("\n" + "="*60)
            print("EVALUATION METRICS")
            print("="*60)
            print(f"\nClasses:       {metrics['classes']}")
            print(f"Attributes:    {metrics['attributes']}")
            print(f"Relationships: {metrics['relationships']}")
            
            average_f1 = (
                metrics['classes'].f1 +
                metrics['attributes'].f1 +
                metrics['relationships'].f1
            ) / 3.0
            
            print(f"\n{'='*60}")
            print(f"OVERALL F1 SCORE: {average_f1:.2f}")
            print(f"{'='*60}")


if __name__ == "__main__":
    main()
