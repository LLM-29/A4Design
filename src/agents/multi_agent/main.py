from src.agents.multi_agent.config import SystemConfig
from src.core.plantuml import PlantUMLTool
from src.evaluation.evaluation import evaluate_diagram
from src.core.utils import ( 
    initialize_multi_agent_system, 
    load_test_exercises, 
    run_exercise
)
from src.core.logger import Logger


def main(api_key: str, evaluation: str = "critic"):
    """Main function to run the system."""
    # Initialize system
    config = SystemConfig(
        api_key=api_key,
        evaluation_mode=evaluation
    )
    app = initialize_multi_agent_system(cfg=config)
    
    Logger.log_info("System initialized successfully")
    
    # Load test exercises
    try:
        test_exercises = load_test_exercises(config.test_exercises_path)
        Logger.log_info(f"Loaded {len(test_exercises)} test exercises")
    except Exception as e:
        Logger.log_error(f"Failed to load test exercises: {e}")
        return
    
    #TODO: When completed, exercises should run in batch
    
    # Select and run a test exercise
    test_idx = 2
    requirements = test_exercises[test_idx]["requirements"]
    final_output = run_exercise(
        app,
        requirements,
        f"Exercise {test_idx + 1}"
    )

    Logger.log_run_output(
        final_output['iterations'], 
        final_output['syntax_valid'], 
        final_output['logic_valid']
    )
    
    if final_output['current_diagram']:
        puml_tool = PlantUMLTool(config.plantuml_host)
        diagram_url = puml_tool.get_diagram_url(final_output['current_diagram'])
        Logger.log_diagram(diagram_url, final_output['current_diagram'])
        
        # Evaluate if gold standard is available
        if "solution_plantuml" in test_exercises[test_idx]:
            gold_standard = test_exercises[test_idx]["solution_plantuml"]
            generated_diagram = final_output["current_diagram"]
            
            # Use automatic threshold selection based on ROC analysis
            metrics = evaluate_diagram(
                gold_standard, 
                generated_diagram, 
                config.evaluation_embedder_model,
                auto_threshold=True
            )

            Logger.log_result_metrics(metrics)
