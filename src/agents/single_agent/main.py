from src.config import SINGLE_AGENT_OUTPUT_DIR
from src.core.logger import Logger
from src.agents.single_agent.config import SystemConfig
from src.core.utils import initialize_single_agent_system
from src.evaluation.evaluation import evaluate_diagram
from src.core.utils import ( 
    initialize_single_agent_system, 
    load_test_exercises, 
    run_exercise
)

def main(api_key: str):
    """
    Main function to run the agent-based diagram generation system.

    Args:
        api_key (str): API key for the language model.
    """
    config = SystemConfig(
        api_key=api_key,
    )

    app = initialize_single_agent_system(cfg=config)

    output_dir = SINGLE_AGENT_OUTPUT_DIR

    Logger.log_info("System initialized successfully")

    try:
        test_exercises = load_test_exercises(config.test_exercises_path)
        validation_exercises = load_test_exercises(config.diagrams_json_path)
        all_exercises = test_exercises + validation_exercises
        all_exercises = all_exercises[3:] # Skip first 3 exercises that are for few-shot
        Logger.log_info(f"Loaded {len(all_exercises)} test exercises")
        #Logger.log_info(f"Loaded {len(test_exercises)} test exercises")
    except Exception as e:
        Logger.log_error(f"Failed to load test exercises: {e}")
        return
    
    #TODO: When completed, only the three test exercises should be used
    # Also, remove attributes not used anymore (e.g., logic_valid)

    
    for idx, exercise in enumerate(test_exercises[:1]):
        Logger.log_info(f"Running exercise: {exercise['title']}")
        requirements = exercise["requirements"]
        final_output = run_exercise(
            app,
            requirements,
            exercise["title"],
            is_single_agent=True
        )

        Logger.log_info(f"Final output for exercise {idx + 1}: \n\n{final_output}")

        if "diagram" in exercise:
            gold_standard = exercise["diagram"]
            generated_diagram = final_output["current_diagram"]
            
            metrics = evaluate_diagram(
                gold_standard, 
                generated_diagram, 
                config.evaluation_embedder_model,
                config.evaluation_similarity_threshold
            )

            Logger.log_result_metrics(metrics)