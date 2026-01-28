import os
import json

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
    except Exception as e:
        Logger.log_error(f"Failed to load test exercises: {e}")
        return
    
    for idx, exercise in enumerate(test_exercises):
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

            metrics_file = os.path.join(output_dir, f"exercise_{idx+1}_metrics.json")
            serializable_metrics = {k: v.model_dump() for k, v in metrics.items()}
            with open(metrics_file, 'w') as f:
                json.dump(serializable_metrics, f, indent=4)

            diagram_file = os.path.join(output_dir, f"exercise_{idx+1}_diagram.puml")
            with open(diagram_file, 'w') as f:
                f.write(generated_diagram)