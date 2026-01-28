import os
import json

from src.agents.multi_agent.config import SystemConfig
from src.core.plantuml import PlantUMLTool
from src.evaluation.evaluation import evaluate_diagram
from src.core.utils import ( 
    initialize_multi_agent_system, 
    load_test_exercises, 
    run_exercise
)
from src.core.models import EvaluationMode
from src.core.logger import Logger
from src.config import MULTI_AGENT_OUTPUT_DIR_CRITIC, MULTI_AGENT_OUTPUT_DIR_SCORER


def main(api_key: str, evaluation: str):
    """Main function to run the system."""

    config = SystemConfig(
        api_key=api_key,
        evaluation_mode=evaluation
    )
    app = initialize_multi_agent_system(cfg=config)
    
    output_dir = None
    if evaluation == EvaluationMode.CRITIC:
        output_dir = MULTI_AGENT_OUTPUT_DIR_CRITIC
    elif evaluation == EvaluationMode.SCORER:
        output_dir = MULTI_AGENT_OUTPUT_DIR_SCORER
    
    Logger.log_info("System initialized successfully")
    
    try:
        test_exercises = load_test_exercises(config.test_exercises_path)
    except Exception as e:
        Logger.log_error(f"Failed to load test exercises: {e}")
        return
    
    
    for idx, exercise in enumerate(test_exercises):
        Logger.log_info(f"Running exercise: {exercise['title']}")
        requirements = exercise["requirements"]
        final_output = run_exercise(
            app,
            requirements,
            exercise["title"]
        )

        Logger.log_run_output(
            final_output.get('no_improvements_iteration', final_output['iterations']), 
            final_output['syntax_valid'], 
            final_output['logic_valid']
        )
    
        if final_output['current_diagram']:
            puml_tool = PlantUMLTool(config.plantuml_host)
            diagram_url = puml_tool.get_diagram_url(final_output['current_diagram'])
            Logger.log_diagram(diagram_url, final_output['current_diagram'])
            
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
            

    

