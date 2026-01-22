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
    # Initialize system
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
    
    # Load test exercises
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
    
    generated_diagrams = []
    
    for idx, exercise in enumerate(test_exercises[:1]):
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

        # Store generated diagram
        generated_diagram_data = {
            'id': exercise['id'],
            'title': exercise['title'],
            'requirements': exercise['requirements'],
            'ground_truth_diagram': exercise['diagram'],
            'generated_diagram': final_output['current_diagram'],
            'iterations': final_output['no_improvements_iteration'],
            'syntax_valid': final_output['syntax_valid'],
            'logic_valid': final_output['logic_valid']
        }
        generated_diagrams.append(generated_diagram_data)

        if output_dir:
            Logger.log_info(f"Saving outputs to {output_dir}")
            with open(output_dir / f"exercise_{idx + 1}_output.txt", 'w', encoding='utf-8') as f:
                f.write(final_output['current_diagram'])

    
        if final_output['current_diagram']:
            puml_tool = PlantUMLTool(config.plantuml_host)
            diagram_url = puml_tool.get_diagram_url(final_output['current_diagram'])
            Logger.log_diagram(diagram_url, final_output['current_diagram'])
            
            # Evaluate if gold standard is available
            if "diagram" in exercise:
                gold_standard = exercise["diagram"]
                generated_diagram = final_output["current_diagram"]
                
                # Use automatic threshold selection based on ROC analysis
                metrics = evaluate_diagram(
                    gold_standard, 
                    generated_diagram, 
                    config.evaluation_embedder_model,
                    config.evaluation_similarity_threshold
                )

                Logger.log_result_metrics(metrics)
            

    

