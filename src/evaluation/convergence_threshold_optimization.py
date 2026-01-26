import json
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm
from os import getenv
from dotenv import load_dotenv
from typing import List, Dict, Any
from dataclasses import dataclass
import concurrent.futures

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.multi_agent.config import SystemConfig
from src.core.utils import initialize_multi_agent_system, run_exercise
from src.evaluation.evaluation import evaluate_diagram
from src.core.logger import Logger
from sentence_transformers import SentenceTransformer


CACHE_DIR = Path(__file__).parent.parent.parent / "output" / "cache"
CACHE_FILE = CACHE_DIR / "threshold_generation_cache.json"
RESULTS_FILE = Path(__file__).parent.parent.parent / "output" / "evaluation" / "convergence_threshold_results.json"

@dataclass
class Exercise:
    """Data structure for an exercise."""
    id: int
    title: str
    requirements: str
    ground_truth_diagram: str = None

def load_validation_exercises() -> List[Exercise]:
    """Load exercises with ground truth diagrams for validation."""
    path = Path(__file__).parent.parent.parent / "data" / "processed" / "diagrams.json"
    if not path.exists():
        raise FileNotFoundError(f"Validation exercises file not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    exercises = []
    for item in data:
        if "diagram" in item and item["diagram"].strip():
            exercises.append(Exercise(
                id=item["id"],
                title=item["title"],
                requirements=item["requirements"],
                ground_truth_diagram=item["diagram"]
            ))

    Logger.log_info(f"Loaded {len(exercises)} validation exercises.")
    return exercises


def load_generation_cache() -> Dict[str, Any]:
    """Load previously generated diagrams to avoid re-running LLM calls."""
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_generation_cache(cache: Dict[str, Any]):
    """Save generated diagrams to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def compute_f1_score(item: Dict[str, Any], eval_thresh: float, exercise_map: Dict[int, Exercise], model) -> float:
    """Compute F1 score for a single exercise-item pair."""
    exercise_id = item["exercise_id"]
    gen_diagram = item["generated_diagram"]
    exercise = exercise_map.get(exercise_id)
    
    if not exercise or not gen_diagram or not exercise.ground_truth_diagram:
        return 0.0
    
    try:
        metrics = evaluate_diagram(
            exercise.ground_truth_diagram,
            gen_diagram,
            model, 
            eval_thresh
        )
        avg_f1 = (metrics['classes'].f1 + metrics['attributes'].f1 + metrics['relationships'].f1) / 3
        return avg_f1
    except Exception as e:
        Logger.log_warning(f"Failed to evaluate {exercise.title}: {e}")
        return 0.0


def run_generation_phase(api_key: str, exercises: List[Exercise], convergence_thresholds: np.ndarray) -> Dict[str, Any]:
    """
    Phase 1: Generate diagrams for each convergence threshold.
    Results are cached to disk immediately.
    """
    cache = load_generation_cache()
    
    config = SystemConfig(api_key=api_key, evaluation_mode="critic")
    
    Logger.log_info("--- Starting Phase 1: Diagram Generation ---")
    
    for conv_thresh in tqdm(convergence_thresholds, desc="Generation Progress (Convergence Thresholds)"):
        thresh_key = f"{conv_thresh:.3f}"
        
        # Check if we already have data for this threshold
        if thresh_key in cache:
            Logger.log_info(f"Skipping generation for threshold {thresh_key} (found in cache)")
            continue


        config.convergence_similarity_threshold = conv_thresh
        app = initialize_multi_agent_system(cfg=config)
        
        batch_results = []
        
        for exercise in tqdm(exercises, desc=f"Generating for thr={thresh_key}", leave=False):
            try:
                final_output = run_exercise(app, exercise.requirements, exercise.title)
                
                batch_results.append({
                    "exercise_id": exercise.id,
                    "exercise_title": exercise.title,
                    "generated_diagram": final_output.get('current_diagram', "")
                })
            except Exception as e:
                Logger.log_warning(f"Failed to generate {exercise.title} at threshold {thresh_key}: {e}")
                batch_results.append({
                    "exercise_id": exercise.id,
                    "exercise_title": exercise.title,
                    "generated_diagram": "",
                    "error": str(e)
                })

        cache[thresh_key] = batch_results
        save_generation_cache(cache)
        
    return cache


def run_evaluation_phase(exercises: List[Exercise], generation_data: Dict[str, Any], eval_thresh: float = 0.45) -> Dict[str, Any]:
    """
    Evaluates the cached diagrams using a fixed evaluation threshold to find the optimal convergence threshold.
    """
    Logger.log_info("--- Starting: Convergence Threshold Evaluation ---")
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    exercise_map = {ex.id: ex for ex in exercises}
    
    results = []
    
    for thresh_key, batch_results in tqdm(generation_data.items(), desc="Evaluating Convergence Thresholds"):
        conv_thresh = float(thresh_key)
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(compute_f1_score, item, eval_thresh, exercise_map, model) for item in batch_results]
            scores = [f.result() for f in concurrent.futures.as_completed(futures) if f.result() is not None]
        
        avg_f1 = np.mean(scores) if scores else 0.0
        
        results.append({
            'convergence_threshold': conv_thresh,
            'average_f1': avg_f1,
            'num_exercises': len(scores)
        })
    
    if not results:
        return {}
    
    best_result = max(results, key=lambda x: x['average_f1'])
    
    return {
        'optimization_metric': 'Fixed Evaluation Threshold F1',
        'fixed_evaluation_threshold': eval_thresh,
        'optimal_convergence_threshold': best_result['convergence_threshold'],
        'max_f1_score': best_result['average_f1'],
        'all_results': results
    }


def optimize_convergence_threshold(api_key: str, eval_thresh: float = 0.45):
    """
    Orchestrates the two-phase optimization process for convergence threshold.
    
    Args:
        api_key: OpenRouter API key
        eval_thresh: Fixed evaluation threshold to use
    """
    exercises = load_validation_exercises()
    convergence_thresholds = np.linspace(0.80, 0.99, 20) 
    generation_data = run_generation_phase(api_key, exercises, convergence_thresholds)
    optimization_results = run_evaluation_phase(exercises, generation_data, eval_thresh)
    return optimization_results


if __name__ == "__main__":
    load_dotenv(Path(__file__).parent.parent.parent / ".env")

    api_key = getenv("OPENROUTER_API_KEY")
    if not api_key:
        Logger.log_error("OPENROUTER_API_KEY environment variable not set")
        sys.exit(1)

    EVAL_THRESH = 0.45
    
    results = optimize_convergence_threshold(api_key, eval_thresh=EVAL_THRESH)

    if results:
        # Save Final Results
        RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)

        Logger.log_info("\n" + "="*50)
        Logger.log_info("OPTIMIZATION COMPLETE")
        Logger.log_info(f"Fixed Evaluation Threshold    : {results['fixed_evaluation_threshold']:.3f}")
        Logger.log_info(f"Optimal Convergence Threshold : {results['optimal_convergence_threshold']:.3f}")
        Logger.log_info(f"Maximum F1 Score              : {results['max_f1_score']:.3f}")
        
        Logger.log_info("="*50)
        Logger.log_info(f"Detailed results saved to {RESULTS_FILE}")