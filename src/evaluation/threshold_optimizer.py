import json
import numpy as np
import sys
import concurrent.futures
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.multi_agent.config import SystemConfig
from src.core.utils import initialize_multi_agent_system, run_exercise
from src.evaluation.evaluation import evaluate_diagram
from src.core.logger import Logger
from sentence_transformers import SentenceTransformer


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


def load_generation_cache(cache_file: Path) -> Dict[str, Any]:
    """Load previously generated diagrams to avoid re-running LLM calls."""
    if cache_file.exists():
        with open(cache_file, "r") as f:
            return json.load(f)
    return {}


def save_generation_cache(cache: Dict[str, Any], cache_file: Path):
    """Save generated diagrams to disk."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
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


def _generate_single_exercise(app, exercise: Exercise, thresh_key: str) -> Dict[str, Any]:
    """Helper function to generate a single exercise (used for parallelization)."""
    try:
        final_output = run_exercise(app, exercise.requirements, exercise.title)
        return {
            "exercise_id": exercise.id,
            "exercise_title": exercise.title,
            "generated_diagram": final_output.get('current_diagram', "")
        }
    except Exception as e:
        Logger.log_warning(f"Failed to generate {exercise.title} at threshold {thresh_key}: {e}")
        return {
            "exercise_id": exercise.id,
            "exercise_title": exercise.title,
            "generated_diagram": "",
            "error": str(e)
        }


def run_generation_phase(api_key: str, exercises: List[Exercise], thresholds: np.ndarray, config_attr: str, evaluation_mode: str, cache_file: Path, threshold_name: str, max_workers: int = 4) -> Dict[str, Any]:
    """
    Phase 1: Generate diagrams for each threshold.
    Results are cached to disk immediately after each exercise and after each threshold.
    
    Args:
        max_workers: Number of parallel workers for exercise generation (default: 4)
    """
    cache = load_generation_cache(cache_file)

    config = SystemConfig(api_key=api_key, evaluation_mode=evaluation_mode)

    Logger.log_info("--- Starting Phase 1: Diagram Generation ---")

    for thresh in tqdm(thresholds, desc=f"Generation Progress ({threshold_name.capitalize()} Thresholds)"):
        thresh_key = f"{thresh:.3f}"

        # Check if we already have data for this threshold
        if thresh_key in cache:
            Logger.log_info(f"Skipping generation for threshold {thresh_key} (found in cache)")
            continue

        setattr(config, config_attr, thresh)
        app = initialize_multi_agent_system(cfg=config)

        batch_results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_generate_single_exercise, app, exercise, thresh_key) for exercise in exercises]
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(exercises), desc=f"Generating for thr={thresh_key}", leave=False):
                result = future.result()
                batch_results.append(result)
                
                cache[thresh_key] = batch_results
                save_generation_cache(cache, cache_file)

    return cache


def run_evaluation_phase(exercises: List[Exercise], generation_data: Dict[str, Any], eval_thresh: float, threshold_name: str) -> Dict[str, Any]:
    """
    Evaluates the cached diagrams using a fixed evaluation threshold to find the optimal threshold.
    """
    Logger.log_info(f"--- Starting: {threshold_name.capitalize()} Threshold Evaluation ---")
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    exercise_map = {ex.id: ex for ex in exercises}

    results = []

    for thresh_key, batch_results in tqdm(generation_data.items(), desc=f"Evaluating {threshold_name.capitalize()} Thresholds"):
        thresh = float(thresh_key)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(compute_f1_score, item, eval_thresh, exercise_map, model) for item in batch_results]
            scores = [f.result() for f in concurrent.futures.as_completed(futures) if f.result() is not None]

        avg_f1 = np.mean(scores) if scores else 0.0

        results.append({
            f'{threshold_name}_threshold': thresh,
            'average_f1': avg_f1,
            'num_exercises': len(scores)
        })

    if not results:
        return {}

    best_result = max(results, key=lambda x: x['average_f1'])

    return {
        'optimization_metric': 'Fixed Evaluation Threshold F1',
        'fixed_evaluation_threshold': eval_thresh,
        f'optimal_{threshold_name}_threshold': best_result[f'{threshold_name}_threshold'],
        'max_f1_score': best_result['average_f1'],
        'all_results': results
    }


def optimize_threshold(
    api_key: str,
    threshold_name: str,
    threshold_range: Tuple[float, float, int],
    config_attr: str,
    evaluation_mode: str,
    cache_file: Path,
    eval_thresh: float = 0.45
):
    """
    Orchestrates the two-phase optimization process for a given threshold.

    Args:
        api_key: OpenRouter API key
        threshold_name: Name of the threshold (e.g., 'convergence', 'scoring')
        threshold_range: Tuple (start, end, num_points) for np.linspace
        config_attr: Attribute name in SystemConfig to set
        evaluation_mode: 'critic' or 'scorer'
        cache_file: Path to cache file
        eval_thresh: Fixed evaluation threshold to use
    """
    exercises = load_validation_exercises()
    thresholds = np.linspace(*threshold_range)
    generation_data = run_generation_phase(api_key, exercises, thresholds, config_attr, evaluation_mode, cache_file, threshold_name)
    optimization_results = run_evaluation_phase(exercises, generation_data, eval_thresh, threshold_name)
    return optimization_results


def save_and_plot_results(results: Dict[str, Any], threshold_name: str, results_file: Path, plot_file: Path, plot_color: str):
    """Save results and generate plot."""
    if results:
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        Logger.log_info("\n" + "="*50)
        Logger.log_info("OPTIMIZATION COMPLETE")
        Logger.log_info(f"Fixed Evaluation Threshold    : {results['fixed_evaluation_threshold']:.3f}")
        Logger.log_info(f"Optimal {threshold_name.capitalize()} Threshold : {results[f'optimal_{threshold_name}_threshold']:.3f}")
        Logger.log_info(f"Maximum F1 Score              : {results['max_f1_score']:.3f}")
        Logger.log_info("="*50)
        Logger.log_info(f"Detailed results saved to {results_file}")

        sorted_results = sorted(results['all_results'], key=lambda x: x[f'{threshold_name}_threshold'])
        thresholds = [res[f'{threshold_name}_threshold'] for res in sorted_results]
        f1_scores = [res['average_f1'] for res in sorted_results]
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, f1_scores, marker='o', linestyle='-', color=plot_color, markersize=4)
        plt.axvline(x=results[f'optimal_{threshold_name}_threshold'], color='r', linestyle='--', label=f'Optimal {threshold_name.capitalize()} Threshold: {results[f"optimal_{threshold_name}_threshold"]:.3f}')
        plt.title(f'Average F1 Score vs {threshold_name.capitalize()} Threshold')
        plt.xlabel(f'{threshold_name.capitalize()} Threshold')
        plt.ylabel('Average F1 Score')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plot_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_file)
        plt.show()
    else:
        Logger.log_error("Optimization failed to produce any results")