"""
Joint Threshold Optimization for UML Diagram Generation and Evaluation.

OPTIMIZED APPROACH:
1. Generation Phase: Runs the agent with different convergence thresholds and 
   caches the resulting diagrams. (Expensive, slow)
2. Evaluation Phase: Sweeps through evaluation thresholds using the cached 
   diagrams to calculate F1 scores. (Cheap, fast)

This reduces API calls from (N_conv * M_eval) to just (N_conv).
"""

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

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.multi_agent.config import SystemConfig
from src.core.utils import initialize_multi_agent_system, run_exercise
from src.evaluation.evaluation import evaluate_diagram
from src.core.logger import Logger
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import KFold


CACHE_DIR = Path(__file__).parent.parent.parent / "output" / "cache"
CACHE_FILE = CACHE_DIR / "threshold_generation_cache.json"
RESULTS_FILE = Path(__file__).parent.parent.parent / "output" / "evaluation" / "joint_threshold_results.json"

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
        return None
    
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
        return None


def run_generation_phase(api_key: str, exercises: List[Exercise], convergence_thresholds: np.ndarray) -> Dict[str, Any]:
    """
    Phase 1: Generate diagrams for each convergence threshold.
    Results are cached to disk immediately.
    """
    cache = load_generation_cache()
    
    # Initialize Config
    config = SystemConfig(api_key=api_key, evaluation_mode="critic")
    
    Logger.log_info("--- Starting Phase 1: Diagram Generation ---")
    
    for conv_thresh in tqdm(convergence_thresholds, desc="Generation Progress (Convergence Thresholds)"):
        thresh_key = f"{conv_thresh:.3f}"
        
        # specific check: if we already have data for this threshold and it covers all exercises, skip
        if thresh_key in cache:
            # Optional: Check if all exercises are present in cache, otherwise partial re-run might be needed
            # For simplicity, we assume if the key exists, the batch was run.
            Logger.log_info(f"Skipping generation for threshold {thresh_key} (found in cache)")
            continue

        # Initialize system with current threshold
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

        # Update and save cache after every threshold batch
        cache[thresh_key] = batch_results
        save_generation_cache(cache)
        
    return cache


def run_evaluation_phase(exercises: List[Exercise], generation_data: Dict[str, Any], evaluation_thresholds: np.ndarray, k_folds: int = 5) -> Dict[str, Any]:
    Logger.log_info(f"--- Starting Phase 2: {k_folds}-Fold Cross-Validation ---")
    
    # Load BERT model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    exercise_ids = np.array([ex.id for ex in exercises])
    exercise_map = {ex.id: ex for ex in exercises}
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    fold_results = []
    
    def compute_scores_parallel(items, eval_thresh, exercise_map, model):
        """Helper function to compute scores in parallel."""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(compute_f1_score, item, eval_thresh, exercise_map, model) for item in items]
            scores = [future.result() for future in concurrent.futures.as_completed(futures)]
        return [s for s in scores if s is not None]
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(exercise_ids)):
        train_ids = set(exercise_ids[train_idx])
        val_ids = set(exercise_ids[val_idx])
        
        best_train_config = {'f1': -1, 'conv': None, 'eval': None}
        
        # 1. FIND BEST PARAMETERS ON TRAINING SET
        for thresh_key, batch_results in generation_data.items():
            conv_thresh = float(thresh_key)
            
            # Filter batch for training exercises only
            train_batch = [b for b in batch_results if b['exercise_id'] in train_ids]
            
            for eval_thresh in evaluation_thresholds:
                # Calculate avg F1 for this configuration on TRAIN set
                scores = compute_scores_parallel(train_batch, eval_thresh, exercise_map, model)
                avg_f1 = np.mean(scores) if scores else 0
                
                if avg_f1 > best_train_config['f1']:
                    best_train_config = {'f1': avg_f1, 'conv': conv_thresh, 'eval': eval_thresh}
        
        # 2. EVALUATE PARAMETERS ON VALIDATION SET
        best_conv_key = f"{best_train_config['conv']:.3f}"
        val_batch = [b for b in generation_data[best_conv_key] if b['exercise_id'] in val_ids]
        
        val_scores = compute_scores_parallel(val_batch, best_train_config['eval'], exercise_map, model)
        val_f1 = np.mean(val_scores) if val_scores else 0
        
        fold_results.append({
            'fold': fold_idx,
            'best_train_conv': best_train_config['conv'],
            'best_train_eval': best_train_config['eval'],
            'train_f1': best_train_config['f1'],
            'validation_f1': val_f1
        })
        
        Logger.log_info(f"Fold {fold_idx}: Selected (Conv={best_train_config['conv']:.3f}, Eval={best_train_config['eval']:.3f}) -> Val F1: {val_f1:.3f}")

    # Aggregating results
    avg_val_f1 = np.mean([r['validation_f1'] for r in fold_results])
    avg_best_conv = np.mean([r['best_train_conv'] for r in fold_results])
    avg_best_eval = np.mean([r['best_train_eval'] for r in fold_results])
    
    return {
        'cross_validation_f1': avg_val_f1,
        'robust_optimal_convergence': avg_best_conv,
        'robust_optimal_evaluation': avg_best_eval,
        'fold_details': fold_results
    }

# --- Main Orchestrator ---

def optimize_joint_thresholds(api_key: str):
    """Orchestrates the two-phase optimization process."""
    exercises = load_validation_exercises()

    # Define ranges
    convergence_thresholds = np.linspace(0.90, 0.99, 10) 
    #TODO provare con range 0.45 - 0.55
    evaluation_thresholds = np.linspace(0.55, 0.70, 10) 

    generation_data = run_generation_phase(api_key, exercises, convergence_thresholds)
    
    optimization_results = run_evaluation_phase(exercises, generation_data, evaluation_thresholds)
    
    return optimization_results

if __name__ == "__main__":
    load_dotenv(Path(__file__).parent.parent.parent / ".env")

    api_key = getenv("OPENROUTER_API_KEY")
    if not api_key:
        Logger.log_error("OPENROUTER_API_KEY environment variable not set")
        sys.exit(1)

    results = optimize_joint_thresholds(api_key)

    if results:
        # Save Final Results
        RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)

        Logger.log_info("\n" + "="*50)
        Logger.log_info("OPTIMIZATION COMPLETE")
        Logger.log_info(f"Optimal Convergence Threshold : {results['robust_optimal_convergence']:.3f}")
        Logger.log_info(f"Optimal Evaluation Threshold  : {results['robust_optimal_evaluation']:.3f}")
        Logger.log_info(f"Cross-Validation F1 Score     : {results['cross_validation_f1']:.3f}")
        Logger.log_info("="*50)
        Logger.log_info(f"Detailed results saved to {RESULTS_FILE}")