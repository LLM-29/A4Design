import sys

from pathlib import Path
from os import getenv
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.logger import Logger
from src.evaluation.threshold_optimizer import optimize_threshold, save_and_plot_results


CACHE_DIR = Path(__file__).parent.parent.parent / "output" / "cache"
CACHE_FILE = CACHE_DIR / "threshold_generation_cache.json"
RESULTS_FILE = Path(__file__).parent.parent.parent / "output" / "evaluation" / "convergence_threshold_results.json"
PLOT_FILE = Path(__file__).parent.parent.parent / "output" / "plots" / "convergence_threshold_plot.png"


if __name__ == "__main__":
    load_dotenv(Path(__file__).parent.parent.parent / ".env")

    api_key = getenv("OPENROUTER_API_KEY")
    if not api_key:
        Logger.log_error("OPENROUTER_API_KEY environment variable not set")
        sys.exit(1)

    EVAL_THRESH = 0.45
    
    results = optimize_threshold(
        api_key=api_key,
        threshold_name="convergence",
        threshold_range=(0.80, 0.99, 20),
        config_attr="convergence_similarity_threshold",
        evaluation_mode="critic",
        cache_file=CACHE_FILE,
        eval_thresh=EVAL_THRESH
    )

    save_and_plot_results(results, "convergence", RESULTS_FILE, PLOT_FILE, "g")