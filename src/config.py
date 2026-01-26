from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOTENV = PROJECT_ROOT / ".env"

DATA_DIR = PROJECT_ROOT / "data"
DATABASE = DATA_DIR 
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEW_SHOT_EXAMPLES = PROCESSED_DATA_DIR / "few_shot.json"
TEST_EXERCISES = PROCESSED_DATA_DIR / "test_exercises.json"
VALIDATION_EXERCISES = PROCESSED_DATA_DIR / "validation_exercises.json"
DIAGRAMS = PROCESSED_DATA_DIR / "diagrams.json"
RAW_DATA_DIR = DATA_DIR / "raw"

OUTPUT_DIR = PROJECT_ROOT / "output"
MULTI_AGENT_OUTPUT_DIR_CRITIC = OUTPUT_DIR / "multi_agent_critic"
MULTI_AGENT_OUTPUT_DIR_SCORER = OUTPUT_DIR / "multi_agent_scorer"   
SINGLE_AGENT_OUTPUT_DIR = OUTPUT_DIR / "single_agent"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DECOMPOSE_MODEL = "mistralai/devstral-2512:free"
GENERATE_MODEL = "mistralai/devstral-2512:free"
EMBEDDER_MODEL = "BAAI/bge-large-en-v1.5"
EVALUATION_EMBEDDER_MODEL = "sentence-transformers/all-mpnet-base-v2"
PLANTUML_HOST = "http://localhost:8080"
MAX_ITERATIONS = 12
MAX_TOKENS_DECOMPOSE = 4096
MAX_TOKENS_GENERATE = 4096
MAX_TOKENS_CRITIQUE = 4096
MAX_TOKENS_SCORING = 4096
TEMPERATURE_GENERATION = 0.0
TEMPERATURE_DECOMPOSE = 0.15
NUM_FEW_SHOTS = 3
EVALUATION_SIMILARITY_THRESHOLD = 0.55
CONVERGENCE_SIMILARITY_THRESHOLD = 0.96


def create_run_dir(agent_type: str, evaluation_mode: str = "critic") -> Path:
    """Create a timestamped directory for a new run.
    
    Args:
        agent_type: Either "multi_agent" or "single_agent"
        evaluation_mode: Either "critic" or "scorer" (only relevant for multi_agent)
    Returns:
        Path to the created run directory
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    
    if agent_type == "multi_agent":
        run_dir = MULTI_AGENT_OUTPUT_DIR_CRITIC / f"run_{timestamp}" if evaluation_mode == "critic" else MULTI_AGENT_OUTPUT_DIR_SCORER / f"run_{timestamp}"
    elif agent_type == "single_agent":
        run_dir = SINGLE_AGENT_OUTPUT_DIR / f"run_{timestamp}"
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def ensure_output_dirs():
    """Create all necessary output directories if they don't exist."""
    dirs = [
        OUTPUT_DIR,
        DATA_DIR,
        PROCESSED_DATA_DIR,
        RAW_DATA_DIR,
        MULTI_AGENT_OUTPUT_DIR_CRITIC,
        MULTI_AGENT_OUTPUT_DIR_SCORER,
        SINGLE_AGENT_OUTPUT_DIR,
    ]

    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)