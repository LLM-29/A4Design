import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from argparse import ArgumentParser, Namespace
from dotenv import load_dotenv
from os import getenv

from src.config import ensure_output_dirs, DOTENV
from src.core.models import EvaluationMode, AgentMode
from src.agents.multi_agent.main import main as run_multi_agent
from src.agents.single_agent.main import main as run_single_agent
from src.core.logger import Logger

def define_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Run multi-agent system with different configurations")
    parser.add_argument("--mode", choices=[AgentMode.SINGLE, AgentMode.MULTI], required=True, help="Mode of operation: single or multi", default=AgentMode.MULTI)
    parser.add_argument("--evaluation", choices=[EvaluationMode.SCORER, EvaluationMode.CRITIC], required=False, help="Mode of operation: scorer or critic", default=EvaluationMode.CRITIC)
    return parser


def retrieve_args() -> Namespace:
    parser = define_parser()
    return parser.parse_args()


def main():
    load_dotenv(dotenv_path=DOTENV)
    ensure_output_dirs()

    args = retrieve_args()
    openrouter_api_key = getenv("OPENROUTER_API_KEY", "")

    if args.mode == AgentMode.SINGLE:
        results = run_single_agent(api_key=openrouter_api_key, evaluation=args.evaluation)
    else:
        results = run_multi_agent(api_key=openrouter_api_key, evaluation=args.evaluation)

    Logger.log_info(f"Completed! Results: {results}")


if __name__ == "__main__":
    main()