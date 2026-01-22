import logging

from typing import Any, Dict, List

from src.core.models import EvaluationMetrics, CritiqueReport, ScoredCritiqueReport


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Logger:
    @staticmethod
    def log_info(line: str) -> None:
        logger.info(line)


    @staticmethod
    def log_error(line: str) -> None:
        logger.error(line)


    @staticmethod
    def log_warning(line: str) -> None:
        logger.warning(line)


    @staticmethod
    def log_debug(line: str) -> None:
        logger.debug(line)


    @staticmethod
    def log_title(title: str) -> None:
        logger.info("="*60)
        logger.info(title)
        logger.info("="*60)


    @staticmethod
    def log_diagram(diagram_url: str, diagram: str) -> None:
        logger.info(f"Diagram URL: {diagram_url}")
        logger.info("Generated Diagram:")
        logger.info(diagram)


    @staticmethod
    def log_generation(final_output: Any) -> None:
        Logger.log_title("WORKFLOW COMPLETED")
        logger.info(f"Iterations: {final_output['iterations']}")
        logger.info(f"Syntax Valid: {final_output['syntax_valid']}")
        logger.info(f"Logic Valid: {final_output['logic_valid']}")
        
        if final_output.get('best_diagram') and not final_output['logic_valid']:
            if final_output['best_diagram'] != final_output['current_diagram']:
                logger.info(
                    "Using BEST diagram instead of final (prevented regression)"
                )
                final_output['current_diagram'] = final_output['best_diagram']


    @staticmethod
    def log_result_metrics(metrics: Dict[str, EvaluationMetrics]) -> None:
        logger.info("EVALUATION METRICS")
        logger.info(f"Classes:       {metrics['classes']}")
        logger.info(f"Attributes:    {metrics['attributes']}")
        logger.info(f"Relationships: {metrics['relationships']}")
        
        average_f1 = (
            metrics['classes'].f1 +
            metrics['attributes'].f1 +
            metrics['relationships'].f1
        ) / 3.0
        
        logger.info(f"OVERALL F1 SCORE: {average_f1:.2f}")
    

    @staticmethod
    def log_run_start(exercise_name: str, requirements: str) -> None:
        Logger.log_title(f"RUNNING EXERCISE: {exercise_name}")
        logger.info(f"Requirements preview: {requirements[:200]}...")
    

    @staticmethod
    def log_run_output(iterations: int, syntax_valid: bool, logic_valid: bool) -> None:
        Logger.log_title("RUN COMPLETED")
        logger.info(f"Iterations: {iterations}")
        logger.info(f"Syntax Valid: {syntax_valid}")
        logger.info(f"Logic Valid: {logic_valid}")
        

    @staticmethod
    def log_models(first_model: str, second_model: str = None) -> None:
        if second_model:
            logger.info("Using models:")
            logger.info(f"  Decompose model: {first_model}")
            logger.info(f"  Generate model:  {second_model}")
        else:
            logger.info(f"Using model: {first_model}")
    

    @staticmethod
    def log_roc_analysis() -> None:
        pass


    @staticmethod
    def log_classes(classes: List) -> None:
        logger.info(f"Extracted {len(classes)} classes")
        logger.info(f"Extracted classes: {classes}")
    

    @staticmethod
    def log_relationships(relationships: List) -> None:
        logger.info(f"Extracted {len(relationships)} relationships")
        logger.info(f"Extracted relationships: {relationships}")
    

    @staticmethod
    def log_critique_report(report: CritiqueReport) -> None:
        logger.info(f"Extracted {len(report.findings)} findings")
        logger.info(f"Extracted report: {report}")
    

    @staticmethod
    def log_scored_report(report: ScoredCritiqueReport) -> None:
        logger.info(f"Extracted scores: {report.scores.syntax_score}")
        logger.info(f"Extracted scores: {report.scores.semantic_score}")
        logger.info(f"Extracted scores: {report.scores.pragmatic_score}")

        