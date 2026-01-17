"""
Evaluation tools for comparing generated diagrams against gold standards.
"""

import logging
from typing import Dict, Set, Tuple

try:
    from sentence_transformers import SentenceTransformer, util
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("sentence-transformers not available. Install with: pip install sentence-transformers")

try:
    from .models import EvaluationMetrics
    from .plantuml_tools import PlantUMLParser
except ImportError:
    from models import EvaluationMetrics
    from plantuml_tools import PlantUMLParser

logger = logging.getLogger(__name__)


class DiagramEvaluator:
    """
    Evaluator for comparing generated diagrams against gold standards.
    
    Computes precision, recall, and F1 scores for classes, attributes,
    and relationships. Supports semantic similarity matching for class names
    using BERT embeddings.
    """
    
    def __init__(
        self,
        gold_plantuml: str,
        pred_plantuml: str,
        embeder_model: str,
        similarity_threshold: float = 0.85
    ):
        """
        Initialize evaluator with gold and predicted diagrams.
        
        Args:
            gold_plantuml: Gold standard PlantUML code
            pred_plantuml: Predicted PlantUML code
            embeder_model: Embedding model name or path
            similarity_threshold: Minimum cosine similarity for class/attribute matches (0.0-1.0)
        
        Raises:
            ImportError: If sentence-transformers is not installed
            RuntimeError: If BERT model fails to load
        """
        if not BERT_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for semantic matching. "
                "Install with: pip install sentence-transformers"
            )
        
        self.gold_parser = PlantUMLParser(gold_plantuml)
        self.pred_parser = PlantUMLParser(pred_plantuml)
        self.similarity_threshold = similarity_threshold
        
        # Initialize BERT model
        try:
            self.bert_model = SentenceTransformer(embeder_model)
            logger.info("BERT model loaded for semantic matching")
        except Exception as e:
            raise RuntimeError(f"Failed to load BERT model: {e}")
    
    def _normalize_attr(self, attr_str: str) -> str:
        """Normalize attribute strings for comparison."""
        return attr_str.split(':')[0].strip().lower()
    
    def _normalize_rel_type(self, rel_type: str) -> str:
        """Normalize relationship types."""
        mapping = {
            '<|--': 'INHERITANCE',
            '--|>': 'INHERITANCE',
            '*--': 'COMPOSITION',
            '--*': 'COMPOSITION',
            'o--': 'AGGREGATION',
            '--o': 'AGGREGATION',
            '--': 'ASSOCIATION',
            '<--': 'ASSOCIATION',
            '-->': 'ASSOCIATION'
        }
        return mapping.get(rel_type, 'ASSOCIATION')
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts using BERT embeddings.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        if text1.lower().strip() == text2.lower().strip():
            return 1.0
        
        embeddings = self.bert_model.encode(
            [text1, text2],
            show_progress_bar=False,
            normalize_embeddings=True
        )
        similarity = util.cos_sim(embeddings[0], embeddings[1])
        similarity_score = float(similarity)
        logger.debug(f"Similarity between '{text1}' and '{text2}': {similarity_score:.4f}")
        return similarity_score
    
    def _generic_fuzzy_match(
        self,
        gold_items: Set,
        pred_items: Set,
        similarity_func
    ) -> Tuple[Set[Tuple], Set, Set]:
        """
        Generic matching algorithm using semantic similarity.
        
        Args:
            gold_items: Set of gold standard items
            pred_items: Set of predicted items
            similarity_func: Function(gold_item, pred_item) -> float for computing similarity
            
        Returns:
            Tuple of (matched_pairs, unmatched_gold, unmatched_pred)
        """
        matched_pairs = set()
        remaining_gold = set(gold_items)
        remaining_pred = set(pred_items)
        
        for gold_item in list(remaining_gold):
            best_match = None
            best_score = 0.0
            
            for pred_item in remaining_pred:
                similarity = similarity_func(gold_item, pred_item)
                if similarity >= self.similarity_threshold and similarity > best_score:
                    best_score = similarity
                    best_match = pred_item
                    logger.debug(f"Found candidate match: {gold_item} -> {pred_item} (score: {best_score:.4f})")
            
            if best_match:
                matched_pairs.add((gold_item, best_match))
                remaining_gold.discard(gold_item)
                remaining_pred.discard(best_match)
                logger.debug(f"Matched: {gold_item} -> {best_match} (score: {best_score:.4f})")
            else:
                logger.debug(f"No match found for: {gold_item} (best score: {best_score:.4f}, threshold: {self.similarity_threshold})")
        
        return matched_pairs, remaining_gold, remaining_pred
    
    def _calc_metrics_from_counts(self, tp: int, fp: int, fn: int) -> EvaluationMetrics:
        """
        Calculate precision, recall, and F1 from TP/FP/FN counts.
        
        Args:
            tp: True positives
            fp: False positives
            fn: False negatives
            
        Returns:
            EvaluationMetrics object
        """
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        
        return EvaluationMetrics(
            precision=round(precision, 2),
            recall=round(recall, 2),
            f1=round(f1, 2)
        )
    
    def _fuzzy_match_classes(
        self,
        gold_classes: Set[str],
        pred_classes: Set[str]
    ) -> Tuple[Set[Tuple[str, str]], Set[str], Set[str]]:
        """
        Match classes using semantic similarity.
        
        Args:
            gold_classes: Set of gold standard class names
            pred_classes: Set of predicted class names
            
        Returns:
            Tuple of (matched_pairs, unmatched_gold, unmatched_pred)
        """
        return self._generic_fuzzy_match(
            gold_classes,
            pred_classes,
            self._compute_similarity
        )
    
    def _fuzzy_match_attributes(
        self,
        gold_attrs: Set[Tuple[str, str]],
        pred_attrs: Set[Tuple[str, str]],
        class_mapping: Dict[str, str]
    ) -> Tuple[Set[Tuple[Tuple[str, str], Tuple[str, str]]], Set[Tuple[str, str]], Set[Tuple[str, str]]]:
        """
        Match attributes using semantic similarity, grouped by class.
        
        Args:
            gold_attrs: Set of (class_name, attribute_name) tuples from gold standard
            pred_attrs: Set of (class_name, attribute_name) tuples from predictions
            class_mapping: Dict mapping predicted class names to gold class names
            
        Returns:
            Tuple of (matched_pairs, unmatched_gold, unmatched_pred)
        """
        matched_pairs = set()
        remaining_gold = set(gold_attrs)
        remaining_pred = set(pred_attrs)
        
        # Group attributes by class for efficient matching
        gold_by_class = {}
        for cls, attr in remaining_gold:
            if cls not in gold_by_class:
                gold_by_class[cls] = set()
            gold_by_class[cls].add(attr)
        
        pred_by_class = {}
        for cls, attr in remaining_pred:
            # Use the mapped class name to group predicted attributes
            mapped_cls = class_mapping.get(cls, cls)
            if mapped_cls not in pred_by_class:
                pred_by_class[mapped_cls] = []
            pred_by_class[mapped_cls].append((cls, attr))
        
        # Match attributes for each gold class
        for gold_cls, gold_attrs_set in gold_by_class.items():
            # Get predicted attributes mapped to this gold class
            pred_attrs_list = pred_by_class.get(gold_cls, [])
            if not pred_attrs_list:
                continue
            
            pred_attrs_set = {attr for _, attr in pred_attrs_list}
            
            # Define similarity function for this class
            def attr_similarity(gold_attr, pred_attr):
                return self._compute_similarity(gold_attr, pred_attr)
            
            # Use generic matcher for attributes within this class
            attr_matches, _, _ = self._generic_fuzzy_match(
                gold_attrs_set,
                pred_attrs_set,
                attr_similarity
            )
            
            # Convert back to (class, attr) tuples using original predicted class names
            for gold_attr, pred_attr in attr_matches:
                # Find the original predicted class name
                orig_pred_cls = next(cls for cls, attr in pred_attrs_list if attr == pred_attr)
                matched_pairs.add(((gold_cls, gold_attr), (orig_pred_cls, pred_attr)))
                remaining_gold.discard((gold_cls, gold_attr))
                remaining_pred.discard((orig_pred_cls, pred_attr))
        # Log attribute matching details
        if matched_pairs:
            logger.info(f"Matched attributes: {matched_pairs}")
        if remaining_gold:
            logger.warning(f"Unmatched gold attributes: {remaining_gold}")
        if remaining_pred:
            logger.warning(f"Unmatched predicted attributes: {remaining_pred}")

        return matched_pairs, remaining_gold, remaining_pred
    
    def _fuzzy_match_relationships(
        self,
        gold_rels: Set[Tuple[str, str, str]],
        pred_rels: Set[Tuple[str, str, str]],
        class_mapping: Dict[str, str]
    ) -> Tuple[Set[Tuple], Set[Tuple[str, str, str]], Set[Tuple[str, str, str]]]:
        """
        Match relationships using semantic similarity on source and target classes.
        
        Args:
            gold_rels: Set of (source, target, type) tuples from gold standard
            pred_rels: Set of (source, target, type) tuples from predictions
            class_mapping: Dict mapping predicted class names to gold class names
            
        Returns:
            Tuple of (matched_pairs, unmatched_gold, unmatched_pred)
        """
        def rel_similarity(gold_rel, pred_rel):
            gold_src, gold_tgt, gold_type = gold_rel
            pred_src, pred_tgt, pred_type = pred_rel
            
            # Relationship type must match exactly
            if gold_type != pred_type:
                return 0.0
            
            # Map predicted classes to gold classes
            mapped_pred_src = class_mapping.get(pred_src, pred_src)
            mapped_pred_tgt = class_mapping.get(pred_tgt, pred_tgt)
            
            # Compute similarity for source and target
            src_sim = self._compute_similarity(gold_src, mapped_pred_src)
            tgt_sim = self._compute_similarity(gold_tgt, mapped_pred_tgt)
            
            # Return average similarity (both must be above threshold)
            avg_sim = (src_sim + tgt_sim) / 2.0
            return avg_sim if src_sim >= self.similarity_threshold and tgt_sim >= self.similarity_threshold else 0.0
        
        matched_pairs, remaining_gold, remaining_pred = self._generic_fuzzy_match(
            gold_rels,
            pred_rels,
            rel_similarity
        )

        # Log relationship matching details
        if matched_pairs:
            logger.info(f"Matched relationships: {matched_pairs}")
        if remaining_gold:
            logger.warning(f"Unmatched gold relationships: {remaining_gold}")
        if remaining_pred:
            logger.warning(f"Unmatched predicted relationships: {remaining_pred}")

        return matched_pairs, remaining_gold, remaining_pred
    
    def _calculate_metrics(
        self,
        gold_set: Set,
        pred_set: Set
    ) -> EvaluationMetrics:
        """
        Calculate precision, recall, and F1 scores.
        
        Args:
            gold_set: Set of gold standard elements
            pred_set: Set of predicted elements
            
        Returns:
            EvaluationMetrics object
        """
        tp = len(gold_set.intersection(pred_set))
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
        
        return self._calc_metrics_from_counts(tp, fp, fn)
    
    def get_metrics(self) -> Dict[str, EvaluationMetrics]:
        """
        Get all evaluation metrics.
        
        Returns:
            Dictionary with metrics for classes, attributes, and relationships
        """
        # Extract and normalize classes
        gold_classes = {c.lower() for c in self.gold_parser.classes.keys()}
        pred_classes = {c.lower() for c in self.pred_parser.classes.keys()}
        
        # Match classes using semantic similarity
        matched_class_pairs, unmatched_gold, unmatched_pred = self._fuzzy_match_classes(
            gold_classes, pred_classes
        )
        
        # Create mapping for matched classes (pred -> gold)
        class_mapping = {pred: gold for gold, pred in matched_class_pairs}
        
        # Log class matching results
        if matched_class_pairs:
            logger.info(f"Matched classes: {matched_class_pairs}")
        if unmatched_gold:
            logger.warning(f"Unmatched gold classes: {unmatched_gold}")
        if unmatched_pred:
            logger.warning(f"Unmatched predicted classes: {unmatched_pred}")
        
        # Calculate class metrics
        class_metrics = self._calc_metrics_from_counts(
            len(matched_class_pairs),
            len(unmatched_pred),
            len(unmatched_gold)
        )
        
        # Extract attributes
        gold_attrs = set()
        for cls, info in self.gold_parser.classes.items():
            for attr in info['attributes']:
                gold_attrs.add((cls.lower(), self._normalize_attr(attr)))
        
        pred_attrs = set()
        for cls, info in self.pred_parser.classes.items():
            cls_lower = cls.lower()
            for attr in info['attributes']:
                pred_attrs.add((cls_lower, self._normalize_attr(attr)))
        # Log extracted attribute sets before matching
        logger.debug(f"Gold attributes extracted: {gold_attrs}")
        logger.debug(f"Pred attributes extracted: {pred_attrs}")

        # Match attributes using semantic similarity
        matched_attr_pairs, unmatched_gold_attrs, unmatched_pred_attrs = self._fuzzy_match_attributes(
            gold_attrs, pred_attrs, class_mapping
        )
        
        # Calculate attribute metrics
        attr_metrics = self._calc_metrics_from_counts(
            len(matched_attr_pairs),
            len(unmatched_pred_attrs),
            len(unmatched_gold_attrs)
        )
        
        # Extract relationships
        gold_rels = {
            (r['source'].lower(), r['target'].lower(), self._normalize_rel_type(r['type']))
            for r in self.gold_parser.relationships
            if r.get('source') and r.get('target')
        }
        pred_rels = {
            (r['source'].lower(), r['target'].lower(), self._normalize_rel_type(r['type']))
            for r in self.pred_parser.relationships
            if r.get('source') and r.get('target')
        }

        # Log extracted relationship sets before matching
        logger.debug(f"Gold relationships extracted: {gold_rels}")
        logger.debug(f"Pred relationships extracted: {pred_rels}")

        # Match relationships using semantic similarity
        matched_rel_pairs, unmatched_gold_rels, unmatched_pred_rels = self._fuzzy_match_relationships(
            gold_rels, pred_rels, class_mapping
        )
        
        # Calculate relationship metrics
        rel_metrics = self._calc_metrics_from_counts(
            len(matched_rel_pairs),
            len(unmatched_pred_rels),
            len(unmatched_gold_rels)
        )
        
        return {
            "classes": class_metrics,
            "attributes": attr_metrics,
            "relationships": rel_metrics,
        }


def evaluate_diagram(
    gold_standard: str,
    generated_diagram: str,
    embeder_model: str,
    similarity_threshold: float = 0.85
) -> Dict[str, EvaluationMetrics]:
    """
    Evaluate a generated diagram against gold standard using semantic similarity.
    
    Args:
        gold_standard: Gold standard PlantUML code
        generated_diagram: Generated PlantUML code
        embeder_model: Embedding model name or path
        similarity_threshold: Minimum similarity for class/attribute matches (0.0-1.0)
        
    Returns:
        Dictionary of evaluation metrics
    
    Raises:
        ImportError: If sentence-transformers is not installed
        RuntimeError: If BERT model fails to load
    """
    evaluator = DiagramEvaluator(
        gold_standard,
        generated_diagram,
        embeder_model,
        similarity_threshold=similarity_threshold
    )
    return evaluator.get_metrics()
