"""
Evaluation tools for comparing generated diagrams against gold standards.
"""

from typing import Dict, Set, Tuple
from sentence_transformers import SentenceTransformer, util
from src.core.models import EvaluationMetrics
from src.core.plantuml import PlantUMLParser
from src.core.logger import Logger


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
        embeder_model,  # Can be str or SentenceTransformer
        similarity_threshold: float,
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

        self.gold_parser = PlantUMLParser(gold_plantuml)
        self.pred_parser = PlantUMLParser(pred_plantuml)
        
        try:
            if isinstance(embeder_model, str):
                self.bert_model = SentenceTransformer(embeder_model)
                Logger.log_info("BERT model loaded for semantic matching")
            else:
                self.bert_model = embeder_model
                Logger.log_info("Using pre-loaded BERT model for semantic matching")
        except Exception as e:
            raise RuntimeError(f"Failed to load BERT model: {e}")
        
        self.similarity_threshold = similarity_threshold
    
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
        Logger.log_debug(f"Similarity between '{text1}' and '{text2}': {similarity_score:.4f}")
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
        

        for gold_item in gold_items:
            if gold_item not in remaining_gold:
                continue
                
            best_match = None
            best_score = 0.0
            
            for pred_item in remaining_pred:
                similarity = similarity_func(gold_item, pred_item)
                if similarity >= self.similarity_threshold and similarity > best_score:
                    best_score = similarity
                    best_match = pred_item
                    Logger.log_debug(f"Found candidate match: {gold_item} -> {pred_item} (score: {best_score:.4f})")
            
            if best_match:
                matched_pairs.add((gold_item, best_match))
                remaining_gold.discard(gold_item)
                remaining_pred.discard(best_match)
                Logger.log_debug(f"Matched: {gold_item} -> {best_match} (score: {best_score:.4f})")
            else:
                Logger.log_debug(f"No match found for: {gold_item} (best score: {best_score:.4f}, threshold: {self.similarity_threshold})")
        
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
        # Group by class for efficiency
        gold_by_class = {}
        for cls, attr in gold_attrs:
            gold_by_class.setdefault(cls, set()).add(attr)
        
        pred_by_class = {}
        for cls, attr in pred_attrs:
            mapped_cls = class_mapping.get(cls, cls)
            pred_by_class.setdefault(mapped_cls, []).append((cls, attr))
        
        matched_pairs = set()
        remaining_gold = set(gold_attrs)
        remaining_pred = set(pred_attrs)
        
        # Match within each class
        for gold_cls, gold_attr_set in gold_by_class.items():
            if gold_cls not in pred_by_class:
                continue
            
            pred_attr_list = pred_by_class[gold_cls]
            pred_attr_set = {attr for _, attr in pred_attr_list}
            
            # Use generic matcher on this subset
            attr_matches, _, _ = self._generic_fuzzy_match(
                gold_attr_set,
                pred_attr_set,
                self._compute_similarity
            )
            
            # Record full matches with class info
            for gold_attr, pred_attr in attr_matches:
                orig_pred_cls = next(cls for cls, attr in pred_attr_list if attr == pred_attr)
                matched_pairs.add(((gold_cls, gold_attr), (orig_pred_cls, pred_attr)))
                remaining_gold.discard((gold_cls, gold_attr))
                remaining_pred.discard((orig_pred_cls, pred_attr))    

        if matched_pairs:
            Logger.log_info(f"Matched attributes: {matched_pairs}")
        if remaining_gold:
            Logger.log_warning(f"Unmatched gold attributes: {remaining_gold}")
        if remaining_pred:
            Logger.log_warning(f"Unmatched predicted attributes: {remaining_pred}")
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

        # Similarity function to use in the generic matcher
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
            return avg_sim
        
        matched_pairs, remaining_gold, remaining_pred = self._generic_fuzzy_match(
            gold_rels,
            pred_rels,
            rel_similarity
        )

        if matched_pairs:
            Logger.log_info(f"Matched relationships: {matched_pairs}")
        if remaining_gold:
            Logger.log_warning(f"Unmatched gold relationships: {remaining_gold}")
        if remaining_pred:
            Logger.log_warning(f"Unmatched predicted relationships: {remaining_pred}")
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
        # normalize classes
        gold_classes = {c.lower() for c in self.gold_parser.classes.keys()}
        pred_classes = {c.lower() for c in self.pred_parser.classes.keys()}
        
        # Match classes using semantic similarity
        matched_class_pairs, unmatched_gold, unmatched_pred = self._fuzzy_match_classes(
            gold_classes, pred_classes
        )
        
        
        class_mapping = {pred: gold for gold, pred in matched_class_pairs}
        

        if matched_class_pairs:
            Logger.log_info(f"Matched classes: {matched_class_pairs}")
        if unmatched_gold:
            Logger.log_warning(f"Unmatched gold classes: {unmatched_gold}")
        if unmatched_pred:
            Logger.log_warning(f"Unmatched predicted classes: {unmatched_pred}")
        
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
        Logger.log_debug(f"Gold attributes extracted: {gold_attrs}")
        Logger.log_debug(f"Pred attributes extracted: {pred_attrs}")

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
        Logger.log_debug(f"Gold relationships extracted: {gold_rels}")
        Logger.log_debug(f"Pred relationships extracted: {pred_rels}")

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
    embeder_model,  # Can be str or SentenceTransformer
    similarity_threshold: float,
) -> Dict[str, EvaluationMetrics]:
    """
    Evaluate a generated diagram against gold standard using semantic similarity.
    
    Args:
        gold_standard: Gold standard PlantUML code
        generated_diagram: Generated PlantUML code
        embeder_model: Embedding model name or path
        similarity_threshold: Minimum similarity for class/attribute matches (0.0-1.0).
        
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
        similarity_threshold=similarity_threshold,
    )
    return evaluator.get_metrics()