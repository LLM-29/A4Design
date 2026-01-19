"""
Evaluation tools for comparing generated diagrams against gold standards.
"""

import matplotlib.pyplot as plt
import numpy as np

from typing import Dict, Set, Tuple, List, Optional
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import auc
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
        embeder_model: str,
        similarity_threshold: float = 0.85,
        auto_threshold: bool = False
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
            self.bert_model = SentenceTransformer(embeder_model)
            Logger.log_info("BERT model loaded for semantic matching")
        except Exception as e:
            raise RuntimeError(f"Failed to load BERT model: {e}")
        
        # Auto-determine threshold using ROC curve if requested
        if auto_threshold:
            self.similarity_threshold = self._find_optimal_threshold()
            Logger.log_info(f"Auto-selected optimal threshold: {self.similarity_threshold:.4f}")
        else:
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
        matched_pairs = set()
        remaining_gold = set(gold_attrs)
        remaining_pred = set(pred_attrs)
        
        # Group attributes by class for matching
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
            pred_attrs_list = pred_by_class.get(gold_cls, [])
            if not pred_attrs_list:
                continue
            
            pred_attrs_set = {attr for _, attr in pred_attrs_list}
            
            # Similarity function for this class to use in the generic matcher
            def attr_similarity(gold_attr, pred_attr):
                return self._compute_similarity(gold_attr, pred_attr)
            

            attr_matches, _, _ = self._generic_fuzzy_match(
                gold_attrs_set,
                pred_attrs_set,
                attr_similarity
            )
            

            for gold_attr, pred_attr in attr_matches:
                orig_pred_cls = next(cls for cls, attr in pred_attrs_list if attr == pred_attr)
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
            return avg_sim if src_sim >= self.similarity_threshold and tgt_sim >= self.similarity_threshold else 0.0
        
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
    
    def _compute_similarity_matrix(
        self,
        gold_items: List,
        pred_items: List
    ) -> np.ndarray:
        """
        Compute pairwise similarity matrix between gold and predicted items.
        
        Args:
            gold_items: List of gold standard items (strings)
            pred_items: List of predicted items (strings)
            
        Returns:
            Similarity matrix of shape (len(gold_items), len(pred_items))
        """
        if not gold_items or not pred_items:
            return np.array([])
        
        similarities = np.zeros((len(gold_items), len(pred_items)))
        for i, gold_item in enumerate(gold_items):
            for j, pred_item in enumerate(pred_items):
                similarities[i, j] = self._compute_similarity(str(gold_item), str(pred_item))
        
        return similarities
    
    def _compute_metrics_at_threshold(
        self,
        similarity_matrix: np.ndarray,
        threshold: float
    ) -> Tuple[float, float, float]:
        """
        Compute TPR, FPR, and F1 at a given threshold.
        
        Args:
            similarity_matrix: Pairwise similarity matrix
            threshold: Similarity threshold to test
            
        Returns:
            Tuple of (TPR, FPR, F1)
        """
        if similarity_matrix.size == 0:
            return 0.0, 0.0, 0.0
        
        n_gold, n_pred = similarity_matrix.shape
        
        # Greedy matching: for each gold item, find best match above threshold
        matched_gold = set()
        matched_pred = set()
        
        for i in range(n_gold):
            best_j = -1
            best_sim = threshold
            
            for j in range(n_pred):
                if j not in matched_pred and similarity_matrix[i, j] > best_sim:
                    best_sim = similarity_matrix[i, j]
                    best_j = j
            
            if best_j >= 0:
                matched_gold.add(i)
                matched_pred.add(best_j)
        
        tp = len(matched_gold)
        fp = n_pred - len(matched_pred)
        fn = n_gold - len(matched_gold)
        tn = 0  # Not applicable in this context
        
        # Calculate metrics
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else (1.0 if fp > 0 else 0.0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0.0
        
        return tpr, fpr, f1
    
    def _find_optimal_threshold(
        self,
        thresholds: Optional[np.ndarray] = None,
        plot: bool = False
    ) -> float:
        """
        Find optimal similarity threshold using ROC curve analysis.
        Uses Youden's J statistic (TPR - FPR) to select the best threshold.
        
        Args:
            thresholds: Array of thresholds to test. If None, uses np.linspace(0.5, 1.0, 51)
            plot: Whether to plot the ROC curve
            
        Returns:
            Optimal similarity threshold
        """
        if thresholds is None:
            thresholds = np.linspace(0.5, 1.0, 51)
        
        # Extract classes for ROC analysis
        gold_classes = list({c.lower() for c in self.gold_parser.classes.keys()})
        pred_classes = list({c.lower() for c in self.pred_parser.classes.keys()})
        
        if not gold_classes or not pred_classes:
            Logger.log_warning("No classes found for ROC analysis, using default threshold")
            return 0.85
        
        # Compute similarity matrix
        similarity_matrix = self._compute_similarity_matrix(gold_classes, pred_classes)
        
        if similarity_matrix.size == 0:
            Logger.log_warning("Empty similarity matrix, using default threshold")
            return 0.85
        
        # Compute metrics at each threshold
        tpr_values = []
        fpr_values = []
        f1_values = []
        
        for threshold in thresholds:
            tpr, fpr, f1 = self._compute_metrics_at_threshold(similarity_matrix, threshold)
            tpr_values.append(tpr)
            fpr_values.append(fpr)
            f1_values.append(f1)
        
        tpr_values = np.array(tpr_values)
        fpr_values = np.array(fpr_values)
        f1_values = np.array(f1_values)

        # --------------------------------
        
        # Calculate ROC AUC using trapezoidal rule
        # Sort by FPR for proper AUC calculation
        sorted_indices = np.argsort(fpr_values)
        fpr_sorted = fpr_values[sorted_indices]
        tpr_sorted = tpr_values[sorted_indices]
        
        roc_auc = auc(fpr_sorted, tpr_sorted)
        Logger.log_info(f"ROC AUC Score: {roc_auc:.4f}")
        
        # Find optimal threshold using multiple strategies
        # Strategy 1: Maximum F1 score
        f1_optimal_idx = np.argmax(f1_values)
        
        # Strategy 2: Youden's J statistic (TPR - FPR)
        youden_j = tpr_values - fpr_values
        youden_optimal_idx = np.argmax(youden_j)
        
        # Strategy 3: Point closest to (0, 1) - perfect classifier
        distances = np.sqrt((1 - tpr_values)**2 + fpr_values**2)
        closest_optimal_idx = np.argmin(distances)
        
        # Use F1 score as primary criterion (better for imbalanced matching)
        optimal_idx = f1_optimal_idx
        optimal_threshold = thresholds[optimal_idx]
        
        Logger.log_info(f"ROC Analysis Results:")
        Logger.log_info(f"  Optimal threshold (F1): {thresholds[f1_optimal_idx]:.4f} (F1={f1_values[f1_optimal_idx]:.3f})")
        Logger.log_info(f"  Optimal threshold (Youden): {thresholds[youden_optimal_idx]:.4f} (J={youden_j[youden_optimal_idx]:.3f})")
        Logger.log_info(f"  Optimal threshold (Distance): {thresholds[closest_optimal_idx]:.4f} (dist={distances[closest_optimal_idx]:.3f})")
        Logger.log_info(f"  Selected threshold: {optimal_threshold:.4f}")
        
        # Plot ROC curve if requested
        if plot:
            self._plot_roc_curve(
                fpr_values, tpr_values, f1_values, thresholds, 
                optimal_idx, f1_optimal_idx, youden_optimal_idx
            )
        
        return float(optimal_threshold)
    
    def _plot_roc_curve(
        self,
        fpr_values: np.ndarray,
        tpr_values: np.ndarray,
        f1_values: np.ndarray,
        thresholds: np.ndarray,
        optimal_idx: int,
        f1_idx: int,
        youden_idx: int
    ) -> None:
        """
        Plot ROC curve with optimal thresholds marked.
        
        Args:
            fpr_values: False positive rates
            tpr_values: True positive rates
            f1_values: F1 scores
            thresholds: Threshold values
            optimal_idx: Index of selected optimal threshold
            f1_idx: Index of F1 optimal threshold
            youden_idx: Index of Youden's J optimal threshold
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Sort by FPR for proper ROC curve
        sorted_indices = np.argsort(fpr_values)
        fpr_sorted = fpr_values[sorted_indices]
        tpr_sorted = tpr_values[sorted_indices]
        
        # Calculate AUC
        roc_auc = auc(fpr_sorted, tpr_sorted)
        
        # Plot 1: ROC Curve
        ax1.plot(fpr_sorted, tpr_sorted, 'b-', linewidth=2, label=f'ROC Curve (AUC={roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
        
        # Mark optimal thresholds
        ax1.plot(fpr_values[optimal_idx], tpr_values[optimal_idx], 'go', 
                markersize=12, label=f'Selected (t={thresholds[optimal_idx]:.2f})', zorder=5)
        ax1.plot(fpr_values[youden_idx], tpr_values[youden_idx], 'mo', 
                markersize=10, label=f'Youden (t={thresholds[youden_idx]:.2f})', zorder=5)
        
        ax1.set_xlabel('False Positive Rate', fontsize=12)
        ax1.set_ylabel('True Positive Rate (Recall)', fontsize=12)
        ax1.set_title('ROC Curve for Similarity Threshold Selection', fontsize=14, fontweight='bold')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([-0.05, 1.05])
        ax1.set_ylim([-0.05, 1.05])
        
        # Plot 2: F1 Score vs Threshold
        ax2.plot(thresholds, f1_values, 'b-', linewidth=2, label='F1 Score')
        ax2.axvline(thresholds[f1_idx], color='g', linestyle='--', linewidth=2, 
                   label=f'Max F1 (t={thresholds[f1_idx]:.2f}, F1={f1_values[f1_idx]:.3f})')
        ax2.plot(thresholds[f1_idx], f1_values[f1_idx], 'go', markersize=12, zorder=5)
        
        ax2.set_xlabel('Similarity Threshold', fontsize=12)
        ax2.set_ylabel('F1 Score', fontsize=12)
        ax2.set_title('F1 Score vs Similarity Threshold', fontsize=14, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([thresholds[0] - 0.05, thresholds[-1] + 0.05])
        
        plt.tight_layout()
        plt.savefig('roc_curve_threshold_analysis.png', dpi=300, bbox_inches='tight')
        Logger.log_info("ROC curve saved to roc_curve_threshold_analysis.png")
        plt.close()
    
    @classmethod
    def find_optimal_threshold(
        cls,
        gold_plantuml: str,
        pred_plantuml: str,
        embeder_model: str,
        thresholds: Optional[np.ndarray] = None,
        plot: bool = False
    ) -> float:
        """
        Class method to find optimal threshold without full evaluation.
        
        Args:
            gold_plantuml: Gold standard PlantUML code
            pred_plantuml: Predicted PlantUML code
            embeder_model: Embedding model name or path
            thresholds: Array of thresholds to test. If None, uses np.linspace(0.5, 1.0, 51)
            plot: Whether to plot the ROC curve
            
        Returns:
            Optimal similarity threshold
        """
        # Create temporary evaluator with default threshold
        temp_evaluator = cls(
            gold_plantuml=gold_plantuml,
            pred_plantuml=pred_plantuml,
            embeder_model=embeder_model,
            similarity_threshold=0.85,
            auto_threshold=False
        )
        
        return temp_evaluator._find_optimal_threshold(thresholds=thresholds, plot=plot)


def evaluate_diagram(
    gold_standard: str,
    generated_diagram: str,
    embeder_model: str,
    similarity_threshold: Optional[float] = None,
    auto_threshold: bool = True
) -> Dict[str, EvaluationMetrics]:
    """
    Evaluate a generated diagram against gold standard using semantic similarity.
    
    Args:
        gold_standard: Gold standard PlantUML code
        generated_diagram: Generated PlantUML code
        embeder_model: Embedding model name or path
        similarity_threshold: Minimum similarity for class/attribute matches (0.0-1.0).
                            If None and auto_threshold=False, defaults to 0.75.
        auto_threshold: If True, automatically determine optimal threshold using ROC analysis
        
    Returns:
        Dictionary of evaluation metrics
    
    Raises:
        ImportError: If sentence-transformers is not installed
        RuntimeError: If BERT model fails to load
    """
    if similarity_threshold is None and not auto_threshold:
        similarity_threshold = 0.75
    
    evaluator = DiagramEvaluator(
        gold_standard,
        generated_diagram,
        embeder_model,
        similarity_threshold=similarity_threshold,
        auto_threshold=auto_threshold
    )
    return evaluator.get_metrics()


# Example usage:
# 
# # Automatic threshold selection:
# metrics = evaluate_diagram(gold, pred, model, auto_threshold=True)
# 
# # Or find optimal threshold first:
# optimal_threshold = DiagramEvaluator.find_optimal_threshold(gold, pred, model, plot=True)
# metrics = evaluate_diagram(gold, pred, model, similarity_threshold=optimal_threshold)
