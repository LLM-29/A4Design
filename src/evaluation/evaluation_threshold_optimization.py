import json
import random
import numpy as np
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

LABELS_FILE = Path(__file__).parent.parent.parent / "data" / "processed" / "labels.json"
MODEL_NAME = "all-mpnet-base-v2"
random.seed(42)
np.random.seed(42)

def load_contrastive_pairs(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    
    positives = [] # Real matches from labels
    hard_negatives = [] # Real mismatches from labels
    all_gt_classes = []
    all_gen_classes = []

    for diag in data["diagram_matches"]:
        for item in diag["class_matches"]:
            gt, gen = item["ground_truth"], item["generated"]
            if gt: all_gt_classes.append(gt)
            if gen: all_gen_classes.append(gen)
            
            if gt and gen:
                pair = {"t1": gt, "t2": gen, "match": 1 if item["match"] else 0}
                if item["match"]: positives.append(pair)
                else: hard_negatives.append(pair)

    # SYNTHETIC NEGATIVE SAMPLING
    synthetic_negatives = []
    for _ in range(len(positives)):
        t1 = random.choice(all_gt_classes)
        t2 = random.choice(all_gen_classes)
        # Ensure we don't accidentally create a positive match
        if t1.lower() != t2.lower():
            synthetic_negatives.append({"t1": t1, "t2": t2, "match": 0})

    return positives + hard_negatives + synthetic_negatives

def calibrate(pairs, model, k=5):
    t1_list = [p["t1"] for p in pairs]
    t2_list = [p["t2"] for p in pairs]
    y_true = [p["match"] for p in pairs]

    emb1 = model.encode(t1_list, convert_to_tensor=True)
    emb2 = model.encode(t2_list, convert_to_tensor=True)
    sims = [util.cos_sim(e1, e2).item() for e1, e2 in zip(emb1, emb2)]

    thresholds = np.linspace(0.1, 0.9, 100)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    avg_f1_scores = []
    for t in thresholds:
        fold_f1s = []
        for train_index, test_index in kf.split(pairs):
            y_pred_test = [1 if sims[i] >= t else 0 for i in test_index]
            y_true_test = [y_true[i] for i in test_index]
            f1 = f1_score(y_true_test, y_pred_test)
            fold_f1s.append(f1)
        avg_f1 = np.mean(fold_f1s)
        avg_f1_scores.append(avg_f1)
    
    best_idx = np.argmax(avg_f1_scores)
    best_t = thresholds[best_idx]
    best_f1 = avg_f1_scores[best_idx]
            
    return best_t, best_f1, thresholds, avg_f1_scores

if __name__ == "__main__":
    model = SentenceTransformer(MODEL_NAME)
    pairs = load_contrastive_pairs(LABELS_FILE)
    
    thresh, f1, thresholds, f1_scores = calibrate(pairs, model, k=10)
    
    print(f"\n--- CALIBRATION WITH 10-FOLD CV ---")
    print(f"Total pairs analyzed: {len(pairs)}")
    print(f"Optimal Threshold:    {thresh:.2f}")
    print(f"Max Average F1 Score: {f1:.4f}")
    
    # Plot F1 scores vs thresholds
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, marker='o', linestyle='-', color='b', markersize=3)
    plt.axvline(x=thresh, color='r', linestyle='--', label=f'Optimal Threshold: {thresh:.2f}')
    plt.title('Average F1 Score vs Similarity Threshold (10-Fold CV)')
    plt.xlabel('Similarity Threshold')
    plt.ylabel('Average F1 Score')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    plot_dir = Path(__file__).parent.parent.parent / "output" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / "threshold_optimization_plot.png"
    plt.savefig(plot_path)
    plt.show()
    
    print(f"Plot saved to: {plot_path}")