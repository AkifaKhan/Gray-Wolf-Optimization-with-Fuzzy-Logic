"""
Fuzzy-GWO optimized Decision Tree (fuzzy_gwo_dt.py) — ENHANCED PERFORMANCE VERSION

Key Improvements:
 - Stronger and adaptive fuzzy influence: weight between fuzzy_a and linear_a depends on diversity.
 - Lévy flight injection is now adaptive (bigger when diversity is very low).
 - Smooth moving average of `a` to avoid oscillations.
 - Re-seeding of worst wolves when stagnation is detected.
 - Increased CV stability (2 repeats).
 - Exploration/exploitation balancing improved: smoother transitions, adaptive step blending.
"""

import os, sys, warnings
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings("ignore")

# ---------------- Fuzzy Import ----------------
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from fuzzy_controller import get_fuzzy_a
    print("Imported get_fuzzy_a from fuzzy_controller.py")
    HAS_FUZZY = True
except Exception as e:
    print(f"Warning: Could not import get_fuzzy_a ({e}). Falling back to linear schedule for `a`.")
    HAS_FUZZY = False
    def get_fuzzy_a(iter_ratio, diversity_val):
        return max(0.0, 2 - 2 * iter_ratio)

# ---------------- Load Data ----------------
DATAFILE = "sepsis_data_encoded.csv"
if not os.path.exists(DATAFILE):
    raise FileNotFoundError(f"Dataset '{DATAFILE}' not found in current folder.")

df = pd.read_csv(DATAFILE, encoding="windows-1252")

# --- Clean and standardize ---
df.columns = df.columns.str.strip().str.lower()
print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

# --- Identify target column ---
target_col = "inhospital_mortality"
if target_col not in df.columns:
    raise ValueError(f"Expected target column '{target_col}' not found. Found: {df.columns.tolist()}")

# --- Encode categorical columns (safety) ---
for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

# --- Define features and target ---
X = df.drop(columns=[target_col]).values
y = df[target_col].values
print(f"✅ Sepsis dataset prepared: {X.shape[0]} samples, {X.shape[1]} features.")

# ---------------- Utility ----------------
def levy_step(dim, scale=0.1):
    gauss = np.random.randn(dim)
    cauchy = np.random.standard_cauchy(dim)
    step = gauss / (np.abs(cauchy) + 1e-9)
    step = step / (np.std(step) + 1e-9)
    return scale * step

# ---------------- Fitness ----------------
def eval_fitness(position, X, y, cv_repeats=2):
    max_depth = int(np.round(position[0])); max_depth = np.clip(max_depth, 1, 20)
    min_samples_split = int(np.round(position[1])); min_samples_split = np.clip(min_samples_split, 2, 50)
    min_samples_leaf = int(np.round(position[2])); min_samples_leaf = np.clip(min_samples_leaf, 1, 20)
    max_features = float(position[3]); max_features = np.clip(max_features, 0.1, 1.0)

    class_counts = Counter(y)
    min_class_count = min(class_counts.values()) if class_counts else 0
    n_splits = min(3, min_class_count) if min_class_count >= 2 else 2

    scores = []
    for _ in range(cv_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=np.random.randint(0, 10**6))
        for tr_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            clf = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=77,
            )
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_val)
            scores.append(f1_score(y_val, y_pred, average="macro", zero_division=0))
    return float(np.mean(scores))

# ---------------- Helpers ----------------
def initialize_positions(n_agents, dim, bounds):
    lb, ub = bounds[:, 0], bounds[:, 1]
    return lb + np.random.rand(n_agents, dim) * (ub - lb)

def compute_diversity(positions, bounds):
    pop_std = np.std(positions, axis=0)
    denom = (bounds[:, 1] - bounds[:, 0]) + 1e-9
    return float(np.mean(pop_std / denom))

# ---------------- Main Fuzzy GWO ----------------
def fuzzy_gwo_optimize(X, y, bounds, n_agents=12, max_iter=40, seed=42,
                       diversity_inject_threshold=0.07, stagnation_limit=7):
    np.random.seed(seed)
    dim = bounds.shape[0]
    positions = initialize_positions(n_agents, dim, bounds)

    fitness = np.array([eval_fitness(positions[i], X, y) for i in range(n_agents)])
    sorted_idx = np.argsort(-fitness)
    alpha_pos, beta_pos, delta_pos = positions[sorted_idx[:3]].copy()
    alpha_score, beta_score, delta_score = fitness[sorted_idx[:3]]

    history = []
    max_possible_diversity = 1.0
    a_ma = 1.9  # moving average for `a`
    best_so_far = alpha_score
    stagnation_counter = 0

    for t in range(1, max_iter + 1):
        iter_ratio = t / max_iter
        diversity = compute_diversity(positions, bounds)
        diversity_norm = np.clip(diversity / max_possible_diversity, 0.0, 1.0)

        try:
            fuzzy_a = float(get_fuzzy_a(iter_ratio, diversity_norm))
        except Exception:
            fuzzy_a = max(0.0, 2 - 2 * iter_ratio)
        linear_a = max(0.0, 2 - 2 * iter_ratio)

        # --- Adaptive fuzzy control for `a` ---
        fuzzy_weight = 0.7 + 0.3 * (1 - diversity_norm)
        a_raw = fuzzy_weight * fuzzy_a + (1 - fuzzy_weight) * linear_a

        # Exponential smoothing — more responsive when diversity is low
        smooth_factor = 0.3 + 0.4 * (1 - diversity_norm)  # 0.3–0.7
        a = (1 - smooth_factor) * a_ma + smooth_factor * a_raw

        # Expand range adaptively (more range when diversity is low)
        low_clip = 0.3 if diversity_norm < 0.3 else 0.5
        high_clip = 2.2 if diversity_norm < 0.3 else 2.0
        a = float(np.clip(a, low_clip, high_clip))

        # Inject mild random jitter when diversity collapses
        if diversity_norm < 0.1:
            a += np.random.uniform(-0.1, 0.1)
            a = np.clip(a, low_clip, high_clip)

        a_ma = a

        if diversity < diversity_inject_threshold:
            scale = 0.08 + 0.12 * (1 - diversity / diversity_inject_threshold)
            n_inject = max(1, int(0.25 * n_agents))
            inject_idxs = np.random.choice(n_agents, n_inject, replace=False)
            for idx in inject_idxs:
                positions[idx] += levy_step(dim, scale=scale)
            positions = np.clip(positions, bounds[:, 0], bounds[:, 1])
            diversity = compute_diversity(positions, bounds)

        for i in range(n_agents):
            for d in range(dim):
                r1, r2 = np.random.rand(), np.random.rand()
                A1, C1 = 2 * a * r1 - a, 2 * r2
                D_alpha = abs(C1 * alpha_pos[d] - positions[i, d])
                X1 = alpha_pos[d] - A1 * D_alpha

                r1, r2 = np.random.rand(), np.random.rand()
                A2, C2 = 2 * a * r1 - a, 2 * r2
                D_beta = abs(C2 * beta_pos[d] - positions[i, d])
                X2 = beta_pos[d] - A2 * D_beta

                r1, r2 = np.random.rand(), np.random.rand()
                A3, C3 = 2 * a * r1 - a, 2 * r2
                D_delta = abs(C3 * delta_pos[d] - positions[i, d])
                X3 = delta_pos[d] - A3 * D_delta

                exploration = max(0.2, 1 - iter_ratio)
                positions[i, d] = (
                    0.7 * positions[i, d] + 0.3 * (X1 + X2 + X3) / 3.0
                    + 0.03 * np.random.randn() * exploration
                )

        positions = np.clip(positions, bounds[:, 0], bounds[:, 1])
        fitness = np.array([eval_fitness(positions[i], X, y) for i in range(n_agents)])
        sorted_idx = np.argsort(-fitness)
        alpha_pos, beta_pos, delta_pos = positions[sorted_idx[:3]].copy()
        alpha_score, beta_score, delta_score = fitness[sorted_idx[:3]]

        if alpha_score <= best_so_far:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
            best_so_far = alpha_score

        if stagnation_counter >= stagnation_limit:
            worst_idxs = sorted_idx[-3:]
            for idx in worst_idxs:
                positions[idx] = initialize_positions(1, dim, bounds)[0]
            stagnation_counter = 0
            print(f"💡 Re-seeded worst wolves at iter {t}")

        history.append({"iter": t, "best_macro_f1": float(alpha_score),
                        "diversity": float(diversity), "a": float(a)})
        print(f"Iter {t:02d}/{max_iter} | best F1: {alpha_score:.4f} | div: {diversity:.4f} | a: {a:.4f}")

    return alpha_pos, alpha_score, history

# ---------------- Run ----------------
if __name__ == "__main__":
    bounds = np.array([[1, 20], [2, 50], [1, 20], [0.1, 1.0]])
    np.random.seed(77)

    best_pos, best_score, history = fuzzy_gwo_optimize(X, y, bounds, n_agents=12, max_iter=40)

    print("\n=== Finished ===")
    print(f"Best F1 from optimizer (CV mean): {best_score:.4f}")
    print("Best position (continuous):", best_pos)

    bd = int(np.round(best_pos[0]))
    bss = int(np.round(best_pos[1]))
    bsl = int(np.round(best_pos[2]))
    bf = float(best_pos[3])
    print("Best DT hyperparameters:")
    print(f"  max_depth = {bd}\n  min_samples_split = {bss}\n  min_samples_leaf = {bsl}\n  max_features = {bf:.3f}")

    class_counts = Counter(y)
    min_class_count = min(class_counts.values()) if class_counts else 0
    n_splits = min(5, min_class_count) if min_class_count >= 2 else 2
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=77)
    final_scores = []
    for tr_idx, val_idx in skf.split(X, y):
        clf = DecisionTreeClassifier(max_depth=bd, min_samples_split=bss,
                                     min_samples_leaf=bsl, max_features=bf, random_state=77)
        clf.fit(X[tr_idx], y[tr_idx])
        y_pred = clf.predict(X[val_idx])
        final_scores.append(f1_score(y[val_idx], y_pred, average="macro", zero_division=0))
    final_mean, final_std = np.mean(final_scores), np.std(final_scores)

    print(f"\nFinal StratifiedKFold CV F1 (mean ± std): {final_mean:.4f} ± {final_std:.4f}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=77)
    clf = DecisionTreeClassifier(max_depth=bd, min_samples_split=bss, min_samples_leaf=bsl,
                                 max_features=bf, random_state=77)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print("\n=== FINAL MODEL PERFORMANCE (train/test split) ===")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1-score : {f1:.3f}")

    pd.DataFrame(history).to_csv("fuzzy_gwo_history.csv", index=False)
    import csv
    with open("model_results.csv", "a", newline="") as f:
        csv.writer(f).writerow(["Fuzzy-GWO-DT", acc, prec, rec, f1])
    print("Saved results to 'model_results.csv' and 'fuzzy_gwo_history.csv'.\nDone ✅")
