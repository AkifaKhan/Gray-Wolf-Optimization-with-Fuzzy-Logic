"""
Standard GWO-based Decision Tree optimizer (gwo_dt.py)

- Optimizes DecisionTree hyperparameters using the original Grey Wolf Optimizer.
- Uses a simple linear schedule for exploration weight `a = 2 - 2*(iteration/max_iter)`.

Expected input file:
    heart.csv  (encoded or numeric dataset with binary target 'target')

Optimized hyperparameters:
    position[0] -> max_depth         (int, 1..20)
    position[1] -> min_samples_split (int, 2..50)
    position[2] -> min_samples_leaf  (int, 1..20)
    position[3] -> max_features      (float, 0.1..1.0)

Fitness:
    Stratified CV macro-F1 score (3 folds or fallback to single split)

Run:
    python gwo_dt.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# ------------------- Data loading -------------------
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

# ------------------- Fitness function -------------------
def eval_fitness(position, X, y):
    """Evaluate F1-score fitness for given GWO position"""
    max_depth = int(np.round(position[0])); max_depth = np.clip(max_depth, 1, 20)
    min_samples_split = int(np.round(position[1])); min_samples_split = np.clip(min_samples_split, 2, 50)
    min_samples_leaf = int(np.round(position[2])); min_samples_leaf = np.clip(min_samples_leaf, 1, 20)
    max_features = float(position[3]); max_features = np.clip(max_features, 0.1, 1.0)

    class_counts = Counter(y)
    min_class_count = min(class_counts.values()) if class_counts else 0
    n_splits = min(3, min_class_count) if min_class_count >= 2 else 2

    try:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []
        for tr_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            clf = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=4
            )
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_val)
            scores.append(f1_score(y_val, y_pred, average='macro', zero_division=0))
        return float(np.mean(scores))
    except Exception:
        # fallback single split
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.3, stratify=y, random_state=4)
        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=77
        )
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_val)
        return float(f1_score(y_val, y_pred, average='macro', zero_division=0))

# ------------------- GWO Core -------------------
def initialize_positions(n_agents, dim, bounds):
    lb = bounds[:, 0]
    ub = bounds[:, 1]
    return lb + np.random.rand(n_agents, dim) * (ub - lb)

def gwo_optimize(X, y, bounds, n_agents=12, max_iter=40):
    dim = bounds.shape[0]
    positions = initialize_positions(n_agents, dim, bounds)

    fitness = np.array([eval_fitness(positions[i], X, y) for i in range(n_agents)])
    sorted_idx = np.argsort(-fitness)

    alpha_pos = positions[sorted_idx[0]].copy(); alpha_score = fitness[sorted_idx[0]]
    beta_pos  = positions[sorted_idx[1]].copy(); beta_score  = fitness[sorted_idx[1]]
    delta_pos = positions[sorted_idx[2]].copy(); delta_score = fitness[sorted_idx[2]]

    history = [alpha_score]

    for t in range(1, max_iter + 1):
        a = 2 - 2 * (t / max_iter)  # Linear schedule for 'a'

        for i in range(n_agents):
            for d in range(dim):
                r1, r2 = np.random.rand(), np.random.rand()
                A1 = 2 * a * r1 - a; C1 = 2 * r2
                D_alpha = abs(C1 * alpha_pos[d] - positions[i, d])
                X1 = alpha_pos[d] - A1 * D_alpha

                r1, r2 = np.random.rand(), np.random.rand()
                A2 = 2 * a * r1 - a; C2 = 2 * r2
                D_beta = abs(C2 * beta_pos[d] - positions[i, d])
                X2 = beta_pos[d] - A2 * D_beta

                r1, r2 = np.random.rand(), np.random.rand()
                A3 = 2 * a * r1 - a; C3 = 2 * r2
                D_delta = abs(C3 * delta_pos[d] - positions[i, d])
                X3 = delta_pos[d] - A3 * D_delta

                positions[i, d] = (X1 + X2 + X3) / 3.0

        lb, ub = bounds[:, 0], bounds[:, 1]
        positions = np.clip(positions, lb, ub)

        fitness = np.array([eval_fitness(positions[i], X, y) for i in range(n_agents)])
        sorted_idx = np.argsort(-fitness)
        alpha_pos = positions[sorted_idx[0]].copy(); alpha_score = fitness[sorted_idx[0]]
        beta_pos  = positions[sorted_idx[1]].copy(); beta_score  = fitness[sorted_idx[1]]
        delta_pos = positions[sorted_idx[2]].copy(); delta_score = fitness[sorted_idx[2]]

        history.append(alpha_score)
        print(f"Iter {t}/{max_iter} | best F1: {alpha_score:.4f} | a: {a:.4f}")

    return alpha_pos, alpha_score, history

# ------------------- Run Optimization -------------------
if __name__ == "__main__":
    np.random.seed(77)
    bounds = np.array([[1, 20], [2, 50], [1, 20], [0.1, 1.0]])

    best_pos, best_score, history = gwo_optimize(X, y, bounds)

    print("\n=== Optimization Complete ===")
    print(f"Best F1 Score: {best_score:.4f}")
    print(f"Best Position: {best_pos}")

    best_max_depth = int(np.round(best_pos[0]))
    best_min_split = int(np.round(best_pos[1]))
    best_min_leaf = int(np.round(best_pos[2]))
    best_max_feat = float(best_pos[3])

    print("Best Decision Tree Hyperparameters:")
    print(f"  max_depth = {best_max_depth}")
    print(f"  min_samples_split = {best_min_split}")
    print(f"  min_samples_leaf = {best_min_leaf}")
    print(f"  max_features = {best_max_feat:.3f}")

    # Train final DT
    clf = DecisionTreeClassifier(
        max_depth=best_max_depth,
        min_samples_split=best_min_split,
        min_samples_leaf=best_min_leaf,
        max_features=best_max_feat,
        random_state=77
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=77)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    print("\n=== FINAL MODEL PERFORMANCE ===")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1-score : {f1:.3f}")

    # Save results
    import csv
    with open("model_results.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["GWO-DT (Heart)", acc, prec, rec, f1])

    with open("gwo_history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "best_macro_f1"])
        for i, val in enumerate(history):
            writer.writerow([i, val])

    print("\n✅ Saved optimization history to 'gwo_history.csv'")
