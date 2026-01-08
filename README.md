# 🐺 Fuzzy Grey Wolf Optimizer for Decision Tree Construction (Master’s Thesis)

This repository contains the complete experimental codebase developed for my **Master’s thesis**, which focuses on improving **Decision Tree (DT)** performance and interpretability for healthcare prediction tasks using **Grey Wolf Optimization (GWO)** and its **fuzzy-enhanced variants**.

The work systematically compares:

* A **baseline Decision Tree**
* **Standard GWO-based Decision Tree induction**
* **GWO variants without the δ-wolf**
* **Fuzzy-controlled GWO**
* **Fuzzy GWO without δ-wolf**

The goal is to evaluate how metaheuristic optimization and fuzzy control mechanisms influence **classification performance, tree simplicity, and stability**, particularly in sensitive medical datasets.

---

## 📌 Thesis Objective

Traditional Decision Trees are interpretable but rely on **greedy, locally optimal splits**, which can limit performance. Metaheuristics like GWO offer global search capability but may suffer from **premature convergence and parameter sensitivity**.

This thesis proposes and evaluates **fuzzy logic–based control strategies** within GWO to:

* Improve convergence behavior
* Reduce dependence on fixed control parameters
* Enhance predictive performance while maintaining interpretability

---

## 🧪 Experimental Pipeline

All models follow a **consistent evaluation pipeline**:

1. Data loading and preprocessing
2. Exploratory Data Analysis (EDA)
3. Baseline Decision Tree training
4. GWO-based optimization of DT parameters/splits
5. Fuzzy logic–based adaptive control (where applicable)
6. Model evaluation and comparison

---

## 🔍 Implemented Models

### 1️⃣ Baseline Decision Tree

* Standard axis-parallel DT
* Greedy split selection
* Serves as the reference model

### 2️⃣ GWO-DT

* Uses Grey Wolf Optimizer to construct/optimize the DT
* Includes α, β, and δ wolves

### 3️⃣ GWO without δ-wolf

* Removes the δ layer to reduce search noise
* Focuses on elite-driven convergence

### 4️⃣ Fuzzy Controller

* Fuzzy logic system to adapt control parameter **a** dynamically
* Inputs: iteration progress and fitness behavior
* Outputs: adaptive parameter adjustments

### 5️⃣ Fuzzy GWO-DT

* Integrates fuzzy controller with GWO
* Aims to balance exploration and exploitation more effectively

### 6️⃣ Fuzzy GWO without δ-wolf

* Combines fuzzy control with reduced wolf hierarchy
* Designed to enhance stability and convergence

---

## 📊 Evaluation Metrics

Models are evaluated using healthcare-relevant classification metrics:

* Accuracy
* Precision
* Recall (Sensitivity)
* Specificity
* F1-score (Macro / Binary)
* Area Under the Curve (AUC)

All experiments use the **same train–test split** for fair comparison.

---

## 📈 Results & Comparison

Results demonstrate that:

* GWO-based models outperform the baseline DT
* Removing the δ-wolf improves stability in some datasets
* Fuzzy-controlled GWO shows smoother convergence behavior
* Interpretability is preserved through compact DT structures

Detailed metrics and plots are available in the `compare_results/` directory.

---

## 📚 Thesis Context

This codebase supports the experimental chapters of my Master’s thesis:

> **Title:** *ENHANCED GREY WOLF OPTIMIZER FOR INTERPRETABLE DECISION TREE INDUCTION IN HEALTHCARE*

The repository is intended for **research and academic use**, with a focus on reproducibility and interpretability.

---

## 👩‍🎓 Author

**Akifa Khan**
Master’s Thesis – Decision Trees, Metaheuristics & Fuzzy Systems

---
