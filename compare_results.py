import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load model results
df = pd.read_csv(
    "results_combined.csv",
    header=None,
    names=["Model", "Accuracy", "Precision", "Recall", "F1"]
)

# Metrics and models
metrics = ["Accuracy", "Precision", "Recall", "F1"]
models = df["Model"].tolist()
n_models = len(models)

# Bar settings
x = np.arange(len(metrics))
bar_width = 0.8 / n_models  # automatically adjust width based on model count
colors = plt.cm.tab10.colors  # good palette for up to 10 models

# Create plot
plt.figure(figsize=(12, 7))

# Plot bars for each model
for i, model in enumerate(models):
    bars = plt.bar(
        x + i * bar_width,
        df.loc[i, metrics],
        width=bar_width,
        label=model,
        color=colors[i % len(colors)]
    )
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

# Styling
plt.xticks(x + bar_width * (n_models - 1) / 2, metrics, fontsize=12)
plt.ylabel("Score", fontsize=12)
plt.ylim(0, 1.05)
plt.title("Performance Comparison on Sepsis Dataset", fontsize=14, pad=15)
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()
