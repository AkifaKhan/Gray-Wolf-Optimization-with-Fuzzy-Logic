# baseline_dt.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder
import csv

# --- Load dataset ---
df = pd.read_csv("sepsis_data_encoded.csv")
print("Original columns:", df.columns.tolist())


# --- Encode categorical columns automatically (safety check) ---
for col in df.columns:
    if df[col].dtype == 'object' or isinstance(df[col].iloc[0], str):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        print(f"Encoded column: {col}")

# --- Clean column names ---
df.columns = df.columns.str.strip().str.lower()

# --- Verify target column ---
if "inhospital_mortality" not in df.columns:
    raise ValueError(f"Target column 'inhospital_mortality' not found. Found: {df.columns.tolist()}")

# --- Define features (X) and target (y) ---
X = df.drop(columns=["inhospital_mortality"]).values
y = df["inhospital_mortality"].values
print(f"\n✅ Loaded Sepsis dataset with {X.shape[0]} rows and {X.shape[1]} features.")

# --- Split dataset ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=77
)

# --- Train baseline Decision Tree ---
dt = DecisionTreeClassifier(random_state=77)
dt.fit(X_train, y_train)

# --- Predict and evaluate ---
y_pred = dt.predict(X_test)
y_prob = dt.predict_proba(X_test)[:, 1] if len(dt.classes_) == 2 else None

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro')
auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

# --- Print results ---
print("\n=== BASELINE DECISION TREE RESULTS (SEPSIS DATA) ===")
print(f"Accuracy : {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall   : {recall:.3f}")
print(f"F1-score : {f1:.3f}")


print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- Save results to CSV for comparison ---
with open("model_results.csv", "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Default-DT-Sepsis", accuracy, precision, recall, f1])

print("\n✅ Results saved to 'model_results.csv'")

# Check class balance (counts)
print("Class Counts:")
print(df['inhospital_mortality'].value_counts())

# Check class percentages
print("\nClass Percentages:")
print(df['inhospital_mortality'].value_counts(normalize=True) * 100)
