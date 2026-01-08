import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv('sepsis_data_encoded.csv')
#df.drop(['studyid_adm'], axis=1, inplace=True)

# Select only numerical columns
# num_df = df.select_dtypes(include=['int64', 'float64'])

# --- Boxplot for all numerical columns ---
def plot_boxplots(df, variance_threshold=80):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Select numeric columns
    num_df = df.select_dtypes(include=['int64', 'float64'])

    filtered_cols = []
    
    for col in num_df.columns:
        series = num_df[col].dropna()
        unique_vals = series.unique()

        # Remove binary (0/1 or two unique values)
        if len(unique_vals) <= 2:
            continue  

        # Remove constant or near-constant columns
        var = np.var(series)
        if var < variance_threshold:     # low variance threshold
            continue

        filtered_cols.append(col)

    if not filtered_cols:
        print("No numerical columns with acceptable variance found.")
        return

    plt.figure(figsize=(18, 8))
    sns.boxplot(data=num_df[filtered_cols])

    plt.xticks(rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('Boxplots of Sepsis Dataset', fontsize=18)

    plt.tight_layout()
    plt.show()

# --- Correlation Heatmap ---
def plot_correlation_heatmap(df):
    num_df = df.select_dtypes(include=['int64', 'float64'])
    corr = num_df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='viridis')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()

# Example usage:
plot_boxplots(df)
##plot_correlation_heatmap(df)
