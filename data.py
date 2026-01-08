import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from imblearn.over_sampling import SMOTE
from collections import Counter

# --- Load dataset ---
df = pd.read_csv("Sepsis.csv", encoding="windows-1252")
print("Original columns:", df.columns.tolist())

# --- Clean column names ---
df.columns = df.columns.str.strip().str.lower()

# --- Select only relevant columns ---
selected_features = [ "agecalc_adm", "height_cm_adm", "hr_bpm_adm", "lengthadm", "glucose_mmolpl_adm", "rr_brpm_app_adm", 
                     "diasbp_mmhg_adm", "hematocrit_gpdl_adm", "weight_kg_adm", "muac_mm_adm", "exclbreastfed_adm", "bcseye_adm",
                       "sysbp_mmhg_adm", "temp_c_adm", "lactate_mmolpl_adm", "spo2site2_pc_oxi_adm", "sqi2_perc_oxi_adm", "momage_adm", 
                       "householdsize_adm", "momagefirstpreg_adm", "sqi1_perc_oxi_adm", "bcsmotor_adm", "spo2site1_pc_oxi_adm", "spo2onoxy_adm",
                         "feedingstatus_adm", "alivechildren_adm", "totalbreastfed_adm", "oxygenavail_adm", "watersource_adm", 
                         "deadchildren_adm", "vaccdpt_adm", "lightfuel_adm", "deliveryloc_adm", "bednet_adm", "diarrheaoften_adm",
                           "vaccpneumoc_adm", "symptoms_adm___3", "vaccmeasles_adm", "sex_adm", "inhospital_mortality" ]

#selected_features =  ['agecalc_adm','height_cm_adm','hr_bpm_adm','glucose_mmolpl_adm','diasbp_mmhg_adm',
 #   'weight_kg_adm','bcseye_adm','sysbp_mmhg_adm','rr_brpm_app_adm','lactate_mmolpl_adm',
  #  'hematocrit_gpdl_adm','bcsverbal_adm','temp_c_adm','spo2site2_pc_oxi_adm',
   # 'exclbreastfed_adm','feedingstatus_adm','watersource_adm','deadchildren_adm',
    #'spo2site1_pc_oxi_adm','deliveryloc_adm','priorweekantimal_adm','malariastatuspos_adm',
   # 'vaccdpt_adm','waterpure_adm','sex_adm','sqi2_perc_oxi_adm','respdistress_adm','muac_mm_adm',
   # 'momage_adm','birthdetail_adm___5','birthdetail_adm___4','vaccpneumoc_adm','birthdetail_adm___1',
   # 'birthdetail_adm___2','birthdetail_adm___3','sqi1_perc_oxi_adm','oxygenavail_adm',
    #'symptoms_adm___3','priorweekabx_adm','bcsmotor_adm','bcgscar_adm','symptoms_adm___9','inhospital_mortality']

df = df[selected_features]
#print(f"\n✅ Using {len(selected_features)} selected columns.")

# --- Encode categorical columns ---
label_encoders = {}
for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"Encoded column: {col}")

# --- Separate features and target ---
X = df.drop(columns=["inhospital_mortality"])
y = df["inhospital_mortality"]

# --- Handle missing values (advanced imputation) ---
#print("\nMissing value summary before imputation:")
#print(X.isna().sum()[X.isna().sum() > 0])

# Use IterativeImputer (multivariate imputation, better than median)
#imputer = IterativeImputer(random_state=42, max_iter=10, initial_strategy="median")
#X_imputed = imputer.fit_transform(X)
#X = pd.DataFrame(X_imputed, columns=X.columns)

#print("\n✅ Missing values imputed using IterativeImputer (MICE-style predictive fill).")

# Optional interpolation pass (for smoother continuous trends)
#X = X.interpolate(method="linear", limit_direction="both", axis=0)

# --- Check class imbalance ---
#print("\nBefore SMOTE:", Counter(y))

# --- Smarter SMOTE ---
# Ensure minority class reaches 35% of majority (not full equalization)
#smote = SMOTE(sampling_strategy=0.35, random_state=42, k_neighbors=5)
#X_res, y_res = smote.fit_resample(X, y)

#print("After SMOTE :", Counter(y_res))

# --- Save processed dataset ---
df_processed = pd.concat([
    pd.DataFrame(X, columns=X.columns),
    pd.Series(y, name="inhospital_mortality")
], axis=1)

df_processed.to_csv("sepsis_data_encoded.csv", index=False)
print("\n✅ Saved final processed & balanced dataset as 'sepsis_data_encoded.csv'")
print(f"Final shape: {df_processed.shape}")

# print("\nSample data:\n", df_processed.head())
