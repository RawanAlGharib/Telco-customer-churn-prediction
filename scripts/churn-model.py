import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import kagglehub
import os

# --- STEP 1: DATA ACQUISITION ---
print("Downloading latest dataset from Kaggle...")
path = kagglehub.dataset_download("blastchar/telco-customer-churn")
csv_path = os.path.join(path, "WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Load the dataframe
df = pd.read_csv(csv_path)

# --- STEP 2: DATA CLEANING (ECONOMIC FIXES) ---
# TotalCharges is imported as a string; convert to numeric and drop 11 empty rows
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(subset=['TotalCharges'], inplace=True)

# Create a binary column for math (1 = Churn, 0 = Stay)
df['Churn_Numeric'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

print(f"Dataset Loaded & Cleaned: {df.shape[0]} rows ready for analysis.\n")

# --- STEP 3: EXPLORATORY DATA ANALYSIS (EDA) ---

# 1. Contract Type Analysis (The 'Incentive' Lever)
plt.figure(figsize=(10, 5))
sns.barplot(x='Contract', y='Churn_Numeric', data=df, palette='magma', ci=None)
plt.title('Churn Rate by Contract Type', fontsize=14)
plt.ylabel('Churn Probability (0 to 1)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 2. Tenure vs. Churn (The 'Loyalty' Lever)
plt.figure(figsize=(10, 5))
sns.kdeplot(df[df['Churn'] == 'Yes']['tenure'], label='Leavers (Churn)', fill=True, color='red', alpha=0.5)
sns.kdeplot(df[df['Churn'] == 'No']['tenure'], label='Stayers', fill=True, color='blue', alpha=0.5)
plt.title('Tenure Distribution: When do customers leave?', fontsize=14)
plt.xlabel('Months with Company')
plt.ylabel('Density')
plt.legend()
plt.show()

# 3. Internet Service vs. Churn (The 'Product' Lever)
plt.figure(figsize=(10, 5))
sns.barplot(x='InternetService', y='Churn_Numeric', data=df, palette='viridis', ci=None)
plt.title('Churn Rate by Internet Service Type', fontsize=14)
plt.ylabel('Churn Probability')
plt.show()

# --- STEP 4: SUMMARY STATISTICS ---
print("--- SUMMARY STATISTICS ---")
contract_churn = df.groupby('Contract')['Churn_Numeric'].mean() * 100
print("Churn Rate per Contract Type (%):")
print(contract_churn)

print("\nAverage Tenure of Leavers:", round(df[df['Churn']=='Yes']['tenure'].mean(), 2), "months")
print("Average Tenure of Stayers:", round(df[df['Churn']=='No']['tenure'].mean(), 2), "months")


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- STEP 4: ENCODING (TEXT TO NUMBERS) ---

# 1. Drop CustomerID (It's a unique label, not a predictor)
df_ml = df.drop(columns=['customerID', 'Churn_Numeric'])

# 2. Binary Encoding (Yes/No to 1/0)
# Columns with only two options (Gender, Partner, Dependents, etc.)
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    df_ml[col] = df_ml[col].apply(lambda x: 1 if x in ['Yes', 'Female'] else 0)

# 3. One-Hot Encoding (Multiple categories to separate columns)
# This handles 'InternetService', 'Contract', 'PaymentMethod', etc.
df_ml = pd.get_dummies(df_ml, drop_first=True)

print(f"Features after encoding: {df_ml.shape[1]}")

# --- STEP 5: FEATURE SCALING ---

# We use StandardScaler so all numbers have a mean of 0 and variance of 1
scaler = StandardScaler()
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
df_ml[num_cols] = scaler.fit_transform(df_ml[num_cols])

# --- STEP 6: TRAIN-TEST SPLIT ---

# X = Everything except the target (Churn)
# y = The target (Churn)
X = df_ml.drop('Churn', axis=1)
y = df_ml['Churn']

# We split 80% for training and 20% for testing the AI's "final exam"
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("--- PHASE 2 COMPLETE ---")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# --- STEP 7: TRAIN THE RANDOM FOREST ---
# n_estimators=100 means we are building 100 individual decision trees
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# --- STEP 8: EVALUATE THE PERFORMANCE ---
y_pred = rf_model.predict(X_test)

print("--- MODEL EVALUATION ---")
print(classification_report(y_test, y_pred))

# --- STEP 9: FEATURE IMPORTANCE (THE BUSINESS INSIGHT) ---
# This is where we see which "levers" the AI prioritized
importances = rf_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='rocket')
plt.title('Top 10 Business Drivers of Churn (AI-Identified)')
plt.show()

# --- STEP 10: GENERATE PROBABILITY FOR TABLEAU ---
# We add the probability back to the original clean dataframe
# Column [:, 1] is the probability of Churning (Class 1)
df['Churn_Probability'] = rf_model.predict_proba(X)[:, 1]

# Save for Phase 4
df.to_csv('AI_Telco_Churn_Results.csv', index=False)
print("Phase 3 Complete: AI Results saved to 'AI_Telco_Churn_Results.csv'")
