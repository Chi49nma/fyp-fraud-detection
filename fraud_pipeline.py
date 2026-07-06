"""
=============================================================================
AI-BASED FRAUD DETECTION SYSTEM - THREE-LAYER PIPELINE
=============================================================================
Implements the three algorithmic components described in Chapter 3:
  Layer 1: Rule-Based Anomaly Filtering
  Layer 2: Machine Learning Classification (Random Forest, trained on a
           SMOTE-balanced training set to address severe class imbalance)
  Layer 3: Behavioural Pattern Analysis

Dataset: NIBSS_cleaned_sample.csv (43,000 transactions: 40,000 legitimate, 3,000 fraud)
Split:   80% train / 20% test (matches Chapter 3 methodology)

Class imbalance handling: SMOTE (Synthetic Minority Over-sampling Technique)
is applied to the TRAINING set only, generating synthetic fraud examples so
the Random Forest model is trained on a balanced 50/50 class distribution.
The test set is left untouched (no synthetic data), so evaluation reflects
real-world class proportions.

Run this script with: python fraud_pipeline.py
All output is also saved to results_output.txt
=============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (confusion_matrix, classification_report,
                              accuracy_score, precision_score, recall_score, f1_score)
from imblearn.over_sampling import SMOTE
import sys

# Send all print output to both screen AND a text file, so nothing is lost
class Tee:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Tee("results_output.txt")

print("=" * 78)
print("AI-BASED FRAUD DETECTION SYSTEM - THREE-LAYER PIPELINE RESULTS")
print("=" * 78)

# -----------------------------------------------------------------------
# STEP 0: LOAD AND PREPARE DATA
# -----------------------------------------------------------------------
print("\n--- STEP 0: Loading dataset ---")
df = pd.read_csv("NIBSS_cleaned_sample.csv")
print(f"Total transactions loaded: {len(df):,}")
print(f"Fraud cases: {df['is_fraud'].sum():,}  |  Legitimate cases: {(df['is_fraud']==0).sum():,}")

# Encode categorical columns for the ML layer (Random Forest needs numbers)
df_ml = df.copy()
categorical_cols = ['channel', 'merchant_category', 'bank', 'location', 'age_group']
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_ml[col] = le.fit_transform(df_ml[col].astype(str))
    encoders[col] = le

# Convert boolean columns to integers (True/False -> 1/0)
for col in ['is_weekend', 'is_peak_hour']:
    df_ml[col] = df_ml[col].astype(int)

# Split into train/test (80/20), preserving row order so we can map back
# to the original (non-encoded) dataframe for Layers 1 and 3.
train_idx, test_idx = train_test_split(
    df.index, test_size=0.20, random_state=42, stratify=df['is_fraud']
)
train_df = df.loc[train_idx].reset_index(drop=True)
test_df = df.loc[test_idx].reset_index(drop=True)
train_ml = df_ml.loc[train_idx].reset_index(drop=True)
test_ml = df_ml.loc[test_idx].reset_index(drop=True)

print(f"\nTraining set: {len(train_df):,} transactions ({len(train_df)/len(df)*100:.0f}%)")
print(f"Testing set:  {len(test_df):,} transactions ({len(test_df)/len(df)*100:.0f}%)")

feature_cols = [c for c in df_ml.columns if c not in ['is_fraud']]

# =========================================================================
# LAYER 1: RULE-BASED ANOMALY FILTERING
# =========================================================================
print("\n" + "=" * 78)
print("LAYER 1: RULE-BASED ANOMALY FILTERING")
print("=" * 78)
print("""
Rule applied (derived from exploratory analysis of the dataset):
  Flag a transaction as suspicious if BOTH of the following hold:
    (a) channel is one of the three highest-fraud-rate channels
        (Web, Mobile, or POS), AND
    (b) EITHER amount_vs_mean_ratio OR velocity_score is in the top 10%
        of all observed values (i.e. unusually large relative to the
        customer's normal spending pattern or transaction velocity).
""")

HIGH_RISK_CHANNELS = ['Web', 'Mobile', 'POS']
amt_threshold = train_df['amount_vs_mean_ratio'].quantile(0.90)
vel_threshold = train_df['velocity_score'].quantile(0.90)
print(f"Thresholds learned from TRAINING set only (no leakage from test set):")
print(f"  amount_vs_mean_ratio 90th percentile: {amt_threshold:.4f}")
print(f"  velocity_score 90th percentile:       {vel_threshold:.4f}")

def apply_rule_layer(data):
    high_risk_channel = data['channel'].isin(HIGH_RISK_CHANNELS)
    high_amt = data['amount_vs_mean_ratio'] >= amt_threshold
    high_vel = data['velocity_score'] >= vel_threshold
    return (high_risk_channel & (high_amt | high_vel)).astype(int)

test_df['layer1_flag'] = apply_rule_layer(test_df)

print(f"\nLayer 1 flagged {test_df['layer1_flag'].sum():,} of {len(test_df):,} "
      f"test transactions as suspicious ({test_df['layer1_flag'].mean()*100:.1f}%).")

print("\n--- Layer 1 Standalone Performance (Rule-Based Filter Only) ---")
print(confusion_matrix(test_df['is_fraud'], test_df['layer1_flag']))
print(classification_report(test_df['is_fraud'], test_df['layer1_flag'],
                             digits=3, target_names=['Legitimate (0)', 'Fraud (1)']))

# =========================================================================
# LAYER 2: MACHINE LEARNING CLASSIFICATION (RANDOM FOREST)
# =========================================================================
print("\n" + "=" * 78)
print("LAYER 2: MACHINE LEARNING CLASSIFICATION (RANDOM FOREST)")
print("=" * 78)

X_train = train_ml[feature_cols]
y_train = train_ml['is_fraud']
X_test = test_ml[feature_cols]
y_test = test_ml['is_fraud']

# -----------------------------------------------------------------------
# Apply SMOTE to the TRAINING set only, to address severe class imbalance
# (3,000 fraud vs 40,000 legitimate transactions overall). The test set is
# left untouched so that evaluation still reflects real-world proportions.
# -----------------------------------------------------------------------
print("\nApplying SMOTE (Synthetic Minority Over-sampling Technique) to "
      "balance the training dataset...")

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("Training data after SMOTE:")
print(pd.Series(y_train).value_counts())

print(f"\nTraining Random Forest on {len(X_train):,} transactions "
      f"(post-SMOTE), {len(feature_cols)} features...")
print("Configuration: n_estimators=100, trained on the SMOTE-balanced "
      "training set (synthetic oversampling of the minority fraud class "
      "is the class-imbalance handling technique used for this model).")

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
test_df['layer2_pred'] = y_pred_rf
test_df['layer2_proba'] = rf_model.predict_proba(X_test)[:, 1]

print("Model training complete.")

print("\n--- Layer 2 Standalone Performance (Random Forest Only) ---")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf, digits=3,
                             target_names=['Legitimate (0)', 'Fraud (1)']))

acc_rf = accuracy_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf)
rec_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
print(f"Summary -> Accuracy: {acc_rf:.4f} | Precision: {prec_rf:.4f} | "
      f"Recall: {rec_rf:.4f} | F-Measure: {f1_rf:.4f}")

print("\nTop 10 most important features (Random Forest feature importance):")
importances = pd.Series(rf_model.feature_importances_, index=feature_cols)
print(importances.sort_values(ascending=False).head(10).round(4))

# =========================================================================
# LAYER 3: BEHAVIOURAL PATTERN ANALYSIS
# =========================================================================
print("\n" + "=" * 78)
print("LAYER 3: BEHAVIOURAL PATTERN ANALYSIS")
print("=" * 78)
print("""
This layer evaluates contextual behavioural deviation using features that
represent a transaction's divergence from the account holder's established
pattern: amount_vs_mean_ratio (deviation from personal average spend),
velocity_score (transaction frequency deviation), channel_diversity, and
location_diversity (deviation from usual channel/location habits).

A transaction is flagged as a BEHAVIOURAL DEVIATION if its combined,
weighted behavioural-deviation score exceeds the 90th percentile observed
in the training data.
""")

def behavioural_score(data):
    # Normalise each behavioural signal to 0-1 range using training-set max,
    # then combine with equal weighting.
    amt_norm = data['amount_vs_mean_ratio'] / train_df['amount_vs_mean_ratio'].max()
    vel_norm = data['velocity_score'] / train_df['velocity_score'].max()
    chan_norm = data['channel_diversity'] / train_df['channel_diversity'].max()
    online_norm = data['online_channel_ratio']  # already 0-1
    return (amt_norm + vel_norm + chan_norm + online_norm) / 4

train_behav_scores = behavioural_score(train_df)
behav_threshold = train_behav_scores.quantile(0.90)
print(f"Behavioural deviation threshold (90th percentile of training scores): "
      f"{behav_threshold:.4f}")

test_df['behav_score'] = behavioural_score(test_df)
test_df['layer3_flag'] = (test_df['behav_score'] >= behav_threshold).astype(int)

print(f"\nLayer 3 flagged {test_df['layer3_flag'].sum():,} of {len(test_df):,} "
      f"test transactions as behaviourally deviant "
      f"({test_df['layer3_flag'].mean()*100:.1f}%).")

print("\n--- Layer 3 Standalone Performance (Behavioural Analysis Only) ---")
print(confusion_matrix(test_df['is_fraud'], test_df['layer3_flag']))
print(classification_report(test_df['is_fraud'], test_df['layer3_flag'],
                             digits=3, target_names=['Legitimate (0)', 'Fraud (1)']))

# =========================================================================
# COMBINED PIPELINE: ALL THREE LAYERS TOGETHER
# =========================================================================
print("\n" + "=" * 78)
print("COMBINED THREE-LAYER PIPELINE DECISION LOGIC")
print("=" * 78)
print("""
Final decision rule (matching Chapter 3 / Figure 3.1 flowchart):

  Every transaction is independently scored by Layer 2 (Random Forest)
  AND checked against Layer 1 (rule filter), regardless of order, since
  in a real-time system both checks can run in parallel. Layer 3 acts as
  a confirming/escalating signal rather than a second independent gate,
  consistent with its role in Section 3.2.3 (distinguishing a borderline
  ML flag from a confirmed high-risk case):

    - If Layer 2 predicts fraud:
        - If Layer 3 also confirms behavioural deviation -> REJECT
        - Otherwise -> FLAG FOR INVESTIGATION
    - If Layer 2 does NOT predict fraud:
        - If Layer 1 (rule filter) independently flags it -> FLAG FOR
          INVESTIGATION (a safety net for cases the ML model misses but
          a simple rule still catches)
        - Otherwise -> APPROVE
""")

def final_decision(row):
    if row['layer2_pred'] == 1:
        if row['layer3_flag'] == 1:
            return 'REJECT'
        else:
            return 'FLAG'
    else:
        if row['layer1_flag'] == 1:
            return 'FLAG'
        else:
            return 'APPROVE'

test_df['final_decision'] = test_df.apply(final_decision, axis=1)

print("Distribution of final pipeline decisions on the test set:")
print(test_df['final_decision'].value_counts())

# For evaluation purposes, treat REJECT and FLAG as "predicted fraud-positive"
# (both represent the pipeline catching something), APPROVE as "predicted
# legitimate" -- this mirrors how the system would behave operationally.
test_df['pipeline_pred'] = test_df['final_decision'].apply(
    lambda d: 1 if d in ['REJECT', 'FLAG'] else 0
)

print("\n--- COMBINED PIPELINE Performance (All Three Layers Together) ---")
print(confusion_matrix(test_df['is_fraud'], test_df['pipeline_pred']))
print(classification_report(test_df['is_fraud'], test_df['pipeline_pred'],
                             digits=3, target_names=['Legitimate (0)', 'Fraud (1)']))

acc_p = accuracy_score(test_df['is_fraud'], test_df['pipeline_pred'])
prec_p = precision_score(test_df['is_fraud'], test_df['pipeline_pred'])
rec_p = recall_score(test_df['is_fraud'], test_df['pipeline_pred'])
f1_p = f1_score(test_df['is_fraud'], test_df['pipeline_pred'])
print(f"Summary -> Accuracy: {acc_p:.4f} | Precision: {prec_p:.4f} | "
      f"Recall: {rec_p:.4f} | F-Measure: {f1_p:.4f}")

# =========================================================================
# FINAL COMPARISON TABLE
# =========================================================================
print("\n" + "=" * 78)
print("FINAL COMPARISON: ALL FOUR APPROACHES")
print("=" * 78)

def get_metrics(y_true, y_pred):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F-Measure': f1_score(y_true, y_pred, zero_division=0),
    }

comparison = pd.DataFrame({
    'Layer 1: Rule-Based Only': get_metrics(test_df['is_fraud'], test_df['layer1_flag']),
    'Layer 2: Random Forest Only': get_metrics(y_test, y_pred_rf),
    'Layer 3: Behavioural Only': get_metrics(test_df['is_fraud'], test_df['layer3_flag']),
    'Combined 3-Layer Pipeline': get_metrics(test_df['is_fraud'], test_df['pipeline_pred']),
}).T.round(4)

print(comparison.to_string())

print("\n" + "=" * 78)
print("END OF RESULTS")
print("=" * 78)

sys.stdout = sys.stdout.terminal  # restore normal stdout before closing log