"""
=============================================================================
FRAUD DETECTION PREDICTOR — WEB APP VERSION (NEAREST-MATCH)
=============================================================================
Loads the model files saved by train_model.py and runs the three-layer
ML-first fraud detection pipeline.

Instead of estimating the 18 engineered features, this version finds the
closest matching transaction in the dataset based on the 7 user inputs,
then uses that row's real pre-computed feature values. This gives much more
accurate and realistic predictions than approximations.

Called by app.py via: from predictor import predict_transaction
=============================================================================
"""

import os
import pandas as pd

# Base directory — works on both Windows and Render
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# LOAD MODEL AND SUPPORT FILES (saved by train_model.py)
# ---------------------------------------------------------------------------

MODEL    = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
ENCODERS = joblib.load(os.path.join(BASE_DIR, "label_encoders.pkl"))
FEATURES = joblib.load(os.path.join(BASE_DIR, "feature_names.pkl"))

# ---------------------------------------------------------------------------
# LOAD DATASET FOR NEAREST-MATCH LOOKUP AND THRESHOLD COMPUTATION
# ---------------------------------------------------------------------------

_df = pd.read_csv(os.path.join(BASE_DIR, "NIBSS_cleaned_sample.csv"))

# Reproduce the exact 80/20 split from train_model.py so thresholds
# are derived from training data only (no data leakage)
_X_train, _, _y_train, _ = train_test_split(
    _df.drop(columns=['is_fraud']),
    _df['is_fraud'],
    test_size=0.20,
    random_state=42,
    stratify=_df['is_fraud']
)
_train_df = _df.loc[_X_train.index].copy()

# ---------------------------------------------------------------------------
# THRESHOLDS — computed from training set only (matching fraud_pipeline.py)
# ---------------------------------------------------------------------------

AMT_THRESHOLD   = float(_train_df['amount_vs_mean_ratio'].quantile(0.90))
VEL_THRESHOLD   = float(_train_df['velocity_score'].quantile(0.90))
MAX_AMT_RATIO   = float(_train_df['amount_vs_mean_ratio'].max())
MAX_VEL         = float(_train_df['velocity_score'].max())
MAX_CHAN_DIV    = float(_train_df['channel_diversity'].max())

_train_behav = (
    (_train_df['amount_vs_mean_ratio'] / MAX_AMT_RATIO) +
    (_train_df['velocity_score']        / MAX_VEL) +
    (_train_df['channel_diversity']     / MAX_CHAN_DIV) +
    _train_df['online_channel_ratio']
) / 4
BEHAV_THRESHOLD = float(_train_behav.quantile(0.90))

# Rule-based: channels with highest fraud rate in dataset
HIGH_RISK_CHANNELS = ['Web', 'Mobile', 'POS']

# ---------------------------------------------------------------------------
# NEAREST-MATCH LOOKUP
# ---------------------------------------------------------------------------

def find_nearest_match(amount, channel, merchant_category,
                        bank, location, age_group):
    """
    Finds the closest matching transaction in the dataset based on the
    7 user-provided inputs, then returns that row's real pre-computed
    feature values (velocity_score, amount_vs_mean_ratio, tx_count_24h,
    channel_diversity, etc.).

    Matching strategy:
      1. Filter to rows where channel, merchant_category, location,
         age_group all match exactly (most important categorical signals)
      2. From those matches, find the row whose amount is closest to
         the input amount
      3. If no exact categorical match exists, fall back to just
         amount-closest row in the full dataset
    """

    # Step 1: try exact categorical match
    mask = (
        (_df['channel']            == channel) &
        (_df['merchant_category']  == merchant_category) &
        (_df['location']           == location) &
        (_df['age_group']          == age_group)
    )
    candidates = _df[mask]

    # Step 2: if no match, try just channel + merchant
    if len(candidates) == 0:
        mask2 = (
            (_df['channel']           == channel) &
            (_df['merchant_category'] == merchant_category)
        )
        candidates = _df[mask2]

    # Step 3: if still no match, use full dataset
    if len(candidates) == 0:
        candidates = _df

    # Step 4: find row with closest amount
    idx = (candidates['amount'] - amount).abs().idxmin()
    return _df.loc[idx]


# ---------------------------------------------------------------------------
# ENCODE HELPER
# ---------------------------------------------------------------------------

def _encode(column, value):
    """Encode a categorical value using the saved LabelEncoder.
    Unknown values default to 0."""
    try:
        return int(ENCODERS[column].transform([str(value)])[0])
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# TIME HELPERS
# ---------------------------------------------------------------------------

def _is_peak_hour(hour):
    """Peak banking hours: 8AM-11AM and 4PM-8PM."""
    return 1 if (8 <= hour <= 11) or (16 <= hour <= 20) else 0


# ---------------------------------------------------------------------------
# MAIN PREDICTION FUNCTION (called by app.py)
# ---------------------------------------------------------------------------

def predict_transaction(
    amount,
    bank,
    channel,
    merchant_category,
    location,
    age_group,
    transaction_time
):
    """
    Runs the full three-layer ML-first fraud detection pipeline.

    Uses nearest-match from the dataset to get real feature values
    for the 18 engineered columns, rather than rough approximations.

    Decision logic matches fraud_pipeline.py exactly:
      - Layer 2 (RF) scores every transaction FIRST
      - If RF predicts fraud:
          - Layer 3 (behavioural) confirms → REJECT
          - Otherwise → FLAG
      - If RF predicts legitimate:
          - Layer 1 (rule filter) flags it → FLAG
          - Otherwise → APPROVE
    """

    # ------------------------------------------------------------------
    # Parse time
    # ------------------------------------------------------------------
    hour         = int(str(transaction_time).split(":")[0])
    now          = datetime.now()
    day_of_week  = now.weekday()
    month        = now.month
    is_weekend   = 1 if now.weekday() >= 5 else 0
    is_peak_hour = _is_peak_hour(hour)

    # ------------------------------------------------------------------
    # Find nearest matching row in dataset and use its real features
    # ------------------------------------------------------------------
    match = find_nearest_match(
        amount, channel, merchant_category,
        bank, location, age_group
    )

    # Use real pre-computed values from the matched row
    tx_count_24h         = match['tx_count_24h']
    amount_sum_24h       = match['amount_sum_24h']
    amount_mean_7d       = match['amount_mean_7d']
    amount_std_7d        = match['amount_std_7d']
    tx_count_total       = match['tx_count_total']
    amount_mean_total    = match['amount_mean_total']
    amount_std_total     = match['amount_std_total']
    channel_diversity    = match['channel_diversity']
    location_diversity   = match['location_diversity']
    online_channel_ratio = match['online_channel_ratio']
    velocity_score       = match['velocity_score']
    merchant_risk_score  = match['merchant_risk_score']
    composite_risk       = match['composite_risk']

    # Recompute amount_vs_mean_ratio using the actual input amount
    # (so it reflects the real transaction amount, not the matched row's)
    amount_vs_mean_ratio = (
        amount / max(float(amount_mean_total), 1.0)
    )

    # ------------------------------------------------------------------
    # Build feature row for the Random Forest
    # ------------------------------------------------------------------
    row = {
        "amount":                amount,
        "channel":               _encode("channel", channel),
        "merchant_category":     _encode("merchant_category", merchant_category),
        "bank":                  _encode("bank", bank),
        "location":              _encode("location", location),
        "age_group":             _encode("age_group", age_group),
        "hour":                  hour,
        "day_of_week":           day_of_week,
        "month":                 month,
        "is_weekend":            is_weekend,
        "is_peak_hour":          is_peak_hour,
        "tx_count_24h":          tx_count_24h,
        "amount_sum_24h":        amount_sum_24h,
        "amount_mean_7d":        amount_mean_7d,
        "amount_std_7d":         amount_std_7d,
        "tx_count_total":        tx_count_total,
        "amount_mean_total":     amount_mean_total,
        "amount_std_total":      amount_std_total,
        "channel_diversity":     channel_diversity,
        "location_diversity":    location_diversity,
        "amount_vs_mean_ratio":  amount_vs_mean_ratio,
        "online_channel_ratio":  online_channel_ratio,
        "velocity_score":        velocity_score,
        "merchant_risk_score":   merchant_risk_score,
        "composite_risk":        composite_risk,
    }

    X = pd.DataFrame([[row[f] for f in FEATURES]], columns=FEATURES)

    # ------------------------------------------------------------------
    # LAYER 2: RANDOM FOREST (runs first on every transaction)
    # ------------------------------------------------------------------
    ml_prediction     = int(MODEL.predict(X)[0])
    fraud_probability = float(MODEL.predict_proba(X)[0][1])

    # ------------------------------------------------------------------
    # LAYER 1: RULE-BASED FILTER
    # (safety net — only used when RF predicts legitimate)
    # ------------------------------------------------------------------
    high_risk_channel = channel in HIGH_RISK_CHANNELS
    high_amt          = amount_vs_mean_ratio >= AMT_THRESHOLD
    high_vel          = float(velocity_score) >= VEL_THRESHOLD
    rule_flag         = bool(high_risk_channel and (high_amt or high_vel))

    # ------------------------------------------------------------------
    # LAYER 3: BEHAVIOURAL PATTERN ANALYSIS
    # (confirmation — only used when RF predicts fraud)
    # ------------------------------------------------------------------
    behaviour_score = (
        (amount_vs_mean_ratio       / MAX_AMT_RATIO) +
        (float(velocity_score)      / MAX_VEL) +
        (float(channel_diversity)   / MAX_CHAN_DIV) +
        float(online_channel_ratio)
    ) / 4
    behaviour_flag = bool(behaviour_score >= BEHAV_THRESHOLD)

    # ------------------------------------------------------------------
    # COMBINED ML-FIRST DECISION (matches fraud_pipeline.py exactly)
    # ------------------------------------------------------------------
    if ml_prediction == 1:
        decision = "REJECT" if behaviour_flag else "FLAG"
    else:
        decision = "FLAG" if rule_flag else "APPROVE"

    # ------------------------------------------------------------------
    # RETURN TO app.py / result.html
    # ------------------------------------------------------------------
    return {
        "decision":          decision,
        "fraud_probability": round(fraud_probability * 100, 2),
        "rule_flag":         rule_flag,
        "behaviour_score":   round(behaviour_score, 4),
        "behaviour_flag":    behaviour_flag,
    }