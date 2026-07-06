import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

print("="*60)
print("TRAINING AI FRAUD DETECTION MODEL")
print("="*60)

# Load dataset
df = pd.read_csv("NIBSS_cleaned_sample.csv")

# Encode categorical variables
categorical_cols = [
    'channel',
    'merchant_category',
    'bank',
    'location',
    'age_group'
]

encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Convert boolean columns
for col in ['is_weekend', 'is_peak_hour']:
    if col in df.columns:
        df[col] = df[col].astype(int)

# Features and target
X = df.drop(columns=["is_fraud"])
y = df["is_fraud"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Apply SMOTE
print("Applying SMOTE...")
smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("Training Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_smote, y_train_smote)

print("Saving model...")

joblib.dump(model, "model.pkl")
joblib.dump(encoders, "label_encoders.pkl")
joblib.dump(list(X.columns), "feature_names.pkl")

print()
print("SUCCESS!")
print("Files created:")
print("model.pkl")
print("label_encoders.pkl")
print("feature_names.pkl")
print("="*60)