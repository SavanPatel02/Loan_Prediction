import os
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


# LOAD DATA

print("\nLoading dataset...")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "Loan_default.csv")

df = pd.read_csv(DATA_PATH)



# DROP ID / HIGH CARDINALITY COLUMNS

ID_COLS = ["LoanID"]
for col in ID_COLS:
    if col in df.columns:
        df = df.drop(col, axis=1)

TARGET = "Default"

if TARGET not in df.columns:
    raise Exception(f"Target column '{TARGET}' not found")


# SPLIT FEATURES / TARGET

y = df[TARGET]
X = df.drop(TARGET, axis=1)


# FEATURE SCHEMA (LOCKED)

NUMERIC_COLS = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
CATEGORICAL_COLS = X.select_dtypes(include=["object", "string"]).columns.tolist()

print("Numeric features:", NUMERIC_COLS)
print("Categorical features:", CATEGORICAL_COLS)


# PREPROCESSING PIPELINE

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC_COLS),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True,
                max_categories=20
            ),
            CATEGORICAL_COLS
        )
    ]
)


# MODEL PIPELINE

model = Pipeline(steps=[
    ("prep", preprocess),
    ("clf", RandomForestClassifier(
        n_estimators=120,
        random_state=42,
        n_jobs=-1
    ))
])


# TRAIN MODEL

print("Training model...")
model.fit(X, y)


if os.path.exists("/model_output"):
    # Running inside Docker
    MODEL_PATH = "/model_output/model.pkl"
else:
    # Running locally
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, "api", "model.pkl")

joblib.dump(
    {
        "model": model,
        "numeric_features": NUMERIC_COLS,
        "categorical_features": CATEGORICAL_COLS,
        "feature_order": NUMERIC_COLS + CATEGORICAL_COLS
    },
    MODEL_PATH
)

print(f"MODEL AND SCHEMA SAVED SUCCESSFULLY at {MODEL_PATH}")
