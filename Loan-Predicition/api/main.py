from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import os
import psycopg2
from datetime import datetime
import json



app = FastAPI()


# GLOBALS

MODEL_PATH = "/app/model.pkl"

model = None
NUMERIC_COLS = []
CATEGORICAL_COLS = []
FEATURE_ORDER = []



#  INSERT DB CONNECTION FUNCTION HERE

def get_db_connection():
    return psycopg2.connect(
        host="db",               # service name from docker-compose
        database="loandb",
        user="loanuser",
        password="loanpass"
    )

# DATABASE INITIALIZATION

def init_db():
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            input_data JSONB,
            prediction INTEGER,
            probability FLOAT,
            created_at TIMESTAMP
        );
    """)

    conn.commit()
    cur.close()
    conn.close()


# LOAD MODEL + SCHEMA

def get_model():
    global model, NUMERIC_COLS, CATEGORICAL_COLS, FEATURE_ORDER

    try:
        if model is not None:
            return model

        if not os.path.exists(MODEL_PATH):
            print("Model file not found yet.")
            return None

        print("Loading trained model bundle...")
        bundle = joblib.load(MODEL_PATH)

        model = bundle["model"]
        NUMERIC_COLS = bundle["numeric_features"]
        CATEGORICAL_COLS = bundle["categorical_features"]
        FEATURE_ORDER = bundle["feature_order"]

        print("Model and schema loaded successfully.")
        return model

    except Exception as e:
        print(f"Model loading error: {str(e)}")
        return None



# HEALTH CHECK

@app.get("/")
def health():
    return {"status": "API running"}



# PREDICTION ENDPOINT

@app.post("/predict")
def predict(data: dict):

    m = get_model()

    if m is None:
        raise HTTPException(status_code=503, detail="Model not ready yet.")

    try:
        # Create DataFrame from request
        df = pd.DataFrame([data])

        # Enforce training schema
        df = df.reindex(columns=FEATURE_ORDER)

        # Numeric features
        for col in NUMERIC_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df[NUMERIC_COLS] = df[NUMERIC_COLS].fillna(0)

        # Categorical features
        for col in CATEGORICAL_COLS:
            df[col] = df[col].astype(str).fillna("unknown")

        # Prediction
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]

        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("""
                        INSERT INTO predictions (input_data, prediction, probability, created_at)
                        VALUES (%s, %s, %s, %s)
                    """, (
        json.dumps(data),   
        int(pred),
        float(prob),
        datetime.utcnow()
        ))

        conn.commit()
        cur.close()
        conn.close()


        return {
            "default_prediction": int(pred),
            "default_probability": float(prob)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.on_event("startup")
def startup_event():
    get_model()
    init_db()
