from fastapi import FastAPI
from fastapi import Request
import keras
from pydantic import BaseModel
from pydantic import BaseModel
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================
#  Load model and scaler
# ================================
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "flight_delay_model.keras"
SCALER_PATH = BASE_DIR / "model" / "scaler.pkl"

origin_categories = joblib.load(BASE_DIR / "model" / "origin_categories.pkl")
dest_categories   = joblib.load(BASE_DIR / "model" / "dest_categories.pkl")

origin_map = {cat: i for i, cat in enumerate(origin_categories)}
dest_map   = {cat: i for i, cat in enumerate(dest_categories)}

model = keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ================================
#  Input Schema
# ================================
class FlightInput(BaseModel):
    ORIGIN_AIRPORT: str
    DESTINATION_AIRPORT: str
    AIRLINE: str
    YEAR: int
    MONTH: int
    DAY: int
    DAY_OF_WEEK: int
    DEPARTURE_DELAY: float
    SCHEDULED_TIME: float
    DISTANCE: float


# ================================
#  Preprocessing Function
# ================================
def preprocess(data: FlightInput) -> np.ndarray:

    df = pd.DataFrame([data.dict()])

    # -----------------------------------
    # 1. Dummy encoding for AIRLINE
    # -----------------------------------
    airline_cols = [
        'AIRLINE_AS','AIRLINE_B6','AIRLINE_DL','AIRLINE_EV','AIRLINE_F9',
        'AIRLINE_HA','AIRLINE_MQ','AIRLINE_NK','AIRLINE_OO','AIRLINE_UA',
        'AIRLINE_US','AIRLINE_VX','AIRLINE_WN'
    ]

    for col in airline_cols:
        df[col] = 0

    col_name = f"AIRLINE_{df.loc[0,'AIRLINE']}"
    if col_name in airline_cols:
        df.loc[0, col_name] = 1

    df.drop(columns=["AIRLINE"], inplace=True)

    # -----------------------------------
    # 2. Cyclic features
    # -----------------------------------
    df["MONTH_sin"] = np.sin(2 * np.pi * df["MONTH"] / 12)
    df["MONTH_cos"] = np.cos(2 * np.pi * df["MONTH"] / 12)

    df["DAY_sin"] = np.sin(2 * np.pi * df["DAY"] / 31)
    df["DAY_cos"] = np.cos(2 * np.pi * df["DAY"] / 31)

    df["DOW_sin"] = np.sin(2 * np.pi * df["DAY_OF_WEEK"] / 7)
    df["DOW_cos"] = np.cos(2 * np.pi * df["DAY_OF_WEEK"] / 7)

    df.drop(columns=["DAY","MONTH","DAY_OF_WEEK","YEAR"], inplace=True)
    
    df["ORIGIN_AIRPORT"] = df["ORIGIN_AIRPORT"].map(origin_map).fillna(-1).astype(int)
    df["DESTINATION_AIRPORT"] = df["DESTINATION_AIRPORT"].map(dest_map).fillna(-1).astype(int)

    # -----------------------------------
    # 3. Scale numeric columns
    # -----------------------------------
    to_scale = ["SCHEDULED_TIME", "DEPARTURE_DELAY", "DISTANCE"]
    df[to_scale] = scaler.transform(df[to_scale])

    # -----------------------------------
    # 4. Reorder columns exactly like training
    # -----------------------------------
    final_columns = [
        'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
        'DEPARTURE_DELAY', 'SCHEDULED_TIME', 'DISTANCE',
        'AIRLINE_AS', 'AIRLINE_B6', 'AIRLINE_DL', 'AIRLINE_EV',
        'AIRLINE_F9', 'AIRLINE_HA', 'AIRLINE_MQ', 'AIRLINE_NK',
        'AIRLINE_OO', 'AIRLINE_UA', 'AIRLINE_US', 'AIRLINE_VX',
        'AIRLINE_WN', 'MONTH_sin', 'MONTH_cos', 'DAY_sin',
        'DAY_cos', 'DOW_sin', 'DOW_cos'
    ]

    df = df[final_columns]

    return df.values.astype("float32")

@app.get("/")
def root():
    return {"message": "Welcome to the Flight Delay Prediction API!"}

# ================================
#  Prediction Endpoint
# ================================
@app.post("/predict")
def predict_delay(flight: FlightInput):
    x = preprocess(flight)
    pred = model.predict(x)[0][0]  # single output
    return {"predicted_arrival_delay_minutes": float(pred)}
