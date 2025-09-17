from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
from prometheus_client import Counter, make_asgi_app, Gauge
from prometheus_fastapi_instrumentator import Instrumentator
from pathlib import Path
from .drift_service import check_drift
from src.alerts.email_alert import send_email_alert

# Counter
REQUEST_COUNT = Counter("total_api_requests", "Number of total API requests")
DRIFT_GAUGE = Gauge("data_drift_detected", "Data Drift flag (1 = drift detected, 0 = not detected)")

app = FastAPI(title="Defect Severity Prediction API")

# Mounting an endpoint named /metrics to localhost   
app.mount("/metrics", make_asgi_app())

# Load Model
with open("data/model/model.pkl", "rb") as m:
    model = pickle.load(m)

# Load Scaler
with open("data/processed/scaler.pkl", "rb") as s:
    scaler = pickle.load(s)


# Input Schema
class DefectInput(BaseModel):
    defect_id: int
    product_id: int
    defect_type: str
    defect_date: str
    defect_location: str
    severity: str
    inspection_method: str
    repair_cost: float


@app.post("/predict/")
def predict_defect(input_data: DefectInput):
    # Incrementing the number of requests received by one
    REQUEST_COUNT.inc()

    df = pd.DataFrame([input_data.dict()])

    # One hot encoding
    cols = ['defect_type', 'defect_location', 'severity', 'inspection_method']
    df = pd.get_dummies(df, columns=cols)

    for col in model.feature_names_in_:
        if col not in df.columns:
            df[col] = 0
    df = df[model.feature_names_in_]

    # Scale the data
    df_scaled = scaler.transform(df)
    
    prediction = model.predict(df_scaled)
    prediction = prediction[0].item() if hasattr(prediction[0], "item") else prediction[0]

    # Append record(s) to csv file
    CURRENT_BATCH_PATH = "data/processed/current_batch.csv"
    CURRENT_TRAIN_PATH = "data/processed/train.csv"
    df_to_append = df.copy()
    df_to_append["predicted"] = prediction
    if not Path(CURRENT_BATCH_PATH).exists():
        df_to_append.to_csv(CURRENT_BATCH_PATH, index=False, mode="w", header=True)
    else:
        df_to_append.to_csv(CURRENT_BATCH_PATH, index=False, mode="a", header=False)

    # Detect Data Drift
    COLUMN_MAPPING = {"target": "target", "prediction": "predicted"}
    train_df = pd.read_csv(CURRENT_TRAIN_PATH)
    new_data = pd.read_csv(CURRENT_BATCH_PATH)
    drift_detected = check_drift(train_df, new_data, COLUMN_MAPPING)
    DRIFT_GAUGE.set(1 if drift_detected else 0) 

    # Notify reciever if drift detected
    if drift_detected:
        send_email_alert(subject="Alert: Drift Detected", "In the API prediction Drift has been detected.")

    return {"Predicted severity": prediction, "Drift detected": drift_detected}
    