import json
import joblib
from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel

MODEL = joblib.load("model.joblib")
with open("labels.json") as f:
    LABELS = json.load(f)
with open("input_spec.json") as f:
    SPEC = json.load(f)
FEATURES = SPEC["features"]

app = FastAPI(title="Model Inference API")

class InputData(BaseModel):
    Internet_Access: str            # "Yes" / "No"
    Attendance_Rate: float
    Assignment_Delay_Days: int
    Travel_Time_Minutes: float
    Part_Time_Job: str              # "Yes" / "No"
    Stress_Index: float
    GPA: float

class PredictResponse(BaseModel):
    prediction: str
    confidence: float

@app.get("/")
def health_check():
    return {"status": "ok", "message": "API siap menerima prediksi"}

@app.post("/predict", response_model=PredictResponse)
def predict(data: InputData):
    df = pd.DataFrame([data.model_dump()])[FEATURES]   # samakan urutan kolom dgn training
    probs = MODEL.predict_proba(df)[0]
    idx = int(probs.argmax())
    return PredictResponse(prediction=str(LABELS[str(idx)]), confidence=float(probs[idx]))
