from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.models import AnomalyDetector, LogSeverityClassifier

app = FastAPI(title="Anomaly Detection & Log Analytics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading models...")
detector = AnomalyDetector()
classifier = LogSeverityClassifier()
detector.load("models/anomaly_model.joblib")
classifier.load("models/log_classifier.joblib")
print("✓ Models loaded!")

class FeaturesRequest(BaseModel):
    features: List[List[float]]

class LogRequest(BaseModel):
    logs: List[str]

@app.get("/")
def root():
    return {"message": "Anomaly Detection API", "status": "running"}

@app.post("/detect")
async def detect_anomalies(request: FeaturesRequest):
    start = time.time()
    predictions = detector.predict(request.features).tolist()
    return {
        "predictions": predictions,
        "anomaly_count": sum(predictions),
        "processing_time_ms": round((time.time() - start) * 1000, 2)
    }

@app.post("/classify")
async def classify_logs(request: LogRequest):
    start = time.time()
    predictions = classifier.predict(request.logs).tolist()
    proba = classifier.predict_proba(request.logs).tolist()
    confidence = []
    for scores in proba:
        confidence.append({
            'INFO': round(scores[0], 3),
            'WARNING': round(scores[1], 3),
            'ERROR': round(scores[2], 3),
            'CRITICAL': round(scores[3], 3)
        })
    return {
        "predictions": predictions,
        "confidence_scores": confidence,
        "processing_time_ms": round((time.time() - start) * 1000, 2)
    }

@app.get("/health")
def health():
    return {"status": "healthy"}