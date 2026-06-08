from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import time
import sys
import os
import json
import redis

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

# Connect to Redis (optional - if Redis not running, caching will be disabled)
try:
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
    redis_client.ping()
    print("✓ Connected to Redis (caching enabled)")
except:
    print("⚠ Redis not running - caching disabled")
    redis_client = None

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
    
    # Check cache if Redis is available
    predictions = None
    if redis_client:
        cache_key = f"detect:{json.dumps(request.features)}"
        cached = redis_client.get(cache_key)
        if cached:
            predictions = json.loads(cached)
            print("✓ Cache hit!")
    
    # If not in cache, compute prediction
    if predictions is None:
        predictions = detector.predict(request.features).tolist()
        if redis_client:
            redis_client.setex(cache_key, 300, json.dumps(predictions))  # Cache for 5 minutes
            print("✓ Cached new prediction")
    
    return {
        "predictions": predictions,
        "anomaly_count": sum(predictions),
        "processing_time_ms": round((time.time() - start) * 1000, 2)
    }

@app.post("/classify")
async def classify_logs(request: LogRequest):
    start = time.time()
    
    # Check cache if Redis is available
    predictions = None
    confidence = None
    
    if redis_client:
        cache_key = f"classify:{json.dumps(request.logs)}"
        cached = redis_client.get(cache_key)
        if cached:
            cached_data = json.loads(cached)
            predictions = cached_data['predictions']
            confidence = cached_data['confidence']
            print("✓ Cache hit!")
    
    # If not in cache, compute prediction
    if predictions is None:
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
        if redis_client:
            cache_data = {
                'predictions': predictions,
                'confidence': confidence
            }
            redis_client.setex(cache_key, 300, json.dumps(cache_data))  # Cache for 5 minutes
            print("✓ Cached new prediction")
    
    return {
        "predictions": predictions,
        "confidence_scores": confidence,
        "processing_time_ms": round((time.time() - start) * 1000, 2)
    }

@app.get("/health")
def health():
    redis_status = redis_client.ping() if redis_client else False
    return {
        "status": "healthy",
        "redis": redis_status
    }