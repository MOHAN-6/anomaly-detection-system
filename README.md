# Anomaly Detection & Log Analytics System

## Features
- Real-time anomaly detection using Isolation Forest
- Log severity classification using NLP (TF-IDF + Naive Bayes)
- FastAPI REST API with <5ms response time

## Run Locally
```bash
pip install -r requirements.txt
uvicorn src.api.main:app --reload --port 8000
