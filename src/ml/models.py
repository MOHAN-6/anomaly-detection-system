import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib
import os

class AnomalyDetector:
    def __init__(self, contamination=0.1):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        
    def train(self, data):
        self.model.fit(data)
        return self
    
    def predict(self, features):
        predictions = self.model.predict(features)
        return (predictions == -1).astype(int)
    
    def save(self, path="models/anomaly_model.joblib"):
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.model, path)
    
    def load(self, path="models/anomaly_model.joblib"):
        self.model = joblib.load(path)
        return self


class LogSeverityClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.classifier = MultinomialNB()
        self.label_encoder = LabelEncoder()
        
    def train(self, log_messages, severity_labels):
        X = self.vectorizer.fit_transform(log_messages)
        y = self.label_encoder.fit_transform(severity_labels)
        self.classifier.fit(X, y)
        return self
    
    def predict(self, log_messages):
        X = self.vectorizer.transform(log_messages)
        predictions = self.classifier.predict(X)
        return self.label_encoder.inverse_transform(predictions)
    
    def predict_proba(self, log_messages):
        X = self.vectorizer.transform(log_messages)
        return self.classifier.predict_proba(X)
    
    def save(self, path="models/log_classifier.joblib"):
        os.makedirs("models", exist_ok=True)
        joblib.dump({
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'label_encoder': self.label_encoder
        }, path)
    
    def load(self, path="models/log_classifier.joblib"):
        data = joblib.load(path)
        self.vectorizer = data['vectorizer']
        self.classifier = data['classifier']
        self.label_encoder = data['label_encoder']
        return self