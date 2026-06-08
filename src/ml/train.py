import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from models import AnomalyDetector, LogSeverityClassifier

def generate_anomaly_data(n_samples=10000):
    normal_data = np.random.normal(0, 1, (int(n_samples * 0.95), 5))
    anomaly_data = np.random.normal(10, 2, (int(n_samples * 0.05), 5))
    return np.vstack([normal_data, anomaly_data])

def generate_log_data(n_samples=5000):
    log_messages = []
    severity_labels = []
    
    log_templates = {
        'INFO': [
            'User {} logged in successfully',
            'Request processed in {}ms',
            'Cache hit for key {}',
            'Scheduled job completed'
        ],
        'WARNING': [
            'High memory usage: {}%',
            'Slow response time: {}ms',
            'Retry attempt {}/3',
            'Deprecated API endpoint called'
        ],
        'ERROR': [
            'Database connection failed: timeout',
            'Null pointer exception in module {}',
            'Authentication failed for user {}',
            'File not found: {}'
        ],
        'CRITICAL': [
            'System outage detected',
            'Data corruption in table {}',
            'Security breach attempt from IP {}',
            'Service unavailable: {}'
        ]
    }
    
    for _ in range(n_samples):
        severity = np.random.choice(['INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                                     p=[0.6, 0.2, 0.15, 0.05])
        template = np.random.choice(log_templates[severity])
        log_messages.append(template.format(np.random.randint(1, 1000)))
        severity_labels.append(severity)
    
    return log_messages, severity_labels

if __name__ == "__main__":
    print("=" * 50)
    print("Training Anomaly Detector...")
    print("=" * 50)
    
    anomaly_data = generate_anomaly_data(10000)
    detector = AnomalyDetector(contamination=0.05)
    detector.train(anomaly_data)
    detector.save()
    print("✓ Anomaly detector trained and saved!\n")
    
    print("=" * 50)
    print("Training Log Classifier...")
    print("=" * 50)
    
    log_messages, severity_labels = generate_log_data(10000)
    classifier = LogSeverityClassifier()
    classifier.train(log_messages, severity_labels)
    classifier.save()
    print("✓ Log classifier trained and saved!\n")
    
    print("=" * 50)
    print("Training Complete!")
    print("Models saved in 'models/' folder")
    print("=" * 50)