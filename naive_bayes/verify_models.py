#!/usr/bin/env python
import joblib
import os

models_dir = os.path.join(os.getcwd(), 'models')
print(f'Models directory: {models_dir}')
print(f'Directory exists: {os.path.exists(models_dir)}')
print()

files = ['model.pkl', 'scaler.pkl', 'features.pkl', 'metrics.pkl']
for f in files:
    path = os.path.join(models_dir, f)
    exists = os.path.exists(path)
    status = "OK" if exists else "MISSING"
    print(f'{f}: {status}')
    
print()
try:
    model = joblib.load(os.path.join(models_dir, 'model.pkl'))
    scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    features = joblib.load(os.path.join(models_dir, 'features.pkl'))
    metrics = joblib.load(os.path.join(models_dir, 'metrics.pkl'))
    print("All model files loaded successfully!")
    print(f"Model: {type(model)}")
    print(f"Scaler: {type(scaler)}")
    print(f"Features: {features}")
    print(f"Metrics keys: {metrics.keys()}")
except Exception as e:
    print(f"Error loading models: {e}")
