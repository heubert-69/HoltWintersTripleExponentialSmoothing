from fastapi import FastAPI
import requests 
import joblib
from HoltRegressor import HoltWintersModel
import sklearn
import numpy as np

# Load your trained Holt-Winters model
model = joblib.load("HoltModel.pkl")

app = FastAPI(title="Holt-Winters Forecast API")

@app.get("/")
def root():
    return {"message": "Holt-Winters Forecast API is running!"}

@app.get("/forecast")
def forecast(steps: int = 5):
    """Predict the next `steps` values"""
    forecast = model.predict(steps)
    return {"forecast": forecast.tolist()}