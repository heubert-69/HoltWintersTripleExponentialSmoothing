# HoltWintersTripleExponentialSmoothing
A Univariate Time Series Model Wrapped in a Scikit Learn Model Wrapper

A lightweight time series forecasting API built with FastAPI and powered by Statsmodels’ Holt-Winters Triple Exponential Smoothing.

This project demonstrates how to integrate classical time series models into production environments using Python, Joblib, and FastAPI, enabling fast and scalable deployment for economic, financial, or seasonal data forecasting.

🚀 Features

Trained Holt-Winters model saved via Joblib

REST API for quick forecast generation

Configurable number of forecast steps

Docker-ready deployment

Lightweight and production-ready setup

🧩 Tech Stack

Python

FastAPI

Statsmodels

Joblib

Uvicorn

⚙️ Example Usage
```bash
GET /forecast?steps=5
```

Response:
```bash
{
  "forecast": [1023.4, 1041.2, 1060.8, 1075.3, 1088.7]
}
```
🐳 Quick Start:
```bash
pip install -r requirements.txt
uvicorn app:app --reload
```
