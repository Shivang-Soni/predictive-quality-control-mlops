from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
import requests
import uvicorn
from src.serving.app import app

# Activate Prometheus Endpoint
Instrumentator().instrument(app).expose(app)


# Sample Endpoint for testing
@app.get("/")
def home():
    return {"message": "Hello World!"}

@app.on_event("startup")
async def start_monitoring():
    # Activate Monitoring
    Instrumentator().instrument(app).expose(app)

    uvicorn.run(app, host="0.0.0.0.0", port=8000)