FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY data/model/model.pkl ./data/model/model.pkl
COPY data/processed/scaler.pkl ./data/processed/scaler.pkl

ENV PYTHONPATH=/app/src

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]