# Predictive Quality Control MLOps

An end-to-end MLOps project for Predictive Quality Control using DVC, Prometheus, Grafana, Evidently AI, MLflow, FastAPI, Docker, Prometheus/Grafana, and GitHub Actions.  
Goal: Reproducible training, versioned datasets & models, model serving, and monitoring.

---

## Features
- Git + DVC: Versioning of code, data, and models  
- Data preprocessing: Download, clean CSV files, track with DVC  
- Model training: Logistic Regression, save with Pickle, track with DVC  
- MLflow: Experiment logging and model registry  
- Evaluation: Accuracy, Precision, Recall, reports  
- DVC pipeline: `extract -> preprocess -> load -> train -> evaluate -> monitor`  
- Serving: FastAPI `/predict` endpoint, Docker deployment  
- Monitoring: Prometheus Exporter + Grafana Dashboard  
- CI/CD: GitHub Actions workflow with tests, DVC Pull, Docker Build/Push  

---

## Setup

### Clone repository
```bash
git clone https://github.com/<username>/predictive-quality-control-mlops.git
cd predictive-quality-control-mlops
````

### Virtual environment & dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

---

## Data & Models with DVC

### Configure remote (e.g., MinIO, S3, GDrive)

```bash
dvc remote add -d storage s3://mybucket/mlops
dvc push   # upload data/models
dvc pull   # download data/models
```

---

## 🏋️ Training & Evaluation

### Run the pipeline

```bash
dvc repro
```

### Check results

* Metrics: `reports/metrics.json`
* Plots: `dvc plots show`
* MLflow UI:

```bash
mlflow ui
```

---

## Serving with FastAPI

### Run locally

```bash
uvicorn src.serving.main:app --reload --port 8000
```

### Example request

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"feature1": 0.5, "feature2": 1.2}'
```

---

## Deployment with Docker

### Build & Run

```bash
docker-compose up --build
```

* API available at: `http://localhost:8000`
* Prometheus: `http://localhost:9090`
* Grafana: `http://localhost:3000`

---

## Monitoring

* Prometheus collects metrics from FastAPI service
* Grafana visualizes dashboards
* Alerts can be configured (Slack/Email)

---

## CI/CD with GitHub Actions

Workflow: `.github/workflows/ci-cd.yml`

* Install dependencies
* Run unit tests
* DVC Pull (sync data/models)
* Build Docker image
* Push to GitHub Container Registry

---

## Future update ideas

* Add more models (RandomForest, XGBoost)
* Hyperparameter tuning with Optuna + MLflow
* Kubernetes deployment (EKS/GKE/AKS)

---

