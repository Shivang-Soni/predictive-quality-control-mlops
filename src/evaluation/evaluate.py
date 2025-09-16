import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import json 
import pickle


def evaluate_model(model_path="data/model/model.pkl",
                    test_data_path="data/processed/test.csv", 
                    report_path="data/reports/metrics.json"):
    # Load test data
    test_data = pd.read_csv(test_data_path)
    X_eval = test_data.drop("target", axis=1)
    y_eval = test_data["target"]

    # Load model
    with open(model_path, "rb") as m:
        model = pickle.load(m)

    # Predict
    y_pred = model.predict(X_eval)

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_eval, y_pred),
        "precision_macro": precision_score(y_eval, y_pred, average="macro"),
        "recall_macro": recall_score(y_eval, y_pred, average="macro"),
        "f1_macro": f1_score(y_eval, y_pred, average="macro")
    }

    # Save report
    with open(report_path, "w") as report:
       json.dump(metrics, report, indent=4)
    
    print(f"Evaluation complete! The report has been saved in {report_path}")


evaluate_model()