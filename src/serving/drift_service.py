import evidently
from evidently import Dataset
from evidently import metrics
from evidently import Report
import pandas as pd
import json
import os

def check_drift(train_df: pd.DataFrame, new_data: pd.DataFrame):
    # Ensure report directory exists
    os.makedirs("data/reports", exist_ok=True)

    # Data definition mapping
    data_definition = {
        "target": "target",
        "prediction": "predicted"
    }

    # Dataset objects
    train_dataset = Dataset.from_pandas(train_df, data_definition=data_definition)
    new_dataset = Dataset.from_pandas(new_data, data_definition=data_definition)

    # Create report using individual metrics
    report = Report(metrics=[
        metrics.DataDriftMetric(),
        metrics.ColumnDriftMetric(column_name="target"),
        metrics.ClassificationPerformanceMetric()
    ])
    report.run(reference_data=train_dataset, current_data=new_dataset)

    # Save JSON
    report_json = report.json()
    with open("data/reports/drift_report.json", "w") as f:
        json.dump(report_json, f, indent=4)

    # Save HTML
    report.save_html("data/reports/drift_report.html")

    # Drift detection implementation
    drift_detected = False
    for metric in report_json["metrics"]:
        if "result" in metric:
            if metric["metric"] in ["DataDriftTable", "ColumnDriftMetric"]:
                for col, res in metric.get("result", {}).get("data", {}).items():
                    if res.get("drift_detected", False):
                        drift_detected = True
                        print(f"Drift detected in column: {col}")
            if metric["metric"] == "ClassificationPerformanceMetrics":
                ref_acc = metric["result"].get("reference", {}).get("accuracy")
                curr_acc = metric["result"].get("current", {}).get("accuracy")
                if ref_acc is not None and curr_acc is not None and curr_acc < ref_acc - 0.05:
                    drift_detected = True
                    print(f"Significant accuracy drop detected: {ref_acc} -> {curr_acc}")

    return drift_detected
